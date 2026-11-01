#!/usr/bin/env python3
"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/

Compile the backend-neutral warp-only artifact: NV12 camera frames + 3x3
transforms -> (2, 6, model_h/2, model_w/2) uint8 warped tensor, on the device
GPU (QCOM). maciqmodeld runs this locally
and feed the output to their backend, so the big model's image pipeline is
bit-identical to comma's fused pkl warp stage.

Run ON the device (needs the QCOM backend):
  cd /data/openpilot && DEV=QCOM WARP_DEV=QCOM IMAGE=1 FLOAT16=1 NOLOCALS=1 JIT_BATCH_SIZE=0 \
    python3 iqpilot/selfdrive/iqmodeld/tools/compile_warp.py \
    --camera-resolutions 1928x1208 --output /data/models/emac_warp.pkl
The artifact is then split per-resolution into Paths.model_root().
"""
from __future__ import annotations

import argparse
import hashlib
import os
import pickle
from functools import partial

import numpy as np

SELFTEST_SEED = 20260817

from iqpilot.selfdrive.iqmodeld.temporal_state import DEFAULT_FRAME_SKIP, MODEL_INPUT_SPEC
from iqpilot.selfdrive.iqmodeld.tools.compile_supercombo import (
  NV12Frame, WARP_INPUTS, compile_jit, make_random_images, make_warp, make_warp_input_queues,
)

MODEL_SIZE = (MODEL_INPUT_SPEC["img"][0][3] * 2, MODEL_INPUT_SPEC["img"][0][2] * 2)  # (512, 256)


def _parse_size(s: str) -> tuple[int, int]:
  w, h = s.lower().split("x")
  return int(w), int(h)


def compile_warp(cam_w: int, cam_h: int, out_path: str | None = None,
                 frame_skip: int = DEFAULT_FRAME_SKIP) -> str:
  """Compile the warp-only QCOM JIT for one camera resolution and write the pkl.
  Returns the artifact path. Callable from the workers so a fresh device
  self-provisions the warp instead of erroring — needs the QCOM backend."""
  # the QCOM warp env must be set before tinygrad is imported here
  os.environ.setdefault("DEV", "QCOM")
  os.environ.setdefault("WARP_DEV", "QCOM")
  os.environ.setdefault("IMAGE", "1")
  os.environ.setdefault("FLOAT16", "1")
  os.environ.setdefault("NOLOCALS", "1")
  os.environ.setdefault("JIT_BATCH_SIZE", "0")
  from tinygrad.engine.jit import TinyJit
  from iqpilot.system.camerad.cameras.nv12_info import get_nv12_info
  from iqpilot.system.hardware.hw import Paths

  model_w, model_h = MODEL_SIZE
  input_shapes = {name: shape for name, (shape, _) in MODEL_INPUT_SPEC.items()}
  nv12 = NV12Frame(cam_w, cam_h, *get_nv12_info(cam_w, cam_h))
  make_random_warp_inputs = partial(make_random_images, keys=["frame", "big_frame"],
                                    shape=nv12.size, device=os.getenv("WARP_DEV"))
  warp_jit = TinyJit(make_warp(nv12, model_w, model_h, frame_skip), prune=True)
  make_warp_queues = partial(make_warp_input_queues, input_shapes, frame_skip)
  compiled = compile_jit(warp_jit, make_random_warp_inputs, WARP_INPUTS, make_warp_queues)

  # historical artifact name: already-provisioned devices keep their warp
  out_path = out_path or os.path.join(Paths.model_root(), f"emac_warp_{cam_w}x{cam_h}_tinygrad.pkl")
  os.makedirs(os.path.dirname(out_path), exist_ok=True)
  tmp = out_path + ".part"
  bundle = {(cam_w, cam_h): compiled, "frame_skip": frame_skip, "model_size": MODEL_SIZE}
  bundle["selftest"] = selftest_digest(compiled, cam_w, cam_h, nv12.size)
  with open(tmp, "wb") as f:
    pickle.dump(bundle, f)
  os.replace(tmp, out_path)  # atomic: a reader never sees a half-written pkl
  return out_path


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--camera-resolutions", type=_parse_size, nargs="+", default=[(1928, 1208)])
  p.add_argument("--output", default=None)
  p.add_argument("--frame-skip", type=int, default=DEFAULT_FRAME_SKIP)
  args = p.parse_args()
  for cam_w, cam_h in args.camera_resolutions:
    out = compile_warp(cam_w, cam_h, args.output, frame_skip=args.frame_skip)
    print(f"saved warp JIT to {out} ({os.path.getsize(out) / 1e6:.2f} MB)")




def selftest_inputs(cam_w: int, cam_h: int, nv12_size: int):
  """A fixed synthetic frame pair and pair of matrices. Deterministic so the
  digest is reproducible on the device that compiled the artifact."""
  rng = np.random.default_rng(SELFTEST_SEED)
  frame = rng.integers(0, 256, nv12_size, dtype=np.uint8)
  big_frame = rng.integers(0, 256, nv12_size, dtype=np.uint8)
  tfm = np.array([[0.7, 0.02, 300.0], [0.01, 0.7, 240.0], [0.0, 0.0, 1.0]], dtype=np.float32)
  big_tfm = np.array([[0.5, 0.01, 380.0], [0.02, 0.5, 300.0], [0.0, 0.0, 1.0]], dtype=np.float32)
  return frame, big_frame, tfm, big_tfm


def selftest_digest(compiled, cam_w: int, cam_h: int, nv12_size: int) -> str:
  """Hash the warp's output for a fixed input.

  A warp artifact pinned to one tinygrad can still unpickle under another and
  then compute silently wrong, which reaches the model as a garbage image and
  looks like a bad model rather than a stale artifact. A version string cannot
  see that; running it can."""
  from tinygrad.tensor import Tensor
  frame, big_frame, tfm, big_tfm = selftest_inputs(cam_w, cam_h, nv12_size)
  dev = os.getenv("WARP_DEV") or "QCOM"
  out = compiled(tfm=Tensor(tfm, device="NPY").realize(),
                 big_tfm=Tensor(big_tfm, device="NPY").realize(),
                 frame=Tensor(frame, device=dev).realize(),
                 big_frame=Tensor(big_frame, device=dev).realize())
  return hashlib.sha256(out.numpy().astype(np.uint8).tobytes()).hexdigest()


if __name__ == "__main__":
  main()
