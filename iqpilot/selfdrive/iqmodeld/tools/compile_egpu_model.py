"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""
from __future__ import annotations

import argparse
import gc
import os
import pickle
import shutil
import sys
import time

os.environ.setdefault("FLOAT16", "1")
os.environ.setdefault("JIT_BATCH_SIZE", "0")
os.environ.setdefault("GMMU", "0")
# TC_OPT=2 lets tinygrad pick tensor-core kernels; on some models a TC kernel miscompiles and biases
# the output (documented on Metal). A parity gate below catches it and ships the TC-off build instead.
os.environ.setdefault("TC_OPT", "0" if ("--tc-off" in sys.argv or os.environ.get("IQ_EGPU_TC_OFF")) else "2")

HOST = "--host" in sys.argv
if HOST:
  from iqpilot.selfdrive.iqmodeld.tools.egpu_host_mock import DEFAULT_ARCH, activate
  activate(sys.argv[sys.argv.index("--arch") + 1] if "--arch" in sys.argv else DEFAULT_ARCH)
os.environ.setdefault("DEV", "USB+AMD:LLVM")

import numpy as np

from iqpilot.selfdrive.iqmodeld.egpu_helpers import egpu_pkl_path, local_onnx, patch_tinygrad_fetch_fw, patch_tinygrad_prune_views
from iqpilot.selfdrive.iqmodeld.egpu_model import EGPU_MODELS, get_egpu_model, resolve_egpu_model
from iqpilot.selfdrive.iqmodeld.temporal_state import MODEL_INPUT_SPEC, spec_from_meta

INPUT_SPEC = dict(MODEL_INPUT_SPEC)

patch_tinygrad_fetch_fw()
patch_tinygrad_prune_views()

SEED = 42


class _ParityFail(RuntimeError):
  pass
KERNEL_PROGRESS_SCALE = 260.0


def _progress_sampler(param: str, base: float, span: float, stop) -> None:
  import math

  from iqpilot.common.params import Params
  from tinygrad.helpers import GlobalCounters
  pm = Params()
  last = -1.0
  while not stop.wait(0.5):
    kernels = float(getattr(GlobalCounters, "kernel_count", 0))
    value = base + span * (1.0 - math.exp(-kernels / KERNEL_PROGRESS_SCALE))
    if value - last >= 0.01:
      last = value
      pm.put(param, f"{min(base + span, value):.3f}")


def set_input_spec(meta: dict) -> None:
  spec = spec_from_meta(meta)
  if spec is not None:
    INPUT_SPEC.clear()
    INPUT_SPEC.update(spec)


def make_run_model(model_runner):
  def run_model(**inputs):
    out = next(iter(model_runner({k: inputs[k] for k in INPUT_SPEC}).values())).cast("float32")
    return out.reshape(-1),
  return run_model


def _random_inputs(seed: int):
  from tinygrad.device import Device
  from tinygrad.tensor import Tensor
  rng = np.random.default_rng(seed)
  out = {}
  for name, (shape, dtype) in INPUT_SPEC.items():
    if dtype == "uint8":
      arr = rng.integers(0, 256, shape).astype(np.uint8)
    else:
      arr = rng.standard_normal(shape).astype(np.float32)
    out[name] = Tensor(arr, device=Device.DEFAULT).realize()
  return out


def _run(fn, seed: int) -> np.ndarray:
  from tinygrad.device import Device
  st = time.perf_counter()
  outs = fn(**_random_inputs(seed))
  Device.default.synchronize()
  print(f"  run(seed={seed}) {(time.perf_counter() - st) * 1e3:6.1f} ms")
  return outs[0].numpy().reshape(-1)


def compile_model(meta: dict, onnx_path: str, out_path: str) -> str:
  from tinygrad.device import Device
  from tinygrad.engine.jit import TinyJit
  from tinygrad.nn.onnx import OnnxRunner

  if meta.get("split"):
    raise RuntimeError(f"model {meta['key']} is a split model; eGPU v1 compiles fused models only")

  jit = TinyJit(make_run_model(OnnxRunner(onnx_path)), prune=True)

  print("capture + replay")
  for _ in range(2):
    baseline = _run(jit, SEED)
  if baseline.shape[0] != meta["output_len"]:
    raise RuntimeError(f"model output length {baseline.shape[0]} != registry {meta['output_len']}")
  if not np.isfinite(baseline).all():
    raise RuntimeError("compiled model produced non-finite outputs")

  bundle = {
    "run_model": jit,
    "model_key": meta["key"],
    "model_sha256": meta["sha256"],
    "output_len": int(meta["output_len"]),
    "frame_skip": int(meta["frame_skip"]),
    "input_spec": {name: (tuple(shape), dtype) for name, (shape, dtype) in INPUT_SPEC.items()},
    "input_device": Device.DEFAULT,
  }
  os.makedirs(os.path.dirname(out_path), exist_ok=True)
  tmp = out_path + ".part"
  print("serialize")
  with open(tmp, "wb") as f:
    pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

  del bundle, jit
  gc.collect()

  print("reload + validate")
  with open(tmp, "rb") as f:
    jit = pickle.load(f)["run_model"]
  if not np.array_equal(_run(jit, SEED), baseline):
    raise RuntimeError("outputs differ from baseline after pickle round trip")
  if np.array_equal(_run(jit, SEED + 1), baseline):
    raise RuntimeError("outputs insensitive to inputs after pickle round trip")

  from tinygrad.tensor import Tensor
  zeros = {name: Tensor(np.zeros(shape, dtype=dtype), device=Device.DEFAULT).realize()
           for name, (shape, dtype) in INPUT_SPEC.items()}
  flat = jit(**zeros)[0].numpy().reshape(-1)
  from iqpilot.selfdrive.iqmodeld.parser import PhaseParser
  from iqpilot.selfdrive.iqmodeld.tools.compile_supercombo import _slice_outputs, _validate_pose_outputs
  _validate_pose_outputs(PhaseParser().parse_vision_outputs(_slice_outputs(flat, meta["output_slices"])))

  os.replace(tmp, out_path)
  return out_path


def _policy_frame(seed: int, input_spec: dict):
  from tinygrad.tensor import Tensor
  rng = np.random.default_rng(seed)
  img = input_spec["img"][0]
  warped = Tensor(rng.integers(0, 256, (2, 6, img[2], img[3])).astype(np.uint8), device="NPY").realize()
  return warped


def _needs_parity(fmt: int) -> bool:
  return fmt in (2, 3, 4, 5) and not HOST and os.environ.get("TC_OPT") != "0" and not os.environ.get("IQ_EGPU_SKIP_PARITY")


def _tc_off_reference(meta: dict, onnx_path: str, fmt: int, resolutions: tuple[tuple[int, int], ...], work_dir: str,
                      progress: tuple[str, float, float] | None = None) -> tuple[str, dict[str, np.ndarray]]:
  """Compile, validate and serialize the model with tensor cores OFF in a child process. Returns the
  validated TC-off artifact and its reloaded outputs: the trusted reference the TC build must match,
  and the artifact that ships instead when a TC kernel miscompiles."""
  import subprocess
  ref = os.path.join(work_dir, "ref.npz")
  pkl = os.path.join(work_dir, "reference.pkl")
  env = {k: v for k, v in os.environ.items() if k not in ("TC_OPT", "BEAM")}
  env["TC_OPT"] = "0"
  env["IQ_EGPU_REFERENCE"] = ref
  source = ["--meta-json", meta["meta_json"]] if meta.get("meta_json") else ["--model", meta["key"]]
  cmd = [sys.executable, "-m", "iqpilot.selfdrive.iqmodeld.tools.compile_egpu_model",
         *source, "--onnx", onnx_path, "--tc-off", "--format", str(fmt), "--output", pkl]
  if fmt in (3, 5):
    cmd += ["--camera-resolutions", *(f"{w}x{h}" for w, h in resolutions)]
  if progress is not None:
    cmd += ["--progress-param", progress[0], "--progress-base", str(progress[1]), "--progress-span", str(progress[2])]
  r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=14400)
  if r.returncode != 0 or not os.path.isfile(ref) or not os.path.isfile(pkl):
    raise RuntimeError(f"parity reference compile failed:\n{r.stderr[-2000:]}")
  with np.load(ref) as archive:
    return pkl, {name: archive[name].copy() for name in archive.files}


def _parity_check(key: str, got: np.ndarray, ref: np.ndarray, label: str = "") -> None:
  rel = float(np.abs(got - ref).mean() / max(1e-3, float(np.abs(ref).mean())))
  if rel > 0.01:
    raise _ParityFail(f"PARITY FAIL: TC kernels miscompiled {key} {label}(rel={rel:.4f} vs TC-off)")
  print(f"  parity vs TC-off reference {label}: rel={rel:.6f} OK")


def _publish(meta: dict, tmp: str, out_path: str, outputs: dict[str, np.ndarray],
             reference: dict[str, np.ndarray] | None) -> str:
  if ref_target := os.environ.get("IQ_EGPU_REFERENCE"):
    np.savez(ref_target, **outputs)
  if reference is not None:
    try:
      for name, got in outputs.items():
        _parity_check(meta["key"], got, reference[name], label=f"{name} ")
    except _ParityFail:
      os.remove(tmp)
      raise
  os.replace(tmp, out_path)
  return out_path


def compile_policy_model(meta: dict, onnx_path: str, out_path: str, reference: dict[str, np.ndarray] | None = None) -> str:
  from tinygrad.device import Device
  from tinygrad.engine.jit import TinyJit
  from tinygrad.nn.onnx import OnnxRunner

  from iqpilot.selfdrive.iqmodeld.egpu_policy import POLICY_FORMAT, PackedInputs, dump_oob, load_bundle, make_queues, make_run_policy

  if meta.get("split"):
    raise RuntimeError(f"model {meta['key']} is a split model; eGPU compiles fused models only")
  input_spec = {name: (tuple(shape), dtype) for name, (shape, dtype) in INPUT_SPEC.items()}
  frame_skip = int(meta["frame_skip"])
  device = Device.DEFAULT
  jit = TinyJit(make_run_policy(OnnxRunner(onnx_path), input_spec, frame_skip, device), prune=True)
  queues = make_queues(input_spec, frame_skip, device)
  packed = PackedInputs(input_spec)

  def step(seed: int) -> np.ndarray:
    packed.views["traffic_convention"][:] = [1, 0]
    packed.views["action_t"][:] = [0.2, 0.3]
    st = time.perf_counter()
    out, = jit(warped=_policy_frame(seed, input_spec), packed_npy_inputs=packed.tensor, **queues)
    flat = out.numpy().reshape(-1)
    print(f"  policy step(seed={seed}) {(time.perf_counter() - st) * 1e3:6.1f} ms")
    packed.views["prev_feat"][:] = flat[meta["output_slices"]["hidden_state"]].reshape(packed.views["prev_feat"].shape)
    return flat

  print("capture + replay")
  for i in range(3):
    baseline = step(SEED + i)
  if baseline.shape[0] != meta["output_len"]:
    raise RuntimeError(f"model output length {baseline.shape[0]} != registry {meta['output_len']}")
  if not HOST and not np.isfinite(baseline).all():
    raise RuntimeError("compiled policy produced non-finite outputs")

  bundle = {
    "format": POLICY_FORMAT,
    "run_policy": jit,
    "model_key": meta["key"],
    "model_sha256": meta["sha256"],
    "output_len": int(meta["output_len"]),
    "frame_skip": frame_skip,
    "input_spec": input_spec,
    "input_device": device,
  }
  os.makedirs(os.path.dirname(out_path), exist_ok=True)
  tmp = out_path + ".part"
  print("serialize (out-of-band buffers)")
  with open(tmp, "wb") as f:
    dump_oob(bundle, f)

  del bundle, jit, queues, packed
  gc.collect()

  print("reload + validate")
  jit = load_bundle(tmp)["run_policy"]
  queues = make_queues(input_spec, frame_skip, device)
  packed = PackedInputs(input_spec)
  outs = []
  for i in range(3):
    packed.views["traffic_convention"][:] = [1, 0]
    packed.views["action_t"][:] = [0.2, 0.3]
    out, = jit(warped=_policy_frame(SEED + i, input_spec), packed_npy_inputs=packed.tensor, **queues)
    flat = out.numpy().reshape(-1)
    packed.views["prev_feat"][:] = flat[meta["output_slices"]["hidden_state"]].reshape(packed.views["prev_feat"].shape)
    outs.append(flat)
  if not HOST:
    if not np.array_equal(outs[-1], baseline):
      raise RuntimeError("policy outputs differ from baseline after pickle round trip")
    if np.array_equal(outs[0], outs[-1]):
      raise RuntimeError("policy outputs insensitive to inputs after pickle round trip")
    if not all(np.isfinite(o).all() for o in outs):
      raise RuntimeError("reloaded policy produced non-finite outputs")
    from iqpilot.selfdrive.iqmodeld.parser import PhaseParser
    from iqpilot.selfdrive.iqmodeld.tools.compile_supercombo import _slice_outputs, _validate_pose_outputs
    _validate_pose_outputs(PhaseParser().parse_vision_outputs(_slice_outputs(outs[-1], meta["output_slices"])))

  return _publish(meta, tmp, out_path, {"policy": outs[-1]}, reference)


DEFAULT_CAMERA_RESOLUTIONS: tuple[tuple[int, int], ...] = ((1928, 1208), (1344, 760))


def camera_nv12(cam_w: int, cam_h: int) -> tuple[int, int, int, int, int]:
  from iqpilot.system.camerad.cameras.nv12_info import get_nv12_info
  stride, y_height, uv_height, _ = get_nv12_info(cam_w, cam_h)
  return (cam_w, cam_h, stride, y_height, uv_height)


def _fill_model_frame(packed, seed: int, res: tuple[int, int], model_w: int, model_h: int) -> None:
  rng = np.random.default_rng(seed)
  cam_w, cam_h = res
  scale = np.array([[cam_w / model_w, 0.0, 0.0], [0.0, cam_h / model_h, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
  for name in ("tfm", "big_tfm"):
    packed.views[name][:, :] = scale * (1.0 + 0.02 * rng.standard_normal((3, 3))).astype(np.float32)
  for v in packed.frames.values():
    v[:] = rng.integers(0, 256, size=v.shape, dtype=np.uint8)
  packed.views["traffic_convention"][:] = [1, 0]
  packed.views["action_t"][:] = [0.2, 0.3]


def compile_model_v3(meta: dict, onnx_path: str, out_path: str,
                     resolutions: tuple[tuple[int, int], ...] = DEFAULT_CAMERA_RESOLUTIONS,
                     reference: dict[str, np.ndarray] | None = None) -> str:
  from tinygrad.device import Device
  from tinygrad.engine.jit import TinyJit
  from tinygrad.nn.onnx import OnnxRunner

  from iqpilot.selfdrive.iqmodeld.egpu_policy import (
    MODEL_FORMAT, dump_oob, load_bundle, make_model_queues, make_run_model, make_run_policy, make_warp, model_size, nv12_copy_size,
  )

  if meta.get("split"):
    raise RuntimeError(f"model {meta['key']} is a split model; eGPU compiles fused models only")
  input_spec = {name: (tuple(shape), dtype) for name, (shape, dtype) in INPUT_SPEC.items()}
  frame_skip = int(meta["frame_skip"])
  hidden = meta["output_slices"]["hidden_state"]
  device = Device.DEFAULT
  model_w, model_h = model_size(input_spec)
  runner = OnnxRunner(onnx_path)
  run_policy = make_run_policy(runner, input_spec, frame_skip, device)

  def step(jit, queues, packed, seed: int, res: tuple[int, int]) -> np.ndarray:
    _fill_model_frame(packed, seed, res, model_w, model_h)
    st = time.perf_counter()
    out, = jit(**queues)
    flat = out.numpy().reshape(-1)
    print(f"  model step(seed={seed}, {res[0]}x{res[1]}) {(time.perf_counter() - st) * 1e3:6.1f} ms")
    packed.views["prev_feat"][:] = flat[hidden].reshape(packed.views["prev_feat"].shape)
    return flat

  def run_three(jit, fcs: int, res: tuple[int, int]) -> list[np.ndarray]:
    queues, packed = make_model_queues(input_spec, frame_skip, device, fcs)
    return [step(jit, queues, packed, SEED + i, res) for i in range(3)]

  jits: dict[tuple[int, int], object] = {}
  sizes: dict[tuple[int, int], int] = {}
  nv12s: dict[tuple[int, int], tuple[int, int, int, int, int]] = {}
  baselines: dict[tuple[int, int], np.ndarray] = {}
  for res in resolutions:
    nv12 = camera_nv12(*res)
    fcs = nv12_copy_size(nv12[2], nv12[3], nv12[4])
    jit = TinyJit(make_run_model(make_warp(nv12, model_w, model_h, device), run_policy, input_spec, fcs, device), prune=True)
    print(f"capture + replay {res[0]}x{res[1]} (frame copy {fcs} B)")
    baseline = run_three(jit, fcs, res)[-1]
    if baseline.shape[0] != meta["output_len"]:
      raise RuntimeError(f"model output length {baseline.shape[0]} != registry {meta['output_len']}")
    if not HOST and not np.isfinite(baseline).all():
      raise RuntimeError("compiled model produced non-finite outputs")
    jits[res], sizes[res], nv12s[res], baselines[res] = jit, fcs, nv12, baseline

  bundle = {
    "format": MODEL_FORMAT,
    "run_model": jits,
    "frame_copy_size": sizes,
    "nv12": nv12s,
    "model_key": meta["key"],
    "model_sha256": meta["sha256"],
    "output_len": int(meta["output_len"]),
    "frame_skip": frame_skip,
    "input_spec": input_spec,
    "input_device": device,
  }
  os.makedirs(os.path.dirname(out_path), exist_ok=True)
  tmp = out_path + ".part"
  print("serialize (out-of-band buffers)")
  with open(tmp, "wb") as f:
    dump_oob(bundle, f)

  del bundle, jits, run_policy, runner
  gc.collect()

  print("reload + validate")
  loaded = load_bundle(tmp)
  outs = {res: run_three(loaded["run_model"][res], loaded["frame_copy_size"][res], res) for res in resolutions}
  if not HOST:
    for res in resolutions:
      if not np.array_equal(outs[res][-1], baselines[res]):
        raise RuntimeError(f"model outputs differ from baseline after pickle round trip ({res[0]}x{res[1]})")
      if np.array_equal(outs[res][0], outs[res][-1]):
        raise RuntimeError(f"model outputs insensitive to inputs after pickle round trip ({res[0]}x{res[1]})")
      if not all(np.isfinite(o).all() for o in outs[res]):
        raise RuntimeError(f"reloaded model produced non-finite outputs ({res[0]}x{res[1]})")
    from iqpilot.selfdrive.iqmodeld.parser import PhaseParser
    from iqpilot.selfdrive.iqmodeld.tools.compile_supercombo import _slice_outputs, _validate_pose_outputs
    _validate_pose_outputs(PhaseParser().parse_vision_outputs(_slice_outputs(outs[resolutions[0]][-1], meta["output_slices"])))

  return _publish(meta, tmp, out_path, {f"{w}x{h}": outs[(w, h)][-1] for (w, h) in resolutions}, reference)


def _stateful_inputs(meta: dict) -> tuple[dict, dict[str, str]]:
  from iqpilot.selfdrive.iqmodeld.egpu_policy import state_pairs
  input_spec = {name: (tuple(shape), dtype) for name, (shape, dtype) in INPUT_SPEC.items()}
  return input_spec, state_pairs(input_spec, meta.get("state"))


def _fill_stateful(views: dict, seed: int) -> None:
  rng = np.random.default_rng(seed)
  if "desire" in views:
    views["desire"][:] = 0
    views["desire"].reshape(-1)[1 + seed % 7] = float(seed % 2)
  views["traffic_convention"][:] = np.array([1, 0], dtype=np.float32).reshape(views["traffic_convention"].shape)
  views["action_t"][:] = (np.array([0.2, 0.3]) + 0.01 * rng.standard_normal(2)).astype(np.float32).reshape(views["action_t"].shape)


def _validate_stateful(meta: dict, outs: list[np.ndarray], baseline: np.ndarray, label: str = "") -> None:
  if HOST:
    return
  if not np.array_equal(outs[-1], baseline):
    raise RuntimeError(f"stateful outputs differ from baseline after pickle round trip {label}")
  if np.array_equal(outs[0], outs[-1]):
    raise RuntimeError(f"stateful outputs insensitive to inputs and state after pickle round trip {label}")
  if not all(np.isfinite(o).all() for o in outs):
    raise RuntimeError(f"reloaded stateful model produced non-finite outputs {label}")
  from iqpilot.selfdrive.iqmodeld.parser import PhaseParser
  from iqpilot.selfdrive.iqmodeld.tools.compile_supercombo import _slice_outputs, _validate_pose_outputs
  _validate_pose_outputs(PhaseParser().parse_vision_outputs(_slice_outputs(outs[-1], meta["output_slices"])))


def compile_stateful_policy(meta: dict, onnx_path: str, out_path: str, reference: dict[str, np.ndarray] | None = None) -> str:
  from tinygrad.device import Device
  from tinygrad.engine.jit import TinyJit
  from tinygrad.nn.onnx import OnnxRunner
  from tinygrad.tensor import Tensor

  from iqpilot.selfdrive.iqmodeld.egpu_policy import (
    STATEFUL_IMAGE_INPUT, STATEFUL_POLICY_FORMAT, StatefulPolicyRunner, dump_oob, load_bundle, make_run_stateful,
  )

  input_spec, pairs = _stateful_inputs(meta)
  device = Device.DEFAULT
  img_shape = input_spec[STATEFUL_IMAGE_INPUT][0]

  def run_three(jit) -> list[np.ndarray]:
    runner = StatefulPolicyRunner(jit, input_spec, pairs, device)
    outs = []
    for i in range(3):
      _fill_stateful(runner._packed.views, SEED + i)
      warped = Tensor(np.random.default_rng(SEED + i).integers(0, 256, img_shape).astype(np.uint8), device="NPY").realize()
      st = time.perf_counter()
      outs.append(runner._step(jit, warped=warped, packed_npy_inputs=runner._packed.tensor))
      print(f"  stateful policy step {i} {(time.perf_counter() - st) * 1e3:6.1f} ms")
    return outs

  jit = TinyJit(make_run_stateful(OnnxRunner(onnx_path), input_spec, pairs, device), prune=True)
  print("capture + replay")
  baseline = run_three(jit)[-1]
  if baseline.shape[0] != meta["output_len"]:
    raise RuntimeError(f"model output length {baseline.shape[0]} != registry {meta['output_len']}")
  if not HOST and not np.isfinite(baseline).all():
    raise RuntimeError("compiled stateful policy produced non-finite outputs")

  bundle = {
    "format": STATEFUL_POLICY_FORMAT,
    "run_policy": jit,
    "state_pairs": pairs,
    "model_key": meta["key"],
    "model_sha256": meta["sha256"],
    "output_len": int(meta["output_len"]),
    "frame_skip": int(meta["frame_skip"]),
    "input_spec": input_spec,
    "input_device": device,
  }
  os.makedirs(os.path.dirname(out_path), exist_ok=True)
  tmp = out_path + ".part"
  print("serialize (out-of-band buffers)")
  with open(tmp, "wb") as f:
    dump_oob(bundle, f)
  del bundle, jit
  gc.collect()

  print("reload + validate")
  outs = run_three(load_bundle(tmp)["run_policy"])
  _validate_stateful(meta, outs, baseline)
  return _publish(meta, tmp, out_path, {"policy": outs[-1]}, reference)


def compile_stateful_model(meta: dict, onnx_path: str, out_path: str,
                           resolutions: tuple[tuple[int, int], ...] = DEFAULT_CAMERA_RESOLUTIONS,
                           reference: dict[str, np.ndarray] | None = None) -> str:
  from tinygrad.device import Device
  from tinygrad.engine.jit import TinyJit
  from tinygrad.nn.onnx import OnnxRunner

  from iqpilot.selfdrive.iqmodeld.egpu_policy import (
    STATEFUL_MODEL_FORMAT, StatefulModelRunner, dump_oob, load_bundle, make_run_stateful, make_run_stateful_model, make_warp,
    nv12_copy_size, stateful_model_size,
  )

  input_spec, pairs = _stateful_inputs(meta)
  device = Device.DEFAULT
  model_w, model_h = stateful_model_size(input_spec)
  runner = OnnxRunner(onnx_path)
  run_stateful = make_run_stateful(runner, input_spec, pairs, device)

  def run_three(jit, fcs: int, res: tuple[int, int]) -> list[np.ndarray]:
    model = StatefulModelRunner(jit, input_spec, pairs, device, fcs)
    outs = []
    for i in range(3):
      _fill_model_frame(model._packed, SEED + i, res, model_w, model_h)
      _fill_stateful(model._packed.views, SEED + i)
      st = time.perf_counter()
      outs.append(model._step(jit, packed_npy_inputs=model._packed.tensor))
      print(f"  stateful model step {i} ({res[0]}x{res[1]}) {(time.perf_counter() - st) * 1e3:6.1f} ms")
    return outs

  jits: dict[tuple[int, int], object] = {}
  sizes: dict[tuple[int, int], int] = {}
  nv12s: dict[tuple[int, int], tuple[int, int, int, int, int]] = {}
  baselines: dict[tuple[int, int], np.ndarray] = {}
  for res in resolutions:
    nv12 = camera_nv12(*res)
    fcs = nv12_copy_size(nv12[2], nv12[3], nv12[4])
    jit = TinyJit(make_run_stateful_model(make_warp(nv12, model_w, model_h, device), run_stateful, input_spec, pairs, fcs, device),
                  prune=True)
    print(f"capture + replay {res[0]}x{res[1]} (frame copy {fcs} B)")
    baseline = run_three(jit, fcs, res)[-1]
    if baseline.shape[0] != meta["output_len"]:
      raise RuntimeError(f"model output length {baseline.shape[0]} != registry {meta['output_len']}")
    if not HOST and not np.isfinite(baseline).all():
      raise RuntimeError("compiled stateful model produced non-finite outputs")
    jits[res], sizes[res], nv12s[res], baselines[res] = jit, fcs, nv12, baseline

  bundle = {
    "format": STATEFUL_MODEL_FORMAT,
    "run_model": jits,
    "state_pairs": pairs,
    "frame_copy_size": sizes,
    "nv12": nv12s,
    "model_key": meta["key"],
    "model_sha256": meta["sha256"],
    "output_len": int(meta["output_len"]),
    "frame_skip": int(meta["frame_skip"]),
    "input_spec": input_spec,
    "input_device": device,
  }
  os.makedirs(os.path.dirname(out_path), exist_ok=True)
  tmp = out_path + ".part"
  print("serialize (out-of-band buffers)")
  with open(tmp, "wb") as f:
    dump_oob(bundle, f)
  del bundle, jits, run_stateful, runner
  gc.collect()

  print("reload + validate")
  loaded = load_bundle(tmp)
  outs = {res: run_three(loaded["run_model"][res], loaded["frame_copy_size"][res], res) for res in resolutions}
  for res in resolutions:
    _validate_stateful(meta, outs[res], baselines[res], label=f"({res[0]}x{res[1]})")
  return _publish(meta, tmp, out_path, {f"{w}x{h}": outs[(w, h)][-1] for (w, h) in resolutions}, reference)


def stateful_format(fmt: int, input_spec: dict) -> int:
  from iqpilot.selfdrive.iqmodeld.egpu_policy import is_stateful_spec
  if not is_stateful_spec(input_spec):
    return fmt
  if fmt == 1:
    raise SystemExit("stateful models need format 2/3 (compiled as 4/5); format 1 has no state buffers")
  return {2: 4, 3: 5}.get(fmt, fmt)


def load_meta_json(path: str) -> dict:
  import json
  with open(path) as f:
    meta = json.load(f)
  meta["output_slices"] = {k: slice(*v) if isinstance(v, list) else v for k, v in meta.get("output_slices", {}).items()}
  missing = [k for k in ("key", "sha256", "output_len", "frame_skip", "input_shapes") if k not in meta]
  if missing:
    raise SystemExit(f"{path} lacks {missing}")
  meta["meta_json"] = os.path.abspath(path)
  return meta


def _parse_resolution(text: str) -> tuple[int, int]:
  w, h = text.lower().split("x")
  return int(w), int(h)


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--model", default=None, help=f"registry key, one of {sorted(EGPU_MODELS)}")
  p.add_argument("--meta-json", default=None, help="model meta as JSON (a catalog row's fields), instead of the synced catalog")
  p.add_argument("--onnx", default=None)
  p.add_argument("--output", default=None)
  p.add_argument("--progress-param", default=None)
  p.add_argument("--progress-base", type=float, default=None)
  p.add_argument("--progress-span", type=float, default=0.0)
  p.add_argument("--format", type=int, default=3, choices=(1, 2, 3, 4, 5),
                 help="3 = warp on the dock from raw NV12 (comma master); 2 = device-warped policy bundle")
  p.add_argument("--camera-resolutions", type=_parse_resolution, nargs="+", default=list(DEFAULT_CAMERA_RESOLUTIONS),
                 help="WxH camera sizes bundled into a format-3 artifact")
  p.add_argument("--host", action="store_true", help="compile on a mock dock (no AMD hardware); outputs need a dock parity gate")
  p.add_argument("--arch", default=None, help="target gfx arch for --host")
  p.add_argument("--tc-off", action="store_true", help="disable tensor-core kernels (conservative; auto-set on parity failure)")
  args = p.parse_args()
  if args.host and args.format == 1:
    raise SystemExit("--host supports formats 2 and 3 only")

  if args.meta_json is not None:
    meta = load_meta_json(args.meta_json)
  elif args.model is not None:
    if args.model in EGPU_MODELS:
      meta = get_egpu_model(args.model)
    else:
      from iqpilot.common.params import Params
      meta = resolve_egpu_model(Params(), args.model)
      if meta is None:
        raise SystemExit(f"unknown model {args.model!r}: not a built-in ({sorted(EGPU_MODELS)}) and not in the synced catalog")
  else:
    meta = get_egpu_model()
  set_input_spec(meta)
  args.format = stateful_format(args.format, INPUT_SPEC)

  onnx_path = args.onnx or local_onnx(meta)
  if onnx_path is None or not os.path.isfile(onnx_path):
    raise SystemExit(f"onnx not found for {meta['key']}; pass --onnx or let iqegpumodeld download it first")

  out_path = os.path.abspath(args.output or egpu_pkl_path(meta))
  resolutions = tuple(args.camera_resolutions)
  progress = (args.progress_param, args.progress_base, args.progress_span) if args.progress_param and args.progress_base is not None else None
  work_dir = out_path + ".reference"
  stop = None
  sampler = None
  try:
    ref_pkl, reference = None, None
    if _needs_parity(args.format):
      # the USB dock flock is exclusive per process: the reference child must finish before this process opens the GPU
      shutil.rmtree(work_dir, ignore_errors=True)
      os.makedirs(work_dir)
      ref_progress = None
      if progress is not None:
        ref_progress = (progress[0], progress[1], progress[2] / 2)
        progress = (progress[0], progress[1] + progress[2] / 2, progress[2] / 2)
      print("compile TC-off parity reference before opening the dock", flush=True)
      ref_pkl, reference = _tc_off_reference(meta, onnx_path, args.format, resolutions, work_dir, ref_progress)
    if progress is not None:
      import threading
      stop = threading.Event()
      sampler = threading.Thread(target=_progress_sampler, args=(*progress, stop), daemon=True)
      sampler.start()
    try:
      if args.format == 5:
        out = compile_stateful_model(meta, onnx_path, out_path, resolutions, reference=reference)
      elif args.format == 4:
        out = compile_stateful_policy(meta, onnx_path, out_path, reference=reference)
      elif args.format == 3:
        out = compile_model_v3(meta, onnx_path, out_path, resolutions, reference=reference)
      elif args.format == 2:
        out = compile_policy_model(meta, onnx_path, out_path, reference=reference)
      else:
        out = compile_model(meta, onnx_path, out_path)
    except _ParityFail as e:
      if ref_pkl is None:
        raise
      print(f"{e}\nshipping the validated TC-off build instead", flush=True)
      os.replace(ref_pkl, out_path)
      out = out_path
  finally:
    if stop is not None:
      stop.set()
    if sampler is not None:
      sampler.join(timeout=2)
    shutil.rmtree(work_dir, ignore_errors=True)
  print(f"saved eGPU jit to {out} ({os.path.getsize(out) / 1e6:.2f} MB)")


if __name__ == "__main__":
  main()
