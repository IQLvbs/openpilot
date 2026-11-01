"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""
from __future__ import annotations

import os
import pickle

import numpy as np

from iqpilot.common.swaglog import cloudlog
from iqpilot.system.hardware.hw import Paths


def _load_bundle(pkl_path: str, cam_w: int, cam_h: int, frame_skip: int) -> dict:
  with open(pkl_path, "rb") as f:
    bundle = pickle.load(f)
  if bundle.get("frame_skip") != frame_skip:
    raise RuntimeError(f"frame_skip {bundle.get('frame_skip')} != {frame_skip}")
  if (cam_w, cam_h) not in bundle:
    raise RuntimeError(f"missing {cam_w}x{cam_h}; has {[k for k in bundle if isinstance(k, tuple)]}")
  _verify_selftest(bundle, cam_w, cam_h)
  return bundle


def _verify_selftest(bundle: dict, cam_w: int, cam_h: int) -> None:
  want = bundle.get("selftest")
  if not want:
    raise RuntimeError("warp artifact predates the self-test; recompiling")
  from iqpilot.system.camerad.cameras.nv12_info import get_nv12_info
  from iqpilot.selfdrive.iqmodeld.tools.compile_warp import selftest_digest
  nv12_size = get_nv12_info(cam_w, cam_h)[3]
  got = selftest_digest(bundle[(cam_w, cam_h)], cam_w, cam_h, nv12_size)
  if got != want:
    raise RuntimeError(f"warp self-test {got[:12]} != {want[:12]}; artifact computes differently here")


class FrameWarp:

  def __init__(self, cam_w: int, cam_h: int, frame_skip: int):
    from tinygrad.tensor import Tensor

    pkl_path = os.path.join(Paths.model_root(), f"emac_warp_{cam_w}x{cam_h}_tinygrad.pkl")
    bundle = None
    if os.path.isfile(pkl_path):
      try:
        bundle = _load_bundle(pkl_path, cam_w, cam_h, frame_skip)
      except Exception as e:
        cloudlog.warning(f"warp artifact unusable ({e}); discarding and recompiling")
        os.remove(pkl_path)
    if bundle is None:
      cloudlog.warning(f"warp artifact missing; compiling for {cam_w}x{cam_h} (one-time)")
      from iqpilot.selfdrive.iqmodeld.tools.compile_warp import compile_warp
      compile_warp(cam_w, cam_h, pkl_path, frame_skip=frame_skip)
      cloudlog.warning(f"warp compiled -> {pkl_path}")
      bundle = _load_bundle(pkl_path, cam_w, cam_h, frame_skip)
    self._jit = bundle[(cam_w, cam_h)]

    self._npy = {"tfm": np.zeros((3, 3), dtype=np.float32), "big_tfm": np.zeros((3, 3), dtype=np.float32)}
    self._tensors = {k: Tensor(v, device="NPY").realize() for k, v in self._npy.items()}
    self._blob_cache: dict[tuple[str, int], object] = {}
    self._Tensor = Tensor

  def _frame_tensor(self, key: str, buf):
    from tinygrad.device import Device
    arr = np.frombuffer(buf.data, dtype=np.uint8)
    ck = (key, arr.ctypes.data)
    t = self._blob_cache.get(ck)
    if t is None:
      t = self._Tensor.from_blob(arr.ctypes.data, (arr.size,), dtype="uint8", device=Device.DEFAULT)
      self._blob_cache[ck] = t
    return t

  def run(self, main_buf, extra_buf, main_tfm: np.ndarray, extra_tfm: np.ndarray) -> np.ndarray:
    self._npy["tfm"][:] = main_tfm
    self._npy["big_tfm"][:] = extra_tfm
    warped = self._jit(tfm=self._tensors["tfm"], big_tfm=self._tensors["big_tfm"],
                       frame=self._frame_tensor("img", main_buf),
                       big_frame=self._frame_tensor("big_img", extra_buf))
    return warped.numpy().astype(np.uint8, copy=False)
