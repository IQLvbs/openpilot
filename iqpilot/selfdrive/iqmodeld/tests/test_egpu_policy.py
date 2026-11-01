"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""
import os

import numpy as np

os.environ["DEV"] = "CPU"

from iqpilot.selfdrive.iqmodeld.egpu_policy import PolicyRunner, make_run_policy, packed_layout, queue_shapes
from iqpilot.selfdrive.iqmodeld.temporal_state import TemporalInputState

SPEC = {
  "img": ((1, 12, 8, 16), "uint8"),
  "big_img": ((1, 12, 8, 16), "uint8"),
  "desire_pulse": ((1, 25, 8), "float32"),
  "traffic_convention": ((1, 2), "float32"),
  "action_t": ((1, 2), "float32"),
  "features_buffer": ((1, 24, 512), "float32"),
}
FS = 4
OUT_LEN = 2580
HIDDEN = slice(1064, 1576)


def _pack(inputs):
  from tinygrad.tensor import Tensor
  parts = [inputs[k].cast("float32").reshape(-1) for k in ("img", "big_img", "features_buffer", "desire_pulse", "traffic_convention", "action_t")]
  flat = Tensor.cat(*parts)
  hidden = (flat[:512] * 0.001).reshape(1, 512)
  return flat, hidden


def _fake_model(inputs):
  from tinygrad.tensor import Tensor
  flat, hidden = _pack(inputs)
  n = flat.shape[0]
  head = flat[:min(n, HIDDEN.start)]
  out = Tensor.cat(head.pad((0, HIDDEN.start - head.shape[0])), hidden.reshape(-1), Tensor.zeros(OUT_LEN - HIDDEN.stop, device="CPU"))
  return {"outputs": out.reshape(1, -1)}


class _Reference:
  def __init__(self):
    self.state = TemporalInputState(FS, SPEC)

  def run(self, warped, desire, traffic, action_t):
    inputs = self.state.push_and_materialize(warped, desire, traffic, action_t)
    from tinygrad.tensor import Tensor
    t = {k: Tensor(np.ascontiguousarray(v), device="CPU") for k, v in inputs.items()}
    out = _fake_model(t)["outputs"].numpy().reshape(-1)
    self.state.note_hidden_state(out, HIDDEN)
    return out


def test_policy_queues_match_temporal_state():
  from tinygrad.engine.jit import TinyJit
  jit = TinyJit(make_run_policy(_fake_model, SPEC, FS, "CPU"), prune=True)
  runner = PolicyRunner(jit, SPEC, FS, HIDDEN, "CPU")
  ref = _Reference()
  rng = np.random.default_rng(3)
  desire = np.zeros(8, dtype=np.float32)
  for i in range(14):
    warped = rng.integers(0, 256, (2, 6, 8, 16), dtype=np.int64).astype(np.uint8)
    if i in (2, 3, 9):
      desire[:] = 0
      desire[1 + (i % 3)] = 1
    elif i == 5:
      desire[:] = 0
    traffic = np.array([1.0, 0.0], dtype=np.float32) if i % 2 else np.array([0.0, 1.0], dtype=np.float32)
    action_t = np.array([0.1 * i, 0.2], dtype=np.float32)
    got = runner.run(warped, desire, traffic, action_t)
    want = ref.run(warped, desire, traffic, action_t)
    np.testing.assert_array_equal(got, want, err_msg=f"frame {i}")


def test_layouts():
  shapes, sizes = packed_layout(SPEC)
  assert list(shapes) == ["desire", "traffic_convention", "action_t", "prev_feat"]
  assert sum(sizes) == 8 + 2 + 2 + 512
  q = queue_shapes(SPEC, FS)
  assert q["img_q"][0] == (5, 6, 8, 16) and q["feat_q"][0] == (96, 1, 512) and q["desire_q"][0] == (100, 1, 8)


CAM = (64, 48)


def _nv12(cam_w, cam_h):
  from iqpilot.system.camerad.cameras.nv12_info import get_nv12_info
  stride, y_height, uv_height, _ = get_nv12_info(cam_w, cam_h)
  return (cam_w, cam_h, stride, y_height, uv_height)


def _numpy_warp_plane(src, m, w_dst, h_dst):
  h_src, w_src = src.shape
  x = np.tile(np.arange(w_dst, dtype=np.float32), h_dst)
  y = np.repeat(np.arange(h_dst, dtype=np.float32), w_dst)
  sx = (m[0, 0] * x + m[0, 1] * y + m[0, 2]) / (m[2, 0] * x + m[2, 1] * y + m[2, 2])
  sy = (m[1, 0] * x + m[1, 1] * y + m[1, 2]) / (m[2, 0] * x + m[2, 1] * y + m[2, 2])
  xi = np.clip(np.round(sx), 0, w_src - 1).astype(np.int64)
  yi = np.clip(np.round(sy), 0, h_src - 1).astype(np.int64)
  return src[yi, xi].reshape(h_dst, w_dst)


def _numpy_frame_prepare(frame, m, nv12, model_w, model_h):
  cam_w, cam_h, stride, y_height, uv_height = nv12
  m = m.astype(np.float32)
  y_src = frame[:cam_h * stride].reshape(cam_h, stride)
  uv = frame[stride * y_height:stride * y_height + uv_height * stride].reshape(uv_height, stride)
  m_uv = m * np.array([[1.0, 1.0, 0.5], [1.0, 1.0, 0.5], [2.0, 2.0, 1.0]], dtype=np.float32)
  y = _numpy_warp_plane(y_src, m, model_w, model_h)
  u = _numpy_warp_plane(uv[:cam_h // 2, :cam_w:2], m_uv, model_w // 2, model_h // 2)
  v = _numpy_warp_plane(uv[:cam_h // 2, 1:cam_w:2], m_uv, model_w // 2, model_h // 2)
  f = np.concatenate([y.ravel(), u.ravel(), v.ravel()]).reshape(model_h * 3 // 2, model_w)
  H, W = model_h, model_w
  return np.stack([f[0:H:2, 0::2], f[1:H:2, 0::2], f[0:H:2, 1::2], f[1:H:2, 1::2],
                   f[H:H + H // 4].reshape(H // 2, W // 2), f[H + H // 4:H + H // 2].reshape(H // 2, W // 2)])


def _jittered_scale(rng, cam, model_w, model_h):
  m = np.array([[cam[0] / model_w, 0.0, 0.0], [0.0, cam[1] / model_h, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
  m += (0.05 * rng.standard_normal((3, 3))).astype(np.float32) * np.array([[1, 1, 1], [1, 1, 1], [0.01, 0.01, 0.1]], dtype=np.float32)
  return m


def test_frame_layout():
  from iqpilot.selfdrive.iqmodeld.egpu_policy import frame_layout, model_size, nv12_copy_size
  shapes, sizes, npy_bytes = frame_layout(SPEC)
  assert list(shapes) == ["tfm", "big_tfm", "desire", "traffic_convention", "action_t", "prev_feat"]
  assert npy_bytes == (18 + 8 + 2 + 2 + 512) * 4
  assert model_size(SPEC) == (32, 16)
  assert nv12_copy_size(128, 64, 32) == 128 * 96


def test_model_runner_matches_device_warp():
  from tinygrad.engine.jit import TinyJit

  from iqpilot.selfdrive.iqmodeld.egpu_policy import ModelRunner, make_run_model, make_warp, model_size, nv12_copy_size
  nv12 = _nv12(*CAM)
  fcs = nv12_copy_size(nv12[2], nv12[3], nv12[4])
  model_w, model_h = model_size(SPEC)
  run_policy = make_run_policy(_fake_model, SPEC, FS, "CPU")
  jit = TinyJit(make_run_model(make_warp(nv12, model_w, model_h, "CPU"), run_policy, SPEC, fcs, "CPU"), prune=True)
  runner = ModelRunner(jit, SPEC, FS, HIDDEN, "CPU", fcs)
  ref = PolicyRunner(TinyJit(make_run_policy(_fake_model, SPEC, FS, "CPU"), prune=True), SPEC, FS, HIDDEN, "CPU")
  rng = np.random.default_rng(7)
  desire = np.zeros(8, dtype=np.float32)
  for i in range(10):
    main = rng.integers(0, 256, fcs, dtype=np.int64).astype(np.uint8)
    extra = rng.integers(0, 256, fcs, dtype=np.int64).astype(np.uint8)
    tfm = _jittered_scale(rng, CAM, model_w, model_h)
    big_tfm = _jittered_scale(rng, CAM, model_w, model_h)
    if i in (2, 6):
      desire[:] = 0
      desire[1 + i % 3] = 1
    traffic = np.array([1.0, 0.0], dtype=np.float32) if i % 2 else np.array([0.0, 1.0], dtype=np.float32)
    action_t = np.array([0.1 * i, 0.2], dtype=np.float32)
    got = runner.run(main, extra, tfm, big_tfm, desire, traffic, action_t)
    warped = np.stack([_numpy_frame_prepare(main, tfm, nv12, model_w, model_h), _numpy_frame_prepare(extra, big_tfm, nv12, model_w, model_h)])
    want = ref.run(warped, desire, traffic, action_t)
    np.testing.assert_array_equal(got, want, err_msg=f"frame {i}")
