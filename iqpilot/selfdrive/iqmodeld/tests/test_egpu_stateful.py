"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""
import os

import numpy as np
import pytest

os.environ["DEV"] = "CPU"

from iqpilot.selfdrive.iqmodeld.egpu_policy import (
  StatefulPolicyRunner, is_stateful_spec, make_run_stateful, state_pairs, stateful_packed_layout,
)

H, W = 8, 16
SPEC = {
  "new_img": ((2, 6, H, W), "uint8"),
  "desire": ((8,), "float32"),
  "traffic_convention": ((1, 2), "float32"),
  "action_t": ((1, 2), "float32"),
  "state_img_q": ((2, 5, 6, H, W), "uint8"),
  "state_desire_q": ((12, 1, 8), "float32"),
  "state_feat_q": ((16, 1, 32), "float32"),
}
OUT_LEN = 40


def _np_model(inp):
  img_q = np.concatenate([inp["state_img_q"][:, 1:], inp["new_img"][:, None]], axis=1)
  desire_q = np.concatenate([inp["state_desire_q"][1:], inp["desire"].reshape(1, 1, 8)], axis=0)
  older = img_q[:, 0].astype(np.float32).mean(axis=(1, 2, 3))
  newest = img_q[:, -1].astype(np.float32).mean(axis=(1, 2, 3))
  feat = np.tile(np.concatenate([older, newest, desire_q.max(axis=0).reshape(-1)]), 3)[:32] * 0.01
  feat = feat + inp["state_feat_q"][-1, 0] * 0.5
  feat_q = np.concatenate([inp["state_feat_q"][1:], feat.reshape(1, 1, 32)], axis=0)
  head = np.concatenate([feat, inp["traffic_convention"].reshape(-1), inp["action_t"].reshape(-1), feat_q.mean(axis=(1, 2))[-4:]])
  return {"outputs": head.reshape(1, OUT_LEN).astype(np.float32), "next_state_img_q": img_q,
          "next_state_desire_q": desire_q, "next_state_feat_q": feat_q.astype(np.float32)}


def _tg_model(inp):
  from tinygrad.tensor import Tensor
  img_q = inp["state_img_q"][:, 1:].cat(inp["new_img"].unsqueeze(1), dim=1)
  desire_q = inp["state_desire_q"][1:].cat(inp["desire"].reshape(1, 1, 8), dim=0)
  older = img_q[:, 0].cast("float32").mean(axis=(1, 2, 3))
  newest = img_q[:, -1].cast("float32").mean(axis=(1, 2, 3))
  base = Tensor.cat(older, newest, desire_q.max(axis=0).reshape(-1))
  feat = base.repeat(3)[:32] * 0.01 + inp["state_feat_q"][-1, 0] * 0.5
  feat_q = inp["state_feat_q"][1:].cat(feat.reshape(1, 1, 32), dim=0)
  head = Tensor.cat(feat, inp["traffic_convention"].reshape(-1), inp["action_t"].reshape(-1), feat_q.mean(axis=(1, 2))[-4:])
  return {"outputs": head.reshape(1, OUT_LEN), "next_state_img_q": img_q, "next_state_desire_q": desire_q, "next_state_feat_q": feat_q}


class _Reference:
  def __init__(self):
    self.reset()

  def reset(self):
    self.state = {name: np.zeros(shape, dtype=dtype) for name, (shape, dtype) in SPEC.items() if name.startswith("state_")}
    self.prev = np.zeros(8, dtype=np.float32)

  def run(self, warped, desire, traffic, action_t):
    pulse = np.where(desire - self.prev > 0.99, desire, 0).astype(np.float32)
    pulse[0] = 0
    self.prev = desire.copy()
    out = _np_model({"new_img": warped, "desire": pulse, "traffic_convention": traffic.reshape(1, 2),
                     "action_t": action_t.reshape(1, 2), **self.state})
    self.state = {name: out[f"next_{name}"] for name in self.state}
    return out["outputs"].reshape(-1)


def _frames(n, seed=5):
  rng = np.random.default_rng(seed)
  desire = np.zeros(8, dtype=np.float32)
  for i in range(n):
    if i in (2, 6):
      desire[:] = 0
      desire[1 + i % 3] = 1
    elif i == 4:
      desire[:] = 0
    yield (rng.integers(0, 256, (2, 6, H, W), dtype=np.int64).astype(np.uint8), desire.copy(),
           np.array([1.0, 0.0], dtype=np.float32), np.array([0.2 + 0.01 * i, 0.3], dtype=np.float32))


def _runner():
  from tinygrad.engine.jit import TinyJit
  pairs = state_pairs(SPEC)
  jit = TinyJit(make_run_stateful(_tg_model, SPEC, pairs, "CPU"), prune=True)
  return StatefulPolicyRunner(jit, SPEC, pairs, "CPU")


def test_spec_detection_and_layout():
  assert is_stateful_spec(SPEC)
  assert state_pairs(SPEC) == {"state_img_q": "next_state_img_q", "state_desire_q": "next_state_desire_q",
                               "state_feat_q": "next_state_feat_q"}
  shapes, sizes = stateful_packed_layout(SPEC, state_pairs(SPEC))
  assert list(shapes) == ["desire", "traffic_convention", "action_t"]
  assert sizes == [8, 2, 2]
  assert not is_stateful_spec({"img": ((1, 12, 8, 16), "uint8"), "features_buffer": ((1, 24, 512), "float32")})


def test_declared_pairs_win_over_naming():
  assert state_pairs(SPEC, {"state_feat_q": "feat_out"}) == {"state_feat_q": "feat_out"}


def test_state_carries_across_frames_and_matches_numpy_loop():
  runner, ref = _runner(), _Reference()
  for i, frame in enumerate(_frames(14)):
    np.testing.assert_allclose(runner.run(*frame), ref.run(*frame), rtol=1e-5, atol=1e-6, err_msg=f"frame {i}")
  assert runner.warm_frames == 14


def test_reset_restarts_from_zero_state():
  runner, ref = _runner(), _Reference()
  frames = list(_frames(10))
  for frame in frames[:6]:
    runner.run(*frame)
  runner.reset()
  assert runner.warm_frames == 0
  for i, frame in enumerate(frames[6:]):
    np.testing.assert_allclose(runner.run(*frame), ref.run(*frame), rtol=1e-5, atol=1e-6, err_msg=f"frame {i} after reset")


@pytest.mark.parametrize("n", [1, 2, 3])
def test_ping_pong_parity_holds_on_odd_and_even_frames(n):
  runner, ref = _runner(), _Reference()
  out = want = None
  for frame in _frames(n, seed=11):
    out, want = runner.run(*frame), ref.run(*frame)
  np.testing.assert_allclose(out, want, rtol=1e-5, atol=1e-6)


def test_daemon_warmup_leaves_state_cold():
  from iqpilot.selfdrive.iqmodeld import iqegpumodeld
  runner = _runner()
  iqegpumodeld._warmup(runner, SPEC, OUT_LEN)
  assert runner.warm_frames == 0
  ref = _Reference()
  for i, frame in enumerate(_frames(3)):
    np.testing.assert_allclose(runner.run(*frame), ref.run(*frame), rtol=1e-5, atol=1e-6, err_msg=f"frame {i} after warmup")


def test_pipeline_reports_warmth_and_resets():
  from iqpilot.selfdrive.iqmodeld.egpu_pipeline import EgpuPipeline
  runner = _runner()
  meta = {"key": "fake", "frame_skip": 4, "output_len": OUT_LEN, "output_slices": {"hidden_state": slice(0, 32)},
          "input_shapes": {k: v[0] for k, v in SPEC.items()}, "input_dtypes": {k: v[1] for k, v in SPEC.items()}}
  pipeline = EgpuPipeline(meta, runner)
  assert pipeline.stateful and pipeline.state is None
  for frame in _frames(4):
    pipeline.run(*frame)
  assert pipeline.warm_frames == 4
  pipeline.reset_state()
  assert pipeline.warm_frames == 0


def test_warp_on_dock_artifact_reads_every_frame_feed():
  from tinygrad.engine.jit import TinyJit
  from tinygrad.tensor import Tensor
  from iqpilot.selfdrive.iqmodeld.egpu_helpers import patch_tinygrad_prune_views
  from iqpilot.selfdrive.iqmodeld.egpu_policy import StatefulModelRunner, make_run_stateful_model, make_warp, nv12_copy_size, stateful_model_size
  patch_tinygrad_prune_views()
  pairs = state_pairs(SPEC)
  nv12 = (64, 48, 64, 48, 24)
  fcs = nv12_copy_size(*nv12[2:])
  w, h = stateful_model_size(SPEC)
  def model(inp):
    feeds = {name: (inp[name] @ Tensor.eye(2)).contiguous() for name in ("traffic_convention", "action_t")}
    return _tg_model({**inp, **feeds})

  run = make_run_stateful(model, SPEC, pairs, "CPU")
  jit = TinyJit(make_run_stateful_model(make_warp(nv12, w, h, "CPU"), run, SPEC, pairs, fcs, "CPU"), prune=True)
  runner = StatefulModelRunner(jit, SPEC, pairs, "CPU", fcs)
  frame, eye = np.zeros(fcs, dtype=np.uint8), np.eye(3, dtype=np.float32)
  for i in range(6):
    traffic = np.array([i % 2, 1 - i % 2], dtype=np.float32)
    action_t = np.array([0.1 * i, -0.05 * i], dtype=np.float32)
    out = runner.run(frame, frame, eye, eye, np.zeros(8, dtype=np.float32), traffic, action_t)
    np.testing.assert_allclose(out[32:36], np.concatenate([traffic, action_t]), rtol=1e-6, atol=1e-7, err_msg=f"frame {i}")
