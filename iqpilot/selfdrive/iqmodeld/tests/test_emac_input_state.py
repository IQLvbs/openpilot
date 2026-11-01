"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""
from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("DEV", "CPU")

from iqpilot.selfdrive.iqmodeld.emac_input_state import EmacInputState
from iqpilot.selfdrive.iqmodeld.emac_model_meta import FRAME_SKIP, OUTPUT_LEN, OUTPUT_SLICES
from iqpilot.selfdrive.iqmodeld.temporal_state import MODEL_INPUT_SPEC as INPUT_SPEC

N_FRAMES_TEST = 30
IMG_SHAPE = INPUT_SPEC["img"][0]
DESIRE_LEN = INPUT_SPEC["desire_pulse"][0][2]


class _CaptureRunner:

  def __init__(self):
    self.captured: dict[str, np.ndarray] | None = None

  def __call__(self, inputs):
    from tinygrad import Tensor
    self.captured = {k: v.numpy().copy() for k, v in inputs.items()}
    return {"outputs": Tensor(np.zeros((1, OUTPUT_LEN), dtype=np.float32))}


@pytest.fixture(scope="module")
def reference():
  from iqpilot.selfdrive.iqmodeld.tools.compile_supercombo import (
    POLICY_INPUTS, make_input_queues, make_run_policy,
  )

  input_shapes = {name: shape for name, (shape, _) in INPUT_SPEC.items()}
  metadata = {"input_shapes": input_shapes}
  capture = _CaptureRunner()
  run_policy = make_run_policy(capture, metadata, FRAME_SKIP)
  queues, npy = make_input_queues(input_shapes, FRAME_SKIP, device="CPU")
  return run_policy, queues, npy, capture, POLICY_INPUTS


def _rising_edge(raw_desire: np.ndarray, prev: np.ndarray) -> np.ndarray:
  cur = raw_desire.astype(np.float32).copy()
  cur[0] = 0
  pulse = np.where(cur - prev > 0.99, cur, 0).astype(np.float32)
  prev[:] = cur
  return pulse


def test_materialized_inputs_match_tinygrad_reference(reference):
  from tinygrad import Tensor

  run_policy, queues, npy, capture, policy_inputs = reference
  rng = np.random.default_rng(1234)
  state = EmacInputState(FRAME_SKIP)
  ref_prev_desire = np.zeros(DESIRE_LEN, dtype=np.float32)

  hidden = np.zeros((1, 512), dtype=np.float32)
  for frame in range(N_FRAMES_TEST):
    warped = rng.integers(0, 256, (2, 6, IMG_SHAPE[2], IMG_SHAPE[3]), dtype=np.int64).astype(np.uint8)
    raw_desire = np.zeros(DESIRE_LEN, dtype=np.float32)
    if frame % 3:
      raw_desire[int(rng.integers(0, DESIRE_LEN))] = 1.0
    traffic = rng.standard_normal(2).astype(np.float32)
    action_t = rng.standard_normal(2).astype(np.float32)

    npy["desire"][:] = _rising_edge(raw_desire, ref_prev_desire)
    npy["traffic_convention"][:] = traffic
    npy["action_t"][:] = action_t
    npy["prev_feat"][:] = hidden
    run_policy(warped=Tensor(warped), **{k: queues[k] for k in policy_inputs})
    ref_inputs = capture.captured

    state.prev_feat[:] = hidden
    mat = state.push_and_materialize(warped, raw_desire, traffic, action_t)

    for name in INPUT_SPEC:
      assert ref_inputs[name].shape == tuple(INPUT_SPEC[name][0]), name
      np.testing.assert_array_equal(
        mat[name].astype(ref_inputs[name].dtype), ref_inputs[name],
        err_msg=f"frame {frame}: materialized {name} diverges from tinygrad reference")

    fake_output = rng.standard_normal(OUTPUT_LEN).astype(np.float32)
    state.note_hidden_state(fake_output, OUTPUT_SLICES["hidden_state"])
    hidden = fake_output[OUTPUT_SLICES["hidden_state"]].reshape(1, 512).copy()


def test_note_hidden_state_slice():
  state = EmacInputState(FRAME_SKIP)
  out = np.arange(OUTPUT_LEN, dtype=np.float32)
  state.note_hidden_state(out, OUTPUT_SLICES["hidden_state"])
  np.testing.assert_array_equal(state.prev_feat.reshape(-1), out[OUTPUT_SLICES["hidden_state"]])


def test_desire_pulse_rising_edge_only_once():
  state = EmacInputState(FRAME_SKIP)
  held = np.zeros(DESIRE_LEN, dtype=np.float32)
  held[3] = 1.0
  warped = np.zeros((2, 6, IMG_SHAPE[2], IMG_SHAPE[3]), dtype=np.uint8)
  zeros2 = np.zeros(2, dtype=np.float32)

  first = state.push_and_materialize(warped, held, zeros2, zeros2)
  assert first["desire_pulse"][0, -1, 3] == 1.0
  second = state.push_and_materialize(warped, held, zeros2, zeros2)
  assert state.desire_q[-1].max() == 0.0
  assert second["desire_pulse"][0, -1, 3] == 1.0


HISTORY_SPEC = {
  "img": ((1, 12, 128, 256), "uint8"),
  "big_img": ((1, 12, 128, 256), "uint8"),
  "features_buffer": ((1, 8, 32, 512), "float32"),
  "desire_pulse": ((1, 33, 8), "float32"),
  "traffic_convention": ((1, 33, 4), "float32"),
  "action_t": ((1, 33, 2), "float32"),
}


def test_history_spec_materializes_speed_history_and_feature_order():
  fs = 4
  state = EmacInputState(fs, HISTORY_SPEC)
  warped = np.zeros((2, 6, 128, 256), dtype=np.uint8)
  desire = np.zeros(8, dtype=np.float32)
  seen = []
  for frame in range(1, 41):
    traffic = np.array([1.0, 0.0, frame / 30.0, 0.5], dtype=np.float32)
    mat = state.push_and_materialize(warped, desire, traffic, np.array([0.2, 0.3], dtype=np.float32))
    hidden = np.full((1, 32 * 512), float(frame), dtype=np.float32)
    state.note_hidden_state(np.concatenate([np.zeros(10, dtype=np.float32), hidden.ravel()]), slice(10, 10 + 32 * 512))
    seen.append(mat)
    assert mat["traffic_convention"].shape == (1, 33, 4)
    assert mat["action_t"].shape == (1, 33, 2)
    assert mat["features_buffer"].shape == (1, 8, 32, 512)
  first = seen[0]["traffic_convention"][0]
  assert np.allclose(first[:, 2], 1 / 30.0) and np.allclose(first[:, 0], 1.0)
  last = seen[-1]["traffic_convention"][0]
  assert np.allclose(last[-1], [1.0, 0.0, 40 / 30.0, 0.5])
  assert np.allclose(last[-2, 2], 36 / 30.0) and np.allclose(last[-9, 2], 8 / 30.0)
  assert np.allclose(seen[-1]["action_t"][0], [[0.2, 0.3]] * 33)
  feats = seen[-1]["features_buffer"][0, :, 0, 0]
  assert np.allclose(feats, [40 - 4 * (8 - i) for i in range(8)])


def test_two_column_spec_truncates_wide_wire_traffic():
  state = EmacInputState(FRAME_SKIP)
  mat = state.push_and_materialize(np.zeros((2, 6, 128, 256), dtype=np.uint8), np.zeros(8, dtype=np.float32),
                                   np.array([0.0, 1.0, 0.7, 0.1], dtype=np.float32), np.array([0.2, 0.3], dtype=np.float32))
  assert mat["traffic_convention"].shape == (1, 2) and np.allclose(mat["traffic_convention"][0], [0.0, 1.0])
  assert mat["action_t"].shape == (1, 2)


def test_split_state_fits_wide_wire_traffic_to_policy_width():
  from iqpilot.selfdrive.iqmodeld.emac_input_state import SplitInputState
  state = SplitInputState(FRAME_SKIP, (1, 12, 128, 256), (1, 25, 512), (1, 25, 8))
  state.materialize_vision(np.zeros((2, 6, 128, 256), dtype=np.uint8), np.zeros(8, dtype=np.float32))
  out = state.materialize_policy(np.zeros(512, dtype=np.float32), np.array([0.0, 1.0, 0.7, 0.1], dtype=np.float32))
  assert out["traffic_convention"].shape == (1, 2) and np.allclose(out["traffic_convention"][0], [0.0, 1.0])
  assert "action_t" not in out
