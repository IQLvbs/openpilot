"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/

eMac split-model "prepared input equivalence": SplitInputState must reproduce,
byte-exact, the queue semantics of compile_split_runtime's execute_bundle —
the real tinygrad reference graph run on CPU with stub vision/policy runners,
over a multi-frame random sequence with desire rising edges.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("DEV", "CPU")

from iqpilot.selfdrive.iqmodeld.emac_input_state import EmacInputState, SplitInputState

N_FRAMES_TEST = 30
FRAME_SKIP = 4
IMG_SHAPE = (1, 12, 16, 32)  # small spatial dims: queue math is shape-generic
FB_SHAPE = (1, 25, 512)
DP_SHAPE = (1, 25, 8)
VISION_OUT_LEN = 1576
HIDDEN_SLICE = slice(1064, 1576)

VISION_SHAPES = {"img": IMG_SHAPE, "big_img": IMG_SHAPE}
POLICY_SHAPES = {"desire_pulse": DP_SHAPE, "traffic_convention": (1, 2), "features_buffer": FB_SHAPE}


class _StubRunner:
  """Stands in for OnnxRunner inside execute_bundle: returns a preset output
  and records the materialized inputs it was fed."""

  def __init__(self, out_len: int):
    self.out_len = out_len
    self.next_output: np.ndarray | None = None
    self.captured: dict[str, np.ndarray] | None = None

  def __call__(self, inputs):
    from tinygrad import Tensor
    self.captured = {k: v.numpy().copy() for k, v in inputs.items()}
    out = self.next_output if self.next_output is not None else np.zeros((1, self.out_len), dtype=np.float32)
    return {"outputs": Tensor(out.astype(np.float32))}


@pytest.fixture(scope="module")
def reference():
  from tinygrad import Tensor
  from iqpilot.selfdrive.iqmodeld.tools.compile_split_runtime import _role_executor

  meta_by_role = {
    "vision": {"input_shapes": dict(VISION_SHAPES), "output_slices": {"hidden_state": HIDDEN_SLICE}},
    "policy": {"input_shapes": dict(POLICY_SHAPES), "output_slices": {}},
  }
  vision, policy = _StubRunner(VISION_OUT_LEN), _StubRunner(1000)
  execute_bundle = _role_executor({"vision": vision, "policy": policy}, meta_by_role, FRAME_SKIP)

  feat_q = Tensor(np.zeros((FRAME_SKIP * (FB_SHAPE[1] - 1) + 1, FB_SHAPE[0], FB_SHAPE[2]), dtype=np.float32),
                  device="CPU").contiguous().realize()
  desire_q = Tensor(np.zeros((FRAME_SKIP * DP_SHAPE[1], DP_SHAPE[0], DP_SHAPE[2]), dtype=np.float32),
                    device="CPU").contiguous().realize()
  return execute_bundle, feat_q, desire_q, vision, policy


def test_split_inputs_match_tinygrad_reference(reference):
  from tinygrad import Tensor

  execute_bundle, feat_q, desire_q, vision_stub, policy_stub = reference
  rng = np.random.default_rng(4321)
  state = SplitInputState(FRAME_SKIP, IMG_SHAPE, FB_SHAPE, DP_SHAPE)
  ref_prev_desire = np.zeros(DP_SHAPE[2], dtype=np.float32)

  for frame in range(N_FRAMES_TEST):
    warped = rng.integers(0, 256, (2, 6, IMG_SHAPE[2], IMG_SHAPE[3]), dtype=np.int64).astype(np.uint8)
    raw_desire = np.zeros(DP_SHAPE[2], dtype=np.float32)
    if frame % 3:
      raw_desire[int(rng.integers(0, DP_SHAPE[2]))] = 1.0
    traffic = rng.standard_normal((1, 2)).astype(np.float32)
    vision_out = rng.standard_normal((1, VISION_OUT_LEN)).astype(np.float32)
    vision_stub.next_output = vision_out

    # --- ours ---
    vis_inputs = state.materialize_vision(warped, raw_desire)
    pol_inputs = state.materialize_policy(vision_out[0, HIDDEN_SLICE], traffic[0])

    # --- reference graph: rising edge happens outside execute_bundle (run_fused) ---
    cur = raw_desire.copy()
    cur[0] = 0
    ref_pulse = np.where(cur - ref_prev_desire > 0.99, cur, 0).astype(np.float32)
    ref_prev_desire[:] = cur

    execute_bundle(
      img=Tensor(vis_inputs["img"], device="CPU").realize(),
      big_img=Tensor(vis_inputs["big_img"], device="CPU").realize(),
      feat_q=feat_q, desire_q=desire_q,
      desire=Tensor(ref_pulse, device="CPU").realize(),
      traffic_convention=Tensor(traffic, device="CPU").realize(),
      action_t=Tensor(np.zeros((1, 2), dtype=np.float32), device="CPU").realize(),
    )
    ref = policy_stub.captured
    assert ref is not None

    assert ref["features_buffer"].tobytes() == pol_inputs["features_buffer"].tobytes(), f"features frame {frame}"
    assert ref["desire_pulse"].tobytes() == pol_inputs["desire_pulse"].tobytes(), f"desire frame {frame}"
    assert ref["traffic_convention"].tobytes() == pol_inputs["traffic_convention"].tobytes()
    # vision saw exactly what our img queues materialized
    vref = vision_stub.captured
    assert vref["img"].tobytes() == vis_inputs["img"].tobytes(), f"img frame {frame}"
    assert vref["big_img"].tobytes() == vis_inputs["big_img"].tobytes(), f"big_img frame {frame}"


def test_split_img_queue_matches_fused_state():
  # img/desire mechanics are shared with the fused mirror: same warps must
  # materialize identical img/big_img in both states
  rng = np.random.default_rng(7)
  fused_spec = {
    "img": (IMG_SHAPE, "uint8"), "big_img": (IMG_SHAPE, "uint8"),
    "desire_pulse": (DP_SHAPE, "float32"), "traffic_convention": ((1, 2), "float32"),
    "features_buffer": ((1, 24, 512), "float32"), "action_t": ((1, 2), "float32"),
  }
  fused = EmacInputState(FRAME_SKIP, fused_spec)
  split = SplitInputState(FRAME_SKIP, IMG_SHAPE, FB_SHAPE, DP_SHAPE)
  for _ in range(12):
    warped = rng.integers(0, 256, (2, 6, IMG_SHAPE[2], IMG_SHAPE[3]), dtype=np.int64).astype(np.uint8)
    desire = np.zeros(DP_SHAPE[2], dtype=np.float32)
    f = fused.push_and_materialize(warped, desire, np.zeros(2, dtype=np.float32), np.zeros(2, dtype=np.float32))
    s = split.materialize_vision(warped, desire)
    assert f["img"].tobytes() == s["img"].tobytes()
    assert f["big_img"].tobytes() == s["big_img"].tobytes()
