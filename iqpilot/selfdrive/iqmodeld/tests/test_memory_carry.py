"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""

import numpy as np
import pytest

from iqpilot.selfdrive.iqmodeld.daemon import NeuralEngineState, ReplayLedger


def test_rank_two_feedback_copies_and_invalid_feedback_resets():
  ledger = ReplayLedger({"memory_state": (1, 192)}, [])
  source = np.ones((1, 192), np.float32)
  ledger.note_feedback("memory_state", source)
  source.fill(2)
  np.testing.assert_array_equal(ledger.inputs["memory_state"], np.ones((1, 192), np.float32))
  with pytest.raises(ValueError):
    ledger.note_feedback("memory_state", np.ones((192,), np.float32))
  assert not ledger.inputs["memory_state"].any()


def test_fused_runtime_feeds_back_memory_and_reset_clears_it():
  class Runner:
    uses_opencl_warp = False
    output_slices = {"memory_state_out": slice(2578, 2770)}

    def __init__(self):
      self.seen = []

    def run_fused(self, vision_bufs, transform_map, fresh_inputs):
      state = fresh_inputs["memory_state"].copy()
      self.seen.append(state)
      return {"memory_state_out": state + 1}

  runtime = object.__new__(NeuralEngineState)
  runtime.model_runner = Runner()
  runtime._ledger = ReplayLedger({"memory_state": (1, 192)}, [])
  runtime.numpy_inputs = runtime._ledger.inputs
  runtime.carry = {"memory_state": "memory_state_out"}
  runtime.run({}, {}, {})
  runtime.run({}, {}, {})
  runtime.reset_memory()
  runtime.run({}, {}, {})
  assert not runtime.model_runner.seen[0].any()
  assert (runtime.model_runner.seen[1] == 1).all()
  assert not runtime.model_runner.seen[2].any()


def test_carry_output_missing_resets_and_raises():
  class Runner:
    uses_opencl_warp = False

    def run_fused(self, vision_bufs, transform_map, fresh_inputs):
      return {}

  runtime = object.__new__(NeuralEngineState)
  runtime.model_runner = Runner()
  runtime._ledger = ReplayLedger({"memory_state": (1, 192)}, [])
  runtime._ledger.inputs["memory_state"].fill(3)
  runtime.numpy_inputs = runtime._ledger.inputs
  runtime.carry = {"memory_state": "memory_state_out"}
  with pytest.raises(ValueError):
    runtime.run({}, {}, {})
  assert not runtime.numpy_inputs["memory_state"].any()


def test_runtime_without_carry_skips_feedback():
  class Runner:
    uses_opencl_warp = False

    def run_fused(self, vision_bufs, transform_map, fresh_inputs):
      assert fresh_inputs == {}
      return {"plan": np.zeros(1)}

  runtime = object.__new__(NeuralEngineState)
  runtime.model_runner = Runner()
  runtime._ledger = ReplayLedger({}, [])
  runtime.numpy_inputs = runtime._ledger.inputs
  runtime.carry = {}
  assert not runtime.has_memory_carry
  assert "plan" in runtime.run({}, {}, {})
