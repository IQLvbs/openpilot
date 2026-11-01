# Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
from types import SimpleNamespace

import numpy as np
import pytest

from iqpilot.selfdrive.iqmodeld import daemon


class RecordingIO:
  services = ["navInstruction"]
  carry: dict = {}

  def __init__(self, extra=None):
    self.extra = extra or {}
    self.calls = []

  def compose(self, numpy_inputs, sub, params, frame_timestamp_eof_ns, vehicle_speed):
    self.calls.append((numpy_inputs, sub, params, frame_timestamp_eof_ns, vehicle_speed))
    return dict(self.extra)


def fake_daemon(io):
  shapes = {"desire_pulse": (1, 25, 8), "traffic_convention": (1, 2), "action_t": (1, 2),
            "lateral_control_params": (1, 2), "extra_context": (1, 32)}
  ledger = daemon.ReplayLedger(shapes, [])
  instance = daemon.InferenceDaemon.__new__(daemon.InferenceDaemon)
  instance._runtime = SimpleNamespace(numpy_inputs=ledger.inputs, desire_key=ledger.pulse_name,
                                      constants=SimpleNamespace(DESIRE_LEN=8))
  instance._desire_logic = SimpleNamespace(desire=0)
  instance._sub = {"driverMonitoringState": SimpleNamespace(isRHD=False)}
  instance._params = object()
  instance._io = io
  return instance, ledger


def test_stock_inputs_are_composed_publicly_and_custom_inputs_stay_zero():
  io = RecordingIO()
  instance, ledger = fake_daemon(io)
  ledger.merge_inputs(instance._compose_inputs(12.0, 0.2, 0.4, 123))
  assert not ledger.inputs["extra_context"].any()
  np.testing.assert_array_equal(ledger.inputs["traffic_convention"], [[1, 0]])
  np.testing.assert_allclose(ledger.inputs["action_t"], [[0.2, 0.4]])
  np.testing.assert_allclose(ledger.inputs["lateral_control_params"], [[12.0, 0.2]])
  assert len(io.calls) == 1
  numpy_inputs, sub, params, timestamp, speed = io.calls[0]
  assert numpy_inputs is ledger.inputs and sub is instance._sub and params is instance._params
  assert (timestamp, speed) == (123, 12.0)


def test_private_inputs_reach_the_ledger():
  expected = np.full((1, 32), 0.5, dtype=np.float32)
  instance, ledger = fake_daemon(RecordingIO({"extra_context": expected}))
  ledger.merge_inputs(instance._compose_inputs(12.0, 0.2, 0.4, 123))
  np.testing.assert_array_equal(ledger.inputs["extra_context"], expected)


def test_fallback_io_is_inert(monkeypatch):
  import importlib.util
  import sys
  from pathlib import Path
  from iqpilot import _proprietary_loader

  def missing(public_name, private_name):
    raise _proprietary_loader.ProprietaryModuleMissing("missing context bundle")

  monkeypatch.setattr(_proprietary_loader, "load_private_module", missing)
  spec = importlib.util.spec_from_file_location("iqmodel_io_fallback_test", Path(daemon.__file__).with_name("iqmodel_io.py"))
  module = importlib.util.module_from_spec(spec)
  monkeypatch.setitem(sys.modules, spec.name, module)
  spec.loader.exec_module(module)
  io = module.IQModelIO(SimpleNamespace(input_shapes={"memory_state": (1, 192)}))
  assert io.carry == {} and io.services == []
  assert io.compose({}, None, None, 0, 0.0) == {}
  assert not io.memory_reset(None, 0, 0)
  assert io.publish(None, None, {}, 0, 0) is None
  assert module.publish_outputs(None, None, {}) is None
  assert not module.fill_reasoning(None, {}, None, 0, 0)


@pytest.mark.parametrize("carry", [{}, {"memory_state": "memory_state_out"}])
def test_runtime_builds_io_once_and_takes_its_carry(monkeypatch, carry):
  shapes = {"desire_pulse": (1, 25, 8)}
  runner = SimpleNamespace(input_shapes=shapes, vision_input_names=[], is_20hz=True,
                           constants=SimpleNamespace(DESIRE_LEN=8))
  built = []

  class IO(RecordingIO):
    def __init__(self, model_runner):
      super().__init__()
      built.append(model_runner)
      self.carry = carry

  monkeypatch.setattr(daemon, "get_model_runner", lambda: runner)
  monkeypatch.setattr(daemon, "get_active_bundle", lambda: None)
  monkeypatch.setattr(daemon, "IQModelIO", IO)
  runtime = daemon.NeuralEngineState(None)
  assert built == [runner]
  assert runtime.carry == carry
  assert runtime.has_memory_carry is bool(carry)
