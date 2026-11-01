# Copyright (c) 2026 IQ.Lvbs. All rights reserved.
from pathlib import Path

import numpy as np
import pytest

from iqpilot.selfdrive.controls.tests.lateral_control_scenarios import CONTROL_CASES, build_controls, load_controls_module, run_scenario
from iqpilot.selfdrive.controls.tests.lateral_control_scenarios import set_frame
from iqpilot.selfdrive.controls.lib.curvature_lookahead import Selection
from iqpilot.selfdrive.controls.lib.drive_helpers import clip_curvature


@pytest.fixture
def controls_module(monkeypatch):
  return load_controls_module(monkeypatch)


@pytest.mark.parametrize("case", CONTROL_CASES)
@pytest.mark.parametrize("slew", [False, True])
def test_unknown_model_cannot_replace_action(controls_module, case, slew):
  disabled = run_scenario(build_controls(controls_module, case, lookahead=False, slew=slew, smooth=True))
  enabled = run_scenario(build_controls(controls_module, case, lookahead=True, slew=slew, smooth=True))
  np.testing.assert_array_equal(enabled, disabled)


@pytest.mark.parametrize("case", CONTROL_CASES)
@pytest.mark.parametrize("slew", [False, True])
def test_toggle_off_matches_pre_fix_outputs(controls_module, case, slew):
  with np.load(Path(__file__).with_name("fixtures") / "lateral_lookahead_off.npz") as baseline:
    for smooth in ((False, True) if case == "curvature" else (False,)):
      actual = run_scenario(build_controls(controls_module, case, lookahead=False, slew=slew, smooth=smooth))
      expected = baseline[f"{case}_slew{int(slew)}_smooth{int(smooth)}"]
      np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("case", ["torque", "tss2", "gm"])
@pytest.mark.parametrize("slew", [False, True])
def test_unknown_model_cannot_replace_action_with_nnff(controls_module, case, slew):
  controls = build_controls(controls_module, case, lookahead=False, slew=slew, nnff=True)
  actual = run_scenario(controls)
  assert controls.LaC.nnff_assist.model.queries > 0
  enabled = build_controls(controls_module, case, lookahead=True, slew=slew, nnff=True)
  np.testing.assert_array_equal(actual, run_scenario(enabled))


@pytest.mark.parametrize("case", CONTROL_CASES)
@pytest.mark.parametrize("gas_override", [False, True])
def test_legacy_selection_precedes_global_limits(controls_module, case, gas_override):
  controls = build_controls(controls_module, case, lookahead=True)
  set_frame(controls, 210)
  model = controls.sm["modelV2"]
  model.action.lateralActionSource = "plan"
  model.action.legacyLookahead = {"contract": "planOrientationV1", "desiredCurvature": 0.2, "horizonSeconds": 0.6}
  controls.sm["carState"].gasPressed = gas_override
  speed = controls.sm["carState"].vEgo
  roll = controls.sm["vehicleParameters"].roll
  eligible = controls.LaC.supports_legacy_curvature_lookahead
  requested = model.action.legacyLookahead.desiredCurvature if eligible else model.action.desiredCurvature
  expected, limited = clip_curvature(speed, controls.desired_curvature, requested, roll, gas_override)
  original_update = controls.LaC.update
  received = []

  def record_update(*args):
    received.append((args[5], args[7]))
    return original_update(*args)

  controls.LaC.update = record_update
  cc, _ = controls.state_control()
  assert controls.desired_curvature == expected
  assert cc.actuators.curvature == pytest.approx(expected, rel=1e-7)
  assert received == [(expected, limited)]
  assert controls.lateral_action_selection == (Selection.legacyLookahead if eligible else Selection.controllerUnsupported)


@pytest.mark.parametrize("case", CONTROL_CASES)
def test_legacy_permission_cannot_change_disabled_behavior(controls_module, case):
  expected_controls = build_controls(controls_module, case, lookahead=False)
  actual_controls = build_controls(controls_module, case, lookahead=False)
  for frame in range(120, 250):
    set_frame(expected_controls, frame)
    set_frame(actual_controls, frame)
    actual_controls.sm["modelV2"].action.lateralActionSource = "plan"
    actual_controls.sm["modelV2"].action.legacyLookahead = {
      "contract": "planOrientationV1", "desiredCurvature": 0.2, "horizonSeconds": 0.6,
    }
    expected_cc, expected_log = expected_controls.state_control()
    actual_cc, actual_log = actual_controls.state_control()
    assert actual_controls.desired_curvature == expected_controls.desired_curvature
    for field in ("curvature", "torque", "steeringAngleDeg", "accel"):
      assert getattr(actual_cc.actuators, field) == getattr(expected_cc.actuators, field)
    for field in expected_log.schema.fields:
      assert getattr(actual_log, field) == getattr(expected_log, field)
