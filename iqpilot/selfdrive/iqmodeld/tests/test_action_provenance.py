# Copyright (c) 2026 IQ.Lvbs. All rights reserved.
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from iqpilot.cereal import log
from iqpilot.cereal import messaging
from iqpilot.selfdrive.controls.lib.curvature_lookahead import Selection, select_lateral_curvature
from iqpilot.selfdrive.controls.lib.drive_helpers import get_curvature_from_plan, smooth_value
from iqpilot.selfdrive.iqmodeld.config import Meta, ModelConstants, Plan
from iqpilot.selfdrive.iqmodeld.daemon import NeuralEngineState
from iqpilot.selfdrive.iqmodeld.driving_action import get_action_from_model
from iqpilot.selfdrive.iqmodeld.messaging import DrivePacketMemory, populate_drive_messages
from iqpilot.selfdrive.iqmodeld.parser import PhaseParser
from iqpilot.selfdrive.iqmodeld.tests.test_iqmodeld_contracts import _phase_sample


def outputs_for(kind):
  times = np.asarray(ModelConstants.T_IDXS)
  plan = np.zeros((1, len(times), ModelConstants.PLAN_WIDTH), dtype=np.float32)
  plan[0, :, Plan.VELOCITY][:, 0] = 20.0
  plan[0, :, Plan.T_FROM_CURRENT_EULER][:, 2] = 0.03 * times ** 2
  plan[0, :, Plan.ORIENTATION_RATE][:, 2] = 0.06 * times
  outputs = {"plan": plan}
  if kind == "nativeAction":
    outputs["action"] = np.asarray([[-0.8, -0.2]], dtype=np.float32)
  if kind in ("nativeCurvature", "synthetic"):
    outputs["desired_curvature"] = np.asarray([[-0.002]], dtype=np.float32)
  if kind == "planplus":
    outputs["planplus"] = plan * 0.25
  return outputs


def runtime(synthetic=False):
  return SimpleNamespace(
    PLANPLUS_CONTROL=1.0, LONG_SMOOTH_SECONDS=0.3, LAT_SMOOTH_SECONDS=0.1,
    MIN_LAT_CONTROL_SPEED=0.3, mlsim=synthetic, generation=12, constants=ModelConstants,
  )


@pytest.mark.parametrize("kind", ["nativeAction", "nativeCurvature", "plan", "planplus", "synthetic"])
@pytest.mark.parametrize("speed", [0.0, 0.2, 20.0, 27.0])
def test_native_producer_tags_actual_branch_without_authorizing_lookahead(kind, speed):
  previous = log.ModelDataV2.Action(desiredCurvature=0.004, desiredAcceleration=0.1)
  state = runtime(kind == "synthetic")
  outputs = outputs_for(kind)
  action = NeuralEngineState.get_action_from_model(state, outputs, previous, 0.45, 0.4, speed)
  expected_source = kind if kind in ("nativeAction", "nativeCurvature") else "plan"
  assert action.lateralActionSource == expected_source
  assert action.legacyLookahead.contract == "unsupported"
  model = log.ModelDataV2.new_message(action=action)
  assert select_lateral_curvature(model, True, 0.0, None, True, True) == (action.desiredCurvature, Selection.modelUnsupported)

  if kind == "nativeAction":
    raw_curvature = outputs["action"][0, 0] / max(1.0, speed) ** 2
    raw_accel = outputs["action"][0, 1]
  else:
    plan = outputs["plan"][0]
    if kind == "planplus":
      plan = plan + (0.75 if speed > 25.0 else 1.0) * outputs["planplus"][0]
    raw_curvature = (outputs["desired_curvature"][0, 0] if kind == "nativeCurvature" else
                     get_curvature_from_plan(plan[:, Plan.T_FROM_CURRENT_EULER][:, 2],
                                             plan[:, Plan.ORIENTATION_RATE][:, 2], ModelConstants.T_IDXS, speed, 0.45))
    raw_accel = 0.0
  expected_curvature = smooth_value(raw_curvature, previous.desiredCurvature, 0.1) if speed > 0.3 else previous.desiredCurvature
  assert action.desiredCurvature == pytest.approx(expected_curvature, rel=1e-7, abs=1e-10)
  assert action.desiredAcceleration == pytest.approx(smooth_value(raw_accel, previous.desiredAcceleration, 0.3), rel=1e-7)


@pytest.mark.parametrize("kind", ["nativeAction", "nativeCurvature", "plan"])
@pytest.mark.parametrize("speed", [0.0, 0.2, 20.0])
def test_offload_shared_action_uses_captured_inputs_and_its_existing_branch(kind, speed):
  previous = log.ModelDataV2.Action(desiredCurvature=0.004, desiredAcceleration=0.1)
  outputs = outputs_for(kind)
  action = get_action_from_model(outputs, previous, speed, 0.45, 0.4, 0.1)
  assert action.lateralActionSource == ("nativeAction" if kind == "nativeAction" else "plan")
  assert action.legacyLookahead.contract == "unsupported"
  state = runtime(kind == "nativeCurvature")
  expected = NeuralEngineState.get_action_from_model(state, outputs, previous, 0.45, 0.4, speed)
  for field in ("desiredCurvature", "desiredAcceleration", "shouldStop"):
    assert getattr(action, field) == getattr(expected, field)


def test_publications_keep_action_contract_with_the_same_frame():
  raw = _phase_sample(np.random.default_rng(31))
  outputs = {**PhaseParser().parse_vision_outputs(copy.deepcopy(raw)), **PhaseParser().parse_policy_outputs(copy.deepcopy(raw))}
  action = NeuralEngineState.get_action_from_model(runtime(), outputs, log.ModelDataV2.Action(), 0.45, 0.4, 20.0)
  driving = messaging.new_message("drivingModelData")
  model = messaging.new_message("modelV2")
  populate_drive_messages(driving, model, outputs, action, DrivePacketMemory(), 42, 43, 44, 0.0, 123456, 0.01, True, Meta)
  for packet in (driving.drivingModelData, model.modelV2):
    assert packet.frameId == 42
    for field in ("desiredCurvature", "desiredAcceleration", "shouldStop", "lateralActionSource"):
      assert getattr(packet.action, field) == getattr(action, field)
    assert packet.action.legacyLookahead.contract == "unsupported"


def test_previous_action_permission_is_not_inherited_by_next_frame():
  previous = log.ModelDataV2.Action(
    desiredCurvature=0.004, lateralActionSource="plan",
    legacyLookahead={"contract": "planOrientationV1", "desiredCurvature": 0.02, "horizonSeconds": 0.6},
  )
  outputs = outputs_for("nativeAction")
  for action in (get_action_from_model(outputs, previous, 0.2, 0.45, 0.4),
                 NeuralEngineState.get_action_from_model(runtime(), outputs, previous, 0.45, 0.4, 0.2)):
    assert action.desiredCurvature == previous.desiredCurvature
    assert action.lateralActionSource == "nativeAction"
    assert action.legacyLookahead.contract == "unsupported"


@pytest.mark.parametrize("producer", ["local", "offload"])
@pytest.mark.parametrize("kind", ["nativeAction", "nativeCurvature", "plan", "planplus", "synthetic"])
@pytest.mark.parametrize("speed", [0.0, 0.2, 20.0])
@pytest.mark.parametrize("lat_active", [False, True])
def test_curvature_seed_tracks_vehicle_only_while_lateral_is_inactive(producer, kind, speed, lat_active):
  previous = log.ModelDataV2.Action(desiredCurvature=0.004, desiredAcceleration=0.1)
  expected_previous = log.ModelDataV2.Action(
    desiredCurvature=0.004 if lat_active else -0.012, desiredAcceleration=0.1,
  )
  outputs = outputs_for(kind)
  if producer == "local":
    state = runtime()
    expected = NeuralEngineState.get_action_from_model(state, outputs, expected_previous, 0.45, 0.4, speed)
    action = NeuralEngineState.get_action_from_model(
      state, outputs, previous, 0.45, 0.4, speed, lat_active=lat_active, current_curvature=-0.012,
    )
  else:
    expected = get_action_from_model(outputs, expected_previous, speed, 0.45, 0.4, 0.1)
    action = get_action_from_model(
      outputs, previous, speed, 0.45, 0.4, 0.1, lat_active=lat_active, current_curvature=-0.012,
    )
  for field in ("desiredCurvature", "desiredAcceleration", "shouldStop", "lateralActionSource"):
    assert getattr(action, field) == getattr(expected, field)
  if speed <= 0.3:
    assert action.desiredCurvature == expected_previous.desiredCurvature


@pytest.mark.parametrize("producer", ["local", "offload"])
def test_engagement_keeps_curvature_history_from_last_inactive_frame(producer):
  previous = log.ModelDataV2.Action(desiredCurvature=0.03, desiredAcceleration=0.1)
  outputs = outputs_for("nativeAction")
  if producer == "local":
    state = runtime()
    inactive = NeuralEngineState.get_action_from_model(
      state, outputs, previous, 0.45, 0.4, 0.2, lat_active=False, current_curvature=-0.012,
    )
    active = NeuralEngineState.get_action_from_model(
      state, outputs, inactive, 0.45, 0.4, 20.0, lat_active=True, current_curvature=0.02,
    )
  else:
    inactive = get_action_from_model(
      outputs, previous, 0.2, 0.45, 0.4, 0.1, lat_active=False, current_curvature=-0.012,
    )
    active = get_action_from_model(
      outputs, inactive, 20.0, 0.45, 0.4, 0.1, lat_active=True, current_curvature=0.02,
    )
  assert inactive.desiredCurvature == pytest.approx(-0.012)
  assert active.desiredCurvature == pytest.approx(smooth_value(-0.8 / 20.0 ** 2, -0.012, 0.1))
