from types import SimpleNamespace

import numpy as np
import pytest

from iqdbc.car.honda.interface import CarInterface
from iqdbc.car.honda.values import CAR
from iqpilot.cereal import log
from iqpilot.common.params import Params
from iqpilot.selfdrive.controls.lib.custom_stop_distance import CustomStopDistance
from iqpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
from iqpilot.selfdrive.controls.lib.longitudinal_planner import CONTROL_N_T_IDX, LongitudinalPlanner
from iqpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import T_IDXS
from iqpilot.selfdrive.test.longitudinal_maneuvers.stop_offset import ActuatorPlant, build_frame, lead_fields


class SensorFrame(dict):
  def __init__(self, values):
    super().__init__(values)
    self.logMonoTime = dict.fromkeys(values, 1_000_000_000)
    self.valid = self.alive = self.freq_ok = dict.fromkeys(values, True)

  def all_checks(self, service_list):
    return True


def planner_and_frame(offset, *, openpilot_long=True):
  cp = CarInterface.get_non_essential_params(CAR.HONDA_CIVIC)
  cp.openpilotLongitudinalControl = openpilot_long
  cp.longitudinalActuatorDelay = 0.5
  cp_iq = CarInterface.get_non_essential_params_iq(cp, CAR.HONDA_CIVIC)
  if offset is None:
    Params().put('IQCustomStopDistance', 0)
  planner = LongitudinalPlanner(cp, cp_iq, init_v=12.0, stop_offset_m=offset)
  sm = SensorFrame(build_frame(ActuatorPlant(12.0), 85.0, 0.0, 0.0, 1.0, True, LongCtrlState.pid))
  return planner, sm


@pytest.mark.parametrize('offset', (None, 0, 2))
def test_published_plan_keeps_measurements_and_stop_target_separate(offset):
  planner, sm = planner_and_frame(offset)
  requested = offset or 0
  planner.update(sm)
  sent = {}
  planner.publish(sm, SimpleNamespace(send=lambda name, message: sent.update({name: message})))
  plan = sent['longitudinalPlan'].longitudinalPlan
  assert plan.leadDistance == 85.0
  assert plan.stopOffset.version == 1
  assert plan.stopOffset.configurationValid
  assert plan.stopOffset.requestedOffset == requested
  assert list(plan.stopOffset.supportedSources) == ['lead']
  assert plan.stopOffset.baselineDistance == 82.0
  assert plan.stopOffset.distance == 82.0 - requested
  assert plan.stopOffset.valid
  assert len(plan.speeds) == len(plan.accels) == len(plan.jerks) == 17
  np.testing.assert_allclose(plan.speeds, planner.stop_plan.trajectory.speeds, atol=1e-6)
  np.testing.assert_allclose(plan.accels, planner.stop_plan.trajectory.accelerations, atol=1e-6)
  assert plan.aTarget == pytest.approx(planner.stop_plan.trajectory.acceleration, abs=1e-6)


def test_offset_does_not_change_physical_mpc_inputs_or_fcw_for_identical_inputs():
  baseline, baseline_sm = planner_and_frame(0)
  adjusted, adjusted_sm = planner_and_frame(2)
  baseline.update(baseline_sm)
  adjusted.update(adjusted_sm)
  np.testing.assert_array_equal(adjusted.mpc.params, baseline.mpc.params)
  np.testing.assert_array_equal(adjusted.mpc.lead_xv_0, baseline.mpc.lead_xv_0)
  np.testing.assert_array_equal(adjusted.mpc.lead_xv_1, baseline.mpc.lead_xv_1)
  assert adjusted.fcw == baseline.fcw
  assert lead_fields(adjusted_sm['radarState'].leadOne) == lead_fields(baseline_sm['radarState'].leadOne)


@pytest.mark.parametrize('offset', (0, 2))
def test_stock_acc_does_not_advertise_an_actuated_offset(offset):
  planner, sm = planner_and_frame(offset, openpilot_long=False)
  planner.update(sm)
  assert not planner.stop_plan.active
  assert planner.stop_plan.trajectory is None


@pytest.mark.parametrize('offset', (-2, -1))
def test_negative_stored_settings_are_invalid(offset):
  planner, sm = planner_and_frame(0)
  planner.custom_stop_distance = CustomStopDistance(params=SimpleNamespace(get=lambda *args, **kwargs: offset))
  planner.update(sm)
  assert planner.stop_plan.state == 'invalidConfiguration'
  assert not planner.stop_plan.active
  assert planner.stop_plan.trajectory is None


def test_invalid_setting_does_not_enable_the_zero_offset_controller():
  planner, sm = planner_and_frame(0)
  planner.custom_stop_distance = CustomStopDistance(params=SimpleNamespace(get=lambda *args, **kwargs: 11))
  planner.update(sm)
  assert planner.stop_plan.state == 'invalidConfiguration'
  assert not planner.stop_plan.active
  assert planner.stop_plan.trajectory is None


def test_stronger_physical_braking_keeps_its_trajectory():
  planner, sm = planner_and_frame(2)
  sm['radarState'].leadOne.dRel = 30.0
  planner.update(sm)
  assert planner.output_a_target < planner.stop_plan.trajectory.acceleration
  np.testing.assert_allclose(planner.v_desired_trajectory, np.interp(CONTROL_N_T_IDX, T_IDXS, planner.mpc.v_solution))


def test_old_plan_defaults_do_not_become_a_stop_target():
  plan = log.LongitudinalPlan.new_message()
  assert plan.stopOffset.version == 0
  assert not plan.stopOffset.valid


def test_unverified_model_stop_is_unavailable_and_keeps_baseline_behavior():
  outputs = []
  for offset in (0, 2):
    planner, sm = planner_and_frame(offset)
    sm['radarState'].leadOne.status = False
    sm['modelV2'].stopPoint.valid = True
    sm['modelV2'].stopPoint.distance = 20.0
    sm['modelV2'].stopPoint.probability = 1.0
    sm['modelV2'].action.shouldStop = True
    planner.update(sm)
    assert not planner.stop_plan.active
    outputs.append((planner.output_a_target, planner.output_should_stop))
  assert outputs[0] == outputs[1]
