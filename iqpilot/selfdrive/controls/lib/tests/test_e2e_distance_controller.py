# Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos

from types import SimpleNamespace
from collections import deque

import numpy as np
import pytest

from iqpilot.cereal import custom
from iqpilot.common.params import Params
from iqpilot.selfdrive.controls.lib import longitudinal_planner
from iqpilot.selfdrive.controls.lib.e2e_distance_controller import E2EDistanceController
from iqpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import T_IDXS
from iqpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import get_T_FOLLOW
from iqpilot.selfdrive.controls.lib.tests.test_longitudinal_planner_accel_state import build_sm, build_planner, CRUISE


class SensorState(dict):
  def __init__(self, *args):
    super().__init__(*args)
    self.logMonoTime = dict.fromkeys(self, 1_000_000_000)
    self.valid = dict.fromkeys(self, True)
    self.alive = dict.fromkeys(self, True)
    self.valid_inputs = True

  def all_checks(self, service_list):
    return self.valid_inputs

  def advance(self):
    self.logMonoTime = {s: t + 50_000_000 for s, t in self.logMonoTime.items()}


def inputs(personality=0, distance=100.0, speed=20.0):
  sm = SensorState(build_sm(speed, distance, speed, CRUISE, v_cruise=speed + 10))
  sm['selfdriveState'].experimentalMode = True
  sm['selfdriveState'].personality = personality
  sm['carControl'].longActive = True
  trajectory = np.column_stack([distance + speed * T_IDXS, np.full(len(T_IDXS), speed)])
  mpc = SimpleNamespace(solution_status=0, lead_xv_0=trajectory, lead_xv_1=trajectory.copy())
  kwargs = dict(a_model=0.0, a_mpc=1.0, a_cruise=1.0, e2e=True, engaged=True, should_stop=False,
                fcw=False, allow_throttle=True, lateral_accel=0.0)
  return sm, mpc, kwargs


def run(controller, sm, mpc, kwargs, steps=100):
  results = []
  for _ in range(steps):
    sm.advance()
    results.append(controller.update(sm, mpc, **kwargs))
  return results


@pytest.mark.parametrize('personality,cap,gap', [(0, 0.5, 1.25), (1, 0.4, 1.45), (2, 0.3, 1.75)])
def test_personality_qualification_caps_and_slew(personality, cap, gap):
  controller = E2EDistanceController(True)
  sm, mpc, kwargs = inputs(personality)
  results = run(controller, sm, mpc, kwargs)
  assert results[:20] == [0.0] * 20
  assert results[-1] == pytest.approx(cap)
  assert np.max(np.diff(results)) <= 0.25 * controller.dt + 1e-9
  assert controller.state.targetTimeGap == gap
  state = custom.IQPlan.new_message(distanceControl=controller.state.to_dict())
  assert state.distanceControl.active


@pytest.mark.parametrize('kind,reason', [
  ('stop', 'stop'), ('gas', 'driverOverride'), ('brake', 'driverOverride'), ('throttle', 'throttle'),
  ('fcw', 'braking'), ('hardBrake', 'braking'), ('laneChange', 'laneChange'), ('curve', 'curve'),
  ('modelCurve', 'curve'), ('modelBrake', 'modelBraking'), ('modelFuture', 'modelBraking'),
  ('modelStop', 'modelBraking'), ('leadBrake', 'leadBraking'), ('leadFuture', 'leadBraking'),
  ('noLead', 'noLead'), ('lowSpeed', 'lowSpeed'), ('confidence', 'unqualifiedLead'),
  ('solver', 'solverFailure'), ('invalid', 'invalidInputs'), ('stale', 'staleInputs'),
  ('staleModel', 'staleInputs'), ('nan', 'invalidInputs'), ('malformed', 'invalidInputs'),
  ('forceDecel', 'stop'), ('mode', 'inactive'), ('disengage', 'inactive'),
])
def test_veto_immediately_clears_correction(kind, reason):
  controller = E2EDistanceController(True)
  sm, mpc, kwargs = inputs()
  assert run(controller, sm, mpc, kwargs)[-1] > 0
  if kind == 'stop':
    kwargs['should_stop'] = True
  if kind == 'gas':
    sm['carState'].gasPressed = True
  if kind == 'brake':
    sm['carState'].brakePressed = True
  if kind == 'throttle':
    kwargs['allow_throttle'] = False
  if kind == 'fcw':
    kwargs['fcw'] = True
  if kind == 'hardBrake':
    sm['modelV2'].meta.hardBrakePredicted = True
  if kind == 'laneChange':
    sm['modelV2'].meta.laneChangeState = 'laneChangeStarting'
  if kind == 'curve':
    kwargs['lateral_accel'] = 1.6
  if kind == 'modelCurve':
    sm['modelV2'].action.desiredCurvature = 0.01
  if kind == 'modelBrake':
    kwargs['a_model'] = -0.5
  if kind == 'modelFuture':
    sm['modelV2'].velocity.x[5] = 18.0
  if kind == 'modelStop':
    sm['modelV2'].velocity.x[-1] = 0.0
  if kind == 'leadBrake':
    sm['radarState'].leadOne.aLeadK = -0.6
  if kind == 'leadFuture':
    mpc.lead_xv_0[3, 1] = 18.0
  if kind == 'noLead':
    sm['radarState'].leadOne.status = False
  if kind == 'lowSpeed':
    sm['carState'].vEgo = 5.0
  if kind == 'confidence':
    sm['radarState'].leadOne.modelProb = 0.8
  if kind == 'solver':
    mpc.solution_status = 1
  if kind == 'invalid':
    sm.valid_inputs = False
  if kind == 'stale':
    sm.logMonoTime['radarState'] -= 300_000_000
  if kind == 'staleModel':
    sm.logMonoTime['modelV2'] -= 300_000_000
  if kind == 'nan':
    sm['modelV2'].velocity.x[0] = float('nan')
  if kind == 'malformed':
    sm['modelV2'].velocity.x = []
  if kind == 'forceDecel':
    sm['controlsState'].forceDecel = True
  if kind == 'mode':
    kwargs['e2e'] = False
  if kind == 'disengage':
    kwargs['engaged'] = False
  sm.advance()
  assert controller.update(sm, mpc, **kwargs) == kwargs['a_model']
  assert controller.correction == 0
  assert not controller.state.active
  assert controller.state.inhibitionReason == reason


@pytest.mark.parametrize('change', ['distance', 'track', 'secondLead'])
def test_lead_changes_requalify(change):
  controller = E2EDistanceController(True)
  sm, mpc, kwargs = inputs()
  run(controller, sm, mpc, kwargs)
  if change == 'distance':
    sm['radarState'].leadOne.dRel += 6.0
  if change == 'track':
    sm['radarState'].leadOne.radarTrackId = 2
  if change == 'secondLead':
    sm['radarState'].leadTwo = sm['radarState'].leadOne
  assert run(controller, sm, mpc, kwargs, 1) == [0.0]
  assert controller.state.inhibitionReason == 'qualifyingLead'
  assert run(controller, sm, mpc, kwargs)[-1] > 0


def test_second_lead_margin_and_personality_change():
  controller = E2EDistanceController(True)
  sm, mpc, kwargs = inputs(distance=36.0)
  assert run(controller, sm, mpc, kwargs)[-1] > 0
  sm['selfdriveState'].personality = 2
  assert run(controller, sm, mpc, kwargs)[-1] == 0
  sm['selfdriveState'].personality = 0
  sm['radarState'].leadTwo = sm['radarState'].leadOne
  sm['radarState'].leadTwo.dRel = 28.0
  assert run(controller, sm, mpc, kwargs)[-1] == 0


@pytest.mark.parametrize('bound', ['a_mpc', 'a_cruise'])
def test_binding_candidate_clears_headroom(bound):
  controller = E2EDistanceController(True)
  sm, mpc, kwargs = inputs()
  run(controller, sm, mpc, kwargs)
  kwargs[bound] = -0.2
  assert run(controller, sm, mpc, kwargs, 1) == [0.0]
  assert controller.correction == 0.0


def test_mild_braking_weight_and_disabled_passthrough():
  sm, mpc, kwargs = inputs()
  kwargs['a_model'] = -0.3
  assert run(E2EDistanceController(True), sm, mpc, kwargs)[-1] == pytest.approx(-0.05)
  controller = E2EDistanceController(False)
  assert controller.update(None, None, **kwargs) == -0.3


def test_planner_disabled_parity_and_existing_cruise_bound():
  baseline, enabled = build_planner(), build_planner()
  baseline.distance_control.enabled = False
  enabled.distance_control.enabled = True
  sm, _, _ = inputs()
  for _ in range(100):
    sm.advance()
    baseline.update(sm)
    enabled.update(sm)
    assert enabled.output_a_target >= baseline.output_a_target - 1e-6
    assert enabled.output_a_target <= enabled.a_cruise + 1e-6
  assert enabled.distance_control.state.active
  enabled.distance_control.enabled = False
  enabled.v_desired_filter.x = baseline.v_desired_filter.x
  enabled.a_desired = baseline.a_desired
  enabled.a_cruise = baseline.a_cruise
  enabled.mpc.reset()
  baseline.mpc.reset()
  sm.advance()
  baseline.update(sm)
  enabled.update(sm)
  assert enabled.output_a_target == baseline.output_a_target
  assert enabled.output_should_stop == baseline.output_should_stop


def test_parameter_is_default_off_and_read_at_initialization(tmp_path, monkeypatch):
  params = Params(str(tmp_path))
  monkeypatch.setattr(longitudinal_planner, 'Params', lambda: params)
  planner = build_planner()
  assert not planner.distance_control.enabled
  params.put_bool('IQE2EDistanceControl', True)
  assert not planner.distance_control.enabled
  assert build_planner().distance_control.enabled


@pytest.mark.parametrize('scenario', ['acc', 'noLead', 'lowSpeed', 'stock'])
def test_ineligible_modes_match_disabled_planner(scenario):
  speed = 4.0 if scenario == 'lowSpeed' else 20.0
  baseline, enabled = build_planner(init_v=speed), build_planner(init_v=speed)
  baseline.distance_control.enabled = False
  enabled.distance_control.enabled = True
  sm, _, _ = inputs(speed=speed)
  if scenario == 'acc':
    sm['selfdriveState'].experimentalMode = False
  if scenario == 'noLead':
    sm['radarState'].leadOne.status = False
  if scenario == 'stock':
    baseline.CP.openpilotLongitudinalControl = False
    enabled.CP.openpilotLongitudinalControl = False
  for _ in range(40):
    sm.advance()
    baseline.update(sm)
    enabled.update(sm)
    assert enabled.output_a_target == baseline.output_a_target
    assert enabled.output_should_stop == baseline.output_should_stop


def simulate(personality, speed, excess, event=None):
  planner = build_planner(init_v=speed)
  planner.CP.longitudinalActuatorDelay = 0.6
  planner.distance_control.enabled = True
  planner.mpc._read_new_lead_mpc = lambda: True
  gap = 3 + get_T_FOLLOW(personality) * speed + excess
  velocity = speed
  acceleration = 0.0
  pending = deque([0.0] * 12)
  trace = []
  for frame in range(1800):
    lead_speed = max(0.0, speed - max(0, frame - 400) * 0.05 * 3) if event == 'brake' else speed
    sm, _, _ = inputs(personality, gap, velocity)
    sm['carState'].aEgo = acceleration
    sm['carState'].vCruise = (speed + 10) * 3.6
    lead = sm['radarState'].leadOne
    lead.vLead = lead.vLeadK = lead_speed
    lead.vRel = lead_speed - velocity
    lead.aLeadK = -3.0 if event == 'brake' and frame > 400 and lead_speed > 0 else 0.0
    if event == 'cutIn' and frame == 400:
      gap = 3 + get_T_FOLLOW(personality) * velocity - 5.0
      lead.dRel = gap
      lead.radarTrackId = 10
    if event == 'leadLoss' and frame > 400:
      lead.status = False
    if event == 'cruise':
      sm['carState'].vCruise = speed * 3.6
    sm.logMonoTime = {s: int((1 + frame * 0.05) * 1e9) for s in sm}
    planner.update(sm)
    pending.append(float(planner.output_a_target))
    acceleration = pending.popleft()
    velocity = max(0.0, velocity + acceleration * 0.05)
    gap += (lead_speed - velocity) * 0.05
    trace.append((gap, velocity, planner.output_a_target, planner.distance_control.correction))
  return np.asarray(trace)


@pytest.mark.parametrize('personality', [0, 1, 2])
@pytest.mark.parametrize('speed', [10.0, 20.0, 30.0])
@pytest.mark.parametrize('excess', [20.0, 60.0])
def test_delayed_closed_loop_convergence(personality, speed, excess):
  trace = simulate(personality, speed, excess)
  target = 3 + get_T_FOLLOW(personality) * speed
  assert np.min(trace[:, 0]) > 0
  assert np.max(np.abs(trace[-100:, 0] - target)) <= 3.0
  assert np.ptp(trace[-100:, 0]) < 0.5
  assert np.min(trace[:, 2]) >= -3.5
  assert np.max(trace[:, 2]) <= 2.0


@pytest.mark.parametrize('event', ['brake', 'cutIn', 'leadLoss', 'cruise'])
def test_delayed_closed_loop_events(event):
  trace = simulate(0, 20.0, 20.0, event)
  assert np.min(trace[:401, 0] if event == 'leadLoss' else trace[:, 0]) > 0
  if event in ['brake', 'leadLoss']:
    assert np.max(trace[401:, 3]) == 0
  if event == 'cutIn':
    assert trace[400, 3] == 0
  if event == 'cruise':
    assert np.max(trace[:, 1]) <= 20.01


@pytest.mark.parametrize('enabled', [False, True])
def test_numpy_accelerations_serialize_into_the_plan_message(enabled):
  controller = E2EDistanceController(enabled)
  sm, mpc, kwargs = inputs()
  kwargs.update(a_model=np.float64(1.368292852838483), a_mpc=np.float64(2.0), a_cruise=np.float32(2.0))
  run(controller, sm, mpc, kwargs, steps=40)
  state = custom.IQPlan.new_message(distanceControl=controller.state.to_dict())
  assert state.distanceControl.originalAcceleration == pytest.approx(1.368292852838483)
