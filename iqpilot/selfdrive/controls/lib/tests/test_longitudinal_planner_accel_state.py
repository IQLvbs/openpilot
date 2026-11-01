"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos
"""
import numpy as np
import pytest

from iqdbc.car.honda.interface import CarInterface
from iqdbc.car.honda.values import CAR
from iqpilot.cereal import custom, log
import iqpilot.cereal.messaging as messaging
from iqpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
from iqpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner
from iqpilot.selfdrive.iqmodeld.config import ModelConstants

CRUISE = "cruise"
SPEED_LIMIT_ASSIST = "speedLimitAssist"
NAV = "nav"
SOURCES = [CRUISE, SPEED_LIMIT_ASSIST, NAV]
PLAN_SOURCE = custom.IQPlan.LongitudinalPlanSource

V_CRUISE_MS = 25.0
NAV_SPEED_TARGET = 11.0
SLC_SPEED_TARGET = 12.0

APPROACH_V_EGO = 11.2
APPROACH_D_REL = 100.0
APPROACH_STEPS = 250
MIN_SAFE_GAP = 2.0
COAST_THROTTLE_PROB = 0.1


def build_planner(init_v=V_CRUISE_MS, init_a=0.0):
  CP = CarInterface.get_non_essential_params(CAR.HONDA_CIVIC)
  CP_IQ = CarInterface.get_non_essential_params_iq(CP, CAR.HONDA_CIVIC)
  return LongitudinalPlanner(CP, CP_IQ, init_v=init_v, init_a=init_a)


def build_sm(v_ego, d_rel, v_lead, source, enabled=True, throttle_prob=1.0, a_ego=0.0, v_cruise=V_CRUISE_MS):
  radar = messaging.new_message('radarState')
  control = messaging.new_message('controlsState')
  ss = messaging.new_message('selfdriveState')
  car_state = messaging.new_message('carState')
  car_control = messaging.new_message('carControl')
  vehicle_params = messaging.new_message('vehicleParameters')
  model = messaging.new_message('modelV2')
  iq_car_state = messaging.new_message('iqCarState')
  iq_nav_state = messaging.new_message('iqNavState')
  iq_live_data = messaging.new_message('iqLiveData')
  gps = messaging.new_message('gpsLocation')

  lead = log.RadarState.LeadData.new_message()
  lead.dRel = float(d_rel)
  lead.vRel = float(v_lead - v_ego)
  lead.vLead = float(v_lead)
  lead.vLeadK = float(v_lead)
  lead.status = True
  lead.modelProb = 1.0
  radar.radarState.leadOne = lead

  t_idxs = np.array(ModelConstants.T_IDXS)
  position = log.XYZTData.new_message()
  position.x = [float(x) for x in v_ego * t_idxs]
  model.modelV2.position = position
  velocity = log.XYZTData.new_message()
  velocity.x = [float(v_ego) for _ in t_idxs]
  model.modelV2.velocity = velocity
  acceleration = log.XYZTData.new_message()
  acceleration.x = [0.0 for _ in t_idxs]
  model.modelV2.acceleration = acceleration
  model.modelV2.action.desiredAcceleration = 0.0
  model.modelV2.meta.disengagePredictions.gasPressProbs = [float(throttle_prob) for _ in range(6)]

  lead_times = np.array(ModelConstants.LEAD_T_IDXS)
  for lead_prediction in model.modelV2.leadsV3:
    lead_prediction.prob = 1.0
    lead_prediction.x = [float(d_rel + v_lead * t) for t in lead_times]
    lead_prediction.v = [float(v_lead) for _ in lead_times]

  control.controlsState.longControlState = LongCtrlState.pid if enabled else LongCtrlState.off
  ss.selfdriveState.enabled = enabled
  car_state.carState.vEgo = float(v_ego)
  car_state.carState.aEgo = float(a_ego)
  car_state.carState.standstill = bool(v_ego < 0.01)
  car_state.carState.vCruise = float(v_cruise * 3.6)
  car_control.carControl.orientationNED = [0.0, 0.0, 0.0]

  if source == NAV:
    iq_nav_state.iqNavState.longitudinalEngaged = True
    iq_nav_state.iqNavState.valid = True
    iq_nav_state.iqNavState.speedTarget = NAV_SPEED_TARGET
    iq_nav_state.iqNavState.accelTarget = 0.0

  return {
    'radarState': radar.radarState,
    'carState': car_state.carState,
    'carControl': car_control.carControl,
    'controlsState': control.controlsState,
    'selfdriveState': ss.selfdriveState,
    'vehicleParameters': vehicle_params.vehicleParameters,
    'modelV2': model.modelV2,
    'iqCarState': iq_car_state.iqCarState,
    'iqNavState': iq_nav_state.iqNavState,
    'iqLiveData': iq_live_data.iqLiveData,
    'gpsLocation': gps.gpsLocation,
  }


def stub_speed_limit_assist(planner):
  planner.slimit.update = lambda *args, **kwargs: SLC_SPEED_TARGET


def run_approach(planner, source, v_ego_0=APPROACH_V_EGO, d_rel_0=APPROACH_D_REL,
                 steps=APPROACH_STEPS, throttle_prob=COAST_THROTTLE_PROB):
  if source == SPEED_LIMIT_ASSIST:
    stub_speed_limit_assist(planner)

  v_ego = v_ego_0
  d_rel = d_rel_0
  prev_output_a_target = None
  trace = []
  for _ in range(steps):
    planner.update(build_sm(v_ego, d_rel, 0.0, source, throttle_prob=throttle_prob))
    trace.append({
      'v_ego': v_ego,
      'd_rel': d_rel,
      'accels_0': float(planner.a_desired_trajectory[0]),
      'prev_output_a_target': prev_output_a_target,
      'output_a_target': float(planner.output_a_target),
    })
    prev_output_a_target = float(planner.output_a_target)
    v_ego = max(0.0, v_ego + prev_output_a_target * planner.dt)
    d_rel = max(0.0, d_rel - v_ego * planner.dt)
  return trace


@pytest.mark.parametrize("source", SOURCES)
def test_mpc_initial_accel_state_carries_previous_command(source):
  planner = build_planner(init_v=APPROACH_V_EGO)
  trace = run_approach(planner, source)

  for i, step in enumerate(trace):
    if step['prev_output_a_target'] is None:
      continue
    assert step['accels_0'] == pytest.approx(step['prev_output_a_target'], abs=1e-6), (
      f"step {i} source={source}: MPC initial accel state was {step['accels_0']:.4f} "
      f"but the previous commanded accel was {step['prev_output_a_target']:.4f}"
    )


@pytest.mark.parametrize("source", SOURCES)
def test_brakes_for_stopped_lead(source):
  planner = build_planner(init_v=APPROACH_V_EGO)
  trace = run_approach(planner, source)

  min_gap = min(step['d_rel'] for step in trace)
  assert min_gap > MIN_SAFE_GAP, (
    f"source={source}: closed to {min_gap:.2f} m of a stopped lead first seen at "
    f"{APPROACH_D_REL:.0f} m while coasting from {APPROACH_V_EGO:.1f} m/s"
  )


@pytest.mark.parametrize("source,expected", [
  (CRUISE, V_CRUISE_MS),
  (SPEED_LIMIT_ASSIST, SLC_SPEED_TARGET),
  (NAV, NAV_SPEED_TARGET),
])
def test_speed_source_arbitration_unchanged(source, expected):
  planner = build_planner(init_v=APPROACH_V_EGO)
  if source == SPEED_LIMIT_ASSIST:
    stub_speed_limit_assist(planner)
  planner.update(build_sm(APPROACH_V_EGO, APPROACH_D_REL, 0.0, source))

  assert planner.output_v_target == pytest.approx(expected, abs=1e-6)
  assert planner.source == getattr(PLAN_SOURCE, source)


def test_cruise_accel_initializes_from_planner_accel():
  planner = build_planner(init_a=-0.35)

  assert planner.a_cruise == pytest.approx(-0.35)


def test_cruise_accel_resets_from_measured_accel():
  a_ego = -0.45
  v_ego = 20.0
  planner = build_planner(init_v=v_ego)
  planner.a_cruise = 0.5
  planner.update(build_sm(v_ego, APPROACH_D_REL, v_ego, CRUISE, enabled=False, a_ego=a_ego, v_cruise=v_ego + a_ego))

  assert planner.a_cruise == pytest.approx(a_ego, abs=1e-6)
