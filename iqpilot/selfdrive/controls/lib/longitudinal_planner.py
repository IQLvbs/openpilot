#!/usr/bin/env python3
import math
import numpy as np

import iqpilot.cereal.messaging as messaging
from iqdbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from iqpilot.common.constants import CV
from iqpilot.common.filter_simple import FirstOrderFilter
from iqpilot.common.params import Params, UnknownKeyName
from iqpilot.common.realtime import DT_MDL
from iqpilot.selfdrive.iqmodeld.config import ModelConstants
from iqpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
from iqpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LongitudinalMpc, LongitudinalPlanSource
from iqpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import T_IDXS as T_IDXS_MPC
from iqpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N, DEFAULT_STOPPING_SPEED, get_accel_from_plan
from iqpilot.selfdrive.car.cruise import V_CRUISE_MAX, V_CRUISE_UNSET
from iqpilot.common.swaglog import cloudlog
from iqpilot.common.issue_debug import log_issue_limited

from iqpilot.selfdrive.controls.lib.iq_longitudinal_planner import LongitudinalPlannerIQ
from iqpilot.selfdrive.controls.lib.iq_longitudinal_planner import LongitudinalPlanSource as IQPlanSource
from iqpilot.selfdrive.controls.lib.accel_boost import AccelBoost
from iqpilot.selfdrive.controls.lib.e2e_distance_controller import E2EDistanceController
from iqpilot.selfdrive.controls.lib.stop_offset import CONTROLLER_VERSION, LeadStopTargets, StopFrame, StopOffset, StopOffsetPlan

A_CRUISE_MAX_VALS = [2.0, 1.6, 0.8, 0.6]
A_CRUISE_MAX_BP = [0., 10.0, 25., 40.]
A_CRUISE_MIN = -1.2
J_CRUISE_VALS = [1.6, 1.2, 0.8, 0.6]
J_CRUISE_BP = [0., 10.0, 25., 40.]
# navd shapes turn/curve targets assuming this deceleration.
A_CRUISE_MIN_NAV = -1.5
CRUISE_STOP_SPEED = 1.0
CONTROL_N_T_IDX = ModelConstants.T_IDXS[:CONTROL_N]
ALLOW_THROTTLE_THRESHOLD = 0.4
MIN_ALLOW_THROTTLE_SPEED = 2.5

LAUNCH_DISARM_SPEED = 2.0
LAUNCH_COMMIT_T = 3.5
LAUNCH_MOVING_SPEED = 1.2
LAUNCH_MAX_ACCEL = 1.5

E2E_CRUISE_CONVERGENCE_TAU = 15.0
E2E_CRUISE_ACCEL_MAX = 0.5
E2E_MODEL_SPEED_HORIZON = 5.0
E2E_ACCEL_INTENT_BP = [-0.05, 0.05]
E2E_MODEL_SPEED_INTENT_BP = [-0.5, 0.0]

_A_TOTAL_MAX_V = [1.7, 3.2]
_A_TOTAL_MAX_BP = [20., 40.]

MODE_BLEND_MAX_TIME = 3.0
MODE_BLEND_JERK_UP = 1.0
MODE_BLEND_JERK_DOWN = 2.5
MODE_BLEND_BRAKE_PASS = -1.0

def get_max_accel(v_ego):
  return np.interp(v_ego, A_CRUISE_MAX_BP, A_CRUISE_MAX_VALS)

def get_coast_accel(pitch):
  return np.sin(pitch) * -5.65 - 0.3

def get_lead_distance(radarState):
  if radarState.leadOne.status and (not radarState.leadTwo.status or radarState.leadOne.dRel < radarState.leadTwo.dRel):
    return radarState.leadOne.dRel
  if radarState.leadTwo.status:
    return radarState.leadTwo.dRel
  return 0

def get_cruise_accel(e2e, v_cruise, v_ego, a_cruise_prev, a_target_prev, angle_steers, CP, dt, accel_coast, allow_throttle, min_accel=A_CRUISE_MIN):
  max_accel = ACCEL_MAX if e2e else get_max_accel(v_ego)

  if not e2e:
    a_total_max = np.interp(v_ego, _A_TOTAL_MAX_BP, _A_TOTAL_MAX_V)
    a_y = v_ego ** 2 * angle_steers * CV.DEG_TO_RAD / (CP.steerRatio * CP.wheelbase)
    a_x_allowed = math.sqrt(max(a_total_max ** 2 - a_y ** 2, 0.))
    max_accel = min(max_accel, a_x_allowed)
  if not allow_throttle:
    clipped_accel_coast = max(accel_coast, ACCEL_MIN)
    coast_limit = np.interp(v_ego, [MIN_ALLOW_THROTTLE_SPEED, MIN_ALLOW_THROTTLE_SPEED*2], [max_accel, clipped_accel_coast])
    max_accel = min(max_accel, coast_limit)

  target_accel = np.clip(v_cruise - v_ego, min_accel, max_accel)
  j_cruise = np.interp(v_ego, J_CRUISE_BP, J_CRUISE_VALS)
  target_accel = float(np.clip(target_accel, min(a_cruise_prev, a_target_prev) - j_cruise * dt, a_cruise_prev + j_cruise * dt))

  cruise_should_stop = v_cruise == 0.0 and v_ego < CRUISE_STOP_SPEED
  return target_accel, cruise_should_stop


def get_e2e_accel(v_ego, v_cruise, model_v, a_target, should_stop):
  if should_stop or v_cruise <= v_ego or len(model_v) != len(T_IDXS_MPC):
    return a_target

  accel = min((v_cruise - v_ego) / E2E_CRUISE_CONVERGENCE_TAU, E2E_CRUISE_ACCEL_MAX)
  if accel <= a_target:
    return a_target
  accel_intent = np.interp(a_target, E2E_ACCEL_INTENT_BP, [0.0, 1.0])
  model_speed = np.interp(E2E_MODEL_SPEED_HORIZON, T_IDXS_MPC, model_v)
  speed_intent = np.interp(model_speed - v_ego, E2E_MODEL_SPEED_INTENT_BP, [0.0, 1.0])
  return float(np.interp(min(accel_intent, speed_intent), [0.0, 1.0], [a_target, accel]))


def limit_mode_transition(a_target, a_prev, dt):
  upper = a_prev + MODE_BLEND_JERK_UP * dt
  if a_target <= MODE_BLEND_BRAKE_PASS:
    return min(a_target, upper)
  lower = a_prev - MODE_BLEND_JERK_DOWN * dt
  return float(np.clip(a_target, lower, upper))


def get_accel_candidates(e2e, has_lead, mpc_candidate, cruise_candidate, e2e_candidate):
  candidates = []
  if not e2e or has_lead:
    candidates.append(mpc_candidate)
  candidates.append(cruise_candidate)
  if e2e:
    candidates.append(e2e_candidate)
  return candidates


class LongitudinalPlanner(LongitudinalPlannerIQ):
  def __init__(self, CP, CP_IQ, init_v=0.0, init_a=0.0, dt=DT_MDL, *, stop_offset_m=None):
    self.CP = CP
    self.stopping_speed = CP_IQ.longitudinalStoppingSpeedOverride or DEFAULT_STOPPING_SPEED
    self.mpc = LongitudinalMpc(dt=dt)
    LongitudinalPlannerIQ.__init__(self, self.CP, CP_IQ, self.mpc, stop_offset_m=stop_offset_m)
    self.fcw = False
    self.dt = dt
    self.allow_throttle = True

    self.a_desired = init_a
    self.v_desired_filter = FirstOrderFilter(init_v, 2.0, self.dt)
    self.a_cruise = init_a
    self.output_a_target = 0.0
    self.output_should_stop = False
    self.launch_armed = False
    self.prev_e2e = False
    self.mode_blend_timer = 0.0
    try:
      distance_control_enabled = Params().get_bool("IQE2EDistanceControl")
    except UnknownKeyName:
      distance_control_enabled = False
    self.distance_control = E2EDistanceController(distance_control_enabled, dt)
    try:
      accel_boost_enabled = Params().get_bool("IQGasOverrideBoost")
    except UnknownKeyName:
      accel_boost_enabled = False
    self.accel_boost = AccelBoost(accel_boost_enabled, dt)

    self.v_desired_trajectory = np.zeros(CONTROL_N)
    self.a_desired_trajectory = np.zeros(CONTROL_N)
    self.j_desired_trajectory = np.zeros(CONTROL_N)
    self.stop_offset = StopOffset(max(self.custom_stop_distance.distance, 0.0), CONTROL_N_T_IDX, delay=CP.longitudinalActuatorDelay, dt=dt)
    self.lead_stops = LeadStopTargets(dt)
    self.stop_plan = StopOffsetPlan()
    self.stop_position = 0.0
    self.stop_time = None
    self.stop_previous_speed = init_v

  def update_stop_offset(self, sm, reset_state):
    cs = sm['carState']
    offset = max(self.custom_stop_distance.distance, 0.0)
    if not self.custom_stop_distance.valid:
      self.stop_offset.reset()
      self.lead_stops.reset()
      self.stop_plan = StopOffsetPlan(state='invalidConfiguration')
      self.stop_time = None
      self.stop_previous_speed = cs.vEgo
      return
    if offset != self.stop_offset.offset:
      self.stop_offset = StopOffset(offset, CONTROL_N_T_IDX, delay=self.CP.longitudinalActuatorDelay, dt=self.dt)
      self.lead_stops.reset()
    now = (self.stop_time + self.dt) if self.stop_time is not None else 0.0
    radar_fresh = True
    if hasattr(sm, 'logMonoTime'):
      now = max(sm.logMonoTime[s] for s in ('carState', 'modelV2', 'radarState')) * 1e-9
      radar_fresh = (sm.valid['radarState'] and sm.alive['radarState']
                     and 0.0 <= now - sm.logMonoTime['radarState'] * 1e-9 <= 0.2)
    elapsed = self.dt if self.stop_time is None else now - self.stop_time
    if not 0.0 <= elapsed <= 0.2:
      self.stop_offset.reset()
      self.lead_stops.reset()
      elapsed = 0.0
    self.stop_position += (self.stop_previous_speed + cs.vEgo) * 0.5 * elapsed
    self.stop_previous_speed, self.stop_time = cs.vEgo, now
    active = (self.CP.openpilotLongitudinalControl and not reset_state
              and sm['selfdriveState'].enabled and sm['carControl'].longActive)
    override = cs.gasPressed or bool(getattr(sm['iqCarState'], 'accelPressed', False))
    frame = StopFrame(now, self.stop_position, cs.vEgo, cs.aEgo, active, override, self.a_desired, self.v_desired_filter.x)
    if active and not override:
      rs = sm['radarState']
      targets = self.lead_stops.update(frame, (rs.leadOne, rs.leadTwo)) if radar_fresh else ()
    else:
      self.lead_stops.reset()
      targets = ()
    self.stop_plan = self.stop_offset.update(frame, targets)

  @staticmethod
  def parse_model(model_msg):
    if (len(model_msg.position.x) == ModelConstants.IDX_N and
      len(model_msg.velocity.x) == ModelConstants.IDX_N and
      len(model_msg.acceleration.x) == ModelConstants.IDX_N):
      x = np.interp(T_IDXS_MPC, ModelConstants.T_IDXS, model_msg.position.x)
      v = np.interp(T_IDXS_MPC, ModelConstants.T_IDXS, model_msg.velocity.x)
      a = np.interp(T_IDXS_MPC, ModelConstants.T_IDXS, model_msg.acceleration.x)
      j = np.zeros(len(T_IDXS_MPC))
    else:
      x = np.zeros(len(T_IDXS_MPC))
      v = np.zeros(len(T_IDXS_MPC))
      a = np.zeros(len(T_IDXS_MPC))
      j = np.zeros(len(T_IDXS_MPC))
    if len(model_msg.meta.disengagePredictions.gasPressProbs) > 1:
      throttle_prob = model_msg.meta.disengagePredictions.gasPressProbs[1]
    else:
      throttle_prob = 1.0
    return x, v, a, j, throttle_prob

  def update(self, sm):
    LongitudinalPlannerIQ.update(self, sm)

    if len(sm['carControl'].orientationNED) == 3:
      accel_coast = get_coast_accel(sm['carControl'].orientationNED[1])
    else:
      accel_coast = ACCEL_MAX

    v_ego = sm['carState'].vEgo
    v_cruise_kph = min(sm['carState'].vCruise, V_CRUISE_MAX)
    v_cruise = v_cruise_kph * CV.KPH_TO_MS
    if sm['controlsState'].forceDecel:
      v_cruise = 0.0

    long_control_off = sm['controlsState'].longControlState == LongCtrlState.off

    reset_state = long_control_off if self.CP.openpilotLongitudinalControl else not sm['selfdriveState'].enabled
    v_cruise_initialized = sm['carState'].vCruise != V_CRUISE_UNSET
    reset_state = reset_state or not v_cruise_initialized
    steer_angle_without_offset = sm['carState'].steeringAngleDeg - sm['vehicleParameters'].angleOffsetDeg

    if reset_state:
      self.v_desired_filter.x = v_ego
      self.a_desired = np.clip(sm['carState'].aEgo, ACCEL_MIN, ACCEL_MAX)
      self.a_cruise = self.a_desired

    self.v_desired_filter.x = max(0.0, self.v_desired_filter.update(v_ego))
    _, model_v, model_a, _, throttle_prob = self.parse_model(sm['modelV2'])
    self.allow_throttle = throttle_prob > ALLOW_THROTTLE_THRESHOLD or v_ego <= MIN_ALLOW_THROTTLE_SPEED

    if sm['controlsState'].forceDecel:
      v_cruise = 0.0

    personality = sm['selfdriveState'].personality
    self.update_stop_offset(sm, reset_state)
    prev_accel_constraint = not (reset_state or sm['carState'].standstill)
    self.mpc.set_weights(prev_accel_constraint, personality=personality)
    self.mpc.set_cur_state(self.v_desired_filter.x, self.a_desired)
    self.mpc.update(sm['modelV2'], sm['radarState'], personality=personality)
    v_cruise = LongitudinalPlannerIQ.update_targets(self, sm, self.v_desired_filter.x, v_cruise)

    self.v_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC, self.mpc.v_solution)
    self.a_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC, self.mpc.a_solution)
    self.j_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC[:-1], self.mpc.j_solution)

    self.fcw = self.mpc.crash_cnt > 2 and not sm['carState'].standstill
    if self.fcw:
      cloudlog.info("FCW triggered")
    a_prev = self.a_desired

    action_t =  self.CP.longitudinalActuatorDelay + DT_MDL
    output_a_target_mpc, output_should_stop_mpc = get_accel_from_plan(self.v_desired_trajectory, self.a_desired_trajectory, CONTROL_N_T_IDX,
                                                                        action_t=action_t, stopping_speed=self.stopping_speed)

    output_a_target_e2e = sm['modelV2'].action.desiredAcceleration
    output_should_stop_e2e = sm['modelV2'].action.shouldStop
    if self.is_e2e(sm) and not self.mpc.status:
      output_a_target_e2e = get_e2e_accel(v_ego, v_cruise, model_v, output_a_target_e2e, output_should_stop_e2e)

    if sm['carState'].standstill:
      self.launch_armed = True
    elif v_ego > LAUNCH_DISARM_SPEED:
      self.launch_armed = False
    if (self.launch_armed and self.is_e2e(sm) and not output_should_stop_e2e and
        np.interp(LAUNCH_COMMIT_T, T_IDXS_MPC, model_v) > LAUNCH_DISARM_SPEED):
      t_cut = min(float(T_IDXS_MPC[np.argmax(model_v > LAUNCH_MOVING_SPEED)]), LAUNCH_COMMIT_T)
      t_shifted = T_IDXS_MPC + t_cut
      v_shifted = np.interp(t_shifted, T_IDXS_MPC, model_v)
      a_shifted = np.interp(t_shifted, T_IDXS_MPC, model_a)
      a_launch = get_accel_from_plan(v_shifted, a_shifted, T_IDXS_MPC, action_t=action_t)[0]
      a_launch_max = np.interp(v_ego, [LAUNCH_MOVING_SPEED, LAUNCH_DISARM_SPEED], [LAUNCH_MAX_ACCEL, 0.])
      output_a_target_e2e = max(output_a_target_e2e, min(a_launch, a_launch_max))

    e2e = self.is_e2e(sm)
    self.a_cruise, cruise_should_stop = get_cruise_accel(e2e, v_cruise, v_ego, self.a_cruise, a_prev,
                                                          steer_angle_without_offset, self.CP, self.dt,
                                                          accel_coast, self.allow_throttle,
                                                          A_CRUISE_MIN_NAV if self.source == IQPlanSource.nav else A_CRUISE_MIN)

    model_limited = e2e and output_a_target_e2e < min(output_a_target_mpc, self.a_cruise) - 0.1
    self.accel_boost.update(sm['selfdriveState'].enabled, v_ego, sm['carState'].gasPressed, model_limited)
    output_a_target_e2e = self.accel_boost.apply(output_a_target_e2e)

    if self.distance_control.enabled:
      lateral_accel = v_ego ** 2 * steer_angle_without_offset * CV.DEG_TO_RAD / (self.CP.steerRatio * self.CP.wheelbase)
    else:
      lateral_accel = 0.0
    output_a_target_e2e = self.distance_control.update(
      sm, self.mpc, a_model=output_a_target_e2e, a_mpc=output_a_target_mpc, a_cruise=self.a_cruise,
      e2e=e2e, engaged=self.CP.openpilotLongitudinalControl and not reset_state and sm['selfdriveState'].enabled and sm['carControl'].longActive,
      should_stop=output_should_stop_e2e or output_should_stop_mpc or cruise_should_stop or self.forcing_stop or self.stop_plan.active,
      fcw=self.fcw, allow_throttle=self.allow_throttle, lateral_accel=lateral_accel,
    )

    candidates = get_accel_candidates(
      e2e,
      self.mpc.status,
      (output_a_target_mpc, self.mpc.source, output_should_stop_mpc),
      (self.a_cruise, LongitudinalPlanSource.cruise, cruise_should_stop),
      (output_a_target_e2e, LongitudinalPlanSource.e2e, output_should_stop_e2e),
    )
    stop_trajectory = self.stop_plan.trajectory
    if stop_trajectory is not None:
      stop_source = LongitudinalPlanSource.lead1 if self.stop_plan.target.source == 'lead1' else LongitudinalPlanSource.lead0
      candidates.append((stop_trajectory.acceleration, stop_source, stop_trajectory.should_stop))

    output_a_target, self.mpc.source, _ = min(candidates, key=lambda c: c[0])
    self.output_should_stop = any(should_stop for _, _, should_stop in candidates)

    self.output_should_stop = self.output_should_stop or self.forcing_stop

    if e2e != self.prev_e2e:
      self.mode_blend_timer = MODE_BLEND_MAX_TIME
    self.prev_e2e = e2e
    if reset_state or self.fcw or self.output_should_stop:
      self.mode_blend_timer = 0.0
    if self.mode_blend_timer > 0.0:
      limited_a_target = limit_mode_transition(output_a_target, a_prev, self.dt)
      self.mode_blend_timer = 0.0 if limited_a_target == output_a_target else max(self.mode_blend_timer - self.dt, 0.0)
      output_a_target = limited_a_target

    if stop_trajectory is not None:
      # A mode blend may ease ordinary braking, but must not override the
      # selected stopping preference after final candidate arbitration.
      output_a_target = min(output_a_target, stop_trajectory.acceleration)
      if output_a_target == stop_trajectory.acceleration:
        self.mpc.source = stop_source
        self.v_desired_trajectory = np.asarray(stop_trajectory.speeds)
        self.a_desired_trajectory = np.asarray(stop_trajectory.accelerations)
        self.j_desired_trajectory = np.asarray(stop_trajectory.jerks)
    self.output_a_target = np.clip(output_a_target, ACCEL_MIN, ACCEL_MAX)

    self.a_desired = float(self.output_a_target)
    self.v_desired_filter.x = self.v_desired_filter.x + self.dt * (self.output_a_target + a_prev) / 2.0

  def _fill_stop_offset(self, stop):
    # The planner's math is numpy; pycapnp on device rejects numpy scalars, so cast every field explicitly.
    stop.version = int(CONTROLLER_VERSION)
    stop.valid = bool(self.stop_plan.trajectory is not None)
    stop.configurationValid = bool(self.custom_stop_distance.valid)
    stop.requestedOffset = float(self.custom_stop_distance.requested_distance)
    stop.supportedSources = ['lead']
    stop.state = str(self.stop_plan.state)
    if self.stop_plan.target is not None:
      stop.eventId = int(self.stop_plan.target.event_id)
      stop.source = str(self.stop_plan.target.source)
      stop.baselineDistance = float(self.stop_plan.target.position - self.stop_position)
      stop.distance = float(self.stop_plan.remaining)
      stop.uncertainty = float(self.stop_plan.target.uncertainty)
      stop.shortfall = float(self.stop_plan.shortfall)

  def publish(self, sm, pm):
    plan_send = messaging.new_message('longitudinalPlan')

    gate_services = ['carState', 'controlsState', 'selfdriveState', 'radarState']
    plan_send.valid = sm.all_checks(service_list=gate_services)
    if not plan_send.valid:
      log_issue_limited(
        "longitudinal_plan_invalid",
        "planner",
        f"longitudinalPlan invalid alive={ {s: sm.alive[s] for s in gate_services} } "
        f"freq_ok={ {s: sm.freq_ok[s] for s in gate_services} } valid={ {s: sm.valid[s] for s in gate_services} } "
        f"subchecks=({sm.all_alive(gate_services)},{sm.all_freq_ok(gate_services)},{sm.all_valid(gate_services)}) "
        f"recheck={sm.all_checks(service_list=gate_services)}",
        interval_sec=5.0,
      )

    longitudinalPlan = plan_send.longitudinalPlan
    longitudinalPlan.modelMonoTime = sm.logMonoTime['modelV2']
    longitudinalPlan.processingDelay = (plan_send.logMonoTime / 1e9) - sm.logMonoTime['modelV2']
    longitudinalPlan.solverExecutionTime = self.mpc.solve_time

    longitudinalPlan.speeds = self.v_desired_trajectory.tolist()
    longitudinalPlan.accels = self.a_desired_trajectory.tolist()
    longitudinalPlan.jerks = self.j_desired_trajectory.tolist()

    longitudinalPlan.hasLead = sm['radarState'].leadOne.status
    longitudinalPlan.leadDistance = get_lead_distance(sm['radarState'])
    self._fill_stop_offset(longitudinalPlan.stopOffset)
    longitudinalPlan.longitudinalPlanSource = self.mpc.source
    longitudinalPlan.fcw = self.fcw

    longitudinalPlan.leadTrajectoryX0 = self.mpc.lead_xv_0[:, 0].tolist()
    longitudinalPlan.leadTrajectoryV0 = self.mpc.lead_xv_0[:, 1].tolist()
    longitudinalPlan.leadTrajectoryX1 = self.mpc.lead_xv_1[:, 0].tolist()
    longitudinalPlan.leadTrajectoryV1 = self.mpc.lead_xv_1[:, 1].tolist()

    longitudinalPlan.aTarget = float(self.output_a_target)
    longitudinalPlan.shouldStop = bool(self.output_should_stop)
    longitudinalPlan.allowBrake = True
    longitudinalPlan.allowThrottle = bool(self.allow_throttle)

    pm.send('longitudinalPlan', plan_send)

    self.publish_longitudinal_plan_iq(sm, pm)
