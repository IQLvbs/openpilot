"""Closed-loop stop measurements through the production planner and LongControl.

The plant's world positions are independent of the distances supplied to perception.
Unlike Plant, this runs controls at 100 Hz and models command dead time and lag.
"""
from collections import deque
from dataclasses import dataclass
import json

import numpy as np

from iqdbc.car.honda.interface import CarInterface
from iqdbc.car.honda.values import CAR
from iqpilot.cereal import messaging
from iqpilot.common.params import Params
from iqpilot.common.realtime import DT_CTRL, DT_MDL
from iqpilot.selfdrive.controls.lib.longcontrol import LongControl, LongCtrlState
from iqpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner, get_lead_distance
from iqpilot.selfdrive.iqmodeld.config import ModelConstants


@dataclass
class ActuatorPlant:
  speed: float
  dead_time: float = 0.2
  lag: float = 0.3
  position: float = 0.0
  acceleration: float = 0.0

  def __post_init__(self):
    self.pending = deque([0.0] * round(self.dead_time / DT_CTRL))

  def step(self, command):
    self.pending.append(float(command))
    delayed = self.pending.popleft()
    alpha = DT_CTRL / (self.lag + DT_CTRL)
    self.acceleration += alpha * (delayed - self.acceleration)
    moving_time = DT_CTRL
    if self.acceleration < 0.0:
      moving_time = min(moving_time, self.speed / -self.acceleration)
    self.position += self.speed * moving_time + 0.5 * self.acceleration * moving_time ** 2
    self.speed = max(0.0, self.speed + self.acceleration * DT_CTRL)


def build_frame(ego, lead_position, lead_speed, lead_accel, now, e2e, long_state, distance_bias=0.0, lead_visible=True):
  names = ('carState', 'carControl', 'controlsState', 'selfdriveState', 'vehicleParameters', 'radarState', 'modelV2',
           'iqCarState', 'iqNavState', 'iqLiveData', 'gpsLocation')
  sm = {name: getattr(messaging.new_message(name), name) for name in names}
  cs, cc, ss, rs, md = (sm[name] for name in ('carState', 'carControl', 'selfdriveState', 'radarState', 'modelV2'))
  cs.vEgo, cs.aEgo = float(ego.speed), float(ego.acceleration)
  cs.standstill = ego.speed < 0.01
  cs.vCruise = cs.vCruiseCluster = 90.0
  cc.longActive = True
  cc.orientationNED = [0.0, 0.0, 0.0]
  ss.enabled, ss.experimentalMode = True, e2e
  sm['controlsState'].longControlState = long_state
  lead = rs.leadOne
  lead.status = lead_visible
  lead.radarTrackId = 7
  lead.dRel = float(lead_position - ego.position + distance_bias)
  lead.vLead = lead.vLeadK = float(lead_speed)
  lead.vRel = float(lead_speed - ego.speed)
  lead.aLeadK = float(lead_accel)
  lead.aLeadTau = 1.5
  lead.modelProb = 1.0
  times = np.array(ModelConstants.T_IDXS)
  md.position.x = ((ego.speed + 0.5) * times).tolist()
  md.velocity.x = [ego.speed] + [ego.speed + 0.5] * (len(times) - 1)
  md.acceleration.x = [0.0] * len(times)
  md.action.desiredAcceleration = 0.5
  md.meta.disengagePredictions.gasPressProbs = [1.0] * 6
  lead_times = np.array(ModelConstants.LEAD_T_IDXS)
  stop_time = -lead_speed / lead_accel if lead_accel < 0 else np.inf
  motion_times = np.minimum(lead_times, stop_time)
  for prediction in md.leadsV3:
    prediction.prob = 1.0 if lead_visible else 0.0
    prediction.x = (lead.dRel + lead_speed * motion_times + 0.5 * lead_accel * motion_times ** 2).tolist()
    prediction.v = np.maximum(lead_speed + lead_accel * lead_times, 0.0).tolist()
  return sm


def lead_fields(lead) -> dict:
  # to_dict() probes which() and relies on catching KjException for union-less structs;
  # on the macOS CI build that exception surfaces as a bare RuntimeError instead
  return {name: getattr(lead, name) for name in lead.schema.non_union_fields}


def run_stop(offset, *, speed=12.0, distance=90.0, moving_lead=False, e2e=False, new_lead_mpc=False,
             dead_time=0.2, lag=0.3, smooth=True, duration=35.0, distance_bias=0.0, vehicle='honda',
             visibility=lambda now: True, resume_at=None, handoff_at=None, only_lead2=False, driver_override=lambda now: False,
             steady_lead=False, dynamic=False):
  params = Params()
  params.put('IQCustomStopDistance', 0)
  params.put_bool('IQForceStops', smooth)
  params.put_bool('newLeadMpc', new_lead_mpc)
  params.put_bool('IQDynamicMode', dynamic)
  interface, model = CarInterface, CAR.HONDA_CIVIC
  if vehicle in ('ev6', 'elantra'):
    from iqdbc.car.hyundai.interface import CarInterface as HyundaiInterface
    from iqdbc.car.hyundai.values import CAR as HyundaiCar
    interface = HyundaiInterface
    model = HyundaiCar.KIA_EV6 if vehicle == 'ev6' else HyundaiCar.HYUNDAI_ELANTRA_2021
  CP = interface.get_non_essential_params(model)
  CP_IQ = interface.get_non_essential_params_iq(CP, model)
  CP.openpilotLongitudinalControl = True
  CP.longitudinalActuatorDelay = dead_time + lag
  ego = ActuatorPlant(speed, dead_time, lag)
  # Inject validation values without expanding the production parameter range.
  planner = LongitudinalPlanner(CP, CP_IQ, init_v=speed, stop_offset_m=offset)
  controller = LongControl(CP, CP_IQ)
  lead_position, lead_speed = distance, speed if moving_lead or steady_lead else 0.0
  trace = []
  for frame in range(round(duration / DT_CTRL)):
    now = frame * DT_CTRL
    lead_accel = -1.5 if moving_lead and now >= 2.0 and lead_speed > 0.0 else 0.0
    if resume_at is not None and now >= resume_at:
      lead_accel = 1.0 if lead_speed < 8.0 else 0.0
    if frame % round(DT_MDL / DT_CTRL) == 0:
      bias = distance_bias(now, lead_position - ego.position) if callable(distance_bias) else distance_bias
      mode = e2e or dynamic
      sm = build_frame(ego, lead_position, lead_speed, lead_accel, now, mode if handoff_at is None or now < handoff_at else not mode,
                       controller.long_control_state if frame else LongCtrlState.pid, bias, visibility(now))
      sm['carState'].gasPressed = driver_override(now)
      if only_lead2:
        sm['radarState'].leadTwo = sm['radarState'].leadOne
        sm['radarState'].leadOne.status = False
      # Exercise the old radard rewrite as well as the new raw-perception path.
      rs = sm['radarState']
      rs.leadOne = planner.custom_stop_distance.apply_lead(lead_fields(rs.leadOne))
      rs.leadTwo = planner.custom_stop_distance.apply_lead(lead_fields(rs.leadTwo))
      planner.update(sm)
    cs = sm['carState']
    cs.vEgo, cs.aEgo, cs.standstill = float(ego.speed), float(ego.acceleration), ego.speed < 0.01
    stop_plan = planner.stop_plan
    limits = interface.get_pid_accel_limits(CP, CP_IQ, ego.speed, cs.vCruise / 3.6)
    command = controller.update(True, cs, planner.output_a_target, planner.output_should_stop, limits,
                                get_lead_distance(sm['radarState']), sm['radarState'].leadOne.status,
                                v_target_now=planner.v_desired_trajectory[0],
                                stop_distance=stop_plan.remaining if stop_plan.trajectory is not None else None,
                                stop_event_id=stop_plan.target.event_id if stop_plan.trajectory is not None else 0)
    ego.step(command)
    lead_dt = min(DT_CTRL, lead_speed / -lead_accel) if lead_accel < 0.0 else DT_CTRL
    lead_position += lead_speed * lead_dt + 0.5 * lead_accel * lead_dt ** 2
    lead_speed = max(0.0, lead_speed + lead_accel * DT_CTRL)
    if frame % round(DT_MDL / DT_CTRL) == 0:
      trace.append((now, ego.position, ego.speed, ego.acceleration, command, lead_position - ego.position,
                    stop_plan.remaining, float(stop_plan.active), float(stop_plan.target.event_id if stop_plan.active else 0),
                    stop_plan.offset, stop_plan.shortfall, planner.output_a_target, float(planner.output_should_stop)))
  return np.asarray(trace)


if __name__ == '__main__':
  for moving in (False, True):
    results = []
    for offset in (0, 2):
      trace = run_stop(offset, moving_lead=moving, distance=35.0 if moving else 90.0)
      results.append({'offset': offset, 'finalGap': float(trace[-1, 5]), 'finalSpeed': float(trace[-1, 2]),
                      'minimumGap': float(min(trace[:, 5]))})
    print(json.dumps({'scenario': 'brakingLead' if moving else 'stationaryLead', 'results': results}))
