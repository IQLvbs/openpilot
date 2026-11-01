# Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos

from dataclasses import dataclass, asdict
import math

import numpy as np

from iqpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import COMFORT_BRAKE, T_IDXS, get_safe_obstacle_distance, get_T_FOLLOW
from iqpilot.selfdrive.iqmodeld.config import ModelConstants

MIN_LEAD_PROBABILITY = float(np.float32(0.8))

@dataclass
class DistanceControlState:
  enabled: bool = False
  active: bool = False
  inhibitionReason: str = "disabled"
  targetTimeGap: float = 0.0
  excessDistance: float = 0.0
  requestedCorrection: float = 0.0
  appliedCorrection: float = 0.0
  originalAcceleration: float = 0.0
  correctedAcceleration: float = 0.0

  def to_dict(self):
    return {key: value if isinstance(value, str) else bool(value) if isinstance(value, (bool, np.bool_)) else float(value)
            for key, value in asdict(self).items()}


class E2EDistanceController:
  def __init__(self, enabled=False, dt=0.05):
    self.enabled = enabled
    self.dt = dt
    self.correction = 0.0
    self.qualified_time = 0.0
    self.previous_leads = {}
    self.state = DistanceControlState(enabled=enabled)

  def inhibit(self, reason, acceleration):
    self.correction = 0.0
    self.qualified_time = 0.0
    self.previous_leads = {}
    self.state.inhibitionReason = reason
    return acceleration

  def update(self, sm, mpc, *, a_model, a_mpc, a_cruise, e2e, engaged, should_stop, fcw, allow_throttle, lateral_accel):
    self.state = DistanceControlState(enabled=self.enabled, originalAcceleration=a_model, correctedAcceleration=a_model)
    if not self.enabled:
      return self.inhibit("disabled", a_model)
    if not e2e or not engaged:
      return self.inhibit("inactive", a_model)
    try:
      services = ['carState', 'carControl', 'controlsState', 'selfdriveState', 'modelV2', 'radarState']
      if not sm.all_checks(service_list=services):
        return self.inhibit("invalidInputs", a_model)
      now = max(sm.logMonoTime[s] for s in services) * 1e-9
      if not math.isfinite(now) or now <= 0 or any(
        not 0 <= now - sm.logMonoTime[s] * 1e-9 <= 0.2 for s in services
      ):
        return self.inhibit("staleInputs", a_model)
      cs, md, rs = sm['carState'], sm['modelV2'], sm['radarState']
      v_ego = float(cs.vEgo)
      personality = sm['selfdriveState'].personality
      personality = int(getattr(personality, 'raw', personality))
      t_follow = get_T_FOLLOW(personality)
      self.state.targetTimeGap = t_follow
      model_v = np.asarray(md.velocity.x, dtype=float)
      if model_v.shape != (ModelConstants.IDX_N,) or not np.all(np.isfinite(model_v)) or not all(
        math.isfinite(x) for x in [v_ego, a_model, a_mpc, a_cruise, lateral_accel, md.action.desiredCurvature]
      ):
        return self.inhibit("invalidInputs", a_model)
      if mpc.solution_status != 0:
        return self.inhibit("solverFailure", a_model)
      if should_stop or sm['controlsState'].forceDecel or md.action.shouldStop:
        return self.inhibit("stop", a_model)
      if cs.gasPressed or cs.brakePressed or getattr(sm['iqCarState'], 'accelPressed', False):
        return self.inhibit("driverOverride", a_model)
      if not allow_throttle:
        return self.inhibit("throttle", a_model)
      if fcw or md.meta.hardBrakePredicted:
        return self.inhibit("braking", a_model)
      if str(md.meta.laneChangeState) != 'off':
        return self.inhibit("laneChange", a_model)
      if max(abs(lateral_accel), abs(md.action.desiredCurvature) * v_ego ** 2) > 1.5:
        return self.inhibit("curve", a_model)
      if v_ego <= 5.0:
        return self.inhibit("lowSpeed", a_model)
      short_v = np.append(model_v[np.asarray(ModelConstants.T_IDXS) <= 5.0], np.interp(5.0, ModelConstants.T_IDXS, model_v))
      if a_model <= -0.5 or min(short_v) < v_ego - 1.0 or min(model_v) <= 2.0:
        return self.inhibit("modelBraking", a_model)
      leads = {}
      excesses = []
      for i, lead in enumerate([rs.leadOne, rs.leadTwo]):
        if not lead.status:
          continue
        values = [lead.dRel, lead.vRel, lead.vLead, lead.aLeadK, lead.modelProb]
        trajectory = np.asarray([mpc.lead_xv_0, mpc.lead_xv_1][i], dtype=float)
        if not all(math.isfinite(x) for x in values) or lead.dRel <= 0 or trajectory.shape != (len(T_IDXS), 2) or not np.all(np.isfinite(trajectory)):
          return self.inhibit("invalidLead", a_model)
        if lead.modelProb <= MIN_LEAD_PROBABILITY or lead.vLead <= 5.0:
          return self.inhibit("unqualifiedLead", a_model)
        short_lead_v = np.append(trajectory[T_IDXS <= 3.0, 1], np.interp(3.0, T_IDXS, trajectory[:, 1]))
        if lead.aLeadK < -0.5 or min(short_lead_v) < lead.vLead - 1.0:
          return self.inhibit("leadBraking", a_model)
        if lead.fcw:
          return self.inhibit("braking", a_model)
        leads[i] = (int(lead.radarTrackId), float(lead.dRel), float(lead.vRel), now)
        excesses.append(lead.dRel + lead.vLead ** 2 / (2 * COMFORT_BRAKE) - get_safe_obstacle_distance(v_ego, t_follow))
      if not leads:
        return self.inhibit("noLead", a_model)
      continuous = leads.keys() == self.previous_leads.keys()
      for i, (track, distance, _, timestamp) in leads.items():
        if i in self.previous_leads:
          old_track, old_distance, old_vrel, old_time = self.previous_leads[i]
          elapsed = timestamp - old_time
          continuous &= track == old_track and 0 < elapsed <= 0.2 and abs(distance - old_distance - old_vrel * elapsed) <= 5.0
      self.previous_leads = leads
      if not continuous:
        self.qualified_time = 0.0
        self.correction = 0.0
      else:
        self.qualified_time += self.dt
      excess = min(excesses)
      self.state.excessDistance = float(excess)
      if self.qualified_time < 1.0:
        self.state.inhibitionReason = "qualifyingLead"
        return a_model
      weight = np.interp(excess, [2.0, 10.0], [0.0, 1.0])
      weight *= np.interp(v_ego, [5.0, 10.0], [0.0, 1.0])
      weight *= np.interp(a_model, [-0.5, -0.1], [0.0, 1.0])
      headroom = max(0.0, min(a_mpc, a_cruise) - a_model)
      requested = float(weight * min(headroom, (0.5, 0.4, 0.3)[personality]))
      self.correction = float(min(headroom, np.clip(requested, self.correction - 0.5 * self.dt, self.correction + 0.25 * self.dt)))
      self.state.requestedCorrection = requested
      self.state.appliedCorrection = self.correction
      self.state.correctedAcceleration = a_model + self.correction
      self.state.active = self.correction > 0.0
      self.state.inhibitionReason = "none" if self.state.active else "bounded" if headroom == 0 else "atTarget"
      return self.state.correctedAcceleration
    except (AttributeError, KeyError, TypeError, ValueError, IndexError, NotImplementedError):
      return self.inhibit("invalidInputs", a_model)
