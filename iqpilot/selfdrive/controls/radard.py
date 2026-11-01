#!/usr/bin/env python3
import math
import numpy as np
from collections import deque
from typing import Any

import capnp
from iqpilot.cereal import messaging, log, car, custom
from iqpilot.common.filter_simple import FirstOrderFilter
from iqpilot.common.params import Params
from iqpilot.common.realtime import DT_MDL, Priority, config_realtime_process
from iqpilot.common.swaglog import cloudlog
from iqpilot.common.simple_kalman import KF1D

from iqdbc.car import structs
from iqdbc.car.honda.values import HONDA_RADAR_SCAN_CAPABLE
from iqdbc.car.hyundai.values import HyundaiFlags, HyundaiFlagsIQ
from iqpilot.selfdrive.controls.lib.custom_stop_distance import CustomStopDistance


# Default lead acceleration decay set to 50% at 1s
_LEAD_ACCEL_TAU = 1.5

# radar tracks
SPEED, ACCEL = 0, 1     # Kalman filter states enum

# stationary qualification parameters
V_EGO_STATIONARY = 4.   # no stationary object flag below this speed

RADAR_TO_CENTER = 2.7   # (deprecated) RADAR is ~ 2.7m ahead from center of car
RADAR_TO_CAMERA = 1.52  # RADAR is ~ 1.5m ahead from center of mesh frame

# Honda radar object scan: 15Hz sweeps consumed at the 20Hz model rate, so measurement absorption is
# gated on fresh sweep data and lead selection carries continuity/staleness evidence
SCAN_SWEEP_DT = 1.0 / 15
SCAN_LEAD_PROB = 0.35
SCAN_LEAD_MIN_CYCLES = 3
SCAN_CHALLENGER_STALE_CYCLES = 2
SCAN_DISTANCE_STALE_CYCLES = 3
SCAN_DISTANCE_STALE_M = 25.0


def uses_scan_radar(CP) -> bool:
  return CP.brand == "honda" and CP.carFingerprint in HONDA_RADAR_SCAN_CAPABLE and not CP.radarUnavailable


class KalmanParams:
  def __init__(self, dt: float):
    # Lead Kalman Filter params, calculating K from A, C, Q, R requires the control library.
    # hardcoding a lookup table to compute K for values of radar_ts between 0.01s and 0.2s
    assert dt > .01 and dt < .2, "Radar time step must be between .01s and 0.2s"
    self.A = [[1.0, dt], [0.0, 1.0]]
    self.C = [1.0, 0.0]
    #Q = np.matrix([[10., 0.0], [0.0, 100.]])
    #R = 1e3
    #K = np.matrix([[ 0.05705578], [ 0.03073241]])
    dts = [i * 0.01 for i in range(1, 21)]
    K0 = [0.12287673, 0.14556536, 0.16522756, 0.18281627, 0.1988689,  0.21372394,
          0.22761098, 0.24069424, 0.253096,   0.26491023, 0.27621103, 0.28705801,
          0.29750003, 0.30757767, 0.31732515, 0.32677158, 0.33594201, 0.34485814,
          0.35353899, 0.36200124]
    K1 = [0.29666309, 0.29330885, 0.29042818, 0.28787125, 0.28555364, 0.28342219,
          0.28144091, 0.27958406, 0.27783249, 0.27617149, 0.27458948, 0.27307714,
          0.27162685, 0.27023228, 0.26888809, 0.26758976, 0.26633338, 0.26511557,
          0.26393339, 0.26278425]
    self.K = [[np.interp(dt, dts, K0)], [np.interp(dt, dts, K1)]]


class Track:
  def __init__(self, identifier: int, v_lead: float, kalman_params: KalmanParams):
    self.identifier = identifier
    self.cnt = 0
    self.aLeadTau = FirstOrderFilter(_LEAD_ACCEL_TAU, 0.45, DT_MDL)
    self.K_A = kalman_params.A
    self.K_C = kalman_params.C
    self.K_K = kalman_params.K
    self.kf = KF1D([[v_lead], [0.0]], self.K_A, self.K_C, self.K_K)

  def update(self, d_rel: float, y_rel: float, v_rel: float, v_lead: float, measured: float,
             absorb_measurement: bool = True):
    # relative values, copy
    self.dRel = d_rel   # LONG_DIST
    self.yRel = y_rel   # -LAT_DIST
    self.vRel = v_rel   # REL_SPEED
    self.vLead = v_lead
    self.measured = measured   # measured or estimate

    # a repeated scan payload between 15Hz sweeps must not be absorbed as a second measurement
    if absorb_measurement and self.cnt > 0:
      self.kf.update(self.vLead)

    self.vLeadK = float(self.kf.x[SPEED][0])
    self.aLeadK = float(self.kf.x[ACCEL][0])

    if absorb_measurement:
      # Learn if constant acceleration
      if abs(self.aLeadK) < 0.5:
        self.aLeadTau.x = _LEAD_ACCEL_TAU
      else:
        self.aLeadTau.update(0.0)

    self.cnt += 1

  def get_RadarState(self, model_prob: float = 0.0, *, model_y: float | None = None):
    return {
      "dRel": float(self.dRel),
      "yRel": float(self.yRel if model_y is None else -model_y),
      "vRel": float(self.vRel),
      "vLead": float(self.vLead),
      "vLeadK": float(self.vLeadK),
      "aLeadK": float(self.aLeadK),
      "aLeadTau": float(self.aLeadTau.x),
      "status": True,
      "fcw": self.is_potential_fcw(model_prob),
      "modelProb": model_prob,
      "radar": True,
      "radarTrackId": self.identifier,
    }

  def potential_low_speed_lead(self, v_ego: float):
    # stop for stuff in front of you and low speed, even without model confirmation
    # Radar points closer than 0.75, are almost always glitches on toyota radars
    return abs(self.yRel) < 1.0 and (v_ego < V_EGO_STATIONARY) and (0.75 < self.dRel < 25)

  def is_potential_fcw(self, model_prob: float):
    return model_prob > .9

  def __str__(self):
    ret = f"x: {self.dRel:4.1f}  y: {self.yRel:4.1f}  v: {self.vRel:4.1f}  a: {self.aLeadK:4.1f}"
    return ret


def laplacian_pdf(x: float, mu: float, b: float):
  b = max(b, 1e-4)
  return math.exp(-abs(x-mu)/b)


def model_association_score(track: Track, lead: capnp._DynamicStructReader, v_ego: float) -> float:
  offset_vision_dist = lead.x[0] - RADAR_TO_CAMERA
  prob_d = laplacian_pdf(track.dRel, offset_vision_dist, lead.xStd[0])
  prob_y = laplacian_pdf(track.yRel, -lead.y[0], lead.yStd[0])
  prob_v = laplacian_pdf(track.vRel + v_ego, lead.v[0], lead.vStd[0])

  # This isn't exactly right, but it's a good heuristic
  return prob_d * prob_y * prob_v


def track_agrees_with_model(track: Track, lead: capnp._DynamicStructReader, v_ego: float, strict: bool) -> bool:
  dist_scale, dist_floor, vel_limit, y_std_scale, y_floor = \
    (0.25, 5.0, 10.0, 1.0, 1.0) if strict else (0.40, 8.0, 13.0, 2.0, 1.5)
  vision_dist = lead.x[0] - RADAR_TO_CAMERA
  dist_ok = abs(track.dRel - vision_dist) < max(abs(vision_dist) * dist_scale, dist_floor)
  vel_ok = (abs(track.vRel + v_ego - lead.v[0]) < vel_limit) or (v_ego + track.vRel > 3)
  lat_ok = abs(track.yRel + lead.y[0]) < max(y_floor, y_std_scale * max(float(lead.yStd[0]), 0.2))
  return dist_ok and vel_ok and lat_ok


def scan_low_speed_candidate(track: Track, v_ego: float) -> bool:
  # require a few real cycles before a radar-only low-speed takeover
  return track.cnt >= SCAN_LEAD_MIN_CYCLES and track.potential_low_speed_lead(v_ego)


def match_vision_to_track(v_ego: float, lead: capnp._DynamicStructReader, tracks: dict[int, Track]):
  offset_vision_dist = lead.x[0] - RADAR_TO_CAMERA

  track = max(tracks.values(), key=lambda c: model_association_score(c, lead, v_ego))

  # if no 'sane' match is found return -1
  # stationary radar points can be false positives
  dist_sane = abs(track.dRel - offset_vision_dist) < max([(offset_vision_dist)*.25, 5.0])
  vel_sane = (abs(track.vRel + v_ego - lead.v[0]) < 10) or (v_ego + track.vRel > 3)
  if dist_sane and vel_sane:
    return track
  else:
    return None


def get_RadarState_from_vision(lead_msg: capnp._DynamicStructReader, v_ego: float, model_v_ego: float):
  lead_v_rel_pred = lead_msg.v[0] - model_v_ego
  return {
    "dRel": float(lead_msg.x[0] - RADAR_TO_CAMERA),
    "yRel": float(-lead_msg.y[0]),
    "vRel": float(lead_v_rel_pred),
    "vLead": float(v_ego + lead_v_rel_pred),
    "vLeadK": float(v_ego + lead_v_rel_pred),
    "aLeadK": float(lead_msg.a[0]),
    "aLeadTau": 0.3,
    "fcw": False,
    "modelProb": float(lead_msg.prob),
    "status": True,
    "radar": False,
    "radarTrackId": -1,
  }


def get_lead(v_ego: float, ready: bool, tracks: dict[int, Track], lead_msg: capnp._DynamicStructReader,
             model_v_ego: float, low_speed_override: bool = True,
             scan_radar: bool = False, filtered_prob: float | None = None, held_track_id: int = -1,
             model_lateral: bool = False) -> dict[str, Any]:
  lead_prob = float(lead_msg.prob if filtered_prob is None else filtered_prob)
  prob_threshold = SCAN_LEAD_PROB if scan_radar else .5

  # Determine leads, this is where the essential logic happens
  if len(tracks) > 0 and ready and lead_prob > prob_threshold:
    track = match_vision_to_track(v_ego, lead_msg, tracks)
  else:
    track = None

  lead_dict = {'status': False}
  if track is not None:
    lead_dict = track.get_RadarState(lead_prob, model_y=lead_msg.y[0] if model_lateral else None)
  elif (track is None) and ready and (lead_prob > prob_threshold):
    lead_dict = get_RadarState_from_vision(lead_msg, v_ego, model_v_ego)

  if low_speed_override:
    if scan_radar:
      low_speed_tracks = [c for c in tracks.values() if scan_low_speed_candidate(c, v_ego)]
    else:
      low_speed_tracks = [c for c in tracks.values() if c.potential_low_speed_lead(v_ego)]

    model_lead_available = ready and lead_prob > prob_threshold

    if scan_radar:
      # Keep the held radar lead through ordinary model-probability fluctuations while it stays
      # coherent. With a valid model lead it must still agree with it; without one, a mature radar
      # track remains eligible for continuity
      held = tracks.get(held_track_id)
      if held is not None and scan_low_speed_candidate(held, v_ego):
        held_matches_model = (not model_lead_available or
                              track_agrees_with_model(held, lead_msg, v_ego, strict=True))
        held_is_current = (not lead_dict.get('status', False) or
                           lead_dict.get('radarTrackId', -1) == held_track_id or
                           (lead_dict.get('status', False) and not lead_dict.get('radar', False)))
        if held_is_current and held_matches_model:
          lead_dict = held.get_RadarState(lead_prob)

      def candidate_established(candidate: Track) -> bool:
        if candidate.cnt < SCAN_LEAD_MIN_CYCLES:
          return False
        if not lead_dict.get('status', False):
          # a mature centered scan point may provide the radar-only low-speed lead
          return True
        if lead_dict.get('radarTrackId', -1) == candidate.identifier:
          return True
        # never replace an established lead with an unrelated closer point without model evidence
        # to arbitrate them
        return model_lead_available and track_agrees_with_model(candidate, lead_msg, v_ego, strict=True)

      low_speed_tracks = [c for c in low_speed_tracks if candidate_established(c)]

    if len(low_speed_tracks) > 0:
      closest_track = min(low_speed_tracks, key=lambda c: c.dRel)

      # Only choose new track if it is actually closer than the previous one
      if (not lead_dict['status']) or (closest_track.dRel < lead_dict['dRel']):
        lead_dict = closest_track.get_RadarState()

  return lead_dict


class RadarD:
  def __init__(self, CP: structs.CarParams, CP_IQ: structs.IQCarParams, delay: float = 0.0):
    self.model_lateral = False
    if CP.brand == "hyundai":
      camera_scc = HyundaiFlags.CANFD_CAMERA_SCC | HyundaiFlags.CAMERA_SCC
      self.model_lateral = bool(CP.flags & camera_scc or CP_IQ.flags & HyundaiFlagsIQ.ENHANCED_SCC)

    self.current_time = 0.0

    self.tracks: dict[int, Track] = {}
    self.scan_radar = uses_scan_radar(CP)
    # the lead KF absorbs scan measurements at the physical 15Hz sweep cadence; lead probability
    # filtering stays on model-loop timing
    self.kalman_params = KalmanParams(SCAN_SWEEP_DT if self.scan_radar else DT_MDL)
    self.lead_prob_filters = [FirstOrderFilter(0.0, 0.2, DT_MDL) for _ in range(2)]
    self.held_lead_ids = [-1, -1]
    self._held_evidence_ids = [-1, -1]
    self._challenger_stale_counts = [0, 0]
    self._distance_stale_counts = [0, 0]
    self._last_tracks_frame = -1

    self.v_ego = 0.0
    self.v_ego_hist = deque([0.0], maxlen=int(round(delay / DT_MDL))+1)
    self.last_v_ego_frame = -1

    self.radar_state: capnp._DynamicStructBuilder | None = None
    self.radar_state_valid = False

    self.ready = False

    self.custom_stop_distance = CustomStopDistance()

  def _refresh_held_lead_evidence(self, lead_index: int, lead: capnp._DynamicStructReader,
                                  lead_prob: float) -> None:
    held_id = self.held_lead_ids[lead_index]
    if self._held_evidence_ids[lead_index] != held_id:
      self._reset_held_evidence(lead_index, held_id)

    held = self.tracks.get(held_id)
    if held_id < 0 or held is None or not self.ready or lead_prob <= SCAN_LEAD_PROB:
      self._reset_held_evidence(lead_index, held_id)
      return

    strict_match = track_agrees_with_model(held, lead, self.v_ego, strict=True)
    relaxed_match = track_agrees_with_model(held, lead, self.v_ego, strict=False)

    # evidence arm 1: another live track scores better against the model while the held one no
    # longer passes even relaxed continuity. Releasing the hold never selects that challenger; the
    # strict-match path in get_lead stays the only way it becomes the radar lead
    if relaxed_match:
      self._challenger_stale_counts[lead_index] = 0
    else:
      best = max(self.tracks.values(), key=lambda c: model_association_score(c, lead, self.v_ego))
      if best.identifier != held_id and \
          model_association_score(best, lead, self.v_ego) > model_association_score(held, lead, self.v_ego):
        self._challenger_stale_counts[lead_index] += 1
      else:
        self._challenger_stale_counts[lead_index] = 0

    # evidence arm 2: gross absolute range disagreement, with a strict match staying authoritative
    # even when model uncertainty would permit the error
    distance_mismatch = abs(held.dRel - (lead.x[0] - RADAR_TO_CAMERA))
    if strict_match:
      self._distance_stale_counts[lead_index] = 0
    elif distance_mismatch > SCAN_DISTANCE_STALE_M:
      self._distance_stale_counts[lead_index] += 1
    else:
      self._distance_stale_counts[lead_index] = 0

    if (self._challenger_stale_counts[lead_index] >= SCAN_CHALLENGER_STALE_CYCLES or
        self._distance_stale_counts[lead_index] >= SCAN_DISTANCE_STALE_CYCLES):
      self.held_lead_ids[lead_index] = -1
      self._reset_held_evidence(lead_index)

  def _reset_held_evidence(self, lead_index: int, held_id: int = -1) -> None:
    self._held_evidence_ids[lead_index] = held_id
    self._challenger_stale_counts[lead_index] = 0
    self._distance_stale_counts[lead_index] = 0

  def update(self, sm: messaging.SubMaster, rr: car.RadarData):
    self.ready = sm.seen['modelV2']
    self.current_time = 1e-9*max(sm.logMonoTime.values())
    self.custom_stop_distance.update()

    if sm.recv_frame['carState'] != self.last_v_ego_frame:
      self.v_ego = sm['carState'].vEgo
      self.v_ego_hist.append(self.v_ego)
      self.last_v_ego_frame = sm.recv_frame['carState']

    sweep_fresh = True
    if self.scan_radar:
      sweep_fresh = sm.recv_frame['radarTracks'] != self._last_tracks_frame
      self._last_tracks_frame = sm.recv_frame['radarTracks']

    ar_pts = {pt.trackId: [pt.dRel, pt.yRel, pt.vRel, pt.measured] for pt in rr.points}

    # *** remove missing points from meta data ***
    for ids in list(self.tracks.keys()):
      if ids not in ar_pts:
        self.tracks.pop(ids, None)

    # *** compute the tracks ***
    for ids in ar_pts:
      rpt = ar_pts[ids]

      # align v_ego by a fixed time to align it with the radar measurement
      v_lead = rpt[2] + self.v_ego_hist[0]

      # create the track if it doesn't exist or it's a new track
      if ids not in self.tracks:
        self.tracks[ids] = Track(ids, v_lead, self.kalman_params)
      if self.scan_radar:
        measured = bool(rpt[3] and sweep_fresh)
        self.tracks[ids].update(rpt[0], rpt[1], rpt[2], v_lead, measured, absorb_measurement=measured)
      else:
        self.tracks[ids].update(rpt[0], rpt[1], rpt[2], v_lead, rpt[3])

    # *** publish radarState ***
    self.radar_state_valid = sm.all_checks()
    self.radar_state = log.RadarState.new_message()
    self.radar_state.mdMonoTime = sm.logMonoTime['modelV2']
    self.radar_state.radarErrors = rr.errors
    self.radar_state.carStateMonoTime = sm.logMonoTime['carState']

    if len(sm['modelV2'].velocity.x):
      model_v_ego = sm['modelV2'].velocity.x[0]
    else:
      model_v_ego = self.v_ego
    leads_v3 = sm['modelV2'].leadsV3
    if len(leads_v3) > 1:
      if self.scan_radar:
        for i in range(2):
          lead_prob = float(leads_v3[i].prob)
          # probability rises instantly, decays filtered: a one-cycle model dip must not drop the lead
          if lead_prob > self.lead_prob_filters[i].x:
            self.lead_prob_filters[i].x = lead_prob
          else:
            self.lead_prob_filters[i].update(lead_prob)
          self._refresh_held_lead_evidence(i, leads_v3[i], self.lead_prob_filters[i].x)

        lead_one = get_lead(self.v_ego, self.ready, self.tracks, leads_v3[0], model_v_ego, model_lateral=self.model_lateral,
                            low_speed_override=True, scan_radar=True, filtered_prob=self.lead_prob_filters[0].x,
                            held_track_id=self.held_lead_ids[0])
        lead_two = get_lead(self.v_ego, self.ready, self.tracks, leads_v3[1], model_v_ego, model_lateral=self.model_lateral,
                            low_speed_override=False, scan_radar=True, filtered_prob=self.lead_prob_filters[1].x,
                            held_track_id=self.held_lead_ids[1])

        for i, lead in enumerate((lead_one, lead_two)):
          if lead.get('status', False) and lead.get('radar', False):
            track_id = int(lead.get('radarTrackId', -1))
            if track_id != self.held_lead_ids[i]:
              self._reset_held_evidence(i, track_id)
            self.held_lead_ids[i] = track_id
          elif (not lead.get('status', False)) or (self.held_lead_ids[i] not in self.tracks):
            self.held_lead_ids[i] = -1
      else:
        lead_one = get_lead(self.v_ego, self.ready, self.tracks, leads_v3[0], model_v_ego, model_lateral=self.model_lateral, low_speed_override=True)
        lead_two = get_lead(self.v_ego, self.ready, self.tracks, leads_v3[1], model_v_ego, model_lateral=self.model_lateral, low_speed_override=False)
      self.radar_state.leadOne = self.custom_stop_distance.apply_lead(lead_one)
      self.radar_state.leadTwo = self.custom_stop_distance.apply_lead(lead_two)

  def publish(self, pm: messaging.PubMaster):
    assert self.radar_state is not None

    radar_msg = messaging.new_message("radarState")
    radar_msg.valid = self.radar_state_valid
    radar_msg.radarState = self.radar_state
    pm.send("radarState", radar_msg)


# fuses camera and radar data for best lead detection
def main() -> None:
  config_realtime_process(5, Priority.CTRL_LOW)

  # wait for stats about the car to come in from controls
  cloudlog.info("radard is waiting for CarParams")
  CP = messaging.log_from_bytes(Params().get("CarParams", block=True), car.CarParams)
  cloudlog.info("radard got CarParams")

  cloudlog.info("radard is waiting for IQCarParams")
  CP_IQ = messaging.log_from_bytes(Params().get("IQCarParams", block=True), custom.IQCarParams)
  cloudlog.info("radard got IQCarParams")

  # *** setup messaging
  sm = messaging.SubMaster(['modelV2', 'carState', 'radarTracks'], poll='modelV2')
  pm = messaging.PubMaster(['radarState'])

  RD = RadarD(CP, CP_IQ, CP.radarDelay)

  while 1:
    sm.update()

    RD.update(sm, sm['radarTracks'])
    RD.publish(pm)


if __name__ == "__main__":
  main()
