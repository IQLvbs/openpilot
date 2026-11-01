import json
import math
import time

import numpy as np

from iqpilot.cereal import car, messaging
from iqpilot.common.params import Params
from iqpilot.ui.engagement import resolve_engagement_display

UI_STREAM_SERVICES = [
  "modelV2", "carState", "selfdriveState", "controlsState", "extrinsicsCalibration",
  "radarState", "longitudinalPlan", "deviceState", "roadCameraState",
  "iqState", "onroadEvents",
]

MAX_BUFFERED_BYTES = 256 * 1024
LOW_BANDWIDTH_BITRATE = 500_000

HEARTBEAT_INTERVAL = 1.0


def _round_list(vals, decimals: int) -> list[float]:
  arr = np.asarray(vals, dtype=np.float64)
  if arr.size == 0:
    return []
  arr = np.round(np.where(np.isfinite(arr), arr, 0.0), decimals)
  return arr.tolist()


def _round_float(val, decimals: int = 3, default: float = 0.0) -> float:
  try:
    v = float(val)
  except (TypeError, ValueError):
    return default
  return round(v, decimals) if math.isfinite(v) else default


def _xyz(line, decimals: int = 2) -> dict[str, list[float]]:
  return {
    "x": _round_list(line.x, decimals),
    "y": _round_list(line.y, decimals),
    "z": _round_list(line.z, decimals),
  }


def _lead(lead) -> dict:
  return {
    "status": bool(lead.status),
    "dRel": _round_float(lead.dRel, 2),
    "yRel": _round_float(lead.yRel, 2),
    "vRel": _round_float(lead.vRel, 2),
  }


def compute_ui_status(ss, iq_state, onroad_events) -> str:
  return resolve_engagement_display(ss, iq_state, onroad_events).status


def build_init_payload(params: Params | None = None) -> dict:
  params = params or Params()

  has_longitudinal_control = False
  cp_bytes = params.get("CarParamsPersistent")
  if cp_bytes is not None:
    try:
      cp = messaging.log_from_bytes(cp_bytes, car.CarParams)
      if cp.alphaLongitudinalAvailable:
        has_longitudinal_control = params.get_bool("AlphaLongitudinalEnabled")
      else:
        has_longitudinal_control = bool(cp.openpilotLongitudinalControl)
    except Exception:
      pass

  camera_offset = 0.0
  if params.get("ModelManager_ActiveBundle"):
    try:
      camera_offset = float(params.get("CameraOffset", return_default=True) or 0.0)
    except (TypeError, ValueError):
      camera_offset = 0.0

  return {
    "hasLongitudinalControl": has_longitudinal_control,
    "cameraOffset": _round_float(camera_offset, 3),
    "isMetric": params.get_bool("IsMetric"),
  }


class UIStreamMessageProxy:
  def __init__(self, sm: messaging.SubMaster | None = None, bitrate_getter=None):
    self.sm = sm if sm is not None else messaging.SubMaster(UI_STREAM_SERVICES)
    self.channels = []
    self.bitrate_getter = bitrate_getter
    self.dropped_frames = 0
    self._last_non_disengaged = "disengaged"
    self._last_emit_time = 0.0
    self._decimate_flip = False
    self._init_payload = build_init_payload()

  def add_channel(self, channel):
    self.channels.append(channel)

  def update(self):
    self.sm.update(0)

    model_updated = self.sm.updated["modelV2"]
    now = time.monotonic()
    if not model_updated:
      if now - self._last_emit_time < HEARTBEAT_INTERVAL:
        return
    elif self._low_bandwidth():
      self._decimate_flip = not self._decimate_flip
      if self._decimate_flip:
        return
    frame = self._build_frame(include_model=model_updated)
    encoded = frame_to_str(frame)
    self._last_emit_time = now
    for channel in self.channels:
      if channel.buffered_amount() > MAX_BUFFERED_BYTES:
        self.dropped_frames += 1
        continue
      channel.send(encoded)

  def _low_bandwidth(self) -> bool:
    if self.bitrate_getter is None:
      return False
    try:
      bitrate = self.bitrate_getter()
    except Exception:
      return False
    return bitrate is not None and bitrate <= LOW_BANDWIDTH_BITRATE

  def _ui_status(self) -> str:
    sm = self.sm
    ss = sm["selfdriveState"]
    iq_state = sm["iqState"]
    status = compute_ui_status(ss, iq_state, sm["onroadEvents"])

    if status != "disengaged":
      self._last_non_disengaged = status
      return status

    if ss.enabled or iq_state.aol.enabled:
      if self._last_non_disengaged != "disengaged":
        return self._last_non_disengaged
      return "engaged" if ss.enabled else "disengaged"

    self._last_non_disengaged = "disengaged"
    return "disengaged"

  def _build_frame(self, include_model: bool = True) -> dict:
    sm = self.sm
    cs = sm["carState"]
    ss = sm["selfdriveState"]
    calib = sm["extrinsicsCalibration"]
    radar = sm["radarState"]
    device_state = sm["deviceState"]

    model_data = None
    if include_model:
      model = sm["modelV2"]
      model_data = {
        "position": _xyz(model.position),
        "laneLines": [_xyz(line) for line in model.laneLines],
        "laneLineProbs": _round_list(model.laneLineProbs, 3),
        "roadEdges": [_xyz(edge) for edge in model.roadEdges],
        "roadEdgeStds": _round_list(model.roadEdgeStds, 3),
        "acceleration": {"x": _round_list(model.acceleration.x, 2)},
      }

    data = {
      "modelV2": model_data,
      "carState": {
        "vEgo": _round_float(cs.vEgo, 2),
        "vEgoCluster": _round_float(cs.vEgoCluster, 2),
        "vCruiseCluster": _round_float(cs.vCruiseCluster, 2),
        "leftBlinker": bool(cs.leftBlinker),
        "rightBlinker": bool(cs.rightBlinker),
      },
      "selfdriveState": {
        "enabled": bool(ss.enabled),
        "experimentalMode": bool(ss.experimentalMode),
        "state": str(ss.state),
        "alertText1": str(ss.alertText1),
        "alertText2": str(ss.alertText2),
        "alertSize": str(ss.alertSize),
        "alertStatus": str(ss.alertStatus),
      },
      "controlsState": {
        "vCruiseDEPRECATED": _round_float(sm["controlsState"].vCruiseDEPRECATED, 2),
      },
      "extrinsicsCalibration": {
        "calStatus": str(calib.calStatus),
        "rpyCalib": _round_list(calib.rpyCalib, 5),
        "wideFromDeviceEuler": _round_list(calib.wideFromDeviceEuler, 5),
        "height": _round_list(calib.height, 3),
      },
      "radarState": {
        "valid": bool(sm.valid["radarState"]),
        "leadOne": _lead(radar.leadOne),
        "leadTwo": _lead(radar.leadTwo),
      },
      "longitudinalPlan": {
        "allowThrottle": bool(sm["longitudinalPlan"].allowThrottle),
      },
      "deviceState": {
        "deviceType": str(device_state.deviceType),
        "started": bool(device_state.started),
      },
      "roadCameraState": {
        "sensor": str(sm["roadCameraState"].sensor),
      },
      "uiStatus": self._ui_status(),
      "init": self._init_payload,
    }

    return {"type": "uiStream", "logMonoTime": sm.logMonoTime["modelV2"], "data": data}

def frame_to_str(frame: dict) -> str:
  return json.dumps(frame, separators=(",", ":"))
