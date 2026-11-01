"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos
"""
from iqdbc.car import structs as _dbc
from iqpilot.common.params import Params as _Store, UnknownKeyName as _UnknownKey
from iqpilot.common.swaglog import cloudlog as _log
from iqpilot.selfdrive.controls.lib.latcontrol_torque import get_nn_model_path as _resolve_nn

import iqpilot.system.sentry as _telemetry

_ANGLE = _dbc.CarParams.SteerControlType.angle

_TUNABLES = (
  "IQSubaruCreepAssist",
  "IQSubaruCreepAssistManualBrake",
  "IQTeslaFsdVisualization",
  "IQTeslaTorqueBlend",
  "IQToyotaFactoryLong",
  "ToyotaSnGHack",
)


def initialize_params(store):
  return [{name: store.get(name, return_default=True)} for name in _TUNABLES]


def log_fingerprint(cp) -> None:
  _telemetry.record_vehicle_identification(cp.carFingerprint, cp.brand)


def set_speed_limit_controller_availability(cp, cp_iq, store=None) -> bool:
  store = store or _Store()
  brand = cp.brand
  off = (brand == "rivian"
         or (brand == "tesla" and store.get_bool("IsReleaseIqBranch"))
         or (not cp.openpilotLongitudinalControl and cp_iq.pcmCruiseSpeed))
  if off and store.get("IQSpeedAssistMode", return_default=True) != 1:
    store.put("IQSpeedAssistMode", 1)
  return not off


def _stamp_lateral_model(cp, cp_iq, store) -> bool:
  where, label, precise = _resolve_nn(cp)
  nn = cp_iq.iqLateralNet
  nn.model.path, nn.model.name, nn.fuzzyFingerprint = where, label, not precise
  if label == "MOCK":
    _log.error({"nnff event": "car doesn't match any Neural Network model"})
    return False
  return cp.steerControlType != _ANGLE and store.get_bool("NeuralNetworkFeedForward")


def _cleanup_unsupported_params(cp, cp_iq, store=None) -> None:
  store = store or _Store()
  doomed = {
    "NeuralNetworkFeedForward": cp.steerControlType == _ANGLE,
    "LongIncrementsEnabled": not cp.openpilotLongitudinalControl and cp_iq.pcmCruiseSpeed,
  }
  for name, gone in doomed.items():
    if gone:
      _log.warning(f"unsupported on this port, clearing {name}")
      store.remove(name)
  set_speed_limit_controller_availability(cp, cp_iq, store)


def _apply_radar_scan_switch(cp, store) -> None:
  if cp.brand != "honda" or cp.radarUnavailable:
    return
  try:
    switch = store.get("IQHondaRadarScan")
  except _UnknownKey:
    return
  if switch in (b"0", "0"):
    _log.warning("IQHondaRadarScan disabled, running without the radar object scan")
    cp.radarUnavailable = True


def _normalize_hyundai_camera_scc(cp, store) -> None:
  if cp.brand != "hyundai":
    return
  from iqdbc.car.hyundai.values import CAMERA_SCC_CAR

  supported = cp.carFingerprint in CAMERA_SCC_CAR
  mode = store.get_int("HyundaiCameraSCC")
  if supported and mode != 2:
    store.put("HyundaiCameraSCC", 2)
  elif not supported and mode != 0:
    store.put("HyundaiCameraSCC", 0)


def apply_iq_car_config(ci, store=None) -> None:
  store = store or _Store()
  _normalize_hyundai_camera_scc(ci.CP, store)
  store.put_bool("IQLaneChangeBsmDelay", bool(ci.CP.enableBsm))
  if ci.CP.brand == "rivian":
    store.put("AolSteeringMode", 2)
  if _stamp_lateral_model(ci.CP, ci.CP_IQ, store):
    ci.configure_torque_tune(ci.CP.carFingerprint, ci.CP.lateralTuning)
  _cleanup_unsupported_params(ci.CP, ci.CP_IQ, store)
  _apply_radar_scan_switch(ci.CP, store)
