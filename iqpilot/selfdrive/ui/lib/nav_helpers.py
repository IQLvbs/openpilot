import json
import platform
import threading
import time
from pathlib import Path

from iqpilot.common.params import Params

_MAPBOX_DEFAULT_HELPER_UNAVAILABLE = False
_MAPBOX_DEFAULT_TOKEN = ""
_MAPBOX_DEFAULT_RETRY_AT = 0.0
_MAPBOX_DEFAULT_LOCK = threading.Lock()
_GPS_SERVICES = ("gpsLocationExternal", "gpsLocation")
_POSITION_PARAM_KEYS = ("LastGPSPosition", "LastGPSPositionIQLoc")


def _decode_param(value) -> str:
  if isinstance(value, bytes):
    return value.decode("utf-8", errors="ignore").strip()
  if isinstance(value, str):
    return value.strip()
  return ""


def active_navigation_route(params: Params) -> dict | None:
  destination = params.get("NavigationDestination")
  route = params.get("NavigationRenderRoute")
  if not isinstance(destination, dict) or not isinstance(route, dict):
    return None
  if not params.get_bool("NavigationActive") or not route.get("active"):
    return None
  if (route.get("destinationLatitude") != destination.get("latitude")
      or route.get("destinationLongitude") != destination.get("longitude")):
    return None
  return route


def _load_default_mapbox_token(params: Params) -> None:
  global _MAPBOX_DEFAULT_HELPER_UNAVAILABLE, _MAPBOX_DEFAULT_TOKEN, _MAPBOX_DEFAULT_RETRY_AT

  try:
    try:
      from iqpilot.system.proprietary_runtime._verified_import import import_verified_module
      runtime_common = import_verified_module("iqpilot_navd_private", "iqpilot_private.navd.runtime_common")
    except Exception:
      _MAPBOX_DEFAULT_HELPER_UNAVAILABLE = True
      return

    for args in ((params,), ()):
      try:
        token = _decode_param(runtime_common.ensure_default_mapbox_token(*args))
      except TypeError:
        continue
      except Exception:
        return
      _MAPBOX_DEFAULT_TOKEN = token or _decode_param(params.get("MapboxToken"))
      return
  finally:
    _MAPBOX_DEFAULT_RETRY_AT = time.monotonic() + 60.0
    _MAPBOX_DEFAULT_LOCK.release()


def resolve_mapbox_token(params: Params | None = None) -> str:
  params = params or Params()
  token = _decode_param(params.get("MapboxToken"))
  if token:
    return token
  if _MAPBOX_DEFAULT_TOKEN or _MAPBOX_DEFAULT_HELPER_UNAVAILABLE:
    return _MAPBOX_DEFAULT_TOKEN

  # Default-token discovery can perform network and runtime verification work.
  if time.monotonic() >= _MAPBOX_DEFAULT_RETRY_AT and _MAPBOX_DEFAULT_LOCK.acquire(blocking=False):
    try:
      threading.Thread(target=_load_default_mapbox_token, args=(params,), daemon=True, name="mapbox-token").start()
    except Exception:
      _MAPBOX_DEFAULT_LOCK.release()
      raise
  return ""


def _valid_lat_lon(lat: float, lon: float) -> bool:
  return abs(lat) <= 90.0 and abs(lon) <= 180.0 and (abs(lat) > 1e-4 or abs(lon) > 1e-4)


def _float_field(data: dict, *names: str) -> float:
  for name in names:
    if name in data:
      return float(data.get(name) or 0.0)
  return 0.0


def _position_from_json(raw) -> tuple[float, float, float, bool]:
  try:
    data = raw if isinstance(raw, dict) else json.loads(_decode_param(raw))
    if not isinstance(data, dict):
      return 0.0, 0.0, 0.0, False
    lat = _float_field(data, "latitude", "lat")
    lon = _float_field(data, "longitude", "lon", "lng")
    if _valid_lat_lon(lat, lon):
      return lat, lon, _float_field(data, "bearing", "bearingDeg"), True
  except (TypeError, ValueError, json.JSONDecodeError):
    pass
  return 0.0, 0.0, 0.0, False


def _position_from_msg(msg, lat_name: str = "latitude", lon_name: str = "longitude",
                       bearing_name: str = "bearingDeg") -> tuple[float, float, float, bool]:
  try:
    lat = float(getattr(msg, lat_name, 0.0))
    lon = float(getattr(msg, lon_name, 0.0))
    if _valid_lat_lon(lat, lon):
      return lat, lon, float(getattr(msg, bearing_name, 0.0)), True
  except Exception:
    pass
  return 0.0, 0.0, 0.0, False


def _position_from_params(params: Params) -> tuple[float, float, float, bool]:
  for key in _POSITION_PARAM_KEYS:
    lat, lon, bearing, valid = _position_from_json(params.get(key))
    if valid:
      return lat, lon, bearing, True
  return 0.0, 0.0, 0.0, False


def current_or_last_gps_position(params: Params | None = None) -> tuple[float, float, float, bool]:
  # This helper also runs before UI initialization has completed.
  import sys
  ui_state = getattr(sys.modules.get("iqpilot.selfdrive.ui.ui_state"), "ui_state", None)

  if ui_state is not None:
    for service in _GPS_SERVICES:
      try:
        lat, lon, bearing, valid = _position_from_msg(ui_state.sm[service])
        if valid:
          return lat, lon, bearing, True
      except Exception:
        pass

    try:
      lat, lon, bearing, valid = _position_from_msg(
        ui_state.sm["iqNavRenderState"],
        lat_name="currentLatitude",
        lon_name="currentLongitude",
        bearing_name="bearingDeg",
      )
      if valid:
        return lat, lon, bearing, True
    except Exception:
      pass

  explicit_params = params is not None
  params = params or (ui_state.params if ui_state is not None else Params())
  lat, lon, bearing, valid = _position_from_params(params)
  if valid:
    return lat, lon, bearing, True

  if not explicit_params and platform.system() != "Darwin" and Path("/dev/shm/params/d").exists():
    try:
      lat, lon, bearing, valid = _position_from_params(Params("/dev/shm/params"))
      if valid:
        return lat, lon, bearing, True
    except Exception:
      pass

  return 0.0, 0.0, 0.0, False
