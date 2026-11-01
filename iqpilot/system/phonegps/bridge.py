# Copyright (c) 2026 IQ.Lvbs. All rights reserved.

import json
import os
import socket
import tempfile

SOCKET_PATH = "/dev/shm/iq-phonegps/rpc.sock"
METHODS = frozenset(("getPhoneGPSStatus", "beginPhoneGPS", "sendPhoneGPS", "stopPhoneGPS"))


def call(method, params=None):
  from iqpilot.common.params import Params
  device_params = Params()
  enabled = device_params.get_bool("IQPhoneGPSEnabled")
  onroad = device_params.get_bool("IsOnroad")
  fallback = {
    "supported": True, "enabled": enabled, "activeForDrive": onroad, "onroad": onroad,
    "source": "device", "state": "waiting" if enabled else "off", "ageMs": None, "accuracyMeters": None,
    "error": "",
  }
  if not onroad:
    if method == "getPhoneGPSStatus":
      return fallback
    return {"success": 0, "error": "PhoneGPS is not active for this drive"}
  if method not in METHODS:
    return {"success": 0, "error": "unsupported PhoneGPS method"}
  try:
    with tempfile.TemporaryDirectory(prefix="iq-phonegps-", dir="/dev/shm") as directory:
      with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:
        sock.bind(os.path.join(directory, "reply"))
        sock.settimeout(0.4)
        packet = json.dumps({"method": method, "params": params or {}}, allow_nan=False).encode()
        if len(packet) > 4096:
          raise ValueError("PhoneGPS request too large")
        sock.sendto(packet, SOCKET_PATH)
        return json.loads(sock.recv(4096))
  except (OSError, ValueError):
    if method == "getPhoneGPSStatus":
      return {**fallback, "state": "unavailable", "error": "PhoneGPS receiver is unavailable; using device GPS"}
    return {"success": 0, "error": "PhoneGPS receiver is unavailable"}


def gps_publish_service(service):
  from iqpilot.system.hardware import TICI
  return f"{service}Device" if TICI and os.getenv("REPLAY") != "1" else service
