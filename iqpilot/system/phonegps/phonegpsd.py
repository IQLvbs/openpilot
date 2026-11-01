# Copyright (c) 2026 IQ.Lvbs. All rights reserved.

import json
import os
import socket
import time
from pathlib import Path

from iqpilot.cereal import messaging
from iqpilot.common.params import Params
from iqpilot.system.phonegps.bridge import SOCKET_PATH
from iqpilot.system.phonegps.protocol import PhoneGPSReceiver
from iqpilot.system.phonegps.mux import PhoneGPSMux


def main():
  params = Params()
  service = "gpsLocationExternal" if params.get_bool("UbloxAvailable") else "gpsLocation"
  raw = messaging.sub_sock(f"{service}Device", conflate=False)
  publisher = messaging.PubMaster([service])
  run(params, service, raw, publisher, SOCKET_PATH)


def run(params, service, raw, publisher, socket_path, stop_event=None):
  receiver = PhoneGPSReceiver()
  directory = Path(socket_path).parent
  directory.mkdir(mode=0o700, parents=True, exist_ok=True)
  Path(socket_path).unlink(missing_ok=True)
  rpc = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
  rpc.bind(socket_path)
  os.chmod(socket_path, 0o600)
  rpc.setblocking(False)
  mux = PhoneGPSMux(service, publisher.send)
  enabled = False
  last_params_read = -1.0
  try:
    while stop_event is None or not stop_event.is_set():
      now = time.monotonic()
      if now - last_params_read >= 0.1:
        enabled = params.get_bool("IQPhoneGPSEnabled") and params.get_bool("IsOnroad")
        last_params_read = now
      if not enabled:
        receiver.clear()
      for _ in range(8):
        try:
          packet, address = rpc.recvfrom(4097)
        except BlockingIOError:
          break
        try:
          if len(packet) > 4096:
            raise ValueError("PhoneGPS request too large")
          request = json.loads(packet)
          method = request["method"]
          arguments = request.get("params", {})
          now = time.monotonic()
          if method == "getPhoneGPSStatus":
            result = receiver.status(now, enabled)
          elif not enabled:
            raise ValueError("PhoneGPS is disabled")
          elif method == "beginPhoneGPS":
            result = receiver.begin(arguments.get("clientId"), now)
          elif method == "sendPhoneGPS":
            result = receiver.accept(arguments, now)
          elif method == "stopPhoneGPS" and arguments.get("sessionId") == receiver.session:
            receiver.clear()
            result = {"success": 1}
          else:
            raise ValueError("invalid PhoneGPS request")
        except (ValueError, KeyError, TypeError, AttributeError) as error:
          receiver.error = str(error)
          result = {"success": 0, "error": str(error)}
        try:
          rpc.sendto(json.dumps(result, allow_nan=False).encode(), address)
        except OSError:
          pass
      fix = receiver.current(time.monotonic()) if enabled else None
      for _ in range(32):
        packet = raw.receive(non_blocking=True)
        if packet is None:
          break
        mux.device(packet, fix)
      mux.update(fix, time.monotonic())
      time.sleep(0.01)
  finally:
    rpc.close()
    Path(socket_path).unlink(missing_ok=True)


if __name__ == "__main__":
  main()
