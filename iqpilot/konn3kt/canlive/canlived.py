#!/usr/bin/env python3
"""
Copyright ©️ IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos
"""
import os
import threading

from websocket import ABNF, create_connection

import iqpilot.cereal.messaging as messaging
from iqpilot.common.api import Api
from iqpilot.common.params import Params
from iqpilot.common.swaglog import cloudlog
CAN_SERVICES = ["can"]
RECONNECT_MIN = 1.0
RECONNECT_MAX = 10.0

def _api_host() -> str:
  host = "wss://api-iqlabs.konn3kt.com"
  host = host.rstrip("/")
  if host.startswith("https://"):
    host = "wss://" + host[len("https://"):]
  elif host.startswith("http://"):
    host = "ws://" + host[len("http://"):]
  return host


def _stream_once(dongle_id: str, ws_uri: str, token: str, exit_event: threading.Event) -> None:
  ws = create_connection(ws_uri, cookie="jwt=" + token, enable_multithread=True, timeout=30.0)
  cloudlog.info("canlived: connected to %s", ws_uri)
  try:
    socks = [messaging.sub_sock(s, conflate=False, timeout=100) for s in CAN_SERVICES]
    while not exit_event.is_set():
      got_any = False
      for sock in socks:
        while True:
          raw = sock.receive(non_blocking=True)
          if raw is None:
            break
          got_any = True
          ws.send_frame(ABNF.create_frame(raw, ABNF.OPCODE_BINARY, 1))
      if not got_any:
        exit_event.wait(0.005)
  finally:
    try:
      ws.close()
    except Exception:
      pass


def main(exit_event: threading.Event | None = None) -> None:
  if exit_event is None:
    exit_event = threading.Event()

  params = Params()
  dongle_id = params.get("DongleId", encoding="utf-8")
  if not dongle_id:
    cloudlog.error("canlived: no DongleId, cannot stream")
    return

  api = Api(dongle_id)
  host = _api_host()
  ws_uri = f"{host}/ws/can/{dongle_id}"

  backoff = RECONNECT_MIN
  while not exit_event.is_set():
    try:
      token = api.get_token(expiry_hours=1)
      _stream_once(dongle_id, ws_uri, token, exit_event)
      backoff = RECONNECT_MIN
    except Exception as e:
      cloudlog.exception("canlived: stream error: %s", e)
      exit_event.wait(backoff)
      backoff = min(backoff * 2, RECONNECT_MAX)


if __name__ == "__main__":
  main()
