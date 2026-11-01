# Copyright (c) 2026 IQ.Lvbs. All rights reserved.

import math
import json
import socket
import tempfile
import threading
import time
from pathlib import Path

import pytest

from iqpilot.cereal import messaging
from iqpilot.system.phonegps.mux import PhoneGPSMux
from iqpilot.system.phonegps.protocol import PhoneGPSReceiver
from iqpilot.system.phonegps.phonegpsd import run


@pytest.fixture
def receiver():
  result = PhoneGPSReceiver()
  result.begin("test-phone", 100)
  return result


def payload(receiver, measured_at=100, sequence=1, **values):
  return {
    "sessionId": receiver.session, "sequence": sequence, "platform": "ios",
    "deviceMonotonicMs": measured_at * 1000, "clockUncertaintyMs": 20,
    "latitude": 41.3, "longitude": -90.1, "altitude": 180,
    "accuracy": 5, "altitudeAccuracy": 8, "speed": 10, "heading": 90,
    "timestamp": 1790000000000, **values,
  }


def hardware(service, measured_at):
  message = messaging.new_message(service, valid=True, logMonoTime=int(measured_at * 1e9))
  getattr(message, service).latitude = 41.3
  getattr(message, service).longitude = -90.1
  getattr(message, service).hasFix = True
  return message.to_bytes()


@pytest.mark.parametrize("service", ["gpsLocation", "gpsLocationExternal"])
def test_bidirectional_live_handoff_without_restarting(receiver, service):
  sent = []
  mux = PhoneGPSMux(service, lambda name, data: sent.append((name, data if isinstance(data, bytes) else data.to_bytes())))
  initial = hardware(service, 99.9)
  mux.device(initial, None)
  assert sent == [(service, initial)]
  receiver.accept(payload(receiver), 100.02)
  phone = receiver.current(100.02)
  mux.update(phone, 100.02)
  assert messaging.log_from_bytes(sent[-1][1]).logMonoTime == 100000000000
  assert getattr(messaging.log_from_bytes(sent[-1][1]), service).source == "iOS"
  count = len(sent)
  mux.update(phone, 100.03)
  assert len(sent) == count
  fallback = hardware(service, 101.9)
  mux.device(fallback, phone)
  assert len(sent) == count
  mux.update(receiver.current(102.01), 102.01)
  assert sent[-1] == (service, fallback)
  receiver.accept(payload(receiver, 102.1, 2), 102.12)
  mux.update(receiver.current(102.12), 102.12)
  assert getattr(messaging.log_from_bytes(sent[-1][1]), service).source == "iOS"
  fallback = hardware(service, 102.2)
  mux.device(fallback, receiver.current(102.2))
  receiver.clear()
  mux.update(receiver.current(102.21), 102.21)
  assert sent[-1] == (service, fallback)
  receiver.begin("test-phone", 102.3)
  receiver.accept(payload(receiver, 102.3), 102.32)
  mux.update(receiver.current(102.32), 102.32)
  assert getattr(messaging.log_from_bytes(sent[-1][1]), service).source == "iOS"


@pytest.mark.parametrize("values", [
  {"latitude": math.nan}, {"accuracy": 51}, {"accuracy": -1}, {"altitudeAccuracy": 101},
  {"speed": math.inf}, {"sequence": True}, {"sequence": 1.2}, {"platform": "web"},
  {"sessionId": "wrong"}, {"mocked": True}, {"clockUncertaintyMs": 251},
  {"deviceMonotonicMs": 99000}, {"deviceMonotonicMs": 101000}, {"heading": -1},
])
def test_rejected_fixes_never_replace_good_fix(receiver, values):
  receiver.accept(payload(receiver), 100.02)
  original = receiver.fix
  with pytest.raises(ValueError):
    receiver.accept({**payload(receiver, 100.1, 2), **values}, 100.12)
  assert receiver.fix is original


def test_replay_jump_expiry_and_owner_isolation(receiver):
  receiver.accept(payload(receiver), 100.02)
  with pytest.raises(ValueError, match="out-of-order"):
    receiver.accept(payload(receiver), 100.03)
  with pytest.raises(ValueError, match="jump"):
    receiver.accept(payload(receiver, 100.1, 2, latitude=42.3), 100.12)
  with pytest.raises(ValueError, match="another phone"):
    receiver.begin("other-phone", 100.12)
  assert receiver.status(100.2, False)["source"] == "device"
  assert receiver.current(102.01) is None
  receiver.begin("other-phone", 102.1)
  receiver.accept(payload(receiver, 102.1, 1, platform="android"), 102.12)
  assert receiver.fix.gps["source"] == "android"
  assert receiver.current(122.1) is None


def test_stale_hardware_not_replayed(receiver):
  sent = []
  mux = PhoneGPSMux("gpsLocation", lambda name, data: sent.append(data))
  receiver.accept(payload(receiver), 100.02)
  mux.update(receiver.fix, 100.02)
  mux.device(hardware("gpsLocation", 99), receiver.fix)
  mux.update(None, 103)
  assert len(sent) == 1


def test_disabled_passthrough_is_byte_identical():
  sent = []
  mux = PhoneGPSMux("gpsLocation", lambda name, data: sent.append(data))
  packets = [hardware("gpsLocation", 100 + index / 10) for index in range(100)]
  for packet in packets:
    mux.device(packet, None)
  assert sent == packets


def test_live_socket_toggle_and_recovery():
  class DeviceParams:
    enabled = False

    def get_bool(self, name):
      return self.enabled if name == "IQPhoneGPSEnabled" else True

  params = DeviceParams()
  stop = threading.Event()
  sent = []

  class Publisher:
    def send(self, name, data):
      sent.append(data if isinstance(data, bytes) else data.to_bytes())

  with tempfile.TemporaryDirectory(prefix="pg-", dir="/tmp") as directory:
    endpoint = f"phonegps-test-{Path(directory).name}"
    hardware_pub = messaging.pub_sock(endpoint)
    hardware_sub = messaging.sub_sock(endpoint, conflate=False)
    path = f"{directory}/rpc"
    worker = threading.Thread(target=run, args=(params, "gpsLocation", hardware_sub, Publisher(), path, stop))
    worker.start()
    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as client:
      client.bind(f"{directory}/client")
      client.settimeout(1)

      def request(method, arguments=None):
        client.sendto(json.dumps({"method": method, "params": arguments or {}}).encode(), path)
        return json.loads(client.recv(4096))

      def wait_for(predicate):
        deadline = time.monotonic() + 2
        while not predicate() and time.monotonic() < deadline:
          time.sleep(0.01)
        assert predicate()

      try:
        wait_for(lambda: Path(path).exists())
        initial = hardware("gpsLocation", time.monotonic())
        hardware_pub.send(initial)
        wait_for(lambda: initial in sent)
        assert request("beginPhoneGPS", {"clientId": "test-phone"})["success"] == 0
        params.enabled = True
        wait_for(lambda: request("getPhoneGPSStatus")["enabled"])
        session = request("beginPhoneGPS", {"clientId": "test-phone"})
        receiver = PhoneGPSReceiver()
        receiver.session = session["sessionId"]
        assert request("sendPhoneGPS", payload(receiver, time.monotonic()))["success"] == 1
        wait_for(lambda: getattr(messaging.log_from_bytes(sent[-1]), "gpsLocation").source == "iOS")
        next_device = hardware("gpsLocation", time.monotonic())
        hardware_pub.send(next_device)
        assert request("getPhoneGPSStatus")["source"] == "phone"
        params.enabled = False
        wait_for(lambda: next_device in sent)
        assert request("getPhoneGPSStatus")["source"] == "device"
        params.enabled = True
        wait_for(lambda: request("getPhoneGPSStatus")["enabled"])
        session = request("beginPhoneGPS", {"clientId": "test-phone"})
        receiver.session = session["sessionId"]
        assert request("sendPhoneGPS", payload(receiver, time.monotonic()))["success"] == 1
        wait_for(lambda: getattr(messaging.log_from_bytes(sent[-1]), "gpsLocation").source == "iOS")
      finally:
        stop.set()
        worker.join(timeout=2)
        assert not worker.is_alive()
