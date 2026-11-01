"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pytest

from iqpilot.system.ui.lib import speed_test
from iqpilot.system.ui.lib.speed_test import CAPS, Phase, SpeedTest, mbps


class Server:
  def __init__(self):
    self.uploaded = 0
    self.download_requests: list[int] = []
    self.fail_downloads = False
    self.stall = threading.Event()
    owner = self

    class Handler(BaseHTTPRequestHandler):
      protocol_version = "HTTP/1.1"

      def log_message(self, *args):
        pass

      def do_GET(self):
        url = urlparse(self.path)
        size = int(parse_qs(url.query).get("bytes", ["0"])[0])
        if owner.fail_downloads and size:
          self.send_response(503)
          self.send_header("Content-Length", "0")
          self.end_headers()
          return
        if size:
          owner.download_requests.append(size)
        self.send_response(200)
        self.send_header("Content-Length", str(size))
        self.end_headers()
        chunk = bytes(64 * 1024)
        remaining = size
        try:
          while remaining > 0:
            if owner.stall.is_set():
              time.sleep(0.01)
              continue
            n = min(remaining, len(chunk))
            self.wfile.write(chunk[:n])
            remaining -= n
        except (BrokenPipeError, ConnectionResetError):
          pass

      def do_POST(self):
        total = 0
        while True:
          line = self.rfile.readline().strip()
          size = int(line, 16)
          if size == 0:
            self.rfile.readline()
            break
          total += len(self.rfile.read(size))
          self.rfile.readline()
        owner.uploaded += total
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    self.httpd.daemon_threads = False
    self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
    threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

  def close(self):
    self.stall.clear()
    self.httpd.shutdown()
    self.httpd.server_close()


@pytest.fixture
def server():
  s = Server()
  yield s
  s.close()


def run(test: SpeedTest, metered=False):
  assert test.start(metered)
  test.join(30)
  return test.state


def test_mbps():
  assert mbps(1_000_000, 1.0) == pytest.approx(8.0)


def test_full_run_reports_all_three_measurements(server):
  seen = []
  test = SpeedTest(server.url, stage_seconds=0.3)
  original = test._update
  test._update = lambda **c: (original(**c), seen.append(test.state))[0]
  state = run(test)
  assert state.phase == Phase.DONE and state.progress == 1.0 and not state.running
  assert state.latency_ms is not None and state.latency_ms >= 0
  assert state.download_mbps > 0 and state.upload_mbps > 0
  assert server.uploaded > 0
  phases = [s.phase for s in seen]
  assert phases.index(Phase.DOWNLOAD) < phases.index(Phase.UPLOAD) < phases.index(Phase.DONE)
  progress = [s.progress for s in seen]
  assert progress == sorted(progress)
  assert any(s.live_mbps > 0 for s in seen if s.phase == Phase.DOWNLOAD)


def test_metered_connection_uses_the_small_caps(server, monkeypatch):
  monkeypatch.setattr(speed_test, "CAPS", {False: (4_000_000, 1_000_000), True: (1_000_000, 256 * 1024)})
  state = run(SpeedTest(server.url, stage_seconds=30), metered=True)
  assert state.phase == Phase.DONE and state.metered
  assert server.download_requests == [1_000_000]
  assert server.uploaded == 256 * 1024
  assert CAPS[True][0] < CAPS[False][0] and CAPS[True][1] < CAPS[False][1]


def test_large_caps_are_split_into_requests_the_server_accepts(server, monkeypatch):
  monkeypatch.setattr(speed_test, "CAPS", {False: (2_500_000, 700_000), True: (1, 1)})
  monkeypatch.setattr(speed_test, "REQUEST_BYTES", 1_000_000)
  state = run(SpeedTest(server.url, stage_seconds=30))
  assert state.phase == Phase.DONE
  assert server.download_requests == [1_000_000, 1_000_000, 500_000]
  assert server.uploaded == 700_000
  assert speed_test.REQUEST_BYTES < 100_000_000


def test_server_errors_fail_with_a_short_reason(server):
  server.fail_downloads = True
  state = run(SpeedTest(server.url, stage_seconds=0.3))
  assert state.phase == Phase.FAILED
  assert state.error == "server error"
  assert state.latency_ms is not None


def test_unreachable_server_fails_without_hanging():
  state = run(SpeedTest("http://127.0.0.1:9", stage_seconds=0.3))
  assert state.phase == Phase.FAILED and state.error == "no connection"


def test_cancel_returns_to_idle(server):
  server.stall.set()
  test = SpeedTest(server.url, stage_seconds=30)
  assert test.start()
  deadline = time.monotonic() + 5
  while test.state.phase != Phase.DOWNLOAD and time.monotonic() < deadline:
    time.sleep(0.01)
  test.cancel()
  server.stall.clear()
  test.join(10)
  assert test.state.phase == Phase.IDLE and test.state.progress == 0.0


def test_start_is_ignored_while_running(server):
  server.stall.set()
  test = SpeedTest(server.url, stage_seconds=30)
  assert test.start()
  assert not test.start()
  test.cancel()
  server.stall.clear()
  test.join(10)


@pytest.mark.parametrize("value,text", [(None, "--"), (0.42, "0.4"), (84.21, "84.2"), (412.7, "413"), (1234.0, "1.23 G")])
def test_format_mbps(value, text):
  assert speed_test.format_mbps(value) == text


def test_status_texts_for_each_phase():
  done = speed_test.SpeedTestState(Phase.DONE, 1.0, 31.6, 84.2, 12.7)
  assert speed_test.summary(done) == "down 84.2 Mbps • up 12.7 Mbps • ping 32 ms"
  assert speed_test.compact_text(done) == "84.2 / 12.7 Mbps"
  assert speed_test.live_text(speed_test.SpeedTestState(Phase.LATENCY)) == "measuring ping"
  assert speed_test.live_text(speed_test.SpeedTestState(Phase.UPLOAD, live_mbps=9.94)) == "upload 9.9 Mbps"
  assert speed_test.live_text(speed_test.SpeedTestState(Phase.FAILED, error="timed out")) == "timed out"
  assert speed_test.compact_text(speed_test.SpeedTestState()) == ""


def test_shared_instance_is_reused():
  assert speed_test.shared_speed_test() is speed_test.shared_speed_test()
