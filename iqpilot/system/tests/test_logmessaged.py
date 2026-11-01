import glob
import json
import os
import time

import iqpilot.cereal.messaging as messaging
from iqpilot.system.manager.process_config import managed_processes
from iqpilot.system.hardware.hw import Paths
from iqpilot.common.swaglog import cloudlog, ipchandler
from iqpilot.system.logmessaged import close_log_handler, shutdown


def test_close_log_handler_tolerates_rollover_interruption():
  class InterruptedHandler:
    def close(self):
      raise ValueError

  close_log_handler(InterruptedHandler())


def test_shutdown_releases_sockets_and_handler():
  calls = []

  class Recorder:
    def __init__(self, label):
      self.label = label

    def close(self):
      calls.append(f"{self.label}.close")

    def term(self):
      calls.append(f"{self.label}.term")

  shutdown(Recorder("sock"), Recorder("ctx"), Recorder("handler"))
  assert calls == ["sock.close", "ctx.term", "handler.close"]


class TestLogmessaged:
  def setup_method(self):
    managed_processes['logmessaged'].start()
    ipchandler.close()
    ipchandler.connect()

    self.sock = messaging.sub_sock("logMessage", timeout=1000, conflate=False)
    self.error_sock = messaging.sub_sock("errorLogMessage", timeout=1000, conflate=False)
    probe = f"logmessaged-ready-{time.monotonic_ns()}"
    deadline = time.monotonic() + 20
    log_ready = error_ready = False
    while not (log_ready and error_ready) and time.monotonic() < deadline:
      cloudlog.error(probe)
      time.sleep(0.05)
      log_ready |= any(json.loads(event.logMessage)["msg"] == probe for event in messaging.drain_sock(self.sock))
      error_ready |= any(json.loads(event.errorLogMessage)["msg"] == probe for event in messaging.drain_sock(self.error_sock))
    assert log_ready and error_ready

  def teardown_method(self):
    del self.sock
    del self.error_sock
    managed_processes['logmessaged'].stop(block=True)

  def _get_log_files(self):
    return list(glob.glob(os.path.join(Paths.swaglog_root(), "swaglog.*")))

  @staticmethod
  def _collect_matching(sock, field, expected, count):
    matches = []
    deadline = time.monotonic() + 2
    while len(matches) < count and time.monotonic() < deadline:
      for event in messaging.drain_sock(sock):
        if json.loads(getattr(event, field))["msg"] in expected:
          matches.append(event)
      time.sleep(0.01)
    return matches

  def test_simple_log(self):
    msgs = [f"abc {i}" for i in range(10)]
    for m in msgs:
      cloudlog.error(m)
    m = self._collect_matching(self.sock, "logMessage", set(msgs), len(msgs))
    received = [json.loads(event.logMessage)["msg"] for event in m]
    assert received[-len(msgs):] == msgs
    assert len(self._get_log_files()) >= 1

  def test_big_log(self):
    n = 10
    msg = "a"*3*1024*1024
    initial_size = sum(os.path.getsize(f) for f in self._get_log_files())
    for _ in range(n):
      cloudlog.info(msg)

    deadline = time.monotonic() + 3
    logsize = initial_size
    while logsize - initial_size <= n * len(msg) and time.monotonic() < deadline:
      time.sleep(0.01)
      logsize = sum(os.path.getsize(f) for f in self._get_log_files())

    msgs = messaging.drain_sock(self.sock)
    assert len(msgs) == 0
    written = logsize - initial_size
    assert n * len(msg) < written < n * (len(msg) + 1024)

  def test_large_log_below_publish_limit(self):
    msg = "a" * (256 * 1024)
    cloudlog.error(msg)
    msgs = self._collect_matching(self.sock, "logMessage", {msg}, 1)
    error_msgs = self._collect_matching(self.error_sock, "errorLogMessage", {msg}, 1)

    assert len(msgs) == 1
    assert json.loads(msgs[0].logMessage)["msg"] == msg
    assert len(error_msgs) == 1
    assert json.loads(error_msgs[0].errorLogMessage)["msg"] == msg
