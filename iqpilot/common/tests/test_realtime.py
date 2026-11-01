#!/usr/bin/env python3
import pytest

from iqpilot.common.realtime import config_background_thread, Ratekeeper


class MonotonicClock:
  def __init__(self) -> None:
    self.now = 0.

  def advance(self, seconds: float) -> None:
    self.now += seconds

  def __call__(self) -> float:
    return self.now


def test_ratekeeper_reset_discards_accumulated_lag(monkeypatch):
  clock = MonotonicClock()
  monkeypatch.setattr("iqpilot.common.realtime.time.monotonic", clock)
  rk = Ratekeeper(100)

  rk.monitor_time()
  clock.advance(0.075)
  rk.monitor_time()
  assert rk.remaining == pytest.approx(-0.055)
  assert rk.lag == pytest.approx(0.055)

  rk.reset()
  assert rk.remaining == 0.
  assert rk.lag == 0.

  rk.monitor_time()
  assert rk.remaining == pytest.approx(0.01)
  assert rk.lag == 0.


def test_ratekeeper_reset_preserves_frame_count(monkeypatch):
  clock = MonotonicClock()
  monkeypatch.setattr("iqpilot.common.realtime.time.monotonic", clock)
  rk = Ratekeeper(100)

  rk.monitor_time()
  clock.advance(0.01)
  rk.monitor_time()
  frame = rk.frame

  rk.reset()
  assert rk.frame == frame


def test_config_background_thread_restores_normal_scheduling(monkeypatch):
  calls = []
  monkeypatch.setattr("iqpilot.common.realtime.sys.platform", "linux")
  monkeypatch.setattr("iqpilot.common.realtime.PC", False)
  monkeypatch.setattr("iqpilot.common.realtime.os.cpu_count", lambda: 8)
  monkeypatch.setattr("iqpilot.common.realtime.os.SCHED_OTHER", 0, raising=False)
  monkeypatch.setattr("iqpilot.common.realtime.os.sched_param", lambda priority: priority, raising=False)
  monkeypatch.setattr("iqpilot.common.realtime.os.sched_setscheduler", lambda pid, policy, param: calls.append((pid, policy, param)), raising=False)
  monkeypatch.setattr("iqpilot.common.realtime.os.sched_setaffinity", lambda pid, cores: calls.append((pid, set(cores))), raising=False)

  config_background_thread()

  assert calls == [(0, 0, 0), (0, set(range(8)))]
