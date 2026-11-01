"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""
from types import SimpleNamespace

import pytest

from iqpilot.selfdrive.ui.mici.layouts.settings.network import speed_test_ui
from iqpilot.selfdrive.ui.mici.layouts.settings.network.speed_test_ui import SpeedTestMici, readouts, status_text
from iqpilot.system.ui.lib.application import MousePos
from iqpilot.system.ui.lib.speed_test import Phase, SpeedTestState


class FakeTest:
  def __init__(self):
    self.state = SpeedTestState()
    self.started: list[bool] = []
    self.cancelled = 0

  def start(self, metered=False):
    self.started.append(metered)
    return True

  def cancel(self):
    self.cancelled += 1


@pytest.fixture
def page(monkeypatch):
  monkeypatch.setattr(speed_test_ui, "ui_state", SimpleNamespace(sm={"deviceState": SimpleNamespace(networkMetered=True)}))
  test = FakeTest()
  widget = SpeedTestMici(test)
  widget._action_rect = speed_test_ui.rl.Rectangle(382, 24, 130, 48)
  return widget, test


@pytest.mark.parametrize("state,status", [
  (SpeedTestState(), "Check your connection speed"),
  (SpeedTestState(Phase.LATENCY), "Measuring latency..."),
  (SpeedTestState(Phase.DOWNLOAD, live_mbps=86.44), "Measuring download..."),
  (SpeedTestState(Phase.UPLOAD, live_mbps=12.9), "Measuring upload..."),
  (SpeedTestState(Phase.DONE, download_mbps=84.2, upload_mbps=12.7), "Test complete"),
  (SpeedTestState(Phase.FAILED, error="no connection"), "no connection"),
])
def test_status_follows_the_phase(state, status):
  assert status_text(state) == status


def test_readouts_keep_completed_results_while_upload_is_running():
  rows = readouts(SpeedTestState(Phase.UPLOAD, download_mbps=90.0, live_mbps=12.0, latency_ms=14))
  assert [(r[1], r[2]) for r in rows] == [("90.0", "Mbps"), ("12.0", "Mbps"), ("14", "ms")]


def test_readouts_show_final_rates_after_completion():
  rows = readouts(SpeedTestState(Phase.DONE, download_mbps=90.0, upload_mbps=12.0, live_mbps=0))
  assert [r[1] for r in rows] == ["90.0", "12.0", "--"]


def test_tap_starts_with_the_metered_flag_and_tap_again_cancels(page):
  widget, test = page
  widget._handle_mouse_press(MousePos(420, 48))
  widget._handle_mouse_release(MousePos(425, 52))
  assert test.started == [True]
  test.state = SpeedTestState(Phase.DOWNLOAD)
  widget._handle_mouse_press(MousePos(420, 48))
  widget._handle_mouse_release(MousePos(420, 48))
  assert test.cancelled == 1


def test_swipes_and_dismissals_do_not_start_a_test(page):
  widget, test = page
  widget._handle_mouse_press(MousePos(200, 40))
  widget._handle_mouse_release(MousePos(200, 200))
  widget._handle_mouse_press(MousePos(100, 120))
  widget._playing_dismiss_animation = True
  widget._handle_mouse_release(MousePos(100, 120))
  widget._handle_mouse_release(MousePos(100, 120))
  assert test.started == []


def test_tapping_results_does_not_start_or_stop(page):
  widget, test = page
  for phase in (Phase.IDLE, Phase.DOWNLOAD):
    test.state = SpeedTestState(phase)
    widget._handle_mouse_press(MousePos(100, 140))
    widget._handle_mouse_release(MousePos(100, 140))
  assert test.started == []
  assert test.cancelled == 0
