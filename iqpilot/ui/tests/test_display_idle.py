# Copyright (c) 2026 IQ.Lvbs. All rights reserved.
import pytest

from iqpilot.ui.display_idle import DisplayIdleTimer


@pytest.fixture
def timer():
  now = [100.0]
  idle = DisplayIdleTimer(lambda: now[0])
  idle.configure(15)
  return idle, now


def test_elapsed_time_does_not_depend_on_rendered_frames(timer):
  idle, now = timer
  now[0] = 114.99
  assert not idle.expired
  now[0] = 115
  assert idle.expired


def test_hidden_view_stays_paused_through_alerts_and_input(timer):
  idle, now = timer
  idle.set_visible(False)
  for active in (True, False):
    idle.set_alert(active)
    idle.wake()
    now[0] += 100
    assert not idle.expired
  idle.set_visible(True)
  now[0] += 14
  assert not idle.expired
  now[0] += 1
  assert idle.expired


def test_alert_inhibits_dimming_until_a_full_delay_after_dismissal(timer):
  idle, now = timer
  now[0] += 20
  assert idle.expired
  idle.set_alert(True)
  now[0] += 100
  assert not idle.expired
  idle.set_alert(False)
  now[0] += 14
  assert not idle.expired
  now[0] += 1
  assert idle.expired


def test_unchanged_settings_do_not_extend_deadline(timer):
  idle, now = timer
  now[0] += 10
  idle.configure(15)
  now[0] += 5
  assert idle.expired


def test_delay_change_rearms_from_current_time(timer):
  idle, now = timer
  now[0] += 10
  idle.configure(30)
  now[0] += 29
  assert not idle.expired
  now[0] += 1
  assert idle.expired


def test_no_delay_disables_dimming(timer):
  idle, now = timer
  idle.configure(None)
  now[0] += 1000
  assert not idle.expired
  idle.wake()
  assert not idle.expired


def test_touch_rearms_expired_timer(timer):
  idle, now = timer
  now[0] += 20
  assert idle.expired
  idle.wake()
  assert not idle.expired
  now[0] += 15
  assert idle.expired


def test_zero_delay_expires_immediately(timer):
  idle, _ = timer
  idle.configure(0)
  assert idle.expired


@pytest.mark.parametrize("mode,delay,before,after", [
  (0, 30, 30.0, 10.0),
  (1, 30, 30.0, 10.0),
  (5, 30, 10.0, 20.0),
])
def test_ui_brightness_modes_use_the_deadline(timer, mode, delay, before, after):
  from types import SimpleNamespace
  from iqpilot.selfdrive.ui.ui_state import IQUIState, IQDevice

  idle, now = timer
  state = object.__new__(IQUIState)
  settings = {"OnroadScreenOffBrightness": mode, "OnroadScreenOffTimer": delay}
  state.params = SimpleNamespace(get=lambda key, **_: settings.get(key), get_bool=lambda _: False)
  state._onroad_idle = idle
  state.started = True
  state.update_params()

  assert IQDevice.set_onroad_brightness(state, True, 10.0) == before
  now[0] += 15
  if mode == 5:
    assert IQDevice.set_onroad_brightness(state, True, 10.0) == before
    now[0] += 15
  assert IQDevice.set_onroad_brightness(state, True, 10.0) == after
  assert IQDevice.set_onroad_brightness(state, False, 10.0) == 10.0


def test_first_touch_wakes_without_activating_controls(timer, monkeypatch):
  from types import SimpleNamespace
  from iqpilot.selfdrive.ui import ui_state as module

  idle, now = timer
  state = object.__new__(module.IQUIState)
  state._onroad_idle = idle
  state.onroad_brightness = 5
  state.started = True
  touches = [SimpleNamespace(left_down=True)]
  monkeypatch.setattr(module, "gui_app", SimpleNamespace(mouse_events=touches))
  now[0] += 15

  module.IQDevice.wake_from_dimmed_onroad_brightness(state, touches)

  assert not idle.expired
  assert touches == []
