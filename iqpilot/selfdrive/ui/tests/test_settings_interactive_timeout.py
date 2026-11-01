from types import SimpleNamespace

import pytest

from iqpilot.selfdrive.ui.layouts.main import MainLayout, MainState


def make_layout(current_mode):
  layout = object.__new__(MainLayout)
  layout._current_mode = current_mode
  layout._set_mode_calls = []
  layout._set_mode_for_state = lambda: layout._set_mode_calls.append(current_mode)
  return layout


class FakeSm:
  def __init__(self, v_ego, carstate_valid):
    self.valid = {"carState": carstate_valid}
    self._v_ego = v_ego

  def __getitem__(self, key):
    return SimpleNamespace(vEgo=self._v_ego)


class TestSettingsInteractiveTimeout:
  def _run(self, current_mode, started, v_ego, carstate_valid=True):
    layout = make_layout(current_mode)
    fake = SimpleNamespace(started=started, sm=FakeSm(v_ego, carstate_valid))
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("iqpilot.selfdrive.ui.layouts.main.ui_state", fake)
    try:
      layout._on_interactive_timeout()
    finally:
      monkeypatch.undo()
    return layout._set_mode_calls

  def test_stationary_in_settings_stays(self):
    # parked/charging hybrid reads onroad; the timeout must not eject from Settings
    assert self._run(MainState.SETTINGS, started=True, v_ego=0.0) == []

  def test_moving_in_settings_returns_to_road(self):
    assert self._run(MainState.SETTINGS, started=True, v_ego=5.0) == [MainState.SETTINGS]

  def test_onroad_layout_always_handled(self):
    assert self._run(MainState.ONROAD, started=True, v_ego=0.0) == [MainState.ONROAD]

  def test_home_layout_always_handled(self):
    assert self._run(MainState.HOME, started=True, v_ego=0.0) == [MainState.HOME]

  def test_offroad_in_settings_handled(self):
    # car off (offroad): existing behavior is unchanged
    assert self._run(MainState.SETTINGS, started=False, v_ego=0.0) == [MainState.SETTINGS]

  def test_invalid_carstate_treated_as_moving(self):
    # if speed is unknown, fail safe to the road view rather than trapping in settings
    assert self._run(MainState.SETTINGS, started=True, v_ego=0.0, carstate_valid=False) == [MainState.SETTINGS]
