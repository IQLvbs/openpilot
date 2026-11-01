"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos
"""
import pytest

from iqpilot.ui.onroad.hud_renderer import IQHudRenderer, LIMIT_HIGHER, LIMIT_LOWER, LIMIT_PULSE_DURATION


def renderer():
  hud = IQHudRenderer.__new__(IQHudRenderer)
  hud._last_displayed_limit = None
  hud._limit_pulse_started = float("-inf")
  hud._limit_change_direction = 0
  return hud


def test_limit_pulse_ignores_initial_limit_and_reacquisition():
  hud = renderer()
  hud._track_limit_change(55, 10.0)
  assert hud._limit_highlight(10.0) == (None, 0.0)

  hud._track_limit_change(65, 10.5)
  assert hud._limit_highlight(10.6)[0] == LIMIT_HIGHER
  hud._track_limit_change(None, 11.0)
  assert hud._limit_highlight(11.0) == (None, 0.0)
  hud._track_limit_change(65, 12.0)
  assert hud._limit_highlight(12.0) == (None, 0.0)


@pytest.mark.parametrize(("old_limit", "new_limit", "expected_color"), [
  (55, 65, LIMIT_HIGHER),
  (65, 45, LIMIT_LOWER),
])
def test_limit_pulse_tracks_direction(old_limit, new_limit, expected_color):
  hud = renderer()
  hud._track_limit_change(old_limit, 10.0)
  hud._track_limit_change(new_limit, 11.0)

  color, strength = hud._limit_highlight(11.15)
  assert color == expected_color
  assert 0.35 <= strength <= 1.0
  assert hud._limit_highlight(11.0 + LIMIT_PULSE_DURATION) == (None, 0.0)
