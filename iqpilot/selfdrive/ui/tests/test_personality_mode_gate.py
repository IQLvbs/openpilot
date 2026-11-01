# Copyright (c) 2026 IQ.Lvbs. All rights reserved.
from types import SimpleNamespace

import pytest

from iqpilot.selfdrive.longitudinal_settings import PERSONALITY_AGGRESSIVE, PERSONALITY_STANDARD
from iqpilot.selfdrive.ui.layouts.settings import toggles
from iqpilot.selfdrive.ui.mici.layouts.settings import iq_widgets


class FakeParams:
  def __init__(self, alpha, experimental, personality=PERSONALITY_STANDARD):
    self.values = {
      "AlphaLongitudinalEnabled": alpha,
      "ExperimentalMode": experimental,
      "IQDynamicMode": False,
      "IQToyotaFactoryLong": False,
      "LongitudinalPersonality": personality,
    }

  def get(self, key, return_default=False):
    return self.values[key]

  def get_bool(self, key):
    return bool(self.values.get(key, False))

  def put(self, key, value):
    self.values[key] = value


MODES = pytest.mark.parametrize("alpha,experimental,selectable", [
  (False, False, False),
  (True, False, True),
  (True, True, True),
], ids=["stock", "chill", "pilot"])


@MODES
def test_big_ui_personality_follows_iq_mode(monkeypatch, mocker, alpha, experimental, selectable):
  layout = mocker.Mock()
  layout._params = FakeParams(alpha, experimental)
  layout._toggles = {"LongitudinalControlMode": mocker.Mock()}
  layout._toggle_defs = {}
  layout._locked_toggles = set()
  layout._get_longitudinal_control_mode_index = lambda: toggles.TogglesLayout._get_longitudinal_control_mode_index(layout)
  cp = SimpleNamespace(alphaLongitudinalAvailable=True, brand="hyundai")
  monkeypatch.setattr(toggles, "ui_state", SimpleNamespace(CP=cp, engaged=False, update_params=lambda: None))
  toggles.TogglesLayout._update_toggles(layout)
  layout._long_personality_setting.action_item.set_enabled.assert_called_once_with(selectable)


@MODES
def test_mici_follow_distance_follows_iq_mode(monkeypatch, mocker, alpha, experimental, selectable):
  selector = mocker.Mock()
  selector.VALUES = iq_widgets.FollowDistanceSelector.VALUES
  selector._display_options = list(iq_widgets.FollowDistanceSelector.OPTIONS)
  selector._params = FakeParams(alpha, experimental)
  iq_widgets.FollowDistanceSelector.refresh(selector)
  selector.set_enabled.assert_called_once_with(selectable)
  monkeypatch.setattr(iq_widgets.BigButton, "_handle_mouse_release", lambda self, pos: None)
  iq_widgets.FollowDistanceSelector._handle_mouse_release(selector, None)
  expected = PERSONALITY_AGGRESSIVE if selectable else PERSONALITY_STANDARD
  assert selector._params.values["LongitudinalPersonality"] == expected
