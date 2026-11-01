"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""

import time

import pyray as rl

from iqpilot.common.params import Params
from iqpilot.ui.onroad.big_model_status import SourceState, draw_source_label, resolve_source
from iqpilot.ui.onroad.emac_status import draw_source_icon
from iqpilot.selfdrive.ui.ui_state import ui_state
from iqpilot.system.ui.lib.application import FontWeight, gui_app
from iqpilot.system.ui.lib.text_measure import measure_text_cached

_POLL_S = 1.0
_FONT_SIZE = 44
_MAC_ICON_H = 46
_MAC_ICON_W = 62
_MAC_FAILED_ICON_W = 78
_WHEEL_H = 50
_MARGIN_R = 12
_MARGIN_B = 14


class EmacSourceIndicator:
  def __init__(self):
    self._params = Params()
    self._font = gui_app.font(FontWeight.SEMI_BOLD)
    self._last_poll = 0.0
    self._label = ""
    self._state = SourceState.HIDDEN

  def update(self) -> None:
    now = time.monotonic()
    if now - self._last_poll < _POLL_S:
      return
    self._last_poll = now
    self._label, self._state = resolve_source(self._params, ui_state.engaged)

  def render(self, rect: rl.Rectangle) -> None:
    if self._state == SourceState.HIDDEN:
      return
    if self._label == "MAC":
      width = _MAC_FAILED_ICON_W if self._state == SourceState.FAILED else _MAC_ICON_W
      center = rl.Vector2(rect.x + rect.width - _MARGIN_R - width / 2,
                          rect.y + rect.height - _MARGIN_B - _WHEEL_H / 2)
      draw_source_icon(self._label, self._state, _MAC_ICON_H, center)
      return
    size = measure_text_cached(self._font, self._label, _FONT_SIZE)
    x = rect.x + rect.width - _MARGIN_R - size.x
    y = rect.y + rect.height - _MARGIN_B - (_WHEEL_H + size.y) / 2
    draw_source_label(self._font, self._label, self._state, rl.Vector2(x, y), _FONT_SIZE)
