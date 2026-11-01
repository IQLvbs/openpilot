"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""

from iqpilot.common.params import Params, UnknownKeyName
from iqpilot.common.ui_settings import LONGITUDINAL_MODE_VALUES
from iqpilot.selfdrive.longitudinal_settings import (
  LONGITUDINAL_MODE_DYNAMIC,
  LONGITUDINAL_MODE_PILOT,
  LONGITUDINAL_MODE_STOCK,
  PERSONALITY_VALUES,
  apply_longitudinal_mode,
  get_follow_distance_state,
  get_longitudinal_mode,
  longitudinal_mode_needs_cycle,
  set_valid_personality,
)
from iqpilot.selfdrive.ui.mici.widgets.stock_button import BigButton, BigMultiToggle
from iqpilot.selfdrive.ui.ui_state import ui_state
from iqpilot.system.ui.lib.multilang import tr
from iqpilot.system.ui.lib.application import gui_app
from iqpilot.selfdrive.ui.mici.widgets.stock_dialog import BigConfirmationDialog


class MappedParamToggle(BigMultiToggle):
  """Multi-option toggle whose options map to arbitrary param values (int or float, drum-style).

  Up to PILL_LIMIT options render as the stock vertical pill column; more options would
  overflow the box, so they instead show the current value as a sub-label and cycle on tap.
  """
  PILL_LIMIT = 4

  def __init__(self, text: str, param: str, options: list[str], values: list | None = None, *, value_only: bool = False):
    self._force_value_only = value_only
    super().__init__(text, options)
    self._param = param
    self._values = values if values is not None else list(range(len(options)))
    self._params = Params()
    self.refresh()

  def _value_only(self) -> bool:
    return self._force_value_only or len(self._options) > self.PILL_LIMIT

  def _width_hint(self) -> int:
    if self._value_only():
      return BigButton._width_hint(self)
    return super()._width_hint()

  def _draw_content(self, btn_x: float, btn_y: float, btn_width: float, btn_height: float):
    if self._value_only():
      BigButton._draw_content(self, btn_x, btn_y, btn_width, btn_height)
    else:
      super()._draw_content(btn_x, btn_y, btn_width, btn_height)

  def refresh(self):
    try:
      raw = self._params.get(self._param, return_default=True)
    except UnknownKeyName:
      raw = self._values[0]
    try:
      cur = float(raw)
    except (TypeError, ValueError):
      cur = float(self._values[0])
    idx = min(range(len(self._values)), key=lambda i: abs(float(self._values[i]) - cur))
    self.set_value(self._options[idx])

  def _handle_mouse_release(self, mouse_pos):
    super()._handle_mouse_release(mouse_pos)
    idx = self._options.index(self.value)
    try:
      self._params.put(self._param, self._values[idx])
    except UnknownKeyName:
      pass


class FollowDistanceSelector(BigMultiToggle):
  OPTIONS = ["relaxed", "standard", "aggressive"]
  VALUES = tuple(reversed(PERSONALITY_VALUES))

  def __init__(self):
    self._display_options = [tr(option) for option in self.OPTIONS]
    super().__init__(tr("Follow Distance"), self._display_options)
    self._params = Params()
    self.refresh()

  def refresh(self):
    selection, enabled = get_follow_distance_state(self._params)
    if selection is None:
      selection = self._params.get("LongitudinalPersonality", return_default=True)
    self.set_value(self._display_options[self.VALUES.index(selection)])
    self.set_enabled(enabled)

  def _handle_mouse_release(self, mouse_pos):
    selection, enabled = get_follow_distance_state(self._params)
    if not enabled:
      self.refresh()
      return
    BigButton._handle_mouse_release(self, mouse_pos)
    next_selection = self.VALUES[(self.VALUES.index(selection) + 1) % len(self.VALUES)]
    set_valid_personality(self._params, next_selection)
    self.set_value(self._display_options[self.VALUES.index(next_selection)])


class IQModeSelector(BigMultiToggle):
  OPTIONS = ["Stock ACC", "IQ.Chill", "IQ.Pilot"]

  def __init__(self, mode_callback=None):
    self._display_options = [tr(option) for option in self.OPTIONS]
    super().__init__(tr("IQ Mode"), self._display_options)
    self._params = Params()
    self._mode_callback = mode_callback
    self._mode = LONGITUDINAL_MODE_STOCK
    self._iq_modes_available = False
    self.refresh()
    self.set_enabled(lambda: self._next() != self._mode)

  def _index(self) -> int:
    mode = get_longitudinal_mode(self._params)
    return LONGITUDINAL_MODE_PILOT if mode == LONGITUDINAL_MODE_DYNAMIC else mode

  def _toyota_factory_long_forced(self) -> bool:
    cp = ui_state.CP
    return bool(cp is not None and cp.brand == "toyota" and self._params.get_bool("IQToyotaFactoryLong"))

  def _read_iq_modes_available(self) -> bool:
    cp = ui_state.CP
    alpha_available = bool(cp is not None and cp.alphaLongitudinalAvailable)
    return alpha_available or self._params.get_bool("AlphaLongitudinalEnabled") or self._toyota_factory_long_forced()

  def _next(self) -> int:
    if not self._iq_modes_available:
      return self._mode
    order = LONGITUDINAL_MODE_VALUES[1:] if ui_state.is_onroad() else LONGITUDINAL_MODE_VALUES
    if self._mode not in order:
      return order[0]
    return order[(order.index(self._mode) + 1) % len(order)]

  def refresh(self):
    self._mode = self._index()
    self._iq_modes_available = self._read_iq_modes_available()
    self.set_value(self._display_options[LONGITUDINAL_MODE_VALUES.index(self._mode)])

  def _apply(self, idx: int):
    previous = self._mode
    toyota_forced = self._toyota_factory_long_forced()
    apply_longitudinal_mode(self._params, idx)
    if idx != LONGITUDINAL_MODE_STOCK and toyota_forced:
      self._params.put_bool("IQToyotaFactoryLong", False)
    if longitudinal_mode_needs_cycle(previous, idx) or (idx != LONGITUDINAL_MODE_STOCK and toyota_forced):
      self._params.put_bool("OnroadCycleRequested", True)

  def _handle_mouse_release(self, mouse_pos):
    nxt = self._next()
    if nxt == self._mode:
      return
    def apply():
      self._apply(nxt)
      if nxt == LONGITUDINAL_MODE_PILOT:
        self._params.put_bool("ExperimentalModeConfirmed", True)
      self.refresh()
      if self._mode_callback:
        self._mode_callback()

    if nxt == LONGITUDINAL_MODE_PILOT and not self._params.get_bool("ExperimentalModeConfirmed"):
      gui_app.push_widget(BigConfirmationDialog(tr("enable IQ.Pilot"),
                                               gui_app.texture("icons_mici/experimental_mode_mici.png", 60, 60), apply))
    else:
      apply()
