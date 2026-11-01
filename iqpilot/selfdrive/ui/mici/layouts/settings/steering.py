"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""

import pyray as rl

from iqpilot.common.swaglog import cloudlog
from iqpilot.common.turn_guard_settings import get_turn_guards_enabled, set_turn_guards_enabled
from iqpilot.common.ui_settings import NO_NUDGE_WARNING
from iqpilot.selfdrive.ui.lib.turn_guard_settings import TURN_GUARDS_DESCRIPTION, turn_guards_availability
from iqpilot.selfdrive.ui.mici.widgets.stock_button import BigButton, BigParamControl, BigToggle
from iqpilot.selfdrive.ui.mici.widgets.stock_dialog import BigDialogBase
from iqpilot.selfdrive.ui.mici.layouts.settings.iq_widgets import MappedParamToggle
from iqpilot.selfdrive.ui.ui_state import ui_state
from iqpilot.system.ui.lib.application import FontWeight, gui_app
from iqpilot.system.ui.widgets.button import SmallButton
from iqpilot.system.ui.widgets.label import MiciLabel
from iqpilot.system.ui.widgets.scroller import NavScroller
from iqpilot.system.ui.widgets.slider import SmallSlider
from iqpilot.system.ui.lib.multilang import tr


def _aol_modes() -> list[str]:
  return [tr("stay engaged"), tr("standby"), tr("disengage")]


def _has_limited_sab_options() -> bool:
  brand = ""
  if ui_state.is_offroad():
    bundle = ui_state.params.get("CarPlatformBundle")
    if bundle:
      brand = bundle.get("brand", "")
  if not brand:
    brand = ui_state.CP.brand if ui_state.CP else ""
  return brand == "rivian"


class SabSettingsPanel(NavScroller):
  def __init__(self):
    super().__init__()
    self._mode = MappedParamToggle(tr("Brake Response Mode"), "AolSteeringMode",
                                   _aol_modes(), [0, 1, 2], value_only=True)
    self._steer_override = BigParamControl(tr("Pause While You Steer"), "AolPauseOnSteeringOverride")
    self._scroller.add_widgets([self._mode, self._steer_override])

  def show_event(self):
    super().show_event()
    limited = _has_limited_sab_options()
    if limited:
      ui_state.params.remove("AolMainCruiseAllowed")
      ui_state.params.put_bool("AolUnifiedEngagementMode", True)
      ui_state.params.put("AolSteeringMode", 2)
    offroad = ui_state.is_offroad()
    for w in (self._mode,):
      w.refresh()
      w.set_enabled(offroad and not limited)
    self._steer_override.refresh()
    self._steer_override.set_enabled(offroad)


class NoNudgeConfirmation(BigDialogBase):
  def __init__(self, confirm_callback):
    super().__init__()
    self._title = MiciLabel(tr("Are you sure?"), font_size=32, font_weight=FontWeight.BOLD)
    self._warning = MiciLabel(tr(NO_NUDGE_WARNING), font_size=24, width=int(self.rect.width - 40),
                              line_height=26, wrap_text=True, elide_right=False)
    self._cancel = self._child(SmallButton(tr("cancel")))
    self._cancel.set_click_callback(self.dismiss)
    self._slider = self._child(SmallSlider(tr("slide to\nenable"), lambda: self.dismiss(confirm_callback), right_to_left=False))
    self._cancel.set_enabled(lambda: self.enabled and not self.is_dismissing and not self._slider.confirmed)
    self._slider.set_enabled(lambda: self.enabled and not self.is_dismissing)

  def _update_state(self):
    super()._update_state()
    if self.is_dismissing and not self._slider.confirmed:
      self._slider.reset()

  def _render(self, rect):
    self._title.render(rl.Rectangle(rect.x + 20, rect.y + 18, rect.width - 40, 38))
    self._warning.set_width(int(rect.width - 40))
    self._warning.render(rl.Rectangle(rect.x + 20, rect.y + 58, rect.width - 40, 78))
    self._cancel.set_opacity(1.0 - self._slider.slider_percentage)
    self._cancel.set_visible(self._slider.slider_percentage < 0.8)
    self._cancel.set_position(rect.x + 8, rect.y + rect.height - self._cancel.rect.height)
    self._cancel.render()
    self._slider.set_position(rect.x + rect.width - self._slider.rect.width, rect.y + rect.height - self._slider.rect.height)
    self._slider.render()


class LaneChangeSelector(MappedParamToggle):
  def __init__(self):
    super().__init__(tr("Auto Lane Change"), "IQLaneChangeTimer", [tr("nudge"), tr("no nudge")], [0, 1])

  def _enable_no_nudge(self):
    self._params.put(self._param, 1)
    self.refresh()

  def _handle_mouse_release(self, mouse_pos):
    BigButton._handle_mouse_release(self, mouse_pos)
    if self._params.get(self._param, return_default=True) == 1:
      self._params.put(self._param, 0)
      self.refresh()
    else:
      gui_app.push_widget(NoNudgeConfirmation(self._enable_no_nudge))


class LaneChangePanel(NavScroller):
  def __init__(self):
    super().__init__()
    self._timer = LaneChangeSelector()
    self._edge_guard = BigParamControl(tr("Lane Edge Guard"), "IQEdgeGuard")
    self._scroller.add_widgets([self._timer, self._edge_guard])

  def show_event(self):
    super().show_event()
    self._timer.refresh()
    self._edge_guard.refresh()


class SteeringLayoutMici(NavScroller):
  def __init__(self):
    super().__init__()

    self._sab_panel = SabSettingsPanel()
    self._lc_panel = LaneChangePanel()

    self._aol = BigParamControl(tr("AOL"), "AolEnabled", toggle_callback=self._on_aol_toggled)
    self._sab_settings_button = BigButton(tr("steering assistance behavior"))
    self._sab_settings_button.set_click_callback(lambda: gui_app.push_widget(self._sab_panel))
    self._lane_change = BigButton(tr("lane change"))
    self._lane_change.set_click_callback(lambda: gui_app.push_widget(self._lc_panel))
    self._turn_guards = BigToggle(tr("turn guards"), initial_state=get_turn_guards_enabled(ui_state.params),
                                  toggle_callback=self._on_turn_guards)
    self._refresh_turn_guards()

    self._scroller.add_widgets([
      self._aol, self._sab_settings_button, self._lane_change, self._turn_guards,
    ])

  def _on_aol_toggled(self, checked: bool):
    if checked:
      ui_state.params.put_bool("AolUnifiedEngagementMode", True)

  def _refresh_turn_guards(self):
    available, reason = turn_guards_availability(ui_state.CP, ui_state.params)
    saved = get_turn_guards_enabled(ui_state.params)
    offroad = ui_state.is_offroad()
    self._turn_guards.set_checked(saved)
    self._turn_guards.set_enabled(offroad and (available or saved))
    description = tr("Experimental") + ": " + tr(TURN_GUARDS_DESCRIPTION)
    if not available:
      description += " " + reason
    if not offroad:
      description += " " + tr("Park the vehicle to change this setting.")
    self._turn_guards.set_value(description)

  def _on_turn_guards(self, enabled: bool):
    if not ui_state.is_offroad() or (enabled and not turn_guards_availability(ui_state.CP, ui_state.params)[0]):
      self._refresh_turn_guards()
      return
    try:
      set_turn_guards_enabled(ui_state.params, enabled)
    except OSError:
      cloudlog.exception("Failed to save turn guards")
      self._refresh_turn_guards()
      return
    ui_state.params.put_bool("OnroadCycleRequested", True)
    self._refresh_turn_guards()

  def _refresh(self):
    self._refresh_turn_guards()
    offroad = ui_state.is_offroad()
    self._aol.refresh()
    self._aol.set_value(tr("on") if self._aol._checked else tr("off"))
    self._aol.set_enabled(offroad)
    self._sab_settings_button.set_enabled(offroad and self._aol._checked)

  def _update_state(self):
    super()._update_state()
    self._refresh()

  def show_event(self):
    super().show_event()
    self._refresh()
