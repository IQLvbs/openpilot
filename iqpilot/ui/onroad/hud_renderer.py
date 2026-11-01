"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos
"""
from __future__ import annotations

import time

import pyray as rl

from iqpilot.ui.onroad.torque_bar import TorqueBar
from iqpilot.selfdrive.ui.ui_state import ui_state
from iqpilot.selfdrive.ui.onroad.hud_renderer import CRUISE_DISABLED_CHAR, HudRenderer
from iqpilot.system.ui.lib.multilang import tr
from iqpilot.system.ui.lib.application import gui_app
from iqpilot.system.ui.lib.text_measure import measure_text_cached
from iqpilot.ui.onroad.hud_overlays import (
  IQDevMetricsOverlay,
  RoadNameRenderer,
  SpeedLimitState,
  IQTurnSignalOverlay,
  IQSpeedOverlay,
)
from iqpilot.ui.onroad.nav_map_panel import NavMapPanel
from iqpilot.ui.onroad.soft_warning import SoftWarningRenderer
from iqpilot.ui.onroad.emac_status import EmacStatusRenderer
from iqpilot.ui.onroad.report_buttons import ReportButtons
from iqpilot.ui.optional_private import optional_private_ui
from iqpilot.ui.onroad.theme import SECONDARY_TEXT, TEXT, pulse


LIMIT_PULSE_DURATION = 2.4
LIMIT_PULSE_PERIOD = 0.6
LIMIT_HIGHER = rl.Color(50, 255, 135, 255)
LIMIT_LOWER = rl.Color(255, 58, 72, 255)


REASONING_BAND_WIDTH_FRACTION = 0.62
REASONING_BAND_HEIGHT = 220.0


def _reasoning_rect(rect: rl.Rectangle) -> rl.Rectangle:
  width = rect.width * REASONING_BAND_WIDTH_FRACTION
  height = min(REASONING_BAND_HEIGHT, rect.height)
  return rl.Rectangle(rect.x + (rect.width - width) / 2.0,
                      rect.y + (rect.height - height) / 2.0,
                      width, height)


class IQHudRenderer(HudRenderer):
  def __init__(self):
    super().__init__()
    self.developer_ui = IQDevMetricsOverlay()
    self.emac_status = EmacStatusRenderer()
    self.nav_map_panel = NavMapPanel()
    self.road_name_renderer = RoadNameRenderer()
    self.speed_limit_renderer = SpeedLimitState()
    self.turn_signal_controller = IQTurnSignalOverlay()
    self.speed_renderer = IQSpeedOverlay()
    self.soft_warning_renderer = SoftWarningRenderer()
    self._torque_bar = TorqueBar(scale=3.0, always=True)
    self._event_tile_visible = False
    self.report_buttons = ReportButtons()
    optional_private_ui.get()
    self._last_displayed_limit: int | None = None
    self._limit_pulse_started = float("-inf")
    self._limit_change_direction = 0

  def set_event_tile_visible(self, visible: bool) -> None:
    self._event_tile_visible = visible

  def _update_state(self) -> None:
    super()._update_state()
    self.nav_map_panel.update()
    self.emac_status.update()
    self.road_name_renderer.update()
    self.speed_limit_renderer.update()
    has_limit = self.speed_limit_renderer.speed_limit_valid or self.speed_limit_renderer.speed_limit_last_valid
    self.limit_available = has_limit
    self.limit_speed_text = str(round(self.speed_limit_renderer.speed_limit_last)) if has_limit else "---"
    offset = round(self.speed_limit_renderer.speed_limit_offset)
    self.limit_offset_text = f"{offset:+d}" if has_limit and offset != 0 else ""
    self._track_limit_change(round(self.speed_limit_renderer.speed_limit_last) if has_limit else None, time.monotonic())
    self.turn_signal_controller.update()
    self.speed_renderer.update()
    self.soft_warning_renderer.update()

  def _draw_current_speed(self, rect: rl.Rectangle) -> None:
    self.speed_renderer.render(rect)

  def _center_text(self, text: str, font: rl.Font, size: int, center_x: float, y: float, color: rl.Color) -> None:
    width = measure_text_cached(font, text, size).x
    rl.draw_text_ex(font, text, rl.Vector2(center_x - width / 2, y), size, 0, color)

  def _track_limit_change(self, limit: int | None, now: float) -> None:
    previous = self._last_displayed_limit
    if limit is None:
      self._limit_change_direction = 0
      self._limit_pulse_started = float("-inf")
    if limit is not None and previous is not None and limit != previous:
      self._limit_change_direction = 1 if limit > previous else -1
      self._limit_pulse_started = now
    self._last_displayed_limit = limit

  def _limit_highlight(self, now: float) -> tuple[rl.Color | None, float]:
    elapsed = now - self._limit_pulse_started
    if self._limit_change_direction == 0 or not 0.0 <= elapsed < LIMIT_PULSE_DURATION:
      return None, 0.0
    strength = 0.35 + 0.65 * pulse(elapsed, LIMIT_PULSE_PERIOD)
    color = LIMIT_HIGHER if self._limit_change_direction > 0 else LIMIT_LOWER
    return color, strength

  @staticmethod
  def _draw_limit_glow(bounds: rl.Rectangle, color: rl.Color, strength: float) -> None:
    for spread, alpha in ((14, 18), (8, 34), (3, 72)):
      glow = rl.Rectangle(bounds.x - spread, bounds.y - spread, bounds.width + spread * 2, bounds.height + spread * 2)
      glow_color = rl.Color(color.r, color.g, color.b, round(alpha * strength))
      rl.draw_rectangle_rounded_lines_ex(glow, 0.3, 12, max(2, spread / 2), glow_color)

  def _draw_set_speed(self, rect: rl.Rectangle) -> None:
    center_x = rect.x + 108
    value = str(round(self.set_speed)) if self.is_cruise_set else CRUISE_DISABLED_CHAR
    self._center_text(tr("SET"), self._font_medium, 32, center_x, rect.y + 38, SECONDARY_TEXT)
    self._center_text(value, self._font_bold, 92, center_x, rect.y + 72, TEXT)

  def _draw_limit_sign(self, rect: rl.Rectangle) -> None:
    center_x = rect.x + rect.width - 108
    value = self.limit_speed_text if self.limit_available else "--"
    highlight, strength = self._limit_highlight(time.monotonic())
    if self.limit_available and highlight is not None:
      self._draw_limit_glow(rl.Rectangle(center_x - 86, rect.y + 32, 172, 156), highlight, strength)
    self._center_text(tr("LIMIT"), self._font_medium, 32, center_x, rect.y + 38, highlight or SECONDARY_TEXT)
    self._center_text(value, self._font_bold, 92, center_x, rect.y + 72, highlight or TEXT)
    if self.limit_available and self.limit_offset_text:
      self._center_text(self.limit_offset_text, self._font_semi_bold, 20, center_x, rect.y + 174, SECONDARY_TEXT)

  def _render(self, rect: rl.Rectangle) -> None:
    rl.draw_rectangle_gradient_v(int(rect.x), int(rect.y), int(rect.width), 320,
                                 rl.Color(0, 0, 0, 230), rl.BLANK)
    if self.is_cruise_available:
      self._draw_set_speed(rect)
    self._draw_limit_sign(rect)
    self._draw_current_speed(rect)
    mode_rect = rl.Rectangle(rect.x + 36, rect.y + rect.height - 180, 144, 144)
    if ui_state.sm['carState'].cruiseFaultLateralMode and ui_state.sm['carControl'].latActive:
      icon = gui_app.texture('icons/onroad/cruise_fault.png', 112, 112)
      rl.draw_texture(icon, int(mode_rect.x + 16), int(mode_rect.y + 16), rl.WHITE)
    else:
      self._exp_button.render(mode_rect)

    torque_rect = rect
    if set_visible := getattr(self._torque_bar, "set_visible", None):
      set_visible(ui_state.torque_bar and not self._event_tile_visible)
      self._torque_bar.render(torque_rect)
    elif ui_state.torque_bar:
      self._torque_bar.render(torque_rect)

    self.developer_ui.render(rect)
    self.emac_status.render(rect)
    self.road_name_renderer.render(torque_rect)
    self.turn_signal_controller.render(rect)
    self.soft_warning_renderer.render(rect)
    optional_private_ui.render_onroad(_reasoning_rect(rect))

    self.report_buttons.layout_under(rl.Rectangle(rect.x + 36, rect.y + 36, 440, 200))
    self.report_buttons.render()

  def render_navigation(self, rect: rl.Rectangle) -> bool:
    if self.nav_map_panel.maps_enabled():
      card = rl.Rectangle(rect.x + rect.width - 456, rect.y + rect.height - 356, 420, 320)
      self.nav_map_panel.render_details(rl.Rectangle(card.x, card.y - 94, card.width, 78))
      self.nav_map_panel.render_split(card)
      return True
    return False
