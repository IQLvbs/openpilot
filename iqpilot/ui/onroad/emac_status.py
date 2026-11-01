"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""

import math
import time

import pyray as rl

from iqpilot.common.params import Params
from iqpilot.ui.onroad.big_model_status import DockStatus, SourceState, draw_source_label, resolve_source
from iqpilot.selfdrive.ui import UI_BORDER_SIZE
from iqpilot.selfdrive.ui.onroad.driver_state import BTN_SIZE
from iqpilot.selfdrive.ui.ui_state import ui_state
from iqpilot.system.ui.lib.application import FontWeight, gui_app
from iqpilot.system.ui.lib.text_measure import measure_text_cached
from iqpilot.system.ui.widgets import Widget
from iqpilot.ui.onroad.theme import SECONDARY_TEXT

_POLL_S = 1.0
_FONT_SIZE = 70
_ICON_H = 76
_POWER_FONT_SIZE = 32
_POWER_GAP = 10
_ICONS = {"MAC": ("mac", 62 / 46), "GPU": ("egpu", 1.0)}
_GREY = rl.Color(165, 165, 170, 235)
_WHITE = rl.Color(255, 255, 255, 255)
_DIM = rl.Color(255, 255, 255, 165)
_ICON_CACHE: dict[tuple[str, int], dict[str, rl.Texture]] = {}


def source_icons(label: str, height: int) -> dict[str, rl.Texture] | None:
  spec = _ICONS.get(label)
  if spec is None:
    return None
  key = (label, height)
  if key not in _ICON_CACHE:
    base, aspect = spec
    width = int(height * aspect)
    _ICON_CACHE[key] = {
      "base": gui_app.texture(f"icons_mici/{base}.png", width, height),
      "green": gui_app.texture(f"icons_mici/{base}_green.png", width, height),
      "orange": gui_app.texture(f"icons_mici/{base}_orange.png", int(width * 1.26), height),
    }
  return _ICON_CACHE[key]


def source_icon(label: str, state: SourceState, height: int) -> tuple[rl.Texture, rl.Color] | None:
  icons = source_icons(label, height)
  if icons is None or state == SourceState.HIDDEN:
    return None
  if state == SourceState.ACTIVE:
    return icons["green"], _WHITE
  if state == SourceState.FAILED:
    return icons["orange"], _WHITE
  if state == SourceState.CROSSED:
    return icons["base"], _DIM
  pulse = 0.35 + 0.65 * (0.5 - 0.5 * math.cos(rl.get_time() * 6.0))
  return icons["base"], rl.Color(_GREY.r, _GREY.g, _GREY.b, int(_GREY.a * pulse))


def draw_source_icon(label: str, state: SourceState, height: int, center: rl.Vector2) -> float:
  icon = source_icon(label, state, height)
  if icon is None:
    return 0.0
  tex, tint = icon
  x = int(center.x - tex.width / 2)
  y = int(center.y - tex.height / 2)
  rl.draw_texture(tex, x, y, tint)
  if state == SourceState.CROSSED:
    cy = y + tex.height // 2
    rl.draw_line_ex(rl.Vector2(x - 4, cy), rl.Vector2(x + tex.width + 4, cy), 4, _DIM)
  return float(tex.width)


def _dock_texture(status: DockStatus, height: int):
  icons = source_icons("GPU", height)
  if icons is None or status == DockStatus.HIDDEN:
    return None
  return {DockStatus.READY: icons["green"], DockStatus.FAULT: icons["orange"]}.get(status, icons["base"])


def dock_icon_width(status: DockStatus, height: int) -> float:
  tex = _dock_texture(status, height)
  return float(tex.width) if tex is not None else 0.0


def draw_dock_icon(status: DockStatus, progress: float, height: int, x: float, cy: float) -> float:
  tex = _dock_texture(status, height)
  if tex is None:
    return 0.0
  icons = source_icons("GPU", height)
  tint = _WHITE if status in (DockStatus.READY, DockStatus.FAULT) else _GREY
  ix, iy = int(x), int(cy - tex.height / 2)
  rl.draw_texture(tex, ix, iy, tint)
  if status == DockStatus.SETUP:
    fill_h = int(tex.height * max(0.0, min(1.0, progress)))
    if fill_h > 0:
      rl.begin_scissor_mode(ix, iy + tex.height - fill_h, tex.width, fill_h)
      rl.draw_texture(icons["green"], ix, iy, _WHITE)
      rl.end_scissor_mode()
  return float(tex.width)


class EmacStatusRenderer(Widget):
  def __init__(self):
    super().__init__()
    self._params = Params()
    self._font = gui_app.font(FontWeight.SEMI_BOLD)
    self._power_font = gui_app.font(FontWeight.MEDIUM)
    self._last_poll = 0.0
    self._label = ""
    self._state = SourceState.HIDDEN

  def update(self):
    now = time.monotonic()
    if now - self._last_poll < _POLL_S:
      return
    self._last_poll = now
    self._label, self._state = resolve_source(self._params, ui_state.engaged)

  def _render(self, rect: rl.Rectangle):
    if self._state == SourceState.HIDDEN:
      return
    if self._label not in _ICONS:
      size = measure_text_cached(self._font, self._label, _FONT_SIZE)
      x = rect.x + UI_BORDER_SIZE + BTN_SIZE // 2 - size.x / 2
      y = rect.y + rect.height / 2 - size.y / 2
      draw_source_label(self._font, self._label, self._state, rl.Vector2(x, y), _FONT_SIZE)
      return
    center = rl.Vector2(rect.x + UI_BORDER_SIZE + BTN_SIZE // 2, rect.y + rect.height / 2)
    draw_source_icon(self._label, self._state, _ICON_H, center)
    if self._label == "GPU":
      self._draw_power_draw(center)

  def _draw_power_draw(self, icon_center: rl.Vector2) -> None:
    sm = ui_state.sm
    if not sm.alive["egpuDockState"] or sm.recv_frame["egpuDockState"] <= 0:
      return
    watts = sm["egpuDockState"].powerDrawW
    if not math.isfinite(watts) or watts < 0:
      return
    text = f"{watts:.0f} W"
    size = measure_text_cached(self._power_font, text, _POWER_FONT_SIZE)
    pos = rl.Vector2(round(icon_center.x - size.x / 2), round(icon_center.y + _ICON_H / 2 + _POWER_GAP))
    rl.draw_text_ex(self._power_font, text, pos, _POWER_FONT_SIZE, 0, SECONDARY_TEXT)
