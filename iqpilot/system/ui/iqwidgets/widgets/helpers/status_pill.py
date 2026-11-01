"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos
"""
from dataclasses import dataclass

import pyray as rl

from iqpilot.system.ui.lib.application import FontWeight, gui_app
from iqpilot.system.ui.lib.text_measure import measure_text_cached

PILL_HEIGHT = 48
PILL_ICON = 26
PILL_FONT_SIZE = 28
PILL_PAD = 16
PILL_GAP = 10
_STROKE = 4.0


@dataclass(frozen=True)
class StatusPill:
  label: str
  color: rl.Color
  glyph: str


def _line(x0: float, y0: float, x1: float, y1: float, color: rl.Color, thick: float = _STROKE) -> None:
  rl.draw_line_ex(rl.Vector2(x0, y0), rl.Vector2(x1, y1), thick, color)


def draw_glyph(glyph: str, x: float, cy: float, size: float, color: rl.Color) -> None:
  cx = x + size / 2
  if glyph == "check":
    _line(x + size * 0.04, cy + size * 0.06, x + size * 0.38, cy + size * 0.34, color)
    _line(x + size * 0.38, cy + size * 0.34, x + size * 0.98, cy - size * 0.34, color)
  elif glyph == "download":
    _line(cx, cy - size * 0.42, cx, cy + size * 0.14, color)
    _line(cx - size * 0.26, cy - size * 0.1, cx, cy + size * 0.18, color)
    _line(cx + size * 0.26, cy - size * 0.1, cx, cy + size * 0.18, color)
    _line(x + size * 0.12, cy + size * 0.42, x + size * 0.88, cy + size * 0.42, color)
  elif glyph == "chip":
    box = rl.Rectangle(x + size * 0.2, cy - size * 0.28, size * 0.6, size * 0.56)
    rl.draw_rectangle_rounded_lines_ex(box, 0.2, 6, 3, color)
    rl.draw_circle(int(cx), int(cy), 3, color)
    for dx in (-size * 0.12, size * 0.12):
      _line(cx + dx, box.y - size * 0.14, cx + dx, box.y, color, 3)
      _line(cx + dx, box.y + box.height, cx + dx, box.y + box.height + size * 0.14, color, 3)


def status_pill_width(pill: StatusPill) -> float:
  text = measure_text_cached(gui_app.font(FontWeight.MEDIUM), pill.label, PILL_FONT_SIZE)
  return PILL_PAD + PILL_ICON + PILL_GAP + text.x + PILL_PAD


def draw_status_pill(pill: StatusPill, x_right: float, cy: float) -> float:
  font = gui_app.font(FontWeight.MEDIUM)
  text = measure_text_cached(font, pill.label, PILL_FONT_SIZE)
  width = PILL_PAD + PILL_ICON + PILL_GAP + text.x + PILL_PAD
  rect = rl.Rectangle(x_right - width, cy - PILL_HEIGHT / 2, width, PILL_HEIGHT)
  c = pill.color
  rl.draw_rectangle_rounded(rect, 0.5, 10, rl.Color(c.r, c.g, c.b, 38))
  rl.draw_rectangle_rounded_lines_ex(rect, 0.5, 10, 2, rl.Color(c.r, c.g, c.b, 210))
  draw_glyph(pill.glyph, rect.x + PILL_PAD, cy, PILL_ICON, c)
  rl.draw_text_ex(font, pill.label, rl.Vector2(rect.x + PILL_PAD + PILL_ICON + PILL_GAP, cy - text.y / 2), PILL_FONT_SIZE, 0, c)
  return rect.x
