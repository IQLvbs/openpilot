"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos

IQ.Pilot drawing surface. Onroad renderers and settings widgets paint through
this facade instead of touching pyray directly, so the HUD has one vocabulary
for shapes/text/textures. Each call forwards verbatim to the backing library —
output is identical to a raw pyray call, only the call site reads in IQ terms.
"""
import pyray as _p
from iqpilot.system.ui.lib.text_measure import measure_text_cached as _measure

Rgba = _p.Color
Box = _p.Rectangle
Pt = _p.Vector2

WHITE = _p.Color(255, 255, 255, 255)
BLACK = _p.Color(0, 0, 0, 255)
RED = _p.Color(230, 41, 55, 255)
CLEAR = _p.Color(0, 0, 0, 0)


def shade(r: int, g: int, b: int, a: int = 255) -> Rgba:
  return _p.Color(r, g, b, a)


def with_opacity(color: Rgba, alpha: float) -> Rgba:
  r, g, b = (color.r, color.g, color.b) if hasattr(color, "r") else color[:3]
  return _p.Color(r, g, b, int(alpha))


# --- filled / stroked shapes -------------------------------------------------

def panel(box: Box, roundness: float, segments: int, color: Rgba) -> None:
  _p.draw_rectangle_rounded(box, roundness, segments, color)


def panel_outline(box: Box, roundness: float, segments: int, thickness: float, color: Rgba) -> None:
  _p.draw_rectangle_rounded_lines_ex(box, roundness, segments, thickness, color)


def h_sweep(x: float, y: float, w: float, h: float, left: Rgba, right: Rgba) -> None:
  _p.draw_rectangle_gradient_h(int(x), int(y), int(w), int(h), left, right)


def disc_at(center: Pt, radius: float, color: Rgba) -> None:
  _p.draw_circle_v(center, radius, color)


def annulus(center: Pt, inner: float, outer: float, start: float, end: float, segments: int, color: Rgba) -> None:
  _p.draw_ring(center, inner, outer, start, end, segments, color)


def wedge(a: Pt, b: Pt, c: Pt, color: Rgba) -> None:
  _p.draw_triangle(a, b, c, color)


# --- text --------------------------------------------------------------------

def span(box_w, text: str, size: int, spacing: float = 0):
  """Measured extent of a text run (Vector2)."""
  return _measure(box_w, text, size, spacing)


def glyphs(font, text: str, at: Pt, size: int, color: Rgba, spacing: float = 0) -> None:
  _p.draw_text_ex(font, text, at, size, spacing, color)


def glyphs_centered(font, text: str, size: int, center: Pt, color: Rgba, spacing: float = 0) -> None:
  extent = _measure(font, text, size, spacing)
  _p.draw_text_ex(font, text, _p.Vector2(center.x - extent.x / 2, center.y - extent.y / 2), size, spacing, color)


# --- textures ----------------------------------------------------------------

def stamp(tex, x: float, y: float, tint: Rgba) -> None:
  _p.draw_texture(tex, int(x), int(y), tint)
