"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos
"""

import time
import pyray as rl

try:
  from iqpilot.common.params import Params
except ImportError:
  Params = None

REFRESH_INTERVAL = 2.0

DEFAULT_ACCENT_HEX = "#00FFF5"


def _hex_to_rgb(hex_str: str) -> tuple[int, int, int]:
  h = hex_str.strip().lstrip("#")
  if len(h) == 3:
    h = h[0] * 2 + h[1] * 2 + h[2] * 2
  if len(h) != 6:
    return (0x00, 0xFF, 0xF5)  # fallback to default
  try:
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
  except ValueError:
    return (0x00, 0xFF, 0xF5)


def _darken(r: int, g: int, b: int, factor: float = 0.08) -> tuple[int, int, int]:
  return (int(r * factor), int(g * factor), int(b * factor))


class _NeonTheme:
  def __init__(self):
    self._params = Params() if Params else None
    self._last_refresh: float = 0.0
    self._hex: str = DEFAULT_ACCENT_HEX
    self._r, self._g, self._b = _hex_to_rgb(DEFAULT_ACCENT_HEX)
    self._refresh()

  def _refresh(self):
    self._last_refresh = time.monotonic()
    if self._params is None:
      return
    try:
      stored = self._params.get("UIAccentColor")
      if stored and isinstance(stored, str) and stored.strip():
        self._hex = stored.strip()
        self._r, self._g, self._b = _hex_to_rgb(self._hex)
    except Exception:
      pass

  def _maybe_refresh(self):
    if time.monotonic() - self._last_refresh >= REFRESH_INTERVAL:
      self._refresh()

  def glow(self, alpha: int = 255) -> rl.Color:
    self._maybe_refresh()
    return rl.Color(self._r, self._g, self._b, alpha)

  def glow_mid(self, alpha: int = 130) -> rl.Color:
    self._maybe_refresh()
    return rl.Color(self._r, self._g, self._b, alpha)

  def glow_outer(self, alpha: int = 45) -> rl.Color:
    self._maybe_refresh()
    return rl.Color(self._r, self._g, self._b, alpha)

  def bg(self) -> rl.Color:
    self._maybe_refresh()
    dr, dg, db = _darken(self._r, self._g, self._b, 0.08)
    dr = max(dr, 0x0A)
    dg = max(dg, 0x0A)
    db = max(db, 0x0A)
    return rl.Color(dr, dg, db, 255)

  def bg_pressed(self) -> rl.Color:
    self._maybe_refresh()
    dr, dg, db = _darken(self._r, self._g, self._b, 0.13)
    dr = max(dr, 0x0D)
    dg = max(dg, 0x0D)
    db = max(db, 0x0D)
    return rl.Color(dr, dg, db, 255)

  @property
  def hex(self) -> str:
    self._maybe_refresh()
    return self._hex

  def set_accent(self, hex_color: str):
    r, g, b = _hex_to_rgb(hex_color)
    self._r, self._g, self._b = r, g, b
    self._hex = "#" + hex_color.strip().lstrip("#").upper()
    self._last_refresh = time.monotonic()
    if self._params:
      try:
        self._params.put_nonblocking("UIAccentColor", self._hex)
      except Exception:
        pass

NeonTheme = _NeonTheme()
