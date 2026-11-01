import math
from functools import lru_cache

import pyray as rl

TEXT = rl.Color(255, 255, 255, 255)
SECONDARY_TEXT = rl.Color(255, 255, 255, 205)

INACTIVE = rl.Color(39, 46, 51, 242)
ENGAGED = rl.Color(0, 137, 126, 245)
PROMPT = rl.Color(218, 111, 27, 248)
CRITICAL = rl.Color(194, 35, 53, 250)


def tone(color: rl.Color, amount: float) -> rl.Color:
  return rl.Color(
    round(color.r * amount),
    round(color.g * amount),
    round(color.b * amount),
    color.a,
  )


def blend(start: rl.Color, end: rl.Color, amount: float) -> rl.Color:
  amount = max(0.0, min(amount, 1.0))
  return rl.Color(*(
    round(a + (b - a) * amount)
    for a, b in zip((start.r, start.g, start.b, start.a), (end.r, end.g, end.b, end.a), strict=True)
  ))


def ease_out(elapsed: float, duration: float, overshoot: float = 0.0) -> float:
  progress = max(0.0, min(elapsed / duration, 1.0))
  if overshoot <= 0.0:
    return 1.0 - (1.0 - progress) ** 4

  shifted = progress - 1.0
  return 1.0 + (overshoot + 1.0) * shifted ** 3 + overshoot * shifted ** 2


def pulse(elapsed: float, period: float) -> float:
  return 0.5 - 0.5 * math.cos(elapsed * math.tau / period)


def tile(rect: rl.Rectangle, color: rl.Color, pressed: bool = False) -> None:
  inset = 4 if pressed else 0
  bounds = rl.Rectangle(rect.x + inset, rect.y + inset, rect.width - inset * 2, rect.height - inset * 2)
  rl.draw_rectangle_rounded(bounds, 0.22, 12, tone(color, 0.88) if pressed else color)


def header_tiles(rect: rl.Rectangle) -> tuple[rl.Rectangle, rl.Rectangle, rl.Rectangle]:
  margin, gap, side = 36, 24, 180
  center_width = min(392, rect.width - 2 * (margin + side + gap))
  y = rect.y + margin
  return (
    rl.Rectangle(rect.x + margin, y, side, 180),
    rl.Rectangle(rect.x + (rect.width - center_width) / 2, y, center_width, 216),
    rl.Rectangle(rect.x + rect.width - margin - side, y, side, 180),
  )


def warning_color(elapsed: float) -> rl.Color:
  return blend(PROMPT, rl.Color(255, 196, 35, 250), 0.85 * pulse(elapsed, 1.3))


@lru_cache(maxsize=4)
def _camera_corners(left: float, top: float, width: float, height: float, radius: float):
  rect = rl.Rectangle(left, top, width, height)
  corners = []
  for x, y, sx, sy in ((rect.x, rect.y, 1, 1), (rect.x + rect.width, rect.y, -1, 1),
                       (rect.x, rect.y + rect.height, 1, -1), (rect.x + rect.width, rect.y + rect.height, -1, -1)):
    points = [rl.Vector2(x, y)]
    for step in range(17):
      angle = math.pi + step * math.pi / 32
      points.append(rl.Vector2(x + sx * radius * (1 + math.cos(angle)),
                               y + sy * radius * (1 + math.sin(angle))))
    if sx * sy < 0:
      points[1:] = reversed(points[1:])
    corners.append(points)
  return corners


def draw_camera_corners(rect: rl.Rectangle, radius: float) -> None:
  for points in _camera_corners(rect.x, rect.y, rect.width, rect.height, radius):
    rl.draw_triangle_fan(points, len(points), rl.BLACK)
