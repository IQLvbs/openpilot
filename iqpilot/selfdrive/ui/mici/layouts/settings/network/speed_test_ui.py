"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""
import pyray as rl

from iqpilot.selfdrive.ui.ui_state import ui_state
from iqpilot.system.ui.lib.application import FontWeight, MousePos, gui_app
from iqpilot.system.ui.lib.multilang import tr
from iqpilot.system.ui.lib.speed_test import Phase, SpeedTest, SpeedTestState, format_latency, format_mbps, shared_speed_test
from iqpilot.system.ui.lib.text_measure import measure_text_cached
from iqpilot.system.ui.widgets.nav_widget import NavWidget
from iqpilot.ui.theme import NeonTheme

TAP_SLOP_PX = 24
CARD = rl.Color(25, 28, 32, 255)
TRACK = rl.Color(255, 255, 255, 28)
GREY = rl.Color(160, 166, 174, 255)
DIM = rl.Color(100, 108, 118, 255)
WHITE = rl.Color(244, 246, 248, 255)
RED = rl.Color(242, 116, 116, 255)


def readouts(state: SpeedTestState) -> tuple:
  return (
    (tr("Download"), format_mbps(state.live_mbps if state.phase == Phase.DOWNLOAD else state.download_mbps), "Mbps", Phase.DOWNLOAD),
    (tr("Upload"), format_mbps(state.live_mbps if state.phase == Phase.UPLOAD else state.upload_mbps), "Mbps", Phase.UPLOAD),
    (tr("Ping"), format_latency(state.latency_ms), "ms", Phase.LATENCY),
  )


def status_text(state: SpeedTestState) -> str:
  if state.phase == Phase.FAILED:
    return state.error or tr("Couldn't connect. Try again.")
  if state.running:
    return {
      Phase.LATENCY: tr("Measuring latency..."),
      Phase.DOWNLOAD: tr("Measuring download..."),
      Phase.UPLOAD: tr("Measuring upload..."),
    }[state.phase]
  if state.phase == Phase.DONE:
    return tr("Test complete")
  return tr("Check your connection speed")


class SpeedTestMici(NavWidget):
  def __init__(self, test: SpeedTest | None = None):
    super().__init__()
    self._test = test or shared_speed_test()
    self._press_pos: MousePos | None = None
    self._action_rect = rl.Rectangle(gui_app.width - 154, 24, 130, 48)

  def _metered(self) -> bool:
    try:
      return bool(ui_state.sm["deviceState"].networkMetered)
    except Exception:
      return False

  def toggle(self) -> None:
    if self._test.state.running:
      self._test.cancel()
    else:
      self._test.start(self._metered())

  def _handle_mouse_press(self, mouse_pos: MousePos) -> None:
    self._press_pos = mouse_pos

  def _handle_mouse_release(self, mouse_pos: MousePos) -> None:
    press, self._press_pos = self._press_pos, None
    if press is None or self._dragging_down or self._playing_dismiss_animation:
      return
    if (abs(mouse_pos.x - press.x) <= TAP_SLOP_PX and abs(mouse_pos.y - press.y) <= TAP_SLOP_PX
        and rl.check_collision_point_rec(press, self._action_rect)
        and rl.check_collision_point_rec(mouse_pos, self._action_rect)):
      self.toggle()

  def _draw_text(self, text: str, weight: FontWeight, size: int, x: float, y: float, color: rl.Color,
                 max_width: float, align: str = "left") -> None:
    font = gui_app.font(weight)
    while size > 18 and measure_text_cached(font, text, size).x > max_width:
      size -= 1
    if measure_text_cached(font, text, size).x > max_width:
      while text and measure_text_cached(font, text + "...", size).x > max_width:
        text = text[:-1]
      text += "..."
    width = measure_text_cached(font, text, size).x
    if align == "center":
      x -= width / 2
    rl.draw_text_ex(font, text, rl.Vector2(round(x), round(y)), size, 0, color)

  def _render(self, rect: rl.Rectangle) -> None:
    rl.draw_rectangle_rec(rect, rl.BLACK)
    state = self._test.state
    accent = NeonTheme.glow()
    left, width = rect.x + 24, rect.width - 48
    self._action_rect = rl.Rectangle(rect.x + rect.width - 154, rect.y + 24, 130, 48)
    self._draw_text(tr("Speed test"), FontWeight.BOLD, 30, left, rect.y + 24, WHITE, width - 150)
    metered = state.metered if state.running or state.phase == Phase.DONE else self._metered()
    subtitle = tr("Metered - data capped") if metered else tr("Network performance")
    self._draw_text(subtitle, FontWeight.ROMAN, 19, left, rect.y + 57, GREY, width - 150)

    pressed = self._press_pos is not None and rl.check_collision_point_rec(self._press_pos, self._action_rect)
    background = NeonTheme.glow(55 if pressed else 30) if not state.running else CARD
    rl.draw_rectangle_rounded(self._action_rect, 0.45, 12, background)
    button_text = tr("Stop") if state.running else {Phase.DONE: tr("Retest"), Phase.FAILED: tr("Retry")}.get(state.phase, tr("Start"))
    self._draw_text(button_text, FontWeight.BOLD, 24, self._action_rect.x + self._action_rect.width / 2,
                    self._action_rect.y + 12, WHITE if state.running else accent, 110, "center")

    card_width = (width - 20) / 3
    for i, (label, value, unit, phase) in enumerate(readouts(state)):
      x = left + i * (card_width + 10)
      active = state.phase == phase
      card = rl.Rectangle(x, rect.y + 88, card_width, 106)
      rl.draw_rectangle_rounded(card, 0.18, 12, NeonTheme.glow(22) if active else CARD)
      if active:
        rl.draw_circle_v(rl.Vector2(x + card_width - 14, rect.y + 106), 3, accent)
      self._draw_text(label, FontWeight.ROMAN, 20, x + 14, rect.y + 101, accent if active else GREY, card_width - 36)
      self._draw_text(value, FontWeight.DISPLAY, 42, x + 14, rect.y + 125, WHITE if value != "--" else DIM, card_width - 28)
      self._draw_text(unit, FontWeight.ROMAN, 18, x + 14, rect.y + 170, GREY, card_width - 28)

    self._draw_text(status_text(state), FontWeight.ROMAN, 20, left, rect.y + 204,
                    RED if state.phase == Phase.FAILED else GREY, width)
    if state.running:
      track = rl.Rectangle(left, rect.y + rect.height - 7, width, 3)
      rl.draw_rectangle_rounded(track, 1.0, 4, TRACK)
      progress = max(0.0, min(1.0, state.progress))
      if progress > 0:
        rl.draw_rectangle_rounded(rl.Rectangle(track.x, track.y, track.width * progress, track.height), 1.0, 4, accent)
