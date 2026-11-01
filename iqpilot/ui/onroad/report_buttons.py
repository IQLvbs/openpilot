# Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos
import time
import pyray as rl

from iqpilot.common.params import Params
from iqpilot.system.ui.lib.application import gui_app
from iqpilot.system.ui.widgets import Widget
from iqpilot.ui.onroad.theme import ENGAGED, INACTIVE, tile

ICON_SIZE = 78
GAP = 24
DROP = 30
IDLE_ALPHA = 120
ACTIVE_ALPHA = 240
CONFIRM_HOLD_S = 1.6


class ReportButton(Widget):
  def __init__(self, icon: str, param: str):
    super().__init__()
    self._params = Params()
    self._param = param
    self._texture = gui_app.texture(icon, ICON_SIZE, ICON_SIZE, keep_aspect_ratio=True)
    self._confirm_until = 0.0

  def _handle_mouse_release(self, _):
    super()._handle_mouse_release(_)
    self._params.put_bool(self._param, True)
    self._confirm_until = time.monotonic() + CONFIRM_HOLD_S

  def _render(self, rect: rl.Rectangle) -> None:
    confirming = time.monotonic() < self._confirm_until
    alpha = ACTIVE_ALPHA if (self.is_pressed or confirming) else IDLE_ALPHA
    cx = int(rect.x + rect.width / 2)
    cy = int(rect.y + rect.height / 2)
    tile(rl.Rectangle(rect.x - 6, rect.y - 6, rect.width + 12, rect.height + 12),
         ENGAGED if confirming else INACTIVE, self.is_pressed)
    rl.draw_texture(self._texture, cx - self._texture.width // 2, cy - self._texture.height // 2,
                    rl.Color(255, 255, 255, alpha))


class ReportButtons(Widget):
  def __init__(self):
    super().__init__()
    self._params = Params()
    self._police = self._child(ReportButton("icons/report_police.png", "IQNavReportPolice"))
    self._accident = self._child(ReportButton("icons/report_accident.png", "IQNavReportCrash"))
    self.set_visible(self._nav_enabled)

  def _nav_enabled(self) -> bool:
    return self._params.get_bool("IQNavAssistDev") and self._params.get_bool("IQAndroidNav")

  def layout_under(self, box: rl.Rectangle) -> None:
    total = 2 * ICON_SIZE + GAP
    x0 = box.x + box.width / 2 - total / 2
    y = box.y + box.height + DROP
    self._police.set_rect(rl.Rectangle(x0, y, ICON_SIZE, ICON_SIZE))
    self._accident.set_rect(rl.Rectangle(x0 + ICON_SIZE + GAP, y, ICON_SIZE, ICON_SIZE))
    self.set_rect(rl.Rectangle(x0, y, total, ICON_SIZE))

  def _render(self, _rect: rl.Rectangle) -> None:
    self._police.render()
    self._accident.render()
