"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""
import pyray as rl
from iqpilot.selfdrive.controls.lib.accel_boost import ACCEL_BOOST_MAX
from iqpilot.selfdrive.ui.mici.onroad import SIDE_PANEL_WIDTH
from iqpilot.selfdrive.ui.ui_state import ui_state
from iqpilot.system.ui.widgets import Widget

BAR_WIDTH = 24
TRACK_COLOR = rl.Color(50, 50, 50, 255)
FILL_COLOR = rl.Color(0, 255, 204, 255)


class BoostBar(Widget):
  def _render(self, rect: rl.Rectangle):
    bar_rect = rl.Rectangle(rect.x + rect.width - (SIDE_PANEL_WIDTH + BAR_WIDTH) / 2, rect.y + 24, BAR_WIDTH, rect.height - 48)
    rl.draw_rectangle_rounded(bar_rect, 1.0, 10, TRACK_COLOR)

    boost = ui_state.sm['iqPlan'].accelBoost if ui_state.engaged else 0.0
    if boost > 0.0:
      fill_height = bar_rect.height * min(boost, ACCEL_BOOST_MAX) / ACCEL_BOOST_MAX
      fill_rect = rl.Rectangle(bar_rect.x, bar_rect.y + bar_rect.height - fill_height, BAR_WIDTH, fill_height)
      rl.draw_rectangle_rounded(fill_rect, 1.0, 10, FILL_COLOR)
