"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""

from iqpilot.selfdrive.ui.mici.onroad.torque_bar import TorqueBar
from iqpilot.selfdrive.ui.ui_state import ui_state
from iqpilot.system.ui.iqwidgets.lib import canvas
from iqpilot.ui.onroad.hud_overlays import RoadNameBanner, clip_to_width


class RoadNameRendererMici(RoadNameBanner):
  TYPE_SIZE = 18
  MARGIN = 120
  GAP = 2
  TORQUE_SCALE = 1.0

  def _render(self, rect):
    if not self.road_name or not ui_state.road_name_toggle:
      return
    label = clip_to_width(self._face, self.road_name, self.TYPE_SIZE, rect.width - self.MARGIN)
    if not label:
      return
    extent = canvas.span(self._face, label, self.TYPE_SIZE)
    top = min(TorqueBar.resting_bottom(rect, self.TORQUE_SCALE) + self.GAP,
              rect.y + rect.height - extent.y)
    canvas.glyphs(self._face, label,
                  canvas.Pt(rect.x + (rect.width - extent.x) / 2, top),
                  self.TYPE_SIZE, canvas.WHITE)
