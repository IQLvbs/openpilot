"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos
"""
import pyray as rl

from iqpilot.selfdrive.ui.mici.onroad.hud_renderer import HudRenderer
from iqpilot.ui.onroad.hud_overlays import IQBlindSpotOverlay
from iqpilot.ui.mici.onroad.emac_source import EmacSourceIndicator
from iqpilot.ui.mici.onroad.speed_limit_sign import MiciSpeedLimitSign

class IQMiciHudRenderer(HudRenderer):
  def __init__(self):
    super().__init__()
    self._overlays = [IQBlindSpotOverlay(), EmacSourceIndicator()]
    self._speed_limit_sign = MiciSpeedLimitSign()

  def _update_state(self) -> None:
    super()._update_state()
    for overlay in self._overlays:
      overlay.update()
    self._speed_limit_sign.update()

  def _render(self, rect: rl.Rectangle) -> None:
    super()._render(rect)
    for overlay in self._overlays:
      overlay.render(rect)
    self._speed_limit_sign.set_obscured(self.drawing_top_icons() or not self._can_draw_top_icons)
    self._speed_limit_sign.render(rect)

  def _has_blind_spot_detected(self) -> bool:
    return any(getattr(overlay, "detected", False) for overlay in self._overlays)
