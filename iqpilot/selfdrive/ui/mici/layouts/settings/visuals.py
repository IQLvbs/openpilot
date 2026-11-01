"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""

from iqpilot.selfdrive.ui.mici.widgets.stock_button import BigParamControl
from iqpilot.system.ui.widgets.scroller import NavScroller
from iqpilot.system.ui.lib.multilang import tr


class VisualsLayoutMici(NavScroller):
  def __init__(self):
    super().__init__()
    self._blind_spot = BigParamControl(tr("Blind Spot Warnings"), "IQBlindSpotAlerts")
    self._turn_signals = BigParamControl(tr("Turn Signals"), "IQBlinkerIndicators")

    self._toggles = [self._blind_spot, self._turn_signals]
    self._scroller.add_widgets(self._toggles)

  def show_event(self):
    super().show_event()
    for w in self._toggles:
      w.refresh()
