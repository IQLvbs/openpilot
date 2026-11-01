from iqpilot.selfdrive.ui.mici.widgets.stock_button import BigButton, BigParamControl
from iqpilot.selfdrive.ui.mici.layouts.settings.iq_widgets import FollowDistanceSelector, MappedParamToggle, IQModeSelector
from iqpilot.system.ui.widgets.scroller import NavScroller
from iqpilot.system.ui.lib.multilang import tr


class GasBoostToggle(BigParamControl):
  def _width_hint(self) -> int:
    return BigButton._width_hint(self)


class CruiseLayoutMici(NavScroller):
  def __init__(self):
    super().__init__()
    self._follow_dist = FollowDistanceSelector()
    self._mode = IQModeSelector(self._follow_dist.refresh)
    self._speed_limit = MappedParamToggle(tr("Speed Limit"), "IQSpeedAssistMode",
                                        [tr("visual"), tr("control")], [1, 3])
    self._gas_boost = GasBoostToggle(tr("Gas Override Boost"), "IQGasOverrideBoost")
    self._main = [self._mode, self._follow_dist, self._speed_limit, self._gas_boost]
    self._scroller.add_widgets(self._main)

  def _refresh(self):
    for item in self._main:
      item.refresh()

  def show_event(self):
    super().show_event()
    self._refresh()
