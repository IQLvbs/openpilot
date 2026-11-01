from iqpilot.common.ui_settings import DISPLAY_BRIGHTNESS_VALUES, SLEEP_TIMER_VALUES
from iqpilot.selfdrive.ui.mici.layouts.settings.iq_widgets import MappedParamToggle
from iqpilot.system.ui.widgets.scroller import NavScroller
from iqpilot.system.ui.lib.multilang import tr


class DisplayLayoutMici(NavScroller):
  def __init__(self):
    super().__init__()
    brightness = list(DISPLAY_BRIGHTNESS_VALUES.values())
    self._display_bright = MappedParamToggle(tr("display brightness"), "Brightness",
                                             [tr("default") if v == 0 else f"{v}%" for v in brightness], brightness)
    self._interact = MappedParamToggle(tr("sleep timer"), "InteractivityTimeout",
                                       [tr("default"), "30s", "1m", "2m"], list(SLEEP_TIMER_VALUES.values()))
    self._items = [self._display_bright, self._interact]
    self._scroller.add_widgets(self._items)

  def show_event(self):
    super().show_event()
    for item in self._items:
      item.refresh()
