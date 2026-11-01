"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""

from iqpilot.system.ui.widgets.scroller import NavScroller
from iqpilot.selfdrive.ui.mici.widgets.stock_button import BigParamControl
from iqpilot.selfdrive.ui.layouts.settings.common import restart_needed_callback
from iqpilot.selfdrive.ui.ui_state import ui_state
from iqpilot.system.ui.lib.multilang import tr


class DashcamLayoutMici(NavScroller):
  def __init__(self):
    super().__init__()
    self._dashcam_mode = BigParamControl(tr("dashcam mode"), "DashcamMode", toggle_callback=restart_needed_callback)
    self._record_front = BigParamControl(tr("record driver camera"), "RecordFront", toggle_callback=restart_needed_callback)
    self._record_audio = BigParamControl(tr("record microphone audio"), "RecordAudio", toggle_callback=restart_needed_callback)

    self._scroller.add_widgets([self._dashcam_mode, self._record_front, self._record_audio])

    self._refresh_toggles = (
      ("DashcamMode", self._dashcam_mode),
      ("RecordFront", self._record_front),
      ("RecordAudio", self._record_audio),
    )

    self._record_front.set_enabled(False if ui_state.params.get_bool("RecordFrontLock") else (lambda: not ui_state.engaged))
    self._dashcam_mode.set_enabled(lambda: not ui_state.engaged)
    self._record_audio.set_enabled(lambda: not ui_state.engaged)
    ui_state.add_engaged_transition_callback(self._update_toggles)

  def show_event(self):
    super().show_event()
    self._update_toggles()

  def _update_toggles(self):
    ui_state.update_params()
    for key, item in self._refresh_toggles:
      item.set_checked(ui_state.params.get_bool(key))
