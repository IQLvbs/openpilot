"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""

from iqpilot.system.ui.widgets.scroller import NavScroller
from iqpilot.selfdrive.ui.mici.widgets.stock_button import BigParamControl
from iqpilot.system.ui.lib.application import gui_app
from iqpilot.selfdrive.ui.ui_state import ui_state
from iqpilot.system.ui.lib.multilang import tr


class TogglesLayoutMici(NavScroller):
  """Equivalent to the BIG UI toggles page, minus cruise items (personality / speed limit /
  longitudinal control live in Cruise) and dashcam items (dashcam / driver-cam / mic live in Dashcam)."""
  def __init__(self):
    super().__init__()
    ui_state.params.put_bool("OpenpilotEnabledToggle", True)

    ldw = BigParamControl(tr("lane departure warnings"), "IsLdwEnabled")
    is_metric = BigParamControl(tr("use metric units"), "IsMetric")
    telemetry = BigParamControl(tr("share anonymous usage stats"), "IQTelemetryEnabled")

    self._scroller.add_widgets([ldw, is_metric, telemetry])

    self._refresh_toggles = (
      ("IsLdwEnabled", ldw),
      ("IsMetric", is_metric),
      ("IQTelemetryEnabled", telemetry),
    )

    if ui_state.params.get_bool("ShowDebugInfo"):
      gui_app.set_show_touches(True)
      gui_app.set_show_fps(True)

  def show_event(self):
    super().show_event()
    self._update_toggles()

  def _update_toggles(self):
    ui_state.update_params()
    for key, item in self._refresh_toggles:
      item.set_checked(ui_state.params.get_bool(key))
