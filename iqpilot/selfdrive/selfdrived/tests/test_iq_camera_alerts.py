import copy
from types import SimpleNamespace

from iqpilot.cereal import car, custom, log
from iqpilot.common.atlas_alerts import HardDisableCard, Tags as ET, Tier as Priority
from iqpilot.selfdrive.selfdrived.alertmanager import AlertManager
from iqpilot.selfdrive.selfdrived.events import EVENTS
from iqpilot.selfdrive.selfdrived import iq_events


def alert(camera_type, *, report_id="", chime=False, distance=300.0):
  nav = SimpleNamespace(
    cameraType=camera_type,
    cameraDistance=distance,
    cameraSpeedLimit=25.0,
    cameraAlertId=report_id,
    cameraChime=chime,
  )
  return iq_events.speed_camera_alert(None, None, {"iqNavState": nav}, False, 0, None)


def test_existing_camera_audio_is_unchanged():
  result = alert(custom.IQNavState.CameraType.fixedSpeed)
  assert result.audible_alert == car.CarControl.HUDControl.AudibleAlert.prompt


def test_police_visual_mode_is_silent():
  result = alert(custom.IQNavState.CameraType.police, report_id="visual", chime=False)
  assert result.audible_alert == car.CarControl.HUDControl.AudibleAlert.none


def test_police_chime_is_deduplicated_by_report():
  iq_events._POLICE_CHIMED_IDS.clear()
  first = alert(custom.IQNavState.CameraType.police, report_id="police-a", chime=True)
  second = alert(custom.IQNavState.CameraType.police, report_id="police-a", chime=True)
  assert first.audible_alert == car.CarControl.HUDControl.AudibleAlert.prompt
  assert second.audible_alert == car.CarControl.HUDControl.AudibleAlert.none


def test_alpr_wording_uses_configured_region(monkeypatch):
  monkeypatch.setattr(iq_events, "_configured_country_code", lambda: "US")
  assert alert(custom.IQNavState.CameraType.alpr).alert_text_1.startswith("Flock / ALPR Camera")
  assert alert(custom.IQNavState.CameraType.alpr, distance=0.0).alert_text_1 == "Flock Camera Detected"

  monkeypatch.setattr(iq_events, "_configured_country_code", lambda: "DE")
  assert alert(custom.IQNavState.CameraType.alpr).alert_text_1.startswith("Traffic / ALPR Camera")
  assert alert(custom.IQNavState.CameraType.alpr, distance=0.0).alert_text_1 == "Traffic / ALPR Camera Detected"


def test_missing_region_is_safe_and_preserves_flock_wording(monkeypatch):
  class UnavailableParams:
    def get(self, key):
      raise OSError(key)

  monkeypatch.setattr(iq_events, "Params", UnavailableParams)
  assert iq_events._configured_country_code() == ""
  assert alert(custom.IQNavState.CameraType.alpr, distance=0.0).alert_text_1 == "Flock Camera Detected"


def test_driver_attention_and_takeover_alerts_preempt_alpr():
  flock = alert(custom.IQNavState.CameraType.alpr, distance=0.0)
  pre_attention = copy.copy(EVENTS[log.OnroadEvent.EventName.preDriverDistracted][ET.PERMANENT])
  prompt_attention = copy.copy(EVENTS[log.OnroadEvent.EventName.promptDriverDistracted][ET.PERMANENT])
  takeover = copy.copy(EVENTS[log.OnroadEvent.EventName.driverDistracted][ET.PERMANENT])
  immediate_disable = HardDisableCard("Regression Test")

  assert flock.priority == Priority.LOW
  assert pre_attention.priority == flock.priority + 1
  assert prompt_attention.priority == flock.priority + 1
  assert takeover.priority > flock.priority
  assert immediate_disable.priority > flock.priority

  for expected in (pre_attention, prompt_attention, takeover, immediate_disable):
    manager = AlertManager()
    flock.alert_type = "flock/warning"
    expected.alert_type = f"expected/{expected.alert_text_1}"
    manager.add_many(0, [flock, expected])
    manager.process_alerts(0, set())
    assert manager.current_alert is expected
