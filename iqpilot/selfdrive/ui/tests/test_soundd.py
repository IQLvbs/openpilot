from types import SimpleNamespace

from iqpilot.cereal import car
from iqpilot.selfdrive.ui import soundd

AudibleAlert = car.CarControl.HUDControl.AudibleAlert


class TestSoundd:
  def test_check_selfdrive_timeout_alert(self, monkeypatch):
    class FakeSubMaster:
      recv_time = {'selfdriveState': 100.0}

      def __init__(self, enabled):
        self.state = SimpleNamespace(enabled=enabled)

      def __getitem__(self, service):
        assert service == 'selfdriveState'
        return self.state

    enabled = FakeSubMaster(True)
    disabled = FakeSubMaster(False)

    monkeypatch.setattr(soundd.time, "monotonic", lambda: 100.0 + soundd.SELFDRIVE_STATE_TIMEOUT)
    assert not soundd.check_selfdrive_timeout_alert(enabled)

    monkeypatch.setattr(soundd.time, "monotonic", lambda: 101.0 + soundd.SELFDRIVE_STATE_TIMEOUT)
    assert soundd.check_selfdrive_timeout_alert(enabled)
    assert not soundd.check_selfdrive_timeout_alert(disabled)

    monkeypatch.setattr(soundd.time, "monotonic", lambda: 110.0 + soundd.SELFDRIVE_STATE_TIMEOUT)
    assert not soundd.check_selfdrive_timeout_alert(enabled)

  # TODO: add test with micd for checking that soundd actually outputs sounds
