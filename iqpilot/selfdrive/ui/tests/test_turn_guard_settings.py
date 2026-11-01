# Copyright (c) 2026 IQ.Lvbs. All rights reserved.
from types import SimpleNamespace

import pytest

from iqdbc.car.volkswagen.values import VolkswagenFlags
from iqpilot.cereal import car
from iqpilot.common.params import Params
from iqpilot.common.turn_guard_settings import get_turn_guards_enabled, set_turn_guards_enabled
from iqpilot.selfdrive.ui.lib.turn_guard_settings import TURN_GUARDS_DESCRIPTION, turn_guards_availability
from iqpilot.selfdrive.ui.mici.layouts.settings import steering
from iqpilot.system.ui.widgets import DialogResult
from iqpilot.ui.layouts.settings import iq_panels


@pytest.fixture
def state(tmp_path):
  cp = SimpleNamespace(brand="hyundai", flags=0, carFingerprint="HYUNDAI_KONA_EV", steerActuatorDelay=0.2,
                       steerControlType=car.CarParams.SteerControlType.torque,
                       lateralTuning=SimpleNamespace(which=lambda: "torque"))
  params = Params(str(tmp_path / "params"))
  params.put("IQSoftwareSteerDelay", 0.2)
  state = SimpleNamespace(CP=cp, params=params, offroad=True, measured_steer_delay=lambda: None)
  state.is_offroad = lambda: state.offroad
  return state


@pytest.fixture(params=["big", "mici"])
def panel(request, state, monkeypatch, mocker):
  module = iq_panels if request.param == "big" else steering
  monkeypatch.setattr(module, "ui_state", state)
  monkeypatch.setattr(module.gui_app, "font", lambda weight: module.rl.Font())
  monkeypatch.setattr(module.gui_app, "texture", lambda path, width=84, height=66, **kwargs:
                      SimpleNamespace(width=width, height=height))
  available = mocker.Mock(return_value=(True, ""))
  monkeypatch.setattr(module, "turn_guards_availability", available)
  if request.param == "big":
    layout = iq_panels.SteeringLayout()
    toggle = layout._turn_guards_toggle.action_item.toggle
    get_state = toggle.get_state
    def get_enabled():
      return layout._turn_guards_toggle.action_item.enabled
  else:
    layout = steering.SteeringLayoutMici()
    toggle = layout._turn_guards
    def get_state():
      return toggle._checked

    def get_enabled():
      return toggle.enabled
  return SimpleNamespace(kind=request.param, module=module, layout=layout, toggle=toggle, state=state,
                          available=available, get_state=get_state, get_enabled=get_enabled)


def test_turn_guards_default_off_and_appear_in_steering_panel(panel):
  assert not get_turn_guards_enabled(panel.state.params)
  assert not panel.get_state()
  if panel.kind == "big":
    row = panel.layout._turn_guards_toggle
    assert row in panel.layout._scroller._items
    assert row.title == "Turn Guards (Experimental)"
    assert TURN_GUARDS_DESCRIPTION in row.description
    assert panel.toggle.param_key is None
  else:
    assert panel.toggle.text == "turn guards"
    assert "Experimental" in panel.toggle.value
    assert TURN_GUARDS_DESCRIPTION in panel.toggle.value
    assert not isinstance(panel.toggle, steering.BigParamControl)


@pytest.mark.parametrize("saved,available,offroad,expected", [
  (False, True, True, True),
  (True, True, True, False),
  (True, False, True, False),
  (False, False, True, False),
  (False, True, False, False),
  (True, True, False, True),
  (True, False, False, True),
])
def test_real_toggle_rechecks_support_and_offroad_before_save(panel, mocker, saved, available, offroad, expected):
  state = panel.state
  set_turn_guards_enabled(state.params, saved)
  panel.layout._refresh_turn_guards()
  state.offroad = offroad
  panel.available.return_value = available, "Unavailable for this steering system."
  original_setter = panel.module.set_turn_guards_enabled
  original_put_bool = state.params.put_bool
  writes = []

  def record_setting(params, enabled):
    writes.append(("setting", enabled))
    original_setter(params, enabled)

  def record_cycle(key, value):
    assert key == "OnroadCycleRequested"
    assert get_turn_guards_enabled(state.params) is expected
    writes.append((key, value))
    original_put_bool(key, value)

  mocker.patch.object(panel.module, "set_turn_guards_enabled", side_effect=record_setting)
  state.params = mocker.Mock(wraps=state.params)
  state.params.put_bool.side_effect = record_cycle
  panel.toggle._handle_mouse_release(None)
  changed = saved != expected
  assert panel.get_state() is expected
  assert get_turn_guards_enabled(state.params) is expected
  assert state.params.get_bool("OnroadCycleRequested") is changed
  assert writes == ([("setting", expected), ("OnroadCycleRequested", True)] if changed else [])


@pytest.mark.parametrize("available,saved,offroad,enabled", [
  (True, False, True, True),
  (True, True, False, False),
  (False, False, True, False),
  (False, True, True, True),
  (False, True, False, False),
])
def test_frame_syncs_availability_and_allows_offroad_disable(panel, available, saved, offroad, enabled):
  set_turn_guards_enabled(panel.state.params, saved)
  panel.state.offroad = offroad
  panel.available.return_value = available, "Unavailable for this steering system."
  panel.layout._update_state()
  assert panel.get_state() is saved
  assert panel.get_enabled() is enabled
  description = panel.layout._turn_guards_toggle.description if panel.kind == "big" else panel.toggle.value
  assert ("Unavailable" in description) is not available
  assert ("Park the vehicle" in description) is not offroad


@pytest.mark.parametrize("saved", [False, True])
def test_failed_save_restores_toggle_without_cycle_request(panel, mocker, saved):
  set_turn_guards_enabled(panel.state.params, saved)
  panel.layout._refresh_turn_guards()
  mocker.patch.object(panel.module, "set_turn_guards_enabled", side_effect=OSError("disk full"))
  logged = mocker.patch.object(panel.module.cloudlog, "exception")
  panel.toggle._handle_mouse_release(None)
  assert panel.get_state() is saved
  assert get_turn_guards_enabled(panel.state.params) is saved
  assert not panel.state.params.get_bool("OnroadCycleRequested")
  logged.assert_called_once()


@pytest.mark.parametrize("reason", ["no_vehicle", "mock_vehicle", "selected_vehicle", "angle", "curvature", "pid", "pq", "mqb"])
def test_availability_excludes_unsupported_steering(state, reason):
  if reason == "no_vehicle":
    state.CP = None
  elif reason == "mock_vehicle":
    state.CP.carFingerprint = "MOCK"
  elif reason == "selected_vehicle":
    state.params.put("CarPlatformBundle", {"platform": "OTHER"})
  elif reason == "angle":
    state.CP.steerControlType = car.CarParams.SteerControlType.angle
  elif reason == "curvature":
    state.CP.steerControlType = car.CarParams.SteerControlType.curvatureDEPRECATED
  elif reason == "pid":
    state.CP.lateralTuning.which = lambda: "pid"
  elif reason in ("pq", "mqb"):
    state.CP.brand = "volkswagen"
    state.CP.flags = VolkswagenFlags.PQ if reason == "pq" else 0
  available, description = turn_guards_availability(state.CP, state.params)
  assert not available
  assert description


def test_hyundai_torque_is_available_without_nnff_model(state):
  state.CP.carFingerprint = "NO_NEURAL_MODEL"
  assert turn_guards_availability(state.CP, state.params) == (True, "")


@pytest.mark.parametrize("brand,flags", [("toyota", 0), ("gm", 0), ("honda", 0), ("volkswagen", VolkswagenFlags.MLB)])
def test_other_brands_are_unavailable(state, brand, flags):
  state.CP.brand, state.CP.flags = brand, flags
  assert turn_guards_availability(state.CP, state.params) == (False, "Available on Hyundai, Kia and Genesis only.")


@pytest.mark.parametrize("save_failure", [False, True])
def test_reset_all_settings_resets_turn_guards_before_reboot(state, monkeypatch, mocker, save_failure):
  set_turn_guards_enabled(state.params, True)
  state.params.put_bool("NeuralNetworkFeedForward", True)
  state.params = mocker.Mock(wraps=state.params)
  state.params.all_keys.return_value = ["NeuralNetworkFeedForward"]
  monkeypatch.setattr(iq_panels, "ui_state", state)
  callbacks = []
  monkeypatch.setattr(iq_panels.gui_app, "set_modal_overlay", lambda dialog, callback: callbacks.append(callback))
  monkeypatch.setattr(iq_panels, "ConfirmDialog", lambda **kwargs: kwargs)
  reboot = mocker.patch.object(iq_panels.HARDWARE, "reboot")
  if save_failure:
    mocker.patch.object(iq_panels, "reset_turn_guards", side_effect=OSError("read only"))
    mocker.patch.object(iq_panels.cloudlog, "exception")
  iq_panels.IQDeviceLayout._reset_settings()
  callbacks[0](DialogResult.CONFIRM)
  callbacks[1](DialogResult.CONFIRM)
  assert get_turn_guards_enabled(state.params) is save_failure
  assert state.params.get_bool("NeuralNetworkFeedForward") is save_failure
  assert reboot.call_count == int(not save_failure)
