# Copyright (c) 2026 IQ.Lvbs. All rights reserved.
from types import SimpleNamespace

import pytest

from iqdbc.car.volkswagen.values import VolkswagenFlags
from iqpilot.cereal import car
from iqpilot.common.params import Params
from iqpilot.selfdrive.ui.lib import nnff_settings
from iqpilot.selfdrive.ui.mici.layouts.settings import vehicle


@pytest.fixture
def state(monkeypatch, tmp_path):
  cp = SimpleNamespace(brand="hyundai", flags=0, carFingerprint="HYUNDAI_KONA_EV", carFw=[],
                       steerControlType=car.CarParams.SteerControlType.torque,
                       lateralTuning=SimpleNamespace(which=lambda: "torque"))
  state = SimpleNamespace(CP=cp, params=Params(str(tmp_path / "params")), is_offroad=lambda: True)
  monkeypatch.setattr(vehicle, "ui_state", state)
  return state


def test_matched_model_available(state, tmp_path, monkeypatch):
  model = tmp_path / "HYUNDAI_KONA_EV.json"
  model.write_text("{}")
  monkeypatch.setattr(nnff_settings, "get_nn_model_path", lambda cp: (str(model), "HYUNDAI_KONA_EV", True))
  available, status = vehicle.VehicleLayoutMici._nnff_status(None)
  assert available
  assert "HYUNDAI KONA EV" in status
  assert "next drive" in status


@pytest.mark.parametrize("reason", ["no_vehicle", "mock_vehicle", "selected_vehicle", "angle", "pid", "pq", "mqb",
                                   "mock_model", "missing_model"])
def test_unavailable_models_and_controllers(state, tmp_path, monkeypatch, reason):
  model = tmp_path / "HYUNDAI_KONA_EV.json"
  model.write_text("{}")
  if reason == "no_vehicle":
    state.CP = None
  elif reason == "mock_vehicle":
    state.CP.carFingerprint = "MOCK"
  elif reason == "selected_vehicle":
    state.params.put("CarPlatformBundle", {"platform": "HYUNDAI_IONIQ_5"})
  elif reason == "angle":
    state.CP.steerControlType = car.CarParams.SteerControlType.angle
  elif reason == "pid":
    state.CP.lateralTuning.which = lambda: "pid"
  elif reason in ("pq", "mqb"):
    state.CP.brand = "volkswagen"
    state.CP.flags = VolkswagenFlags.PQ if reason == "pq" else 0
  elif reason == "mock_model":
    model = nnff_settings.MOCK_MODEL_PATH
  else:
    model.unlink()
  monkeypatch.setattr(nnff_settings, "get_nn_model_path", lambda cp: (str(model), "HYUNDAI_KONA_EV", True))
  available, status = vehicle.VehicleLayoutMici._nnff_status(None)
  assert not available
  assert status


@pytest.mark.parametrize("available,saved,offroad,enabled", [
  (True, False, True, True),
  (True, True, False, False),
  (False, False, True, False),
  (False, True, True, True),
  (False, True, False, False),
])
def test_availability_offroad_and_saved_on_gate(state, mocker, available, saved, offroad, enabled):
  state.params.put_bool("NeuralNetworkFeedForward", saved)
  state.is_offroad = lambda: offroad
  layout = mocker.Mock()
  layout._nnff_status.return_value = available, "model status"
  vehicle.VehicleLayoutMici._refresh_nnff(layout)
  assert layout._nnff_available is available
  layout._nnff.set_value.assert_called_once_with("model status")
  layout._nnff.set_enabled.assert_called_once_with(enabled)
  layout._nnff.refresh.assert_called_once()


@pytest.mark.parametrize("checked", [True, False])
def test_changes_saved_before_cycle_requested(state, mocker, checked):
  state.params.put_bool("NeuralNetworkFeedForward", not checked)
  state.params.put_bool("IQHkgReducedTorqueFeedback", True)
  put_bool = state.params.put_bool
  events = []

  def record_put(key, value):
    if key == "OnroadCycleRequested":
      assert state.params.get_bool("NeuralNetworkFeedForward") is checked
    events.append((key, value))
    put_bool(key, value)

  state.params = mocker.Mock(wraps=state.params)
  state.params.put_bool.side_effect = record_put
  layout = mocker.Mock()
  layout._nnff_status.return_value = True, "model"
  vehicle.VehicleLayoutMici._on_nnff(layout, checked)
  assert state.params.get_bool("NeuralNetworkFeedForward") is checked
  assert state.params.get_bool("OnroadCycleRequested")
  assert events[-1] == ("OnroadCycleRequested", True)
  assert state.params.get_bool("IQHkgReducedTorqueFeedback") is not checked
  assert layout._hkg_reduced_feedback.refresh.call_count == int(checked)
  layout._nnff.refresh.assert_not_called()


@pytest.mark.parametrize("offroad,available", [(False, True), (True, False)])
def test_cannot_enable_onroad_or_unavailable(state, mocker, offroad, available):
  state.is_offroad = lambda: offroad
  layout = mocker.Mock()
  layout._nnff_status.return_value = available, "unavailable"
  vehicle.VehicleLayoutMici._on_nnff(layout, True)
  assert not state.params.get_bool("NeuralNetworkFeedForward")
  assert not state.params.get_bool("OnroadCycleRequested")
  layout._nnff.refresh.assert_called_once()


def test_can_disable_unavailable_model(state, mocker):
  state.params.put_bool("NeuralNetworkFeedForward", True)
  layout = mocker.Mock()
  layout._nnff_status.return_value = False, "missing model"
  vehicle.VehicleLayoutMici._on_nnff(layout, False)
  assert not state.params.get_bool("NeuralNetworkFeedForward")
  assert state.params.get_bool("OnroadCycleRequested")
  layout._nnff_status.assert_not_called()


@pytest.mark.parametrize("saved,available,offroad,expected", [
  (False, True, True, True),
  (True, True, True, False),
  (True, False, True, False),
  (False, False, True, False),
  (True, True, False, True),
])
def test_real_param_control_saves_callback_result(state, monkeypatch, mocker, saved, available, offroad, expected):
  monkeypatch.setattr(vehicle.BigButton, "_handle_mouse_release", lambda self, pos: None)
  state.is_offroad = lambda: offroad
  state.params.put_bool("NeuralNetworkFeedForward", saved)
  control = vehicle.BigParamControl.__new__(vehicle.BigParamControl)
  control.params = state.params
  control.param = "NeuralNetworkFeedForward"
  control._checked = saved
  control.set_value = mocker.Mock()
  layout = mocker.Mock()
  layout._nnff = control
  layout._nnff_status.return_value = available, "model status"
  control._toggle_callback = lambda checked: vehicle.VehicleLayoutMici._on_nnff(layout, checked)
  control._handle_mouse_release(None)
  assert control._checked is expected
  assert state.params.get_bool("NeuralNetworkFeedForward") is expected
  assert state.params.get_bool("OnroadCycleRequested") is (saved != expected)


def test_frame_updates_do_not_scan_model_directory(state, monkeypatch, mocker):
  monkeypatch.setattr(vehicle.NavScroller, "_update_state", lambda self: None)
  layout = vehicle.VehicleLayoutMici.__new__(vehicle.VehicleLayoutMici)
  layout._vehicle_btn = mocker.Mock()
  layout._vehicle_status = mocker.Mock(return_value="Kona")
  layout._hkg_reduced_feedback = mocker.Mock()
  layout._nnff = mocker.Mock()
  layout._nnff_available = True
  layout._nnff_status = mocker.Mock()
  layout._update_state()
  layout._nnff_status.assert_not_called()
  layout._nnff.set_enabled.assert_called_once_with(True)
