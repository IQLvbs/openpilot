# Copyright (c) 2026 IQ.Lvbs. All rights reserved.
from types import SimpleNamespace

import pytest

from iqpilot.common.params import Params
from iqpilot.cereal import car
from iqpilot.selfdrive.ui.mici.layouts.settings import vehicle


def test_feedback_selection_disables_nnff(monkeypatch, mocker, tmp_path):
  params = Params(str(tmp_path))
  params.put_bool("NeuralNetworkFeedForward", True)
  monkeypatch.setattr(vehicle, "ui_state", SimpleNamespace(params=params, is_offroad=lambda: True))
  layout = mocker.Mock()
  vehicle.VehicleLayoutMici._on_hkg_reduced_feedback(layout, True)
  assert not params.get_bool("NeuralNetworkFeedForward")
  assert params.get_bool("IQHkgReducedTorqueFeedback")
  assert params.get_bool("OnroadCycleRequested")
  layout._nnff.refresh.assert_called_once()


def test_feedback_deselection_preserves_nnff(monkeypatch, mocker, tmp_path):
  params = Params(str(tmp_path))
  params.put_bool("NeuralNetworkFeedForward", True)
  monkeypatch.setattr(vehicle, "ui_state", SimpleNamespace(params=params, is_offroad=lambda: True))
  vehicle.VehicleLayoutMici._on_hkg_reduced_feedback(mocker.Mock(), False)
  assert params.get_bool("NeuralNetworkFeedForward")
  assert not params.get_bool("IQHkgReducedTorqueFeedback")
  assert params.get_bool("OnroadCycleRequested")


def test_feedback_cannot_change_onroad(monkeypatch, mocker, tmp_path):
  params = Params(str(tmp_path))
  params.put_bool("NeuralNetworkFeedForward", True)
  monkeypatch.setattr(vehicle, "ui_state", SimpleNamespace(params=params, is_offroad=lambda: False))
  layout = mocker.Mock()
  vehicle.VehicleLayoutMici._on_hkg_reduced_feedback(layout, True)
  assert params.get_bool("NeuralNetworkFeedForward")
  assert not params.get_bool("IQHkgReducedTorqueFeedback")
  assert not params.get_bool("OnroadCycleRequested")
  layout._hkg_reduced_feedback.refresh.assert_called_once()


@pytest.mark.parametrize("brand,angle,tuning,offroad,visible", [
  ("hyundai", False, "torque", True, True),
  ("hyundai", False, "torque", False, True),
  ("hyundai", True, "torque", True, False),
  ("hyundai", False, "pid", True, False),
  ("toyota", False, "torque", True, False),
])
def test_visibility_and_offroad_gate(monkeypatch, mocker, brand, angle, tuning, offroad, visible):
  layout = mocker.Mock()
  control = layout._hkg_reduced_feedback
  layout._all_brand_widgets = [control]
  layout._brand_widgets = {"hyundai": [control]}
  layout._get_current_brand.return_value = brand
  cp = SimpleNamespace(steerControlType=car.CarParams.SteerControlType.angle if angle else car.CarParams.SteerControlType.torque,
                       lateralTuning=SimpleNamespace(which=lambda: tuning))
  monkeypatch.setattr(vehicle, "ui_state", SimpleNamespace(CP=cp, is_offroad=lambda: offroad))
  vehicle.VehicleLayoutMici._refresh(layout)
  control.set_visible.assert_called_once_with(visible)
  control.set_enabled.assert_called_once_with(offroad)
