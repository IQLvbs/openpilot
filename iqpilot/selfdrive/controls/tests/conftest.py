# Copyright (c) 2026 IQ.Lvbs. All rights reserved.
import pytest

from iqdbc.car.car_helpers import interfaces
from iqdbc.car.vehicle_model import VehicleModel
from iqpilot.cereal import log
from iqpilot.common.params import Params
from iqpilot.selfdrive.car import interfaces as iqpilot_interfaces
from iqpilot.selfdrive.car.helpers import convert_to_capnp
from iqpilot.selfdrive.controls.lib.latcontrol_torque import LatControlTorque
from iqpilot.selfdrive.controls.tests.lateral_control_scenarios import ScenarioNetwork
from iqpilot.selfdrive.iqmodeld.config import ModelConstants


@pytest.fixture
def controller_factory(monkeypatch, tmp_path):
  monkeypatch.setenv("PARAMS_ROOT", str(tmp_path / "params"))
  settings = Params()
  settings.put_bool("NeuralNetworkFeedForward", False)
  settings.put_bool("IQHkgReducedTorqueFeedback", False)
  settings.put_bool("IQLateralAccelSlew", False)
  settings.put("IQLatJerkGain", "1.0")

  def factory(platform="HYUNDAI_KONA_EV", nnff=False, dt=0.01):
    interface_type = interfaces[platform]
    cp = interface_type.get_non_essential_params(platform)
    cp_iq = interface_type.get_non_essential_params_iq(cp, platform)
    ci = interface_type.__new__(interface_type)
    ci.CP, ci.CP_IQ = cp, cp_iq
    iqpilot_interfaces.apply_iq_car_config(ci)
    controller = LatControlTorque(cp.as_reader(), convert_to_capnp(cp_iq).as_reader(), ci, dt)
    if nnff:
      helper = controller.nnff_assist
      helper.enabled = True
      helper.has_nn_model = True
      helper.model = ScenarioNetwork()
      count = len(ModelConstants.T_IDXS)
      packet = log.ModelDataV2.new_message()
      packet.orientation.x = [0.0] * count
      packet.orientation.y = [0.0] * count
      packet.acceleration.y = [0.0] * count
      helper.update_model_v2(packet)
    return controller, VehicleModel(cp), cp

  return factory
