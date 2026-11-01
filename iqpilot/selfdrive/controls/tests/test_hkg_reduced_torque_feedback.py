# Copyright (c) 2026 IQ.Lvbs. All rights reserved.
import numpy as np
import pytest

from iqdbc.car.car_helpers import interfaces
from iqdbc.car.lateral import FRICTION_THRESHOLD, get_friction
from iqdbc.car.vehicle_model import VehicleModel
from iqpilot.cereal import car, log
from iqpilot.common.params import Params
from iqpilot.common.realtime import DT_CTRL
from iqpilot.common.constants import ACCELERATION_DUE_TO_GRAVITY
from iqpilot.selfdrive.car import interfaces as iqpilot_interfaces
from iqpilot.selfdrive.car.helpers import convert_to_capnp
from iqpilot.selfdrive.controls.lib import latcontrol_torque
from iqpilot.selfdrive.controls.lib.latcontrol_torque import JERK_GAIN, LatControlTorque, friction_error_with_preview


def make_controller(platform, enabled, nnff=False):
  params = Params()
  params.put_bool("IQHkgReducedTorqueFeedback", enabled)
  params.put_bool("NeuralNetworkFeedForward", nnff)
  interface_type = interfaces[platform]
  cp = interface_type.get_non_essential_params(platform)
  cp_iq = interface_type.get_non_essential_params_iq(cp, platform)
  ci = interface_type.__new__(interface_type)
  ci.CP, ci.CP_IQ = cp, cp_iq
  iqpilot_interfaces.apply_iq_car_config(ci)
  controller = LatControlTorque(cp.as_reader(), convert_to_capnp(cp_iq).as_reader(), ci, DT_CTRL)
  return controller, VehicleModel(cp), cp


def drive(controller, vm, cp, speed, active=True):
  state = car.CarState.new_message(vEgo=speed, steeringAngleDeg=0.4)
  params = log.VehicleParameters.new_message(stiffnessFactor=1.0, steerRatio=cp.steerRatio, roll=0.01)
  for _ in range(80):
    output, _, trace = controller.update(active, state, vm, params, True, 0.0002, None, False, 0.3)
  return output, trace


def expected_friction_error(controller, baseline):
  if controller.hkg_fix:
    return friction_error_with_preview(baseline.error, baseline.actualLateralAccel, baseline.desiredLateralJerk)
  return baseline.error + JERK_GAIN * baseline.desiredLateralJerk


def use_generic_torque_path(monkeypatch):
  for name in ("LAT_ACCEL_FACTOR_SPEED_TABLE", "LAT_TUNE_V6", "LAT_TUNE_V7", "LAT_TUNE_V8"):
    monkeypatch.setattr(latcontrol_torque, name, {})


def test_defaults_off():
  assert not Params().get_bool("IQHkgReducedTorqueFeedback")
  assert not Params().get("IQHkgReducedTorqueFeedback", return_default=True)


@pytest.mark.parametrize("platform", ["HYUNDAI_TUCSON_4TH_GEN", "KIA_K5_2021", "GENESIS_G80"])
@pytest.mark.parametrize("speed", [4.0, 7.5, 12.5, 21.0, 30.0])
def test_hkg_scales_feedback_and_friction_only(platform, speed, monkeypatch):
  use_generic_torque_path(monkeypatch)
  base, vm, cp = make_controller(platform, False)
  tuned, _, _ = make_controller(platform, True)
  base_output, baseline = drive(base, vm, cp, speed)
  output, trace = drive(tuned, vm, cp, speed)
  assert tuned.hkg_reduced_torque_feedback
  assert trace.error == baseline.error
  assert trace.p == pytest.approx(baseline.p * 0.8)
  assert trace.i == baseline.i
  assert trace.desiredLateralAccel == baseline.desiredLateralAccel
  assert trace.actualLateralAccel == baseline.actualLateralAccel
  assert trace.desiredLateralJerk == baseline.desiredLateralJerk
  for field in ["latAccelFactor", "latAccelOffset", "friction", "steeringAngleDeadzoneDeg"]:
    assert getattr(tuned.torque_params, field) == getattr(base.torque_params, field)
  friction = get_friction(expected_friction_error(base, baseline), 0.0, FRICTION_THRESHOLD, base.torque_params)
  assert trace.f == pytest.approx(baseline.f - 0.3 * friction, abs=1e-7)
  expected = -tuned.torque_from_lateral_accel(trace.p + trace.i + trace.f, tuned.torque_params)
  assert output == pytest.approx(np.clip(expected, -1, 1), abs=1e-7)
  assert output != base_output


@pytest.mark.parametrize("platform", ["TOYOTA_RAV4", "CHEVROLET_BOLT_EUV"])
def test_non_hkg_is_exactly_unchanged(platform):
  base, vm, cp = make_controller(platform, False)
  tuned, _, _ = make_controller(platform, True)
  assert not tuned.hkg_reduced_torque_feedback
  for speed in [4.0, 7.5, 12.5, 21.0, 30.0]:
    base_output, baseline = drive(base, vm, cp, speed)
    output, trace = drive(tuned, vm, cp, speed)
    assert output == base_output
    for field in ["p", "i", "d", "f", "error", "output", "desiredLateralAccel", "actualLateralAccel", "desiredLateralJerk", "active"]:
      assert getattr(trace, field) == getattr(baseline, field)


def test_nnff_selection_excludes_experimental_tune():
  controller, _, _ = make_controller("HYUNDAI_TUCSON_4TH_GEN", True, nnff=True)
  assert not controller.hkg_reduced_torque_feedback
  assert controller.friction_scale == 1.0


def test_live_torque_updates_preserve_scale_and_disengagement(monkeypatch):
  use_generic_torque_path(monkeypatch)
  base, vm, cp = make_controller("HYUNDAI_TUCSON_4TH_GEN", False)
  tuned, _, _ = make_controller("HYUNDAI_TUCSON_4TH_GEN", True)
  for controller in [base, tuned]:
    controller.update_live_torque_params(3.2, 0.05, 0.14)
  _, baseline = drive(base, vm, cp, 12.5)
  _, trace = drive(tuned, vm, cp, 12.5)
  friction = get_friction(expected_friction_error(base, baseline), 0.0, FRICTION_THRESHOLD, base.torque_params)
  assert trace.f == pytest.approx(baseline.f - 0.3 * friction, abs=1e-7)
  output, trace = drive(tuned, vm, cp, 12.5, active=False)
  assert output == 0.0
  assert not trace.active


@pytest.mark.parametrize("speed", [4.0, 7.5, 12.5, 21.0, 30.0])
def test_tucson_schedule_preserves_hkg_feedback_and_friction_scaling(speed):
  base, vm, cp = make_controller("HYUNDAI_TUCSON_4TH_GEN", False)
  tuned, _, _ = make_controller("HYUNDAI_TUCSON_4TH_GEN", True)
  base_output, baseline = drive(base, vm, cp, speed)
  output, trace = drive(tuned, vm, cp, speed)
  assert trace.error == baseline.error
  assert trace.p == pytest.approx(baseline.p * 0.8)
  assert trace.i == baseline.i
  assert trace.desiredLateralAccel == baseline.desiredLateralAccel
  assert trace.desiredLateralJerk == baseline.desiredLateralJerk
  core = 0.0002 * speed ** 2 - 0.01 * ACCELERATION_DUE_TO_GRAVITY - base.torque_params.latAccelOffset
  assert trace.f - core == pytest.approx(0.7 * (baseline.f - core), abs=1e-7)
  assert np.isfinite(base_output) and np.isfinite(output)
  assert -1.0 <= output <= 1.0
  output, trace = drive(tuned, vm, cp, speed, active=False)
  assert output == 0.0
  assert not trace.active
