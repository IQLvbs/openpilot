# Copyright (c) 2026 IQ.Lvbs. All rights reserved.
import math

import pytest

from iqpilot.cereal import car, log
from iqpilot.selfdrive.controls.lib import latcontrol_torque
from iqpilot.selfdrive.controls.lib.actuator_feedback import SteeringActuatorFeedback
from iqpilot.selfdrive.controls.lib.hkg_tuning import DEDICATED_TUNE_FINGERPRINTS

NON_HYUNDAI = ["TOYOTA_RAV4_TSS2", "CHEVROLET_BOLT_EUV", "TOYOTA_RAV4"]


def step(controller, vm, cp, requested, measured, speed=20.0, pressed=False, lat_delay=0.3):
  angle = math.degrees(vm.get_steer_from_curvature(-measured / speed ** 2, speed, 0.0))
  state = car.CarState.new_message(vEgo=speed, steeringAngleDeg=angle, steeringPressed=pressed)
  params = log.VehicleParameters.new_message(stiffnessFactor=1.0, steerRatio=cp.steerRatio)
  return controller.update(True, state, vm, params, True, requested / speed ** 2, None, False, lat_delay)


def test_gate_follows_brand_and_excludes_dedicated_tunes(controller_factory):
  assert controller_factory("HYUNDAI_KONA_EV")[0].hkg_fix
  assert controller_factory("KIA_K5_2021")[0].hkg_fix
  assert not controller_factory("HYUNDAI_TUCSON_4TH_GEN")[0].hkg_fix
  for platform in NON_HYUNDAI:
    assert not controller_factory(platform)[0].hkg_fix


@pytest.mark.parametrize("platform", NON_HYUNDAI)
def test_other_brands_keep_original_gain_curve(controller_factory, platform):
  controller, _, _ = controller_factory(platform)
  assert list(controller.pid._k_p[0]) == latcontrol_torque.INTERP_SPEEDS
  assert list(controller.pid._k_p[1]) == latcontrol_torque.KP_INTERP
  assert controller.pid.k_p > 100 or controller.pid.speed == 0.0


def test_hyundai_caps_low_speed_gain(controller_factory):
  controller, _, _ = controller_factory("HYUNDAI_KONA_EV")
  assert list(controller.pid._k_p[0]) == latcontrol_torque.HYUNDAI_INTERP_SPEEDS
  controller.pid.speed = 1.0
  assert controller.pid.k_p == pytest.approx(5.5)


@pytest.mark.parametrize("platform", NON_HYUNDAI)
def test_other_brands_keep_integrator_across_reset(controller_factory, platform):
  controller, _, _ = controller_factory(platform)
  controller.pid.i = 1.5
  controller.reset()
  assert controller.pid.i == 1.5


def test_hyundai_reset_clears_integrator(controller_factory):
  controller, _, _ = controller_factory("HYUNDAI_KONA_EV")
  controller.pid.i = 1.5
  controller.reset()
  assert controller.pid.i == 0.0


@pytest.mark.parametrize("platform", NON_HYUNDAI)
def test_other_brands_keep_delayed_setpoint(controller_factory, platform):
  controller, vm, cp = controller_factory(platform)
  controller.lat_accel_request_buffer.extend([2.0] * controller.lat_accel_request_buffer_len)
  _, _, trace = step(controller, vm, cp, 0.8, 1.2)
  assert trace.desiredLateralAccel > 1.5


def test_hyundai_projects_setpoint_onto_current_request(controller_factory):
  controller, vm, cp = controller_factory("HYUNDAI_KONA_EV")
  controller.lat_accel_request_buffer.extend([2.0] * controller.lat_accel_request_buffer_len)
  _, _, trace = step(controller, vm, cp, 0.8, 1.2)
  assert 0.0 <= trace.desiredLateralAccel <= 0.8 + 1e-6


@pytest.mark.parametrize("platform", NON_HYUNDAI)
@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_other_brands_never_recover_through_a_limit(controller_factory, platform, direction):
  controller, vm, cp = controller_factory(platform)
  controller.lat_accel_request_buffer.extend([direction * 1.0] * controller.lat_accel_request_buffer_len)
  controller.pid.i = direction * 2.0
  controller.update_actuator_feedback(SteeringActuatorFeedback(True, -direction * 0.2, -direction * 0.1))
  _, _, trace = step(controller, vm, cp, direction * 1.0, direction * 1.2)
  assert trace.i == pytest.approx(direction * 2.0)


@pytest.mark.parametrize("platform", NON_HYUNDAI)
def test_other_brands_never_enable_turn_guards(controller_factory, platform, monkeypatch):
  monkeypatch.setattr(latcontrol_torque, "get_turn_guards_enabled", lambda params: True)
  controller, _, _ = controller_factory(platform)
  assert not controller.turn_guards_enabled


def test_dedicated_tune_tables_are_all_excluded_from_the_gate():
  tuned = set(latcontrol_torque.LAT_ACCEL_FACTOR_SPEED_TABLE)
  for table in (latcontrol_torque.LAT_TUNE_V6, latcontrol_torque.LAT_TUNE_V7, latcontrol_torque.LAT_TUNE_V8):
    tuned |= set(table)
  assert tuned <= DEDICATED_TUNE_FINGERPRINTS


def test_dedicated_tune_keeps_its_own_standard_path(controller_factory):
  controller, vm, cp = controller_factory("HYUNDAI_TUCSON_4TH_GEN")
  assert not controller.turn_guards_enabled
  assert controller.v6 is not None
  controller.pid.i = 1.5
  controller.reset()
  assert controller.pid.i == 1.5
  controller.lat_accel_request_buffer.extend([2.0] * controller.lat_accel_request_buffer_len)
  _, _, trace = step(controller, vm, cp, 0.8, 1.2)
  assert trace.desiredLateralAccel > 1.5
