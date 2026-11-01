# Copyright (c) 2026 IQ.Lvbs. All rights reserved.
import math

import numpy as np
import pytest

from iqdbc.car.hyundai.values import HyundaiFlags
from iqpilot.cereal import car, log
from iqpilot.selfdrive.iqmodeld.config import ModelConstants

def run_frame(controller, vm, cp, speed=2.0, active=True, angle=0.0, curvature=0.0001):
  state = car.CarState.new_message(vEgo=speed, steeringAngleDeg=angle, steeringRateDeg=0.0)
  params = log.VehicleParameters.new_message(stiffnessFactor=1.0, steerRatio=cp.steerRatio)
  return controller.update(active, state, vm, params, True, curvature, None, False, 0.3)


@pytest.mark.parametrize("speed", [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.49, 7.5])
def test_standard_low_speed_feedback_is_bounded_without_output_clipping(controller_factory, speed):
  controller, vm, cp = controller_factory()
  output, _, trace = run_frame(controller, vm, cp, speed, angle=0.1)
  assert controller.pid.k_p == pytest.approx(5.5)
  assert trace.p == pytest.approx(5.5 * trace.error, abs=1e-7)
  assert np.isfinite(output)
  assert abs(output) < 1.0


@pytest.mark.parametrize("speed", [7.5, 8.0, 10.0, 12.0, 15.0, 20.0, 30.0, 40.0])
def test_standard_above_transition_retains_original_feedback(controller_factory, speed):
  controller, vm, cp = controller_factory()
  _, _, trace = run_frame(controller, vm, cp, speed, angle=0.1)
  expected_gain = float(np.interp(speed, [7.5, 10.0, 15.0, 30.0], [5.5, 3.5, 2.0, 0.8]))
  assert controller.pid.k_p == pytest.approx(expected_gain)
  assert trace.p == pytest.approx(expected_gain * trace.error, abs=1e-7)


@pytest.mark.parametrize("speed", [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0])
def test_nnff_uses_torque_feedback_gain_at_every_speed(controller_factory, speed):
  controller, vm, cp = controller_factory(nnff=True)
  for curvature in [0.0005, -0.0005]:
    output, _, trace = run_frame(controller, vm, cp, speed, curvature=curvature)
    assert controller.pid.k_p == pytest.approx(0.8)
    assert trace.p == pytest.approx(0.8 * trace.error, abs=1e-7)
    assert controller.pid.pos_limit == 1.0
    assert np.isfinite(output)
    assert abs(output) < 0.5


def test_nnff_gain_and_units_follow_model_availability(controller_factory):
  controller, vm, cp = controller_factory(nnff=True)
  helper = controller.nnff_assist
  for available in [True, False, True, False]:
    helper.model_valid = available
    controller.pid.i = 0.7
    run_frame(controller, vm, cp)
    assert controller.pid.k_p == pytest.approx(0.8 if available else 5.5)
    assert controller.pid.i == 0.0
    expected_limit = 1.0 if available else controller.lateral_accel_from_torque(1.0, controller.torque_params)
    assert controller.pid.pos_limit == pytest.approx(expected_limit)


def test_inference_failure_restores_standard_gain_in_same_frame(controller_factory, monkeypatch):
  controller, vm, cp = controller_factory(nnff=True)
  run_frame(controller, vm, cp)
  monkeypatch.setattr(controller.nnff_assist.model, "evaluate", lambda inputs: float("nan"))
  baseline, baseline_vm, baseline_cp = controller_factory()
  actual, _, trace = run_frame(controller, vm, cp)
  expected, _, baseline_trace = run_frame(baseline, baseline_vm, baseline_cp)
  assert not controller.nnff_assist.has_nn_model
  assert controller.pid.k_p == pytest.approx(5.5)
  assert actual == pytest.approx(expected)
  assert trace.p == pytest.approx(baseline_trace.p)


@pytest.mark.parametrize("dt", [0.005, 0.01, 0.02])
@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_classic_hyundai_jerk_uses_signed_angle_derivative(controller_factory, dt, direction):
  controller, vm, _ = controller_factory(dt=dt)
  helper = controller.nnff_assist
  state = car.CarState.new_message(vEgo=8.0, steeringAngleDeg=20.0, steeringRateDeg=100.0)
  assert helper._measured_jerk(state, vm) == 0.0
  slope = direction * 2.0
  for frame in range(1, int(2.0 / dt) + 1):
    state.steeringAngleDeg = 20.0 + slope * frame * dt
    measured = helper._measured_jerk(state, vm)
  expected = -vm.calc_curvature(math.radians(slope), state.vEgo, 0.0) * state.vEgo ** 2
  assert measured == pytest.approx(expected, rel=2e-5)


def test_quantized_duplicate_angles_do_not_erase_jerk_or_lose_direction(controller_factory):
  controller, vm, _ = controller_factory()
  helper = controller.nnff_assist
  state = car.CarState.new_message(vEgo=8.0, steeringRateDeg=4.0)
  measured = []
  for frame in range(250):
    state.steeringAngleDeg = -0.1 * (frame // 5)
    measured.append(helper._measured_jerk(state, vm))
  expected = -vm.calc_curvature(math.radians(-2.0), state.vEgo, 0.0) * state.vEgo ** 2
  assert min(measured[-50:]) > 0.0
  assert np.mean(measured[-50:]) == pytest.approx(expected, rel=0.01)
  for frame in range(1, 101):
    state.steeringAngleDeg = -4.9 + 0.02 * frame
    latest = helper._measured_jerk(state, vm)
  assert latest == pytest.approx(-expected, rel=2e-5)


@pytest.mark.parametrize("platform", ["TOYOTA_RAV4", "HYUNDAI_TUCSON_4TH_GEN"])
@pytest.mark.parametrize("rate", [-12.0, 12.0])
def test_other_vehicles_retain_native_signed_rate(controller_factory, platform, rate):
  controller, vm, cp = controller_factory(platform)
  if cp.brand == "hyundai":
    assert cp.flags & HyundaiFlags.CANFD
  state = car.CarState.new_message(vEgo=12.0, steeringAngleDeg=100.0, steeringRateDeg=rate)
  expected = -vm.calc_curvature(math.radians(rate), state.vEgo, 0.0) * state.vEgo ** 2
  assert controller.nnff_assist._measured_jerk(state, vm) == pytest.approx(expected)
  state.steeringAngleDeg = -100.0
  assert controller.nnff_assist._measured_jerk(state, vm) == pytest.approx(expected)


@pytest.mark.parametrize("nnff", [False, True])
def test_disengagement_clears_feedback_and_derivative_history(controller_factory, nnff):
  controller, vm, cp = controller_factory(nnff=nnff)
  run_frame(controller, vm, cp, angle=20.0)
  run_frame(controller, vm, cp, angle=20.2)
  controller.pid.i = 0.7
  output, _, trace = run_frame(controller, vm, cp, active=False, angle=80.0)
  assert output == 0.0
  assert not trace.active
  assert controller.pid.i == 0.0
  state = car.CarState.new_message(vEgo=8.0, steeringAngleDeg=-80.0, steeringRateDeg=100.0)
  assert controller.nnff_assist._measured_jerk(state, vm) == 0.0


def test_nonfinite_angle_clears_jerk_history_before_error(controller_factory):
  controller, vm, _ = controller_factory()
  helper = controller.nnff_assist
  state = car.CarState.new_message(vEgo=8.0, steeringAngleDeg=10.0)
  helper._measured_jerk(state, vm)
  state.steeringAngleDeg = float("nan")
  with pytest.raises(ValueError):
    helper._measured_jerk(state, vm)
  state.steeringAngleDeg = -30.0
  assert helper._measured_jerk(state, vm) == 0.0


def test_nnff_friction_override_is_in_torque_units(controller_factory):
  controller, vm, cp = controller_factory(nnff=True)
  helper = controller.nnff_assist
  helper.model.friction_override = True
  controller.update_live_torque_params(4.0, 0.0, 0.12)
  _, _, with_friction = run_frame(controller, vm, cp, speed=2.0, curvature=0.01)
  helper.model.friction_override = False
  _, _, without_friction = run_frame(controller, vm, cp, speed=2.0, curvature=0.01)
  assert with_friction.error - without_friction.error == pytest.approx(0.12, abs=1e-6)


def test_friction_weight_recovers_after_zero_jerk_cycle(controller_factory):
  controller, vm, _ = controller_factory(nnff=True)
  helper = controller.nnff_assist
  state = car.CarState.new_message(vEgo=10.0)
  helper.update_calculations(state, vm, 0.0)
  assert helper.lat_accel_friction_factor == 1.0
  helper.model_v2.acceleration.y = [float(t) for t in ModelConstants.T_IDXS]
  helper.update_calculations(state, vm, 0.0)
  assert helper.jerk_ahead > 0.0
  assert helper.lat_accel_friction_factor == 0.7
