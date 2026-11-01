# Copyright (c) 2026 IQ.Lvbs. All rights reserved.
import math

import numpy as np
import pytest

from iqpilot.cereal import car, log
from iqpilot.selfdrive.controls.lib.actuator_feedback import SteeringActuatorFeedback
from iqpilot.selfdrive.controls.lib.latcontrol_torque import friction_error_with_preview

def cycle(controller, vm, cp, requested, measured, active=True, *, speed=20.0, pressed=False):
  angle = math.degrees(vm.get_steer_from_curvature(-measured / speed ** 2, speed, 0.0))
  state = car.CarState.new_message(vEgo=speed, steeringAngleDeg=angle, steeringPressed=pressed)
  params = log.VehicleParameters.new_message(stiffnessFactor=1.0, steerRatio=cp.steerRatio)
  return controller.update(active, state, vm, params, True, requested / speed ** 2, None, False, 0.3)


@pytest.mark.parametrize("direction", [-1.0, 1.0])
@pytest.mark.parametrize("jerk", [-3.0, 0.0, 3.0])
@pytest.mark.parametrize("old,current,measured", [(0.8, 1.1, 1.0), (0.2, 1.0, 0.4), (0.0, 1.0, 0.4)])
def test_preview_cannot_reverse_outward_friction(controller_factory, direction, jerk, old, current, measured):
  controller, vm, cp = controller_factory()
  controller.lat_accel_request_buffer.extend([direction * old] * controller.lat_accel_request_buffer_len)
  controller.jerk_filter.x = jerk
  output, _, trace = cycle(controller, vm, cp, direction * current, direction * measured)
  assert direction * trace.error < 0.0
  friction = trace.f - direction * current + controller.torque_params.latAccelOffset
  assert direction * friction <= 1e-7
  assert np.isfinite(output)


@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_friction_is_continuous_when_measurement_crosses_setpoint(controller_factory, direction):
  values = []
  for epsilon in (-1e-5, 0.0, 1e-5):
    controller, vm, cp = controller_factory()
    controller.lat_accel_request_buffer.extend([direction] * controller.lat_accel_request_buffer_len)
    controller.jerk_filter.x = direction * 2.0
    _, _, trace = cycle(controller, vm, cp, direction, direction + epsilon)
    values.append(trace.f)
  assert max(values) - min(values) < 1e-4


@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_outward_preview_is_retained_before_feedback_turns_outward(direction):
  assert friction_error_with_preview(direction * 0.1, direction * 0.8, -direction * 2.0) == pytest.approx(-direction * 0.5)


@pytest.mark.parametrize("boundary", ["error", "measurement"])
@pytest.mark.parametrize("jerk", [-4.0, -0.1, 0.0, 0.1, 4.0])
def test_preview_has_no_sign_switch_step(boundary, jerk):
  for fixed in (-2.0, -0.01, 0.0, 0.01, 2.0):
    values = [friction_error_with_preview(epsilon, fixed, jerk) if boundary == "error"
              else friction_error_with_preview(fixed, epsilon, jerk) for epsilon in (-1e-8, 0.0, 1e-8)]
    assert max(values) - min(values) <= 2.01e-8


def test_preview_cannot_deepen_an_overtracked_turn():
  for error, measurement, jerk in np.random.default_rng(42).uniform(-5.0, 5.0, (10000, 3)):
    friction = friction_error_with_preview(error, measurement, jerk)
    if error * measurement < 0.0:
      assert friction * measurement <= 0.0
    assert abs(friction) <= max(abs(error), abs(measurement))


@pytest.mark.parametrize("slew", [False, True])
def test_measurement_does_not_create_requested_jerk(controller_factory, slew):
  controller, vm, cp = controller_factory()
  controller.lateral_acceleration_slew_limiter.enabled = slew
  controller.lateral_acceleration_slew_limiter.reset(1.0)
  controller.reference_slew_limiter.reset(1.0)
  for index in range(200):
    _, _, trace = cycle(controller, vm, cp, 1.0, 1.2 + 0.15 * math.sin(index * 0.4))
    assert trace.desiredLateralJerk == pytest.approx(0.0, abs=1e-10)


@pytest.mark.parametrize("direction", [-1.0, 1.0])
@pytest.mark.parametrize("slew", [False, True])
def test_request_ramp_retains_jerk(controller_factory, direction, slew):
  controller, vm, cp = controller_factory()
  controller.lateral_acceleration_slew_limiter.enabled = slew
  for index in range(300):
    _, _, trace = cycle(controller, vm, cp, direction * 0.3 * index * controller.dt, direction * 1.2)
  assert trace.desiredLateralJerk == pytest.approx(direction * 0.3, abs=2e-5)


@pytest.mark.parametrize("direction", [-1.0, 1.0])
@pytest.mark.parametrize("speed", [6.0, 20.0, 30.0])
@pytest.mark.parametrize("recovering", [False, True])
def test_real_clamp_allows_integral_recovery_only(controller_factory, direction, speed, recovering):
  controller, vm, cp = controller_factory()
  requested, measured = (1.0, 1.2) if recovering else (1.2, 1.0)
  controller.lat_accel_request_buffer.extend([direction * requested] * controller.lat_accel_request_buffer_len)
  controller.pid.i = direction * 2.0
  controller.update_actuator_feedback(SteeringActuatorFeedback(True, -direction * 0.2, -direction * 0.1))
  _, _, trace = cycle(controller, vm, cp, direction * requested, direction * measured, speed=speed)
  expected = direction * 2.0 + (controller.pid.k_i * controller.dt * trace.error if recovering else 0.0)
  assert trace.i == pytest.approx(expected, abs=1e-7)
  if recovering:
    assert abs(trace.i) < 2.0


@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_previous_clamp_cannot_authorize_integration_after_reversal(controller_factory, direction):
  controller, vm, cp = controller_factory()
  controller.lat_accel_request_buffer.extend([-direction] * controller.lat_accel_request_buffer_len)
  controller.pid.i = -direction * 2.0
  controller.update_actuator_feedback(SteeringActuatorFeedback(True, -direction * 0.2, -direction * 0.1))
  _, _, trace = cycle(controller, vm, cp, -direction, -direction * 0.8)
  assert trace.i == pytest.approx(-direction * 2.0)


@pytest.mark.parametrize("speed,pressed", [(4.0, False), (20.0, True)])
def test_driver_and_low_speed_still_freeze_recovery(controller_factory, speed, pressed):
  controller, vm, cp = controller_factory()
  controller.lat_accel_request_buffer.extend([1.0] * controller.lat_accel_request_buffer_len)
  controller.pid.i = 2.0
  controller.update_actuator_feedback(SteeringActuatorFeedback(True, -0.2, -0.1))
  _, _, trace = cycle(controller, vm, cp, 1.0, 1.2, speed=speed, pressed=pressed)
  assert trace.i == pytest.approx(2.0)


@pytest.mark.parametrize("feedback", [None, SteeringActuatorFeedback(True), SteeringActuatorFeedback(False, 0.0, -0.1)])
def test_unknown_limit_direction_remains_frozen(controller_factory, feedback):
  controller, vm, cp = controller_factory()
  controller.lat_accel_request_buffer.extend([1.0] * controller.lat_accel_request_buffer_len)
  controller.pid.i = 2.0
  if feedback is not None:
    controller.update_actuator_feedback(feedback)
  _, _, trace = cycle(controller, vm, cp, 1.0, 1.2)
  assert trace.i == pytest.approx(2.0)
  controller.reset()
  assert controller._actuator_feedback is None
