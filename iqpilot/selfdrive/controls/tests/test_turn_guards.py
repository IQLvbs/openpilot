# Copyright (c) 2026 IQ.Lvbs. All rights reserved.
import math
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

from iqpilot.selfdrive.controls.lib import latcontrol_torque
from iqpilot.selfdrive.controls.lib.lateral_acceleration_slew_limiter import LateralAccelerationSlewLimiter
from iqpilot.selfdrive.controls.lib.turn_guards import apply_turn_guards
from iqpilot.selfdrive.controls.tests.capnp_values import plain
from iqpilot.selfdrive.controls.tests.lateral_control_scenarios import build_controls, load_controls_module, run_scenario, set_frame


@pytest.mark.parametrize("desired,actual,delayed,expected", [
  (3.0, 4.0, 3.0, 2.0),
  (-3.0, -4.0, -3.0, -2.0),
  (4.0, 5.0, 4.0, 3.0),
  (-3.0, 4.0, -3.0, -3.0),
  (3.0, -4.0, 3.0, 3.0),
  (0.2, 3.0, 0.2, 0.0),
  (-0.2, 3.0, -0.2, -0.2),
  (0.0, 3.0, 0.0, 0.0),
  (3.0, 3.02, 3.0, 2.98),
  (3.0, 2.98, 3.0, 3.0),
  (3.0, 3.0, 3.0, 3.0),
  (3.0, 6.5, 3.0, 0.0),
  (1.0, 3.0, 1.0, 0.0),
  (0.000001, 3.0, 0.000001, 0.0),
  (-0.000001, 3.0, -0.000001, -0.000001),
  (1.0, 3.0, 3.0, 1.0),
  (1.0, 3.5, 3.0, 0.5),
  (3.0, 1.0, 1.0, 3.0),
  (3.0, 4.0, -3.0, 3.0),
  (3.0, 4.0, 0.0, 3.0),
])
def test_retained_gains_and_corrected_acceleration_cases(desired, actual, delayed, expected):
  assert apply_turn_guards(desired, actual, delayed) == pytest.approx(expected, abs=1e-12)
  assert apply_turn_guards(-desired, -actual, -delayed) == pytest.approx(-expected, abs=1e-12)


def test_guard_output_cannot_amplify_or_reverse_requested_acceleration():
  cases = np.random.default_rng(42).uniform(-20.0, 20.0, (10000, 3))
  for desired, actual, delayed in cases:
    result = apply_turn_guards(desired, actual, delayed)
    assert abs(result) <= abs(desired)
    assert result * desired >= 0.0


@pytest.mark.parametrize("epsilon", [1e-3, 1e-6, 1e-9, 1e-12])
def test_correction_is_continuous_at_zero_desired_acceleration(epsilon):
  for actual in (-3.0, 3.0):
    for delayed in (-3.0, 0.0, 3.0):
      assert abs(apply_turn_guards(epsilon, actual, delayed)) <= epsilon
      assert abs(apply_turn_guards(-epsilon, actual, delayed)) <= epsilon
      assert apply_turn_guards(0.0, actual, delayed) == 0.0


@pytest.fixture
def controls_module(monkeypatch):
  return load_controls_module(monkeypatch)


def make_controls(module, monkeypatch, enabled, nnff=False, slew=False, case="hkg"):
  with monkeypatch.context() as context:
    context.setattr(latcontrol_torque, "get_turn_guards_enabled", lambda params: enabled)
    return build_controls(module, case, nnff=nnff, slew=slew)


def cycle(controls, desired_curvature, actual_curvature, speed=20.0, active=True, lat_delay=0.1):
  set_frame(controls, 120)
  state = controls.sm["carState"]
  state.vEgo = speed
  state.steeringAngleDeg = 1.0
  state.steeringRateDeg = 0.0
  state.steeringPressed = False
  params = controls.sm["vehicleParameters"]
  params.roll = 0.0
  params.angleOffsetDeg = 0.0
  vm = SimpleNamespace(calc_curvature=lambda angle, *args: -actual_curvature if angle == math.radians(1.0)
                       else -actual_curvature * angle / math.radians(1.0))
  model = controls.sm["modelV2"]
  if controls.LaC.nnff_assist.has_nn_model:
    model.acceleration.y = [desired_curvature * speed ** 2] * len(model.acceleration.y)
  controls.LaC.nnff_assist.update_model_v2(model)
  controls.LaC.nnff_assist.update_lateral_lag(lat_delay)
  return controls.LaC.update(active, state, vm, params, True, desired_curvature, None, False, lat_delay)


@pytest.mark.parametrize("nnff", [False, True])
@pytest.mark.parametrize("slew", [False, True])
@pytest.mark.parametrize("case", ["torque", "tss2", "gm", "hkg"])
def test_disabled_guards_do_not_enter_shaping_path(controls_module, monkeypatch, nnff, slew, case):
  controls = make_controls(controls_module, monkeypatch, False, nnff=nnff, slew=slew, case=case)
  expected = run_scenario(controls)

  def forbidden(*args):
    pytest.fail("Disabled guards must not evaluate or modify the baseline request")

  monkeypatch.setattr(latcontrol_torque, "apply_turn_guards", forbidden)
  unchanged = make_controls(controls_module, monkeypatch, False, nnff=nnff, slew=slew, case=case)
  np.testing.assert_array_equal(run_scenario(unchanged), expected)


@pytest.mark.parametrize("initial", [False, True])
def test_setting_is_captured_at_controller_construction(controls_module, monkeypatch, initial):
  setting = {"enabled": initial}
  reads = []

  def read_setting(params):
    reads.append(setting["enabled"])
    return setting["enabled"]

  monkeypatch.setattr(latcontrol_torque, "get_turn_guards_enabled", read_setting)
  controls = build_controls(controls_module, "hkg")
  setting["enabled"] = not initial
  for _ in range(11):
    cycle(controls, 3.0 / 400, 4.0 / 400)
  assert controls.LaC.turn_guards_enabled == initial
  assert controls.LaC.lat_accel_request_buffer[-1] == pytest.approx(2.0 if initial else 3.0)
  assert reads == [initial]
  replacement = build_controls(controls_module, "hkg")
  assert replacement.LaC.turn_guards_enabled == (not initial)
  assert reads == [initial, not initial]


@pytest.mark.parametrize("direction", [-1.0, 1.0])
@pytest.mark.parametrize("nnff", [False, True])
def test_guards_adjust_shared_request_once_and_preserve_limits(controls_module, monkeypatch, direction, nnff):
  controls = make_controls(controls_module, monkeypatch, True, nnff=nnff)
  calls = []
  pid_calls = []
  original_pid_update = controls.LaC.pid.update

  def record_guard(*args):
    calls.append(args)
    return apply_turn_guards(*args)

  def record_pid(*args, **kwargs):
    pid_calls.append((args, kwargs))
    return original_pid_update(*args, **kwargs)

  monkeypatch.setattr(latcontrol_torque, "apply_turn_guards", record_guard)
  monkeypatch.setattr(controls.LaC.pid, "update", record_pid)
  for index in range(30):
    output, _, trace = cycle(controls, direction * 3.0 / 400, direction * 4.0 / 400)
    assert np.isfinite(output)
    assert abs(output) <= controls.LaC.steer_max
    assert len(pid_calls) == index + 1
    assert len(calls) == max(0, index - 9)
    if calls:
      assert calls[-1] == pytest.approx((direction * 3.0, direction * 4.0, direction * 3.0))
    assert controls.LaC.lat_accel_request_buffer[-1] == pytest.approx(direction * (2.0 if index >= 10 else 3.0))
    assert controls.LaC.turn_guard_reference_buffer[-1] == pytest.approx(direction * 3.0)
  assert trace.desiredLateralAccel == pytest.approx(direction * 2.0)
  assert trace.actualLateralAccel == pytest.approx(direction * 4.0)
  if nnff:
    assert controls.LaC.nnff_assist.model.queries > 0
    assert controls.LaC.pid.pos_limit == controls.LaC.steer_max
  else:
    assert controls.LaC.pid.pos_limit == controls.LaC.lateral_accel_from_torque(controls.LaC.steer_max, controls.LaC.torque_params)


@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_guarded_request_passes_through_existing_slew_limiter(controls_module, monkeypatch, direction):
  controls = make_controls(controls_module, monkeypatch, True, slew=True)
  limiter = controls.LaC.lateral_acceleration_slew_limiter
  limiter.reset(direction * 3.0)
  controls.LaC.reference_slew_limiter.reset(direction * 3.0)
  controls.LaC.turn_guard_reference_buffer.extend([direction * 3.0] * 10)
  cycle(controls, direction * 3.0 / 400, direction * 4.0 / 400)
  expected = direction * (3.0 - limiter.jerk_max(20.0) * controls.LaC.dt)
  assert controls.LaC.lat_accel_request_buffer[-1] == pytest.approx(expected)
  assert limiter.a_lim == pytest.approx(expected)


@pytest.mark.parametrize("nnff", [False, True])
@pytest.mark.parametrize("slew", [False, True])
def test_inactive_and_zero_speed_do_not_evaluate_guards(controls_module, monkeypatch, nnff, slew):
  controls = make_controls(controls_module, monkeypatch, True, nnff=nnff, slew=slew)
  baseline = make_controls(controls_module, monkeypatch, False, nnff=nnff, slew=slew)

  def forbidden(*args):
    pytest.fail("No guard evaluation while inactive or at zero speed")

  monkeypatch.setattr(latcontrol_torque, "apply_turn_guards", forbidden)
  for active, speed in ((False, 20.0), (True, 0.0), (False, 0.0)):
    output, _, trace = cycle(controls, 3.0 / 400, 4.0 / 400, speed=speed, active=active)
    expected, _, expected_trace = cycle(baseline, 3.0 / 400, 4.0 / 400, speed=speed, active=active)
    assert output == expected
    assert np.isfinite(output)
    assert plain(trace) == plain(expected_trace)
    assert list(controls.LaC.lat_accel_request_buffer) == list(baseline.LaC.lat_accel_request_buffer)
    if not active:
      assert output == 0.0
      assert not trace.active


@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_nnff_queries_use_guarded_acceleration_and_consistent_curvature(controls_module, monkeypatch, direction):
  controls = make_controls(controls_module, monkeypatch, True, nnff=True)
  nnff = controls.LaC.nnff_assist
  queries = []
  original_evaluate = nnff.model.evaluate

  def record_query(inputs):
    queries.append(list(inputs))
    return original_evaluate(inputs)

  monkeypatch.setattr(nnff.model, "evaluate", record_query)
  for _ in range(11):
    queries.clear()
    cycle(controls, direction * 3.0 / 400, direction * 4.0 / 400)
  assert nnff._want_la == pytest.approx(direction * 2.0)
  assert nnff._want_cv == pytest.approx(direction * 2.0 / 400)
  assert nnff._have_la == pytest.approx(direction * 4.0)
  assert nnff._window.lat_accel_desired[-1] == pytest.approx(direction * 2.0)
  assert queries[0][1] == pytest.approx(direction * (2.0 + 2.0 / 400))
  assert queries[-1][1] == pytest.approx(direction * 2.0)
  assert queries[-1][7:11] == pytest.approx([direction * 3.0] * 4)


@pytest.mark.parametrize("direction", [-1.0, 1.0])
@pytest.mark.parametrize("nnff", [False, True])
@pytest.mark.parametrize("slew", [False, True])
def test_perfect_delayed_tracking_and_normal_turn_exit_are_unchanged(controls_module, monkeypatch, direction, nnff, slew):
  controls = make_controls(controls_module, monkeypatch, True, nnff=nnff, slew=slew)
  baseline = make_controls(controls_module, monkeypatch, False, nnff=nnff, slew=slew)
  reference = LateralAccelerationSlewLimiter(slew)
  reference.reset(direction * 2.0)
  reference_history = deque([direction * 2.0 / 400] * 31, maxlen=31)
  for controller in (controls, baseline):
    cycle(controller, direction * 2.0 / 400, direction * 2.0 / 400, active=False, lat_delay=0.3)

  for frame in range(100):
    desired = direction * (2.0 if frame < 35 else max(0.5, 2.0 - 0.05 * (frame - 34)))
    reference_history.append(reference.update(desired / 400, 20.0, controls.LaC.dt))
    measured_curvature = reference_history[0]
    actual, _, trace = cycle(controls, desired / 400, measured_curvature, lat_delay=0.3)
    expected, _, expected_trace = cycle(baseline, desired / 400, measured_curvature, lat_delay=0.3)
    assert actual == expected
    assert plain(trace) == plain(expected_trace)
    assert controls.LaC.lat_accel_request_buffer[-1] == baseline.LaC.lat_accel_request_buffer[-1]
    assert controls.LaC.turn_guard_reference_buffer[-1] == baseline.LaC.lat_accel_request_buffer[-1]


@pytest.mark.parametrize("nnff", [False, True])
def test_disengagement_clears_reference_and_requires_fresh_history(controls_module, monkeypatch, nnff):
  controls = make_controls(controls_module, monkeypatch, True, nnff=nnff)
  for _ in range(15):
    cycle(controls, 3.0 / 400, 4.0 / 400)
  assert controls.LaC.lat_accel_request_buffer[-1] == pytest.approx(2.0)
  cycle(controls, 3.0 / 400, 4.0 / 400, active=False)
  assert not controls.LaC.turn_guard_reference_buffer
  for _ in range(10):
    cycle(controls, 3.0 / 400, 4.0 / 400)
    assert controls.LaC.lat_accel_request_buffer[-1] == pytest.approx(3.0)
  cycle(controls, 3.0 / 400, 4.0 / 400)
  assert controls.LaC.lat_accel_request_buffer[-1] == pytest.approx(2.0)


def test_reference_slew_enable_and_reset_follow_existing_limiter(controls_module, monkeypatch):
  controls = make_controls(controls_module, monkeypatch, True, slew=False)
  actual_limiter = controls.LaC.lateral_acceleration_slew_limiter
  reference_limiter = controls.LaC.reference_slew_limiter
  for enabled in (True, False, True):
    actual_limiter.enabled = enabled
    cycle(controls, 2.0 / 400, 2.0 / 400, active=False)
    assert reference_limiter.enabled == enabled
    assert reference_limiter.a_lim == actual_limiter.a_lim
    cycle(controls, 1.0 / 400, 1.0 / 400)
    assert reference_limiter.a_lim == actual_limiter.a_lim
    assert controls.LaC.turn_guard_reference_buffer[-1] == controls.LaC.lat_accel_request_buffer[-1]


@pytest.mark.parametrize("case", ["torque", "tss2", "gm"])
@pytest.mark.parametrize("nnff", [False, True])
def test_enabled_setting_never_shapes_non_hyundai(controls_module, monkeypatch, case, nnff):
  expected = run_scenario(make_controls(controls_module, monkeypatch, False, nnff=nnff, case=case))

  def forbidden(*args):
    pytest.fail("Guards are Hyundai-only")

  monkeypatch.setattr(latcontrol_torque, "apply_turn_guards", forbidden)
  controls = make_controls(controls_module, monkeypatch, True, nnff=nnff, case=case)
  assert not controls.LaC.turn_guards_enabled
  np.testing.assert_array_equal(run_scenario(controls), expected)
