import numpy as np

from iqpilot.selfdrive.controls.lib.lateral_acceleration_slew_limiter import (
  A_LAT_MAX,
  AVOIDANCE_BYPASS_ACCEL_DELTA,
  LateralAccelerationSlewLimiter,
)


def test_disabled_is_exact_passthrough_without_state_change():
  limiter = LateralAccelerationSlewLimiter(False)
  limiter.reset(1.25)
  rng = np.random.default_rng(0)
  for curvature in rng.standard_normal(100):
    assert limiter.update(curvature, 25.0, 0.01) is curvature
    assert limiter.a_lim == 1.25


def test_step_is_limited_by_speed_scheduled_jerk():
  limiter = LateralAccelerationSlewLimiter(True)
  limiter.reset(0.0)
  v_ego = 20.0
  dt = 0.01
  target = 1.5 / v_ego ** 2
  previous = limiter.a_lim
  for _ in range(100):
    limiter.update(target, v_ego, dt)
    assert abs(limiter.a_lim - previous) <= limiter.jerk_max(v_ego) * dt + 1e-12
    previous = limiter.a_lim


def test_converges_to_held_target():
  limiter = LateralAccelerationSlewLimiter(True)
  v_ego = 20.0
  target_accel = 1.0
  limiter.reset(0.0)
  for _ in range(100):
    limiter.update(target_accel / v_ego ** 2, v_ego, 0.01)
  assert limiter.a_lim == target_accel


def test_reset_prevents_reengagement_jump():
  limiter = LateralAccelerationSlewLimiter(True)
  v_ego = 20.0
  target_accel = 1.0
  limiter.reset(target_accel)
  curvature = limiter.update(target_accel / v_ego ** 2, v_ego, 0.01)
  assert curvature == target_accel / v_ego ** 2
  assert limiter.a_lim == target_accel


def test_low_speed_passes_through_and_resets():
  limiter = LateralAccelerationSlewLimiter(True)
  limiter.reset(-1.0)
  curvature = 0.2
  assert limiter.update(curvature, 4.0, 0.01) == curvature
  assert limiter.a_lim == A_LAT_MAX


def test_speed_schedule_changes_slew_rate():
  limiter = LateralAccelerationSlewLimiter(True)
  limiter.reset(0.0)
  limiter.update(1.0 / 8.0 ** 2, 8.0, 0.01)
  low_speed_step = limiter.a_lim
  limiter.reset(0.0)
  limiter.update(1.0 / 35.0 ** 2, 35.0, 0.01)
  high_speed_step = limiter.a_lim
  assert low_speed_step > high_speed_step


def test_sharp_avoidance_bypasses_limiter():
  limiter = LateralAccelerationSlewLimiter(True)
  v_ego = 20.0
  limiter.reset(0.0)
  target_accel = AVOIDANCE_BYPASS_ACCEL_DELTA + 0.1
  curvature = limiter.update(target_accel / v_ego ** 2, v_ego, 0.01)
  assert limiter.a_lim == target_accel
  assert curvature == target_accel / v_ego ** 2


def test_acceleration_space_couples_speed_and_curvature_changes():
  limiter = LateralAccelerationSlewLimiter(True)
  limiter.reset(0.5)
  limiter.update(0.005, 10.0, 0.01)
  previous = limiter.a_lim
  limiter.update(0.003, 20.0, 0.01)
  assert limiter.a_lim - previous <= limiter.jerk_max(20.0) * 0.01 + 1e-12
