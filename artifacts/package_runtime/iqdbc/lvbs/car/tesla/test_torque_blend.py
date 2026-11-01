"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos
"""
from types import SimpleNamespace

import pytest

from iqdbc.car.tesla.interface import CarInterface
from iqdbc.car.vehicle_model import VehicleModel
from iqdbc.lvbs.car.tesla.torque_blend import (
  DT_LAT_CTRL,
  STEER_OVERRIDE_MAX_LAT_ACCEL,
  STEER_OVERRIDE_MIN_TORQUE,
  STEER_OVERRIDE_TORQUE_RANGE,
  STEER_OVERRIDE_TORQUE_ZERO_MAX,
  SteeringTorqueZero,
  TorqueBlendController,
  calc_override_angle_limited,
  get_steer_from_lat_accel,
)
from iqdbc.lvbs.car.tesla.values import TeslaFlagsIQ

VM = VehicleModel(CarInterface.get_non_essential_params("TESLA_MODEL_Y"))

COOP_ON = SimpleNamespace(flags=TeslaFlagsIQ.COOP_STEERING.value)
COOP_OFF = SimpleNamespace(flags=0)

HAND_REST_TORQUE = 0.55
ZERO_WEIGHT_SPEED = 20.0
FULL_WEIGHT_SPEED = 8.0
SETTLING_SECONDS = 300.0


def _car_state(torque, v_ego, angle=0.0):
  return SimpleNamespace(
    out=SimpleNamespace(steeringTorque=torque, vEgo=v_ego, vEgoRaw=v_ego, steeringAngleDeg=angle, steeringRateDeg=0.0),
    hands_on_level=0,
  )


def _drive(blend, torque, v_ego, seconds, angle=0.0, cp_iq=COOP_ON, lat_active=True):
  steps = int(seconds / DT_LAT_CTRL)
  out = 0.0
  for _ in range(steps):
    out = blend.update(angle, lat_active, cp_iq, _car_state(torque, v_ego, angle), VM).steeringAngleDeg
  return out


def _override(blend):
  return blend.coop_apply_angle_last - blend.debug_angle_desired_limited


MAX_NULLABLE_PRELOAD = STEER_OVERRIDE_MIN_TORQUE + STEER_OVERRIDE_TORQUE_ZERO_MAX


@pytest.mark.parametrize("rest", [0.3, HAND_REST_TORQUE, MAX_NULLABLE_PRELOAD])
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_torque_zero_pulls_a_sustained_hand_rest_inside_the_deadzone(rest, sign):
  zero = SteeringTorqueZero()
  for _ in range(int(SETTLING_SECONDS / DT_LAT_CTRL)):
    corrected = zero.update(sign * rest, True)
  assert abs(corrected) <= STEER_OVERRIDE_MIN_TORQUE


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_a_settled_hand_rest_at_the_nullable_limit_produces_no_override(sign):
  blend = TorqueBlendController()
  _drive(blend, sign * MAX_NULLABLE_PRELOAD, ZERO_WEIGHT_SPEED, seconds=SETTLING_SECONDS)
  assert abs(_override(blend)) == pytest.approx(0.0, abs=1e-6)


def test_override_unwinds_once_the_driver_releases():
  blend = TorqueBlendController()
  _drive(blend, 2.0, FULL_WEIGHT_SPEED, seconds=10.0)
  assert abs(_override(blend)) > 1.0
  _drive(blend, 0.0, FULL_WEIGHT_SPEED, seconds=20.0)
  assert _override(blend) == pytest.approx(0.0, abs=1e-6)
  assert blend.override_angle_accu == pytest.approx(0.0, abs=1e-6)


def test_torque_zero_passes_a_step_through_untouched():
  zero = SteeringTorqueZero()
  assert zero.update(2.0, True) == pytest.approx(2.0, abs=1e-3)


def test_torque_zero_is_bounded():
  zero = SteeringTorqueZero()
  for _ in range(int(SETTLING_SECONDS / DT_LAT_CTRL)):
    zero.update(10.0, True)
  assert zero.update(0.0, False) == pytest.approx(-STEER_OVERRIDE_TORQUE_ZERO_MAX)


def test_torque_zero_holds_while_not_learning():
  zero = SteeringTorqueZero()
  zero.update(1.0, False)
  assert zero.update(1.0, False) == pytest.approx(1.0)


def test_sustained_hand_rest_stops_steering_the_car():
  blend = TorqueBlendController()
  _drive(blend, HAND_REST_TORQUE, FULL_WEIGHT_SPEED, seconds=1.0)
  early = abs(_override(blend))
  _drive(blend, HAND_REST_TORQUE, FULL_WEIGHT_SPEED, seconds=SETTLING_SECONDS)
  assert early > 0.1
  assert abs(_override(blend)) < 0.1 * early


def test_a_deliberate_push_still_overrides_after_the_zero_has_settled():
  blend = TorqueBlendController()
  _drive(blend, HAND_REST_TORQUE, FULL_WEIGHT_SPEED, seconds=SETTLING_SECONDS)
  _drive(blend, HAND_REST_TORQUE + 1.5, FULL_WEIGHT_SPEED, seconds=1.0)
  assert abs(_override(blend)) > 1.0


def test_accumulator_cannot_charge_where_its_weight_is_zero():
  blend = TorqueBlendController()
  _drive(blend, 2.0, ZERO_WEIGHT_SPEED, seconds=30.0)
  assert blend.override_angle_accu == pytest.approx(0.0, abs=1e-6)


def test_accumulator_drains_once_the_car_passes_the_crossover_speed():
  blend = TorqueBlendController()
  _drive(blend, 2.5, FULL_WEIGHT_SPEED, seconds=10.0)
  assert abs(blend.override_angle_accu) > 1.0
  _drive(blend, 2.5, ZERO_WEIGHT_SPEED, seconds=5.0)
  assert blend.override_angle_accu == pytest.approx(0.0, abs=1e-6)


def test_low_speed_override_can_exceed_the_direct_term_authority():
  blend = TorqueBlendController()
  _drive(blend, 2.5, FULL_WEIGHT_SPEED, seconds=60.0)
  authority = calc_override_angle_limited(STEER_OVERRIDE_TORQUE_RANGE, FULL_WEIGHT_SPEED, VM, STEER_OVERRIDE_MAX_LAT_ACCEL)
  assert abs(blend.override_angle_accu) > authority


def test_a_sustained_low_speed_push_reaches_a_large_override():
  blend = TorqueBlendController()
  _drive(blend, 2.5, FULL_WEIGHT_SPEED, seconds=10.0)
  assert abs(_override(blend)) > 20.0


@pytest.mark.parametrize("v_ego", [3.0, 5.0, FULL_WEIGHT_SPEED, 12.0])
def test_override_stays_inside_the_max_lat_accel_envelope_plus_the_direct_term(v_ego):
  blend = TorqueBlendController()
  _drive(blend, 2.5, v_ego, seconds=60.0)
  envelope = get_steer_from_lat_accel(STEER_OVERRIDE_MAX_LAT_ACCEL, v_ego, VM)
  direct_authority = calc_override_angle_limited(STEER_OVERRIDE_TORQUE_RANGE, v_ego, VM, STEER_OVERRIDE_MAX_LAT_ACCEL)
  assert abs(_override(blend)) <= envelope + direct_authority


@pytest.mark.parametrize("v_ego", [3.0, 5.0, FULL_WEIGHT_SPEED, 12.0])
def test_a_pinned_push_cannot_run_the_contribution_away(v_ego):
  blend = TorqueBlendController()
  _drive(blend, 2.5, v_ego, seconds=130.0)
  envelope = get_steer_from_lat_accel(STEER_OVERRIDE_MAX_LAT_ACCEL, v_ego, VM)
  capability = calc_override_angle_limited(STEER_OVERRIDE_TORQUE_RANGE, v_ego, VM, STEER_OVERRIDE_MAX_LAT_ACCEL) / envelope
  assert abs(blend.override_angle_accu) * (1.0 - capability) <= envelope + 1e-6


def test_the_envelope_opens_up_as_speed_drops():
  low = get_steer_from_lat_accel(STEER_OVERRIDE_MAX_LAT_ACCEL, 3.0, VM)
  high = get_steer_from_lat_accel(STEER_OVERRIDE_MAX_LAT_ACCEL, 12.0, VM)
  assert low > 10 * high


def test_relative_weight_is_zero_above_the_direct_capability_crossover():
  capability = (calc_override_angle_limited(STEER_OVERRIDE_TORQUE_RANGE, ZERO_WEIGHT_SPEED, VM, STEER_OVERRIDE_MAX_LAT_ACCEL) /
                get_steer_from_lat_accel(STEER_OVERRIDE_MAX_LAT_ACCEL, ZERO_WEIGHT_SPEED, VM))
  assert capability == pytest.approx(1.0)


HEAVY_REST = 0.80


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_the_zero_nulls_a_hand_rest_heavier_than_the_deadzone(sign):
  assert HEAVY_REST > 2 * STEER_OVERRIDE_MIN_TORQUE
  blend = TorqueBlendController()
  _drive(blend, sign * HEAVY_REST, FULL_WEIGHT_SPEED, seconds=SETTLING_SECONDS)
  assert abs(_override(blend)) == pytest.approx(0.0, abs=1e-6)


LIGHT_PUSH = 0.45


@pytest.mark.parametrize("v_ego", [FULL_WEIGHT_SPEED, 17.0])
def test_a_light_push_overrides_without_needing_the_old_half_newton(v_ego):
  assert STEER_OVERRIDE_MIN_TORQUE < LIGHT_PUSH < 0.5
  blend = TorqueBlendController()
  _drive(blend, LIGHT_PUSH, v_ego, seconds=5.0)
  assert abs(_override(blend)) > 0.2


def test_torque_below_the_deadzone_never_overrides():
  blend = TorqueBlendController()
  _drive(blend, STEER_OVERRIDE_MIN_TORQUE * 0.5, FULL_WEIGHT_SPEED, seconds=5.0)
  assert abs(_override(blend)) == pytest.approx(0.0, abs=1e-6)


def test_coop_disabled_leaves_the_command_untouched():
  blend = TorqueBlendController()
  angle = 5.0
  out = _drive(blend, 2.0, FULL_WEIGHT_SPEED, seconds=5.0, angle=angle, cp_iq=COOP_OFF)
  assert out == pytest.approx(angle, abs=1e-6)
  assert blend.override_angle_accu == pytest.approx(0.0)
