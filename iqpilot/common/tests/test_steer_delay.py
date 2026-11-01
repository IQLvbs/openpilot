"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos
"""
import time

import pytest

import iqpilot.cereal.messaging as messaging
from iqpilot.cereal import car
from iqpilot.common.params import Params
from iqpilot.common.steer_delay import (
  SteerDelayPublisher,
  cached_steer_delay,
  fixed_steer_delay,
  lateral_action_delay,
  resolve_steer_delay,
)

ANGLE = car.CarParams.SteerControlType.angle
TORQUE = car.CarParams.SteerControlType.torque

LIVE_DELAY = 0.4387
RACK_DELAY = 0.10
OFFSET = 0.05


@pytest.fixture
def params(tmp_path, monkeypatch):
  monkeypatch.setenv("PARAMS_ROOT", str(tmp_path))
  p = Params()
  p.put("IQSteerDelayCache", LIVE_DELAY)
  p.put("IQSoftwareSteerDelay", OFFSET)
  return p


def _car_params(steer_control_type):
  cp = car.CarParams.new_message()
  cp.steerControlType = steer_control_type
  cp.steerActuatorDelay = RACK_DELAY
  return cp


def _lateral_delay_msg(value):
  msg = messaging.new_message("lateralDelay")
  msg.lateralDelay.lateralDelay = value
  return msg.as_reader()


def test_params_fixture_is_isolated_from_the_real_device(params, tmp_path):
  assert str(tmp_path) in params.get_param_path("")


@pytest.mark.parametrize("live_enabled", [True, False])
def test_torque_cars_always_use_live_delay(params, live_enabled):
  params.put_bool("IQLiveSteerDelay", live_enabled)
  assert lateral_action_delay(params, _car_params(TORQUE), LIVE_DELAY) == pytest.approx(LIVE_DELAY)


def test_angle_cars_ignore_live_delay_when_self_tuning_is_off(params):
  params.put_bool("IQLiveSteerDelay", False)
  delay = lateral_action_delay(params, _car_params(ANGLE), LIVE_DELAY)
  assert delay == pytest.approx(RACK_DELAY + OFFSET)
  assert delay != pytest.approx(LIVE_DELAY)


def test_angle_cars_use_cached_delay_when_self_tuning_is_on(params):
  params.put_bool("IQLiveSteerDelay", True)
  assert lateral_action_delay(params, _car_params(ANGLE), LIVE_DELAY) == pytest.approx(LIVE_DELAY)


@pytest.mark.parametrize("offset", [0.05, 0.20, 0.50])
def test_manual_offset_reaches_the_path_and_matches_what_the_ui_reports(params, offset):
  params.put_bool("IQLiveSteerDelay", False)
  params.put("IQSoftwareSteerDelay", offset)
  ui_total = RACK_DELAY + offset
  assert fixed_steer_delay(params, RACK_DELAY) == pytest.approx(ui_total)
  assert lateral_action_delay(params, _car_params(ANGLE), LIVE_DELAY) == pytest.approx(ui_total)


@pytest.mark.parametrize("live_enabled", [False, True])
def test_publisher_writes_the_value_the_resolver_reads(params, live_enabled):
  params.put_bool("IQLiveSteerDelay", live_enabled)
  params.put("IQSteerDelayCache", -1.0)
  SteerDelayPublisher(_car_params(ANGLE)).update(_lateral_delay_msg(LIVE_DELAY))

  expected = LIVE_DELAY if live_enabled else RACK_DELAY + OFFSET
  deadline = time.monotonic() + 5.0
  while cached_steer_delay() != pytest.approx(expected) and time.monotonic() < deadline:
    time.sleep(0.01)
  assert cached_steer_delay() == pytest.approx(expected)
  assert resolve_steer_delay(params, RACK_DELAY) == pytest.approx(expected)
