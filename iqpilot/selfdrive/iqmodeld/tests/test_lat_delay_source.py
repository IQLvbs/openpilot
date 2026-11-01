"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos
"""
from types import SimpleNamespace

import pytest

from iqpilot.cereal import car
from iqpilot.common.params import Params
from iqpilot.selfdrive.iqmodeld.daemon import InferenceDaemon

LIVE_DELAY = 0.4387
RACK_DELAY = 0.10
OFFSET = 0.05


@pytest.fixture
def params(tmp_path, monkeypatch):
  monkeypatch.setenv("PARAMS_ROOT", str(tmp_path))
  p = Params()
  p.put("IQSteerDelayCache", LIVE_DELAY)
  p.put("IQSoftwareSteerDelay", OFFSET)
  p.put_bool("ModelSmoothingEnabled", False)
  p.put("ModelLatSmoothSec", 0)
  p.put("PlanplusControl", 1.0)
  p.put("CameraOffset", 0.0)
  return p


def _daemon(params, steer_control_type):
  car_params = car.CarParams.new_message()
  car_params.steerControlType = steer_control_type
  car_params.steerActuatorDelay = RACK_DELAY
  return SimpleNamespace(
    _params=params,
    _car_params=car_params,
    _channel=None,
    _sub={"lateralDelay": SimpleNamespace(lateralDelay=LIVE_DELAY)},
    _runtime=SimpleNamespace(lat_delay=None, PLANPLUS_CONTROL=None, model_smoothing_max_extra_sec=None),
    _warps=SimpleNamespace(set_offset=lambda _: None),
  )


@pytest.mark.parametrize("live_enabled, expected", [(False, RACK_DELAY + OFFSET), (True, LIVE_DELAY)])
def test_angle_cars_honour_the_self_tuning_toggle(params, live_enabled, expected):
  params.put_bool("IQLiveSteerDelay", live_enabled)
  daemon = _daemon(params, car.CarParams.SteerControlType.angle)
  InferenceDaemon._refresh_tunables(daemon, 0)
  assert daemon._runtime.lat_delay == pytest.approx(expected)


def test_angle_cars_never_plan_against_the_live_estimate_when_disabled(params):
  params.put_bool("IQLiveSteerDelay", False)
  daemon = _daemon(params, car.CarParams.SteerControlType.angle)
  InferenceDaemon._refresh_tunables(daemon, 0)
  assert daemon._runtime.lat_delay != pytest.approx(LIVE_DELAY)


@pytest.mark.parametrize("live_enabled", [True, False])
def test_torque_cars_keep_the_live_estimate(params, live_enabled):
  params.put_bool("IQLiveSteerDelay", live_enabled)
  daemon = _daemon(params, car.CarParams.SteerControlType.torque)
  InferenceDaemon._refresh_tunables(daemon, 0)
  assert daemon._runtime.lat_delay == pytest.approx(LIVE_DELAY)


def test_refresh_is_throttled_to_every_sixtieth_tick(params):
  params.put_bool("IQLiveSteerDelay", False)
  daemon = _daemon(params, car.CarParams.SteerControlType.angle)
  InferenceDaemon._refresh_tunables(daemon, 1)
  assert daemon._runtime.lat_delay is None
  InferenceDaemon._refresh_tunables(daemon, 60)
  assert daemon._runtime.lat_delay == pytest.approx(RACK_DELAY + OFFSET)
