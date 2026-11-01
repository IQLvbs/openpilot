import pytest

from iqdbc.car import DT_CTRL, structs
from iqdbc.car.ford.interface import CarInterface
from iqdbc.car.ford.values import CAR, CarControllerParams
from iqdbc.car.lateral import (CurvatureSteeringLimits, MAX_LATERAL_ACCEL, MAX_LATERAL_JERK, apply_std_curvature_limits)

LIMITS = CarControllerParams.CURVATURE_LIMITS
STEER_STEP = CarControllerParams.STEER_STEP


def test_ford_curvature_max():
  assert LIMITS.CURVATURE_MAX == 0.02


@pytest.mark.parametrize("v_ego", [1., 5., 10., 20., 35.])
def test_lateral_accel_limit(v_ego):
  limit = min(MAX_LATERAL_ACCEL / v_ego ** 2, LIMITS.CURVATURE_MAX)
  assert LIMITS.apply_limits(1.0, limit, v_ego, 0., True, STEER_STEP) == pytest.approx(limit)
  assert LIMITS.apply_limits(-1.0, -limit, v_ego, 0., True, STEER_STEP) == pytest.approx(-limit)


@pytest.mark.parametrize("v_ego", [10., 20., 35.])
def test_lateral_jerk_limit(v_ego):
  max_delta = MAX_LATERAL_JERK / v_ego ** 2 * STEER_STEP * DT_CTRL
  assert LIMITS.apply_limits(0.01, 0., v_ego, 0., True, STEER_STEP) == pytest.approx(min(max_delta, 0.01))
  assert LIMITS.apply_limits(-0.01, 0., v_ego, 0., True, STEER_STEP) == pytest.approx(-min(max_delta, 0.01))
  assert LIMITS.apply_limits(0., 0.01, v_ego, 0., True, STEER_STEP) == pytest.approx(max(0.01 - max_delta, 0.))


def test_inactive_returns_measured_curvature():
  assert LIMITS.apply_limits(0.01, 0.01, 20., 0., False, STEER_STEP) == 0.
  assert LIMITS.apply_limits(0.01, 0.01, 20., 0.003, False, STEER_STEP) == 0.003


def test_curvature_max_clips_at_low_speed():
  assert LIMITS.apply_limits(1.0, 0.019, 0., 0., True, STEER_STEP) == LIMITS.CURVATURE_MAX
  assert LIMITS.apply_limits(-1.0, -0.019, 0., 0., True, STEER_STEP) == -LIMITS.CURVATURE_MAX


def test_other_platforms_ignore_ford_limit_fields():
  default = CurvatureSteeringLimits(0.195)
  tweaked = CurvatureSteeringLimits(0.195, MAX_LATERAL_ACCEL=0.1, MAX_LATERAL_JERK=0.1)
  args = (0.05, 0.04, 15., 0.03, False, STEER_STEP, True)
  assert apply_std_curvature_limits(*args[:5], *args[5:6], *args[6:], default) == \
    apply_std_curvature_limits(*args[:5], *args[5:6], *args[6:], tweaked)


@pytest.mark.parametrize("candidate", [CAR.FORD_F_150_MK14, CAR.FORD_ESCAPE_MK4])
def test_controller_applies_native_curvature_limits(candidate, monkeypatch, tmp_path):
  monkeypatch.setenv("PARAMS_ROOT", str(tmp_path))
  fingerprint = {bus: {} for bus in range(7)}
  cp = CarInterface.get_params(candidate, fingerprint, [], alpha_long=False, is_release=False, docs=False)
  cp_iq = CarInterface.get_params_iq(cp, candidate, fingerprint, [], alpha_long=False, is_release_iq=False, docs=False)
  interface = CarInterface(cp, cp_iq)
  cc_iq = structs.IQCarControl()

  now = 0
  for lat_active, expected in ((True, LIMITS.CURVATURE_MAX), (False, 0.)):
    cc = structs.CarControl()
    cc.enabled = lat_active
    cc.latActive = lat_active
    cc.actuators.curvature = 0.05
    for _ in range(STEER_STEP * 2):
      now += int(DT_CTRL * 1e9)
      interface.update([])
      interface.apply(cc.as_reader(), cc_iq, now)
    assert interface.CC.apply_curvature_last == pytest.approx(expected)
