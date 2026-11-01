# Copyright © IQ.Lvbs, part of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos
import numpy as np
import pytest

from iqdbc.car.volkswagen.values import VolkswagenFlags
from iqpilot.cereal import car, messaging
from iqpilot.common.params import Params
from iqpilot.selfdrive.locationd.paramsd import MEB_ANGLE_SCALE_REV, retrieve_initial_vehicle_params

DBC_SCALE_STEER_RATIO = 15.7


def car_params(brand, fingerprint, flags, steer_ratio=18.4):
  CP = car.CarParams.new_message()
  CP.brand = brand
  CP.carFingerprint = fingerprint
  CP.flags = int(flags)
  CP.steerRatio = steer_ratio
  return CP.as_reader()


def cache_learned_before_the_fix(params, CP):
  msg = messaging.new_message("vehicleParameters")
  msg.vehicleParameters.steerRatio = DBC_SCALE_STEER_RATIO
  msg.vehicleParameters.stiffnessFactor = 1.0
  msg.vehicleParameters.angleOffsetAverageDeg = 0.4
  params.put("LiveParametersV2", msg.to_bytes())
  params.put("CarParamsPrevRoute", CP.as_builder().to_bytes())


@pytest.fixture
def params(tmp_path, monkeypatch):
  monkeypatch.setenv("PARAMS_ROOT", str(tmp_path))
  return Params()


MEB_GEN2 = VolkswagenFlags.MEB | VolkswagenFlags.MEB_GEN2


@pytest.mark.parametrize("fingerprint,flags", [
  ("VOLKSWAGEN_ID4_MK2", MEB_GEN2),
  ("VOLKSWAGEN_ID4_MK1", VolkswagenFlags.MEB),
  ("FORD_CAPRI_MK1", MEB_GEN2),
])
def test_meb_cache_learned_on_the_old_angle_scale_is_dropped_once(params, fingerprint, flags):
  CP = car_params("volkswagen", fingerprint, flags)
  cache_learned_before_the_fix(params, CP)

  steer_ratio, _, angle_offset, _ = retrieve_initial_vehicle_params(params, CP, replay=False, debug=False)
  assert steer_ratio == pytest.approx(CP.steerRatio)
  assert angle_offset == 0.0
  assert params.get("MebAngleScaleRev") == MEB_ANGLE_SCALE_REV

  cache_learned_before_the_fix(params, CP)
  steer_ratio, _, angle_offset, _ = retrieve_initial_vehicle_params(params, CP, replay=False, debug=False)
  assert steer_ratio == pytest.approx(DBC_SCALE_STEER_RATIO)
  assert angle_offset == pytest.approx(0.4)


def test_meb_cache_learned_after_the_fix_survives(params):
  CP = car_params("volkswagen", "VOLKSWAGEN_ID4_MK2", MEB_GEN2)
  params.put("MebAngleScaleRev", MEB_ANGLE_SCALE_REV)
  cache_learned_before_the_fix(params, CP)

  steer_ratio, _, _, _ = retrieve_initial_vehicle_params(params, CP, replay=False, debug=False)
  assert steer_ratio == pytest.approx(DBC_SCALE_STEER_RATIO)


@pytest.mark.parametrize("brand,fingerprint,flags", [
  ("volkswagen", "VOLKSWAGEN_GOLF_MK8", VolkswagenFlags.MQB_EVO),
  ("volkswagen", "VOLKSWAGEN_GOLF_MK7", 0),
  ("toyota", "TOYOTA_RAV4_TSS2", 0),
])
def test_other_cars_keep_their_cache_and_marker(params, brand, fingerprint, flags):
  CP = car_params(brand, fingerprint, flags)
  cache_learned_before_the_fix(params, CP)

  steer_ratio, _, _, _ = retrieve_initial_vehicle_params(params, CP, replay=False, debug=False)
  assert steer_ratio == pytest.approx(DBC_SCALE_STEER_RATIO)
  assert params.get("MebAngleScaleRev") is None


def test_replay_never_touches_the_cache_or_marker(params):
  CP = car_params("volkswagen", "VOLKSWAGEN_ID4_MK2", MEB_GEN2)
  cache_learned_before_the_fix(params, CP)

  steer_ratio, _, _, _ = retrieve_initial_vehicle_params(params, CP, replay=True, debug=False)
  np.testing.assert_allclose(steer_ratio, DBC_SCALE_STEER_RATIO, rtol=1e-6)
  assert params.get("MebAngleScaleRev") is None
