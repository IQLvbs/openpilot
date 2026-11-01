"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos
"""

import numpy as np

from iqpilot.selfdrive.iqlocd.models.car_kf import CarKalman, States as CarStates
from iqpilot.selfdrive.iqlocd.models.constants import ObservationKind
from iqpilot.selfdrive.locationd.models.pose_kf import PoseKalman, States as PoseStates


def configured_car() -> CarKalman:
  estimator = CarKalman()
  estimator.set_globals(1800.0, 2500.0, 1.2, 1.6, 90000.0, 100000.0)
  estimator.init_state(CarKalman.initial_x, CarKalman.P_initial, 0.0)
  return estimator


def test_car_mutable_parameters_affect_prediction() -> None:
  light = configured_car()
  heavy = configured_car()
  heavy.set_globals(3600.0, 5000.0, 1.2, 1.6, 90000.0, 100000.0)
  state = CarKalman.initial_x.copy()
  state[CarStates.STEER_ANGLE] = 0.1
  light.init_state(state, CarKalman.P_initial, 0.0)
  heavy.init_state(state, CarKalman.P_initial, 0.0)
  light.predict(0.01)
  heavy.predict(0.01)
  assert abs(light.x[CarStates.YAW_RATE].item()) > abs(heavy.x[CarStates.YAW_RATE].item())


def test_car_long_sequence_stays_finite() -> None:
  estimator = configured_car()
  for index in range(500):
    time = index * 0.01
    estimator.predict_and_observe(time, ObservationKind.STEER_ANGLE, np.array([0.02 * np.sin(time)]))
    estimator.predict_and_observe(time, ObservationKind.ROAD_FRAME_X_SPEED, np.array([15.0]))
  assert np.isfinite(estimator.x).all()
  assert np.isfinite(estimator.P).all()
  assert np.linalg.eigvalsh(estimator.P).min() >= -1e-10


def test_pose_delayed_sensor_sequence_is_stable() -> None:
  estimator = PoseKalman(0.8)
  estimator.init_state(PoseKalman.initial_x, PoseKalman.initial_P, 0.0)
  estimator.predict_and_observe(0.02, ObservationKind.PHONE_GYRO, np.array([0.01, -0.02, 0.03]))
  estimator.predict_and_observe(0.04, ObservationKind.PHONE_ACCEL, np.array([0.0, 0.0, -9.81]))
  estimator.predict_and_observe(0.03, ObservationKind.CAMERA_ODO_ROTATION, np.array([0.01, -0.02, 0.03]))
  assert np.isfinite(estimator.x).all()
  assert np.isfinite(estimator.P).all()
  np.testing.assert_allclose(estimator.P, estimator.P.T, atol=1e-12)


def test_pose_zero_rotation_preserves_orientation() -> None:
  estimator = PoseKalman(0.8)
  estimator.init_state(PoseKalman.initial_x, PoseKalman.initial_P, 0.0)
  estimator.predict(1.0)
  np.testing.assert_allclose(estimator.x[PoseStates.NED_ORIENTATION], np.zeros(3), atol=1e-12)


def test_legacy_car_estimator_imports_share_canonical_types():
  from iqpilot.selfdrive.iqlocd.models import car_kf, constants
  from iqpilot.selfdrive.locationd.models import car_kf as legacy_car, constants as legacy_constants

  assert legacy_car.CarKalman is car_kf.CarKalman
  assert legacy_car.States is car_kf.States
  assert legacy_constants.ObservationKind is constants.ObservationKind
  assert legacy_constants.SAT_OBS is constants.SAT_OBS
