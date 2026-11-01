"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos
"""

import numpy as np

from iqpilot.common.transformations.orientation import euler_from_rot, rot_from_euler
from iqpilot.selfdrive.iqlocd.models.constants import ObservationKind
from iqpilot.selfdrive.state_estimation import EstimatorModel, ModelDefinition, StateEstimator
try:
  from iqpilot.selfdrive.state_estimation.native_binding_pyx import pose_predict, pose_update
except ModuleNotFoundError:
  pose_predict = None
  pose_update = None


EARTH_G = 9.81


class States:
  NED_ORIENTATION = slice(0, 3)
  DEVICE_VELOCITY = slice(3, 6)
  ANGULAR_VELOCITY = slice(6, 9)
  GYRO_BIAS = slice(9, 12)
  ACCELERATION = slice(12, 15)
  ACCEL_BIAS = slice(15, 18)


def _transition(state: np.ndarray, dt: float, _: dict[str, float]) -> np.ndarray:
  result = state.copy()
  result[States.DEVICE_VELOCITY] += dt * state[States.ACCELERATION]
  rotation = rot_from_euler(state[States.NED_ORIENTATION]) @ rot_from_euler(dt * state[States.ANGULAR_VELOCITY])
  result[States.NED_ORIENTATION] = euler_from_rot(rotation)
  return result


def _phone_acceleration(state: np.ndarray, _: dict[str, float]) -> np.ndarray:
  device_from_ned = rot_from_euler(state[States.NED_ORIENTATION]).T
  centripetal = np.cross(state[States.ANGULAR_VELOCITY], state[States.DEVICE_VELOCITY])
  return device_from_ned @ np.array([0.0, 0.0, -EARTH_G]) + state[States.ACCELERATION] + centripetal + state[States.ACCEL_BIAS]


class PoseKalman(EstimatorModel):
  name = "pose"
  initial_x = np.zeros(18)
  initial_P = np.diag([0.01**2] * 3 + [10**2] * 3 + [1**2] * 6 + [100**2] * 3 + [0.01**2] * 3)
  Q = np.diag([0.001**2] * 3 + [0.01**2] * 3 + [0.1**2] * 3 + [(0.005 / 100)**2] * 3 + [3**2] * 3 + [0.005**2] * 3)
  obs_noise = {
    ObservationKind.PHONE_GYRO: np.diag([0.025**2] * 3),
    ObservationKind.PHONE_ACCEL: np.diag([0.5**2] * 3),
    ObservationKind.CAMERA_ODO_TRANSLATION: np.diag([0.5**2] * 3),
    ObservationKind.CAMERA_ODO_ROTATION: np.diag([0.05**2] * 3),
  }

  def __init__(self, max_rewind_age: float):
    measurements = {
      ObservationKind.PHONE_GYRO: lambda state, _: state[States.ANGULAR_VELOCITY] + state[States.GYRO_BIAS],
      ObservationKind.PHONE_ACCEL: _phone_acceleration,
      ObservationKind.CAMERA_ODO_TRANSLATION: lambda state, _: state[States.DEVICE_VELOCITY],
      ObservationKind.CAMERA_ODO_ROTATION: lambda state, _: state[States.ANGULAR_VELOCITY],
    }
    def native_predict(state, covariance, dt, process_noise, _):
      pose_predict(state, covariance, process_noise, dt)

    model = ModelDefinition(18, 18, _transition, measurements, self.Q, self.obs_noise,
                            native_predict=native_predict if pose_predict is not None else None, native_update=pose_update)
    super().__init__(StateEstimator(model, self.initial_x, self.initial_P, max_rewind_age=max_rewind_age))
