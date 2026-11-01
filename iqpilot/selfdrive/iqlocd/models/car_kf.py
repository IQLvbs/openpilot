"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos
"""

import math
from typing import Any

import numpy as np

from iqpilot.common.constants import ACCELERATION_DUE_TO_GRAVITY
from iqpilot.selfdrive.iqlocd.models.constants import ObservationKind
from iqpilot.selfdrive.state_estimation import EstimatorModel, ModelDefinition, StateEstimator
try:
  from iqpilot.selfdrive.state_estimation.native_binding_pyx import car_predict, car_update
except ModuleNotFoundError:
  car_predict = None
  car_update = None


class States:
  STIFFNESS = slice(0, 1)
  STEER_RATIO = slice(1, 2)
  ANGLE_OFFSET = slice(2, 3)
  ANGLE_OFFSET_FAST = slice(3, 4)
  VELOCITY = slice(4, 6)
  YAW_RATE = slice(6, 7)
  STEER_ANGLE = slice(7, 8)
  ROAD_ROLL = slice(8, 9)


def _transition(state: np.ndarray, dt: float, values: dict[str, float]) -> np.ndarray:
  result = state.copy()
  stiffness = state[0]
  steer_ratio = state[1]
  angle = state[7] - state[2] - state[3]
  speed, lateral_speed = state[4:6]
  yaw_rate = state[6]
  mass = values["mass"]
  inertia = values["rotational_inertia"]
  front = values["center_to_front"]
  rear = values["center_to_rear"]
  front_stiffness = stiffness * values["stiffness_front"]
  rear_stiffness = stiffness * values["stiffness_rear"]
  lateral_dot = -(front_stiffness + rear_stiffness) * lateral_speed / (mass * speed)
  lateral_dot += (-(front_stiffness * front - rear_stiffness * rear) / (mass * speed) - speed) * yaw_rate
  lateral_dot += front_stiffness * angle / (mass * steer_ratio) - ACCELERATION_DUE_TO_GRAVITY * state[8]
  yaw_dot = -(front_stiffness * front - rear_stiffness * rear) * lateral_speed / (inertia * speed)
  yaw_dot -= (front_stiffness * front**2 + rear_stiffness * rear**2) * yaw_rate / (inertia * speed)
  yaw_dot += front_stiffness * front * angle / (inertia * steer_ratio)
  result[5] += dt * lateral_dot
  result[6] += dt * yaw_dot
  return result


class CarKalman(EstimatorModel):
  name = "car"
  initial_x = np.array([1.0, 15.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0])
  Q = np.diag([(.05 / 100)**2, .01**2, math.radians(0.02)**2, math.radians(0.25)**2,
               .1**2, .01**2, math.radians(0.1)**2, math.radians(0.1)**2, math.radians(1)**2])
  P_initial = Q.copy()
  obs_noise: dict[int, Any] = {
    ObservationKind.STEER_ANGLE: np.atleast_2d(math.radians(0.05)**2),
    ObservationKind.ANGLE_OFFSET_FAST: np.atleast_2d(math.radians(10.0)**2),
    ObservationKind.ROAD_ROLL: np.atleast_2d(math.radians(1.0)**2),
    ObservationKind.STEER_RATIO: np.atleast_2d(5.0**2),
    ObservationKind.STIFFNESS: np.atleast_2d(0.5**2),
    ObservationKind.ROAD_FRAME_X_SPEED: np.atleast_2d(0.1**2),
  }

  def __init__(self):
    self.native_parameters = np.zeros(6)
    measurements = {
      ObservationKind.ROAD_FRAME_YAW_RATE: lambda state, _: state[6:7],
      ObservationKind.ROAD_FRAME_XY_SPEED: lambda state, _: state[4:6],
      ObservationKind.ROAD_FRAME_X_SPEED: lambda state, _: state[4:5],
      ObservationKind.STEER_ANGLE: lambda state, _: state[7:8],
      ObservationKind.ANGLE_OFFSET_FAST: lambda state, _: state[3:4],
      ObservationKind.STEER_RATIO: lambda state, _: state[1:2],
      ObservationKind.STIFFNESS: lambda state, _: state[0:1],
      ObservationKind.ROAD_ROLL: lambda state, _: state[8:9],
    }
    def native_predict(state, covariance, dt, process_noise, _):
      car_predict(state, covariance, process_noise, dt, self.native_parameters)

    model = ModelDefinition(9, 9, _transition, measurements, self.Q, self.obs_noise,
                            native_predict=native_predict if car_predict is not None else None, native_update=car_update)
    super().__init__(StateEstimator(model, self.initial_x, self.P_initial, max_rewind_age=0.8))

  def set_globals(self, mass: float, rotational_inertia: float, center_to_front: float, center_to_rear: float,
                  stiffness_front: float, stiffness_rear: float) -> None:
    self.native_parameters[:] = mass, rotational_inertia, center_to_front, center_to_rear, stiffness_front, stiffness_rear
    for name, value in locals().copy().items():
      if name != "self":
        self.filter.set_global(name, value)
