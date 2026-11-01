"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos
"""

import json
import time

import numpy as np

from iqpilot.selfdrive.iqlocd.models.car_kf import CarKalman
from iqpilot.selfdrive.iqlocd.models.constants import ObservationKind
from iqpilot.selfdrive.locationd.models.pose_kf import PoseKalman


def measure(function, count: int) -> dict[str, float]:
  samples = np.empty(count)
  for index in range(count):
    started = time.perf_counter_ns()
    function(index)
    samples[index] = (time.perf_counter_ns() - started) / 1000.0
  return {"p50_us": float(np.percentile(samples, 50)), "p99_us": float(np.percentile(samples, 99)), "mean_us": float(samples.mean())}


def main() -> None:
  car = CarKalman()
  car.set_globals(1800.0, 2500.0, 1.2, 1.6, 90000.0, 100000.0)
  car.init_state(CarKalman.initial_x, CarKalman.P_initial, 0.0)
  pose = PoseKalman(0.8)
  pose.init_state(PoseKalman.initial_x, PoseKalman.initial_P, 0.0)
  result = {
    "car": measure(lambda index: car.predict_and_observe(index * 0.01, ObservationKind.ROAD_FRAME_X_SPEED, np.array([15.0])), 1000),
    "pose": measure(lambda index: pose.predict_and_observe(index * 0.01, ObservationKind.PHONE_GYRO, np.zeros(3)), 1000),
  }
  print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
  main()
