"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos
"""
import numpy as np
JERK_SPEED_BP = [0.0, 8.0, 20.0, 35.0]
JERK_MAX_BP = [5.0, 4.0, 2.5, 2.0]
A_LAT_MAX = 3.0
MIN_LIMIT_SPEED = 5.0
AVOIDANCE_BYPASS_ACCEL_DELTA = 2.0
CURVATURE_SPEED_FLOOR = 0.1


class LateralAccelerationSlewLimiter:
  def __init__(self, enabled: bool):
    self.enabled = enabled
    self.a_lim = 0.0

  def reset(self, a_lat: float) -> None:
    self.a_lim = float(np.clip(a_lat, -A_LAT_MAX, A_LAT_MAX))

  def jerk_max(self, v_ego: float) -> float:
    return float(np.interp(v_ego, JERK_SPEED_BP, JERK_MAX_BP))

  def update(self, desired_curvature: float, v_ego: float, dt: float) -> float:
    if not self.enabled:
      return desired_curvature

    a_des = v_ego ** 2 * desired_curvature
    if v_ego < MIN_LIMIT_SPEED:
      self.reset(a_des)
      return desired_curvature

    if abs(a_des - self.a_lim) > AVOIDANCE_BYPASS_ACCEL_DELTA:
      self.reset(a_des)
    else:
      da_max = self.jerk_max(v_ego) * dt
      self.a_lim += float(np.clip(a_des - self.a_lim, -da_max, da_max))
      self.a_lim = float(np.clip(self.a_lim, -A_LAT_MAX, A_LAT_MAX))

    speed = max(abs(v_ego), CURVATURE_SPEED_FLOOR)
    return self.a_lim / speed ** 2
