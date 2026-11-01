"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""
import numpy as np

from iqpilot.common.constants import CV

ACCEL_BOOST_MAX = 0.2
ACCEL_BOOST_RATE = 0.025
ACCEL_BOOST_DECAY_RATE = 0.1
ACCEL_BOOST_PER_OVERRIDE = 0.05
ACCEL_BOOST_MIN_SPEED = 10 * CV.MPH_TO_MS


class AccelBoost:
  def __init__(self, enabled: bool, dt: float):
    self.enabled = enabled
    self.dt = dt
    self.value = 0.0
    self.override_boost = 0.0
    self.model_limited = False

  def update(self, engaged: bool, v_ego: float, gas_pressed: bool, model_limited: bool) -> float:
    enabled = self.enabled and engaged
    if not enabled or not gas_pressed:
      self.override_boost = 0.0
      self.model_limited = enabled and model_limited

    if not enabled:
      self.value = 0.0
    elif v_ego < ACCEL_BOOST_MIN_SPEED:
      self.value = max(0.0, self.value - ACCEL_BOOST_DECAY_RATE * self.dt)
    elif gas_pressed and self.model_limited:
      increase = min(ACCEL_BOOST_RATE * self.dt, ACCEL_BOOST_PER_OVERRIDE - self.override_boost, ACCEL_BOOST_MAX - self.value)
      self.value += increase
      self.override_boost += increase
    return self.value

  def apply(self, accel: float) -> float:
    return accel + np.interp(accel, [-1.0, -0.5, 5.0], [0.0, self.value, self.value], right=0.0)
