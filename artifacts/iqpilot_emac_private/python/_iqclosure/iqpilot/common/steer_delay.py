"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos

Chooses which steer-actuator delay the lateral controllers run with: the value the
live estimator learned, or the driver's fixed software delay — gated by the
"IQLiveSteerDelay" param. The pick is mirrored into "IQSteerDelayCache" so consumers that do
not subscribe to lateralDelay can still read the current value.
"""
from iqpilot.cereal import car
from iqpilot.common.params import Params

_ENABLE_KEY = "IQLiveSteerDelay"
_FIXED_KEY = "IQSoftwareSteerDelay"
_CACHE_KEY = "IQSteerDelayCache"


def fixed_steer_delay(params, stock_delay):
  """The rack's own delay plus the driver's IQSoftwareSteerDelay offset, as the UI reports it."""
  return stock_delay + float(params.get(_FIXED_KEY, return_default=True))


def resolve_steer_delay(params, stock_delay):
  """Learned lateral delay while live-learning is enabled, otherwise the driver's fixed delay."""
  if not params.get_bool(_ENABLE_KEY):
    return fixed_steer_delay(params, stock_delay)
  return float(params.get(_CACHE_KEY, return_default=True))


def lateral_action_delay(params, car_params, live_delay):
  """Delay the lateral path should be planned against.

  Angle cars honour the IQLiveSteerDelay toggle so that with live learning off the
  estimate never reaches the path: lagd cross-correlates against localizer lateral
  accel, so it reports whole-vehicle response (~0.36 s measured on VW MQB, 0.44 s on
  Tesla) where the lookahead wants actuator delay (~0.10 s). Torque cars keep the
  live estimate.
  """
  if car_params.steerControlType == car.CarParams.SteerControlType.angle:
    return resolve_steer_delay(params, car_params.steerActuatorDelay)
  return live_delay


def cached_steer_delay():
  """Last value SteerDelayPublisher mirrored into the param — usable without a
  lateralDelay subscription (e.g. at process startup)."""
  return Params().get(_CACHE_KEY, return_default=True)


class SteerDelayPublisher:
  """Refreshes IQSteerDelayCache every lag message: the learned live delay when the
  toggle is on, else the actuator delay plus the driver's fixed software offset."""

  def __init__(self, car_params):
    self._params = Params()
    self._actuator_delay = car_params.steerActuatorDelay

  def update(self, lag_msg):
    live = self._params.get_bool(_ENABLE_KEY)
    value = lag_msg.lateralDelay.lateralDelay if live else fixed_steer_delay(self._params, self._actuator_delay)
    self._params.put_nonblocking(_CACHE_KEY, value)
