"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos
"""
import math
import numpy as np
from collections import deque

from iqpilot.cereal import log
from iqdbc.car.lateral import get_friction
from iqpilot.common.constants import ACCELERATION_DUE_TO_GRAVITY
from iqpilot.common.filter_simple import FirstOrderFilter
from iqpilot.common.params import Params
from iqpilot.selfdrive.controls.lib.latcontrol import LatControl
from iqpilot.selfdrive.controls.lib.lateral_acceleration_slew_limiter import LateralAccelerationSlewLimiter
from iqpilot.common.pid import PIDController

FRICTION_THRESHOLD_PQ = 1.0
KP = 0.8
KI = 0.15
KI_HIGHWAY = 0.3
KI_SPEEDS = [10.0, 15.0]

INTERP_SPEEDS = [1, 1.5, 2.0, 3.0, 5, 7.5, 10, 15, 30]
KP_INTERP = [250, 120, 65, 30, 11.5, 5.5, 3.5, 2.0, KP]

LP_FILTER_CUTOFF_HZ = 1.5
JERK_LOOKAHEAD_SECONDS = 0.34
JERK_GAIN = 0.3
LAT_ACCEL_REQUEST_BUFFER_SECONDS = 1.0
VERSION = 1

DEFAULT_LAT_ACCEL_FACTOR = 2.2
DEFAULT_LAT_ACCEL_OFFSET = -0.13
DEFAULT_FRICTION = 0.1
FREEZE_LIVE_TORQUE_PARAMS = False
LIVE_LAT_ACCEL_FACTOR_GAIN = 1.1
LIVE_LAT_ACCEL_FACTOR_BOUNDS = (1.2, 3.5)
LIVE_LAT_ACCEL_OFFSET_BOUNDS = (-0.3, 0.3)
LIVE_TORQUE_PARAMS_TAU_S = 5.0

ASSIST_COMPENSATION = True
ASSIST_SPEEDS_KPH = [0.0, 50.0, 120.0]
ASSIST_GAIN = [0.688, 0.883, 1.211]
ASSIST_REF_KPH = 100.0


def _assist_comp(v_ego_ms):
  import numpy as _np
  ref = _np.interp(ASSIST_REF_KPH, ASSIST_SPEEDS_KPH, ASSIST_GAIN)
  g = _np.interp(v_ego_ms * 3.6, ASSIST_SPEEDS_KPH, ASSIST_GAIN)
  return float(_np.clip(ref / g, 0.7, 1.6))


class LatControlTorquePQ(LatControl):
  supports_legacy_curvature_lookahead = True

  def __init__(self, CP, CP_IQ, CI, dt):
    super().__init__(CP, CP_IQ, CI, dt)
    self.torque_params = CP.lateralTuning.torque.as_builder()
    self.torque_params.latAccelFactor = DEFAULT_LAT_ACCEL_FACTOR
    self.torque_params.latAccelOffset = DEFAULT_LAT_ACCEL_OFFSET
    self.torque_params.friction = DEFAULT_FRICTION
    self.torque_from_lateral_accel = CI.torque_from_lateral_accel()
    self.lateral_accel_from_torque = CI.lateral_accel_from_torque()
    self.pid = PIDController([INTERP_SPEEDS, KP_INTERP], [KI_SPEEDS, [KI, KI_HIGHWAY]], rate=1/self.dt)
    self.live_torque_valid = False
    self.lat_accel_factor_filter = FirstOrderFilter(DEFAULT_LAT_ACCEL_FACTOR, LIVE_TORQUE_PARAMS_TAU_S, self.dt)
    self.lat_accel_offset_filter = FirstOrderFilter(DEFAULT_LAT_ACCEL_OFFSET, LIVE_TORQUE_PARAMS_TAU_S, self.dt)
    self.update_limits()
    self.steering_angle_deadzone_deg = self.torque_params.steeringAngleDeadzoneDeg
    self.lat_accel_request_buffer_len = int(LAT_ACCEL_REQUEST_BUFFER_SECONDS / self.dt)
    self.lat_accel_request_buffer = deque([0.] * self.lat_accel_request_buffer_len, maxlen=self.lat_accel_request_buffer_len)
    self.lookahead_frames = int(JERK_LOOKAHEAD_SECONDS / self.dt)
    self.jerk_filter = FirstOrderFilter(0.0, 1 / (2 * np.pi * LP_FILTER_CUTOFF_HZ), self.dt)
    self.lateral_acceleration_slew_limiter = LateralAccelerationSlewLimiter(Params().get_bool("IQLateralAccelSlew"))

  def set_live_torque_valid(self, valid):
    self.live_torque_valid = bool(valid)

  def update_live_torque_params(self, latAccelFactor, latAccelOffset, friction):
    if FREEZE_LIVE_TORQUE_PARAMS:
      return
    if self.live_torque_valid and math.isfinite(latAccelFactor) and math.isfinite(latAccelOffset):
      target_factor = float(np.clip(latAccelFactor * LIVE_LAT_ACCEL_FACTOR_GAIN, *LIVE_LAT_ACCEL_FACTOR_BOUNDS))
      target_offset = float(np.clip(latAccelOffset, *LIVE_LAT_ACCEL_OFFSET_BOUNDS))
    else:
      target_factor = DEFAULT_LAT_ACCEL_FACTOR
      target_offset = DEFAULT_LAT_ACCEL_OFFSET
    self.torque_params.latAccelFactor = self.lat_accel_factor_filter.update(target_factor)
    self.torque_params.latAccelOffset = self.lat_accel_offset_filter.update(target_offset)
    self.update_limits()

  def update_limits(self):
    self.pid.set_limits(self.lateral_accel_from_torque(self.steer_max, self.torque_params),
                        self.lateral_accel_from_torque(-self.steer_max, self.torque_params))

  def update(self, active, CS, VM, params, steer_limited_by_safety, desired_curvature, calibrated_pose, curvature_limited, lat_delay):
    pid_log = log.ControlsState.LateralTorqueState.new_message()
    pid_log.version = VERSION
    measured_curvature = -VM.calc_curvature(math.radians(CS.steeringAngleDeg - params.angleOffsetDeg), CS.vEgo, params.roll)
    measurement = measured_curvature * CS.vEgo ** 2
    target_curvature = desired_curvature
    if not active and self.lateral_acceleration_slew_limiter.enabled:
      self.lateral_acceleration_slew_limiter.reset(target_curvature * CS.vEgo ** 2)
    limited_curvature = self.lateral_acceleration_slew_limiter.update(target_curvature, CS.vEgo, self.dt)
    future_desired_lateral_accel = limited_curvature * CS.vEgo ** 2
    self.lat_accel_request_buffer.append(future_desired_lateral_accel)

    roll_compensation = params.roll * ACCELERATION_DUE_TO_GRAVITY
    curvature_deadzone = abs(VM.calc_curvature(math.radians(self.steering_angle_deadzone_deg), CS.vEgo, 0.0))
    lateral_accel_deadzone = curvature_deadzone * CS.vEgo ** 2

    delay_frames = int(np.clip(lat_delay / self.dt + 1, 1, self.lat_accel_request_buffer_len))
    expected_lateral_accel = self.lat_accel_request_buffer[-delay_frames]
    setpoint = expected_lateral_accel
    error = setpoint - measurement

    lookahead_idx = int(np.clip(-delay_frames + self.lookahead_frames, -self.lat_accel_request_buffer_len + 1, -2))
    raw_lateral_jerk = (self.lat_accel_request_buffer[lookahead_idx + 1] - self.lat_accel_request_buffer[lookahead_idx - 1]) / (2 * self.dt)
    desired_lateral_jerk = self.jerk_filter.update(raw_lateral_jerk)
    gravity_adjusted_future_lateral_accel = future_desired_lateral_accel - roll_compensation
    ff = gravity_adjusted_future_lateral_accel
    ff -= self.torque_params.latAccelOffset
    ff += get_friction(error + JERK_GAIN * desired_lateral_jerk, lateral_accel_deadzone, FRICTION_THRESHOLD_PQ, self.torque_params)

    if not active:
      output_torque = 0.0
      pid_log.active = False
    else:
      pid_log.error = float(error)
      freeze_integrator = steer_limited_by_safety or CS.steeringPressed or CS.vEgo < 5
      output_lataccel = self.pid.update(pid_log.error, speed=CS.vEgo, feedforward=ff, freeze_integrator=freeze_integrator)
      output_torque = self.torque_from_lateral_accel(output_lataccel, self.torque_params)
      if ASSIST_COMPENSATION:
        output_torque = float(np.clip(output_torque * _assist_comp(CS.vEgo),
                                      -self.steer_max, self.steer_max))

      pid_log.active = True
      pid_log.p = float(self.pid.p)
      pid_log.i = float(self.pid.i)
      pid_log.d = float(self.pid.d)
      pid_log.f = float(self.pid.f)
      pid_log.output = float(-output_torque)
      pid_log.actualLateralAccel = float(measurement)
      pid_log.desiredLateralAccel = float(setpoint)
      pid_log.desiredLateralJerk = float(desired_lateral_jerk)
      pid_log.saturated = bool(self._check_saturation(self.steer_max - abs(output_torque) < 1e-3, CS, steer_limited_by_safety, curvature_limited))

    return -output_torque, 0.0, pid_log
