import math

import numpy as np

from iqdbc.car.common.filter_simple import FirstOrderFilter
from iqdbc.car.common.conversions import Conversions as CV
from iqdbc.car.common.pid import PIDController
from iqdbc.car import DT_CTRL


def longitudinal_comfort_values(CC, CC_IQ, legacy_band):
  """Return positive/negative jerk and VW oben/unten deviation fields."""
  comfort = CC_IQ.longitudinalComfort
  legacy = (4.0, 4.0, legacy_band, legacy_band)
  if not CC.longComfortMode or not comfort.valid:
    return legacy
  values = (comfort.jerkUp, comfort.jerkDown, comfort.accelDeviationAbove, comfort.accelDeviationBelow)
  if not all(np.isfinite(v) for v in values) or min(values) < 0 or min(values[:2]) <= 0:
    return legacy
  if str(CC.hudControl.visualAlert) == 'fcw' or comfort.bypassJerkLimit:
    return 5.0, 5.0, 0.0, 0.0

  up, down = min(comfort.jerkUp, 5.0), min(comfort.jerkDown, 5.0)
  above, below = comfort.accelDeviationAbove, comfort.accelDeviationBelow
  if comfort.bypassJerkDownLimit:
    down = 5.0
  if not comfort.comfortEnabled or comfort.bypassJerkDownLimit:
    above = below = 0.0
  # Physical above -> unten, below -> oben. Stock Golf ACC (cae14e88932eb364 /
  # 2021-03-26--14-43-28, segments 2-3) tightens unten to zero while braking,
  # and commonly sends unten/oben = 0.192/0.125 while accelerating. Sustained
  # cruise reaches oben = 0.375; segments 0-5 never exceed unten/oben =
  # 0.192/0.5 while active. Cap the shared policy to that stock envelope.
  # This supports the original port's comments; ECU semantics are inferred.
  # Round down to DBC steps: encoding must never widen the requested band.
  unten = np.floor(min(above, 0.192) / 0.024) * 0.024
  oben = np.floor(min(below, 0.5) / 0.0625) * 0.0625
  return up, down, oben, unten


class LongStopHold():
  HOLD_CLEAR_TIME = 0.3
  DRIVE_OFF_CONFIRM_TIME = 0.25
  HOLD_RELEASE_SPEED = 0.3
  HOLD_ACCEL_CEILING = 0.0

  def __init__(self, dt=DT_CTRL):
    self.dt = dt
    self.esp_hold = False
    self.hold_clear_timer = 0.
    self.stopping = False
    self.drive_off_timer = 0.

  def reset(self, stopping, esp_hold):
    self.esp_hold = esp_hold
    self.hold_clear_timer = 0.
    self.stopping = stopping
    self.drive_off_timer = 0.

  def _update_hold(self, esp_hold, v_ego):
    if esp_hold:
      self.esp_hold = True
      self.hold_clear_timer = 0.
    elif self.esp_hold:
      self.hold_clear_timer += self.dt
      if self.hold_clear_timer >= self.HOLD_CLEAR_TIME or v_ego > self.HOLD_RELEASE_SPEED:
        self.esp_hold = False
        self.hold_clear_timer = 0.

  def update(self, long_active, stopping, starting, accel, esp_hold, v_ego, override):
    if not long_active or override:
      self.reset(stopping, esp_hold)
      return stopping, starting, esp_hold, accel

    self._update_hold(esp_hold, v_ego)

    holding = self.stopping and (self.esp_hold or v_ego <= self.HOLD_RELEASE_SPEED)
    if holding and not stopping:
      if starting or accel > self.HOLD_ACCEL_CEILING:
        self.drive_off_timer += self.dt
      else:
        self.drive_off_timer = 0.
      committed = self.drive_off_timer >= self.DRIVE_OFF_CONFIRM_TIME or v_ego > self.HOLD_RELEASE_SPEED
      self.stopping = not committed
    else:
      self.drive_off_timer = 0.
      self.stopping = stopping

    if self.stopping:
      return True, False, self.esp_hold, min(accel, self.HOLD_ACCEL_CEILING)

    return self.stopping, starting, self.esp_hold, accel


def sigmoid_curvature_boost_meb(kappa: float, v_ego: float, kappa_thresh: float = 0.0) -> float:
  # compensate non linear behaviour: boost low curvatures
  # this is either a model issue (nerfing low curvatures) or a specific steering rack behaviour
  v_points = np.array([20.0, 40.0])
  boost_values = np.array([1.5, 2.1]) # increase boost amplitude with speed
  boost = float(np.interp(v_ego, v_points, boost_values))
  steepness_values = np.array([5000.0, 3200.0]) # increase boost area with speed
  steepness = float(np.interp(v_ego, v_points, steepness_values))

  abs_kappa = abs(kappa)
  boost_factor = 1.0 + (boost - 1.0) / (1 + np.exp(steepness * (abs_kappa - kappa_thresh)))

  return np.sign(kappa) * abs_kappa * boost_factor


def map_speed_to_acc_tempolimit(v_ms):
  acc_tempolimit_kph = { # DBC Mapping
    1: 5, 2: 7, 3: 10, 4: 15, 5: 20, 6: 25, 7: 30, 8: 35,
    9: 40, 10: 45, 11: 50, 12: 55, 13: 60, 14: 65, 15: 70,
    16: 75, 17: 80, 18: 85, 19: 90, 20: 95, 21: 100, 22: 110,
    23: 120, 24: 130, 25: 140, 26: 150, 27: 160, 28: 200,
    30: 250
  }

  v_kph = int(round(v_ms * CV.MS_TO_KPH))
  acc_value = 0

  for val, limit in sorted(acc_tempolimit_kph.items()):
    if v_kph >= limit:
      acc_value = val
    else:
      break

  return acc_value

def get_acc_warning_meb(self, acc_hud):
  # this works as long our radar does not fault while using OP
  if (acc_hud["ACC_Status_ACC"] in (3, 4) # ACC active or in override mode
      and acc_hud["ACC_EGO_Fahrzeug"] == 2 # a warning for the lead car is active
      and acc_hud["ACC_Optischer_Fahrerhinweis"] != 0 # there is an optical warning
      and acc_hud["ACC_Akustischer_Fahrerhinweis"] != 0 # there is a sound warning
      and acc_hud["ACC_Display_Prio"] == 0): # this warning has highest priority
    return True
  return False

class MultiplicativeUnwindPID(PIDController):
  def __init__(self, k_p, k_i, k_f=0., k_d=0., pos_limit=1e308, neg_limit=-1e308, rate=100, min_cmd=1e-10, ki_red_time=1.0):
    super().__init__(k_p, k_i, k_f=k_f, k_d=k_d, pos_limit=pos_limit, neg_limit=neg_limit, rate=rate)
    self.min_cmd = abs(min_cmd)
    self.ki_red_time = float(ki_red_time)
    self.rate = rate
    self.override_prev = False
    self.i_unwind_factor = 1.0

  def _calc_unwind_factor(self, override):
    if not override or self.override_prev:
      return
    if self.ki_red_time <= 0.0:
      self.i_unwind_factor = 1.0
      return
    if abs(self.i) <= self.min_cmd:
      self.i_unwind_factor = 0.0
      return
    steps = max(int(self.ki_red_time * self.rate), 1)
    factor = (self.min_cmd / abs(self.i)) ** (1.0 / steps)
    self.i_unwind_factor = min(factor, 1.0)

  def update(self, error, error_rate=0.0, speed=0.0, override=False, feedforward=0., freeze_integrator=False):
    self.speed = speed

    self.p = float(error) * self.k_p
    self.f = feedforward * self.k_f
    self.d = error_rate * self.k_d

    if override:
      self._calc_unwind_factor(override)
      self.i *= self.i_unwind_factor
      if abs(self.i) < self.min_cmd:
        self.i = 0.0
    else:
      if not freeze_integrator:
        self.i = self.i + error * self.k_i * self.i_rate

        # Clip i to prevent exceeding control limits
        control_no_i = self.p + self.d + self.f
        control_no_i = np.clip(control_no_i, self.neg_limit, self.pos_limit)
        self.i = np.clip(self.i, self.neg_limit - control_no_i, self.pos_limit - control_no_i)

    control = self.p + self.i + self.d + self.f

    self.control = np.clip(control, self.neg_limit, self.pos_limit)
    self.override_prev = override
    return self.control

class CurvatureSmoother():
  def __init__(self, speed_bp, tau_v, dt):
    self.speed_bp = speed_bp
    self.tau_v = tau_v
    self.dt = dt
    self.stage_1 = 0.
    self.stage_2 = 0.

  def reset(self, curvature):
    self.stage_1 = curvature
    self.stage_2 = curvature

  def update(self, curvature, v_ego):
    tau = float(np.interp(v_ego, self.speed_bp, self.tau_v))
    alpha = 1. - math.exp(-self.dt / tau)
    self.stage_1 += alpha * (curvature - self.stage_1)
    self.stage_2 += alpha * (self.stage_1 - self.stage_2)
    return self.stage_2


class LatControlCurvature():
  def __init__(self, pid_params, limit, rate):
    self.pid = MultiplicativeUnwindPID((pid_params.kpBP, pid_params.kpV),
                                       (pid_params.kiBP, pid_params.kiV),
                                       k_f=pid_params.kf, pos_limit=limit, neg_limit=-limit,
                                       rate=rate, min_cmd=6.7e-6, ki_red_time=2.0)
  def reset(self):
    self.pid.reset()

  def update(self, CS, CC, desired_curvature):
    actual_curvature_vm    = CC.currentCurvature # includes roll
    speed_floor            = max(CS.vEgo, 0.1)
    yaw_rate_pose          = CC.angularVelocity[2] if len(CC.angularVelocity) > 2 else actual_curvature_vm * speed_floor
    actual_curvature_pose  = yaw_rate_pose / speed_floor
    actual_curvature       = np.interp(CS.vEgo, [2.0, 5.0], [actual_curvature_vm, actual_curvature_pose])
    desired_curvature_corr = desired_curvature - CC.rollCompensation
    error                  = desired_curvature - actual_curvature
    freeze_integrator      = CC.steerLimited or CS.vEgo < 5
    output_curvature       = self.pid.update(error, feedforward=desired_curvature_corr, speed=CS.vEgo,
                                             freeze_integrator=freeze_integrator, override=CS.steeringPressed)
    return output_curvature
