"""Comfort policy shared by native jerk actuation and the acceleration PID.

update() runs after LongControl selects its state and returns the PID bounds.
finish() computes the native deviation band from the final clipped command.
All timers advance at the control rate, including while an emergency bypasses
the ramp. Normal software actuation has only the car's physical bounds.
"""
import math

from iqdbc.car import structs
from iqpilot.common.realtime import DT_CTRL
from iqpilot.selfdrive.car.cruise import V_CRUISE_UNSET

RAMP_START = 0.8
RAMP_RATE = 1.05
RAMP_END = 5.0
RAMP_DURATION = (RAMP_END - RAMP_START) / RAMP_RATE
SPEED_CHANGE_KPH = 0.5
SETTLE_TIME = 0.5
# Stock Golf ACC permits up to 0.375 below target during sustained steady
# cruise. Ports cap each physical side to their observed stock envelope.
STEADY_BAND = 0.375
BAND_WIDEN_RATE = 0.24
# Native jerk while the driver overrides with gas: a high jerk makes the car cut
# its own acceleration abruptly when openpilot's request drops under the pedal.
OVERRIDE_JERK = 0.5
# A lead counts as braking us only when the target drops by at least this much
# between plan updates (~1 m/s^3 at 20 Hz) or the command lags the target by
# CATCH_UP_MARGIN. Plan ripple while following must not close the comfort band.
LEAD_DROP = 0.05
CATCH_UP_MARGIN = 0.1


def clip(value, lower, upper):
  return min(max(value, lower), upper)


class LongitudinalComfort:
  def __init__(self, actuation: structs.IQCarParams.LongitudinalActuation, dt=DT_CTRL):
    self.native_jerk = actuation.jerkControl == 'nativeAsymmetric'
    self.native_band = actuation.comfortBand == 'asymmetric'
    self.max_up = actuation.maxJerkUp or RAMP_END
    self.max_down = actuation.maxJerkDown or RAMP_END
    self.dt = dt
    self.reset()

  def reset(self):
    self.control = structs.IQCarControl.LongitudinalComfort()
    self._active = False
    self._ramp_time = RAMP_DURATION
    self._speed_baseline = None
    self._episode_speed = None
    self._edit_episode = False
    self._edit_quiet_time = 0.0
    self._suppressed = False
    self._resume_time = 0.0
    self._last_target = None
    self._brake_hold = 0.0
    self._steady_time = 0.0
    self._above = self._below = STEADY_BAND

  def _speed_change(self, v_cruise):
    valid_speed = math.isfinite(v_cruise) and 0 < v_cruise < V_CRUISE_UNSET
    changed = False
    if valid_speed:
      if self._speed_baseline is None:
        self._speed_baseline = self._episode_speed = v_cruise
      elif abs(v_cruise - self._episode_speed) >= SPEED_CHANGE_KPH:
        changed = not self._edit_episode and abs(v_cruise - self._speed_baseline) >= SPEED_CHANGE_KPH
        self._edit_episode = True
        self._episode_speed = v_cruise
        self._edit_quiet_time = 0.0
        return changed
    if self._edit_episode:
      self._edit_quiet_time += self.dt
      if self._edit_quiet_time + 1e-9 >= SETTLE_TIME:
        self._edit_episode = False
        self._speed_baseline = self._episode_speed
    return changed

  def update(self, *, enabled, active, car_state, a_target, should_stop, stopping, previous_command,
             physical_limits, driver_override=False, fcw=False, plan_source='cruise'):
    if not enabled:
      self.reset()
      return physical_limits

    # Gas can leave longActive true on some ports. Never constrain the driver.
    if not active or driver_override or car_state.gasPressed or car_state.brakePressed:
      self.reset()
      self.control.valid = True
      gas = driver_override or car_state.gasPressed
      jerk_up, jerk_down = (OVERRIDE_JERK, OVERRIDE_JERK) if gas and self.native_jerk else (self.max_up, self.max_down)
      self.control.jerkUp, self.control.jerkDown = jerk_up, jerk_down
      return physical_limits

    engaged = not self._active
    self._active = True
    if engaged or self._speed_change(car_state.vCruise):
      self._ramp_time = 0.0
    if engaged:
      # The first command must start from measured acceleration, not the off
      # state's zero output. Speed changes keep the previous command instead.
      self._speed_change(car_state.vCruise)
      self._above = self._below = STEADY_BAND
      previous_command = clip(car_state.aEgo if math.isfinite(car_state.aEgo) else 0.0, *physical_limits)

    lead = str(plan_source) in ('lead0', 'lead1', 'lead2')
    decreasing = self._last_target is not None and a_target <= self._last_target - LEAD_DROP
    self._brake_hold = max(0.0, self._brake_hold - self.dt)
    if (lead and decreasing) or a_target <= -1.5:
      self._brake_hold = SETTLE_TIME
    down_bypass = self._brake_hold > 1e-9 or (lead and a_target < previous_command - CATCH_UP_MARGIN)
    self._last_target = a_target

    if should_stop or stopping or car_state.vEgo < 2.0:
      self._suppressed = True
      self._resume_time = 0.0
    elif self._suppressed:
      self._resume_time = self._resume_time + self.dt if car_state.vEgo > 2.5 else 0.0
      if self._resume_time + 1e-9 >= SETTLE_TIME:
        self._suppressed = False
    if self._suppressed:
      # Departure and hold acquisition must never wait for a comfort ramp.
      # Discard the interrupted ramp rather than resuming it after departure.
      self._ramp_time = RAMP_DURATION

    ramping = self._ramp_time + 1e-9 < RAMP_DURATION
    jerk = min(RAMP_END, RAMP_START + RAMP_RATE * self._ramp_time)
    self._ramp_time += self.dt
    self.control = structs.IQCarControl.LongitudinalComfort(
      valid=True,
      jerkUp=self.max_up if fcw or not ramping else min(jerk, self.max_up),
      jerkDown=self.max_down if fcw or down_bypass or not ramping else min(jerk, self.max_down),
      comfortEnabled=self.native_band and not (fcw or down_bypass or self._suppressed),
      bypassJerkLimit=bool(fcw),
      bypassJerkDownLimit=bool(down_bypass),
    )

    if self.native_jerk or not ramping or fcw:
      return physical_limits
    lower, upper = physical_limits
    if not down_bypass:
      lower = max(lower, previous_command - self.control.jerkDown * self.dt)
    upper = min(upper, previous_command + self.control.jerkUp * self.dt)
    # Moving physical bounds can exclude the reference entirely. The physical
    # bounds take precedence over comfort on both sides of an empty intersection.
    return (lower, upper) if lower <= upper else physical_limits

  def finish(self, a_target, a_command, a_ego):
    if not self.control.comfortEnabled:
      self._above = self._below = self._steady_time = 0.0
    else:
      self._steady_time = self._steady_time + self.dt if math.isfinite(a_ego) and abs(a_ego) < 0.15 else 0.0
      demand = clip((max(abs(a_target), abs(a_command)) - 0.10) / 0.70, 0.0, 1.0)
      above = STEADY_BAND + (0.10 - STEADY_BAND) * demand
      below = STEADY_BAND + (0.02 - STEADY_BAND) * demand
      sign = a_target if abs(a_target) > 0.10 else a_command
      if sign < 0:
        above, below = below, above
      widening = BAND_WIDEN_RATE * self.dt if self._steady_time + 1e-9 >= SETTLE_TIME else 0.0
      self._above = min(above, self._above + widening)
      self._below = min(below, self._below + widening)
    self.control.accelDeviationAbove = self._above
    self.control.accelDeviationBelow = self._below
