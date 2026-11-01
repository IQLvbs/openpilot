from iqdbc.car.common.conversions import Conversions as CV
from iqdbc.car.tesla.values import CANBUS, CarControllerParams, TeslaFlags
from iqdbc.car import DT_CTRL


class TeslaCAN:
  COAST_ENTRY_ACCEL = 0.03
  COAST_EXIT_ACCEL = 0.08
  IMMEDIATE_REGEN_ACCEL = -0.5
  COAST_JERK_LIMIT = 0.7

  def __init__(self, CP, packer):
    self.CP = CP
    self.packer = packer
    self.l_jerk = 0.0
    self.longitudinal_mode = "coast"

  def create_steering_control(self, angle, enabled, control_type):
    # control_type comes from torque_blend: ANGLE_CONTROL (1) normally, LANE_KEEP_ASSIST (2) when cooperative steering is enabled
    control_type = control_type if enabled else 0
    if self.CP.flags & TeslaFlags.LEGACY_DAS_STEERING:
      control_type <<= 1  # legacy firmware uses a 2-bit field, one bit up from the 3-bit signal

    values = {
      "DAS_steeringAngleRequest": -angle,
      "DAS_steeringHapticRequest": 0,
      "DAS_steeringControlType": control_type,
    }

    return self.packer.make_can_msg("DAS_steeringControl", CANBUS.party, values)

  def _update_longitudinal_mode(self, accel, active, cruise_override, stopping, cancel):
    if not active or cruise_override or cancel:
      self.longitudinal_mode = "coast"
    elif stopping or accel <= self.IMMEDIATE_REGEN_ACCEL:
      self.longitudinal_mode = "regen"
    elif self.longitudinal_mode == "drive":
      if accel <= -self.COAST_EXIT_ACCEL:
        self.longitudinal_mode = "regen"
      elif accel <= self.COAST_ENTRY_ACCEL:
        self.longitudinal_mode = "coast"
    elif self.longitudinal_mode == "regen":
      if accel >= self.COAST_EXIT_ACCEL:
        self.longitudinal_mode = "drive"
      elif accel >= -self.COAST_ENTRY_ACCEL:
        self.longitudinal_mode = "coast"
    elif accel >= self.COAST_EXIT_ACCEL:
      self.longitudinal_mode = "drive"
    elif accel <= -self.COAST_EXIT_ACCEL:
      self.longitudinal_mode = "regen"

  def create_longitudinal_command(self, acc_state, accel, counter, v_ego, active, cruise_override, set_speed_kph=None,
                                  comfort_mode=False, stopping=False, cancel=False):
    from iqdbc.car.interfaces import V_CRUISE_MAX

    set_speed = max(v_ego * CV.MS_TO_KPH, 0)
    if active:
      self.l_jerk = 0.0 if cruise_override else min(
        self.l_jerk + CarControllerParams.JERK_UP * DT_CTRL * 4,
        CarControllerParams.JERK_LIMIT_MAX,
      )
    else:
      self.l_jerk = 0.0

    accel_min = accel
    accel_max = max(accel, 0)
    jerk_min = CarControllerParams.JERK_LIMIT_MIN
    jerk_max = self.l_jerk

    if active and comfort_mode:
      self._update_longitudinal_mode(accel, active, cruise_override, stopping, cancel)
      if self.longitudinal_mode == "drive":
        set_speed = V_CRUISE_MAX if set_speed_kph is None else max(0.0, min(V_CRUISE_MAX, float(set_speed_kph)))
      elif self.longitudinal_mode == "regen":
        set_speed = 0
      else:
        accel_min = min(accel, 0)
        accel_max = max(accel, 0)
        jerk_min = -self.COAST_JERK_LIMIT
        jerk_max = self.COAST_JERK_LIMIT
    elif active:
      self._update_longitudinal_mode(accel, False, cruise_override, stopping, cancel)
      set_speed = 0 if accel < 0 else V_CRUISE_MAX
      if set_speed_kph is not None and accel >= 0:
        set_speed = max(0.0, min(V_CRUISE_MAX, float(set_speed_kph)))
    else:
      self._update_longitudinal_mode(accel, active, cruise_override, stopping, cancel)

    values = {
      "DAS_setSpeed": set_speed,
      "DAS_accState": acc_state,
      "DAS_aebEvent": 0,
      "DAS_jerkMin": jerk_min,
      "DAS_jerkMax": jerk_max,
      "DAS_accelMin": accel_min,
      "DAS_accelMax": accel_max,
      "DAS_controlCounter": counter,
    }
    return self.packer.make_can_msg("DAS_control", CANBUS.party, values)

  def create_steering_allowed(self):
    values = {
      "APS_eacAllow": 1,
    }

    return self.packer.make_can_msg("APS_eacMonitor", CANBUS.party, values)

  def create_body_controls(self, stock_dat, left_blinker, right_blinker, cancel=False):
    # Ride alongside the car's native DAS_bodyControls: copy the raw frame, override only the
    # turn-indicator bits, and stamp counter + 1 so our frame supersedes the stock one.
    dat = bytearray(stock_dat)
    if len(dat) < 8:
      dat.extend(b"\x00" * (8 - len(dat)))

    if left_blinker or right_blinker:
      turn_req = 1 if left_blinker else 2  # DAS_TURN_INDICATOR_LEFT / _RIGHT
      dat[1] = (dat[1] & ~0x07) | (turn_req & 0x07)
      dat[2] = (dat[2] & ~0x3C) | (1 << 2)  # DAS_ACTIVE_NAV_LANE_CHANGE
    elif cancel:
      dat[1] = (dat[1] & ~0x07) | 0x03  # DAS_TURN_INDICATOR_CANCEL
      dat[2] = (dat[2] & ~0x3C) | (4 << 2)  # DAS_CANCEL_LANE_CHANGE

    counter = (((dat[6] >> 4) + 1) & 0x0F)
    dat[6] = (dat[6] & ~0xF0) | (counter << 4)

    addr = 0x3E9
    checksum = (addr & 0xFF) + ((addr >> 8) & 0xFF)
    for i in range(7):
      checksum += dat[i]
    dat[7] = checksum & 0xFF

    return addr, bytes(dat), CANBUS.vehicle


def tesla_checksum(address: int, sig, d: bytearray) -> int:
  checksum = (address & 0xFF) + ((address >> 8) & 0xFF)
  checksum_byte = sig.start_bit // 8
  for i in range(len(d)):
    if i != checksum_byte:
      checksum += d[i]
  return checksum & 0xFF
