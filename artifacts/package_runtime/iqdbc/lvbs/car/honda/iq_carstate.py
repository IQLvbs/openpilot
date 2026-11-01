"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos
"""
from enum import StrEnum

from iqdbc.car import Bus, structs
from iqdbc.car.common.conversions import Conversions as CV
from iqdbc.car.honda.values import HONDA_BOSCH, HONDA_BOSCH_CANFD, HONDA_BOSCH_RADARLESS
from iqdbc.can.parser import CANParser
from iqdbc.lvbs.car.honda.iq_values import HondaFlagsIQ


class IQCarState:
  def __init__(self, CP, CP_IQ):
    self.CP = CP
    self.CP_IQ = CP_IQ

  def update(self, ret: structs.CarState, ret_iq: structs.IQCarState, can_parsers: dict[StrEnum, CANParser]) -> None:
    cp = can_parsers[Bus.pt]
    cp_cam = can_parsers[Bus.cam]

    if self.CP_IQ.flags & HondaFlagsIQ.HAS_CAMERA_MESSAGES:
      speed_bus = cp if (self.CP.carFingerprint in (HONDA_BOSCH - HONDA_BOSCH_RADARLESS - HONDA_BOSCH_CANFD)) else cp_cam
      speed_limit_raw = speed_bus.vl["CAMERA_MESSAGES"]["SPEED_LIMIT_SIGN"] % 32
      ret_iq.speedLimit = speed_limit_raw * 5.0 * CV.MPH_TO_MS if (1 <= speed_limit_raw <= 17) else 0.0

    if self.CP_IQ.flags & HondaFlagsIQ.NIDEC_HYBRID:
      ret.accFaulted = bool(cp.vl["HYBRID_BRAKE_ERROR"]["BRAKE_ERROR_1"] or cp.vl["HYBRID_BRAKE_ERROR"]["BRAKE_ERROR_2"])
      ret.stockAeb = bool(cp_cam.vl["BRAKE_COMMAND"]["AEB_REQ_1"] and cp_cam.vl["BRAKE_COMMAND"]["COMPUTER_BRAKE_HYBRID"] > 1e-5)

    if self.CP_IQ.flags & HondaFlagsIQ.HYBRID_ALT_BRAKEHOLD:
      ret.brakeHoldActive = cp.vl["BRAKE_HOLD_HYBRID_ALT"]["BRAKE_HOLD_ACTIVE"] == 1

    if self.CP_IQ.enableGasInterceptor:
      # Same threshold as panda, equivalent to 1e-5 with previous DBC scaling
      gas = (cp.vl["GAS_SENSOR"]["INTERCEPTOR_GAS"] + cp.vl["GAS_SENSOR"]["INTERCEPTOR_GAS2"]) // 2
      ret.gasPressed = gas > 492
