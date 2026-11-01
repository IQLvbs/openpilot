import copy
from iqdbc.can import CANDefine, CANParser
from iqdbc.car import Bus, create_button_events, structs
from iqdbc.car.common.conversions import Conversions as CV
from iqdbc.car.interfaces import CarStateBase
from iqdbc.car.tesla import TESLA_BLINKERS
from iqdbc.car.tesla.values import (DBC, CANBUS, CAR, GEAR_MAP, LEGACY_CARS, LEGACY_HW3_CARS, STEER_THRESHOLD, TeslaFlags,
                                    get_legacy_canbus)

from iqdbc.lvbs.car.tesla.iq_carstate import IQCarState
from iqdbc.lvbs.car.tesla.values import TeslaFlagsIQ, TeslaSafetyFlagsIQ
from iqpilot.common.params import Params

ButtonType = structs.CarState.ButtonEvent.Type
ACC_CANCEL_GENERIC_SILENT = 13


def stock_autosteer_invalid(CP, CP_IQ, autopilot_state: int, autosteer_enabled: int = 0) -> bool:
  if CP.flags & TeslaFlags.MISSING_DAS_SETTINGS or CP_IQ.iqSafetyFlags & TeslaSafetyFlagsIQ.FSD_VISUALIZATION:
    return False

  # HW4 gen2 never sends DAS_status, so the engaged state can't be read and only the setting is left
  if CP.flags & TeslaFlags.HW4_GEN2:
    return autosteer_enabled != 0

  return autopilot_state not in (0, 1, 2)


class CarState(CarStateBase, IQCarState):
  def __init__(self, CP, CP_IQ):
    CarStateBase.__init__(self, CP, CP_IQ)
    IQCarState.__init__(self, CP, CP_IQ)
    self.legacy = CP.carFingerprint in LEGACY_CARS
    self.can_define = CANDefine(DBC[CP.carFingerprint][Bus.party])

    self.summon = False
    self.summon_prev = False
    self.cruise_enabled_prev = False

    self.hands_on_level = 0
    self.prev_cancel_request = False
    self.das_control = None
    self.das_body_controls_dat = b""
    self.cruise_override = False

    if self.legacy:
      self.can_defines = {
        **self.can_define.dv,
        **CANDefine(DBC[CP.carFingerprint][Bus.pt]).dv,
        **CANDefine(DBC[CP.carFingerprint][Bus.chassis]).dv,
      }
      self.shifter_values = self.can_defines["DI_torque2"]["DI_gear"]
      self._odometer_store = None
    else:
      self.shifter_values = self.can_define.dv["DI_systemStatus"]["DI_gear"]
      try:
        from iqpilot.system.proprietary_runtime._verified_import import import_verified_module
        iq_lvbs_alc = import_verified_module("iqpilot_alc_private", "iqpilot_private.konn3kt.iqlvbs.alc")
        self._odometer_store = iq_lvbs_alc.create_vehicle_odometer_store(CP, Params())
      except Exception:
        self._odometer_store = None

  def update_summon_state(self, summon_state: str, cruise_enabled: bool):
    summon_now = summon_state in ("ACTIVE", "COMPLETE", "SELFPARK_STARTED")
    if summon_now and not self.summon_prev and not self.cruise_enabled_prev:
      self.summon = True
    if not summon_now:
      self.summon = False
    self.summon_prev = summon_now
    self.cruise_enabled_prev = cruise_enabled

  def update_legacy(self, can_parsers) -> tuple[structs.CarState, structs.IQCarState]:
    cp_party = can_parsers[Bus.party]
    cp_ap_party = can_parsers[Bus.ap_party]
    cp_pt = can_parsers[Bus.pt]
    cp_ap_pt = can_parsers[Bus.ap_pt]
    cp_chassis = can_parsers[Bus.chassis]
    ret = structs.CarState()
    ret_iq = structs.IQCarState()

    ret.vEgoRaw = cp_chassis.vl["ESP_B"]["ESP_vehicleSpeed"] * CV.KPH_TO_MS
    ret.vEgo, ret.aEgo = self.update_speed_kf(ret.vEgoRaw)

    ret.gasPressed = cp_pt.vl["DI_torque1"]["DI_pedalPos"] > 0

    ret.brakePressed = cp_chassis.vl["BrakeMessage"]["driverBrakeStatus"] == 2

    if self.CP.carFingerprint in LEGACY_HW3_CARS:
      epas_status = cp_party.vl["EPAS_sysStatus"]
    else:
      epas_status = cp_chassis.vl["EPAS_sysStatus"]
    self.hands_on_level = epas_status["EPAS_handsOnLevel"]
    ret.steeringAngleDeg = -epas_status["EPAS_internalSAS"]
    ret.steeringRateDeg = -cp_chassis.vl["STW_ANGLHP_STAT"]["StW_AnglHP_Spd"]
    ret.steeringTorque = -epas_status["EPAS_torsionBarTorque"]

    ret.steeringPressed = self.update_steering_pressed(abs(ret.steeringTorque) > STEER_THRESHOLD, 5)

    eac_status = self.can_defines["EPAS_sysStatus"]["EPAS_eacStatus"].get(int(epas_status["EPAS_eacStatus"]), None)
    ret.steerFaultPermanent = eac_status == "EAC_FAULT"
    ret.steerFaultTemporary = eac_status == "EAC_INHIBITED"

    eac_error_code = self.can_defines["EPAS_sysStatus"]["EPAS_eacErrorCode"].get(int(epas_status["EPAS_eacErrorCode"]), None)
    ret.steeringDisengage = self.hands_on_level >= 3 or (eac_status == "EAC_INHIBITED" and
                                                         eac_error_code == "EAC_ERROR_HIGH_ANGLE_RATE_SAFETY")

    cruise_state = self.can_defines["DI_state"]["DI_cruiseState"].get(int(cp_chassis.vl["DI_state"]["DI_cruiseState"]), None)
    speed_units = self.can_defines["DI_state"]["DI_speedUnits"].get(int(cp_chassis.vl["DI_state"]["DI_speedUnits"]), None)

    cruise_enabled = cruise_state in ("ENABLED", "STANDSTILL", "OVERRIDE", "PRE_FAULT", "PRE_CANCEL")

    ret.cruiseState.enabled = cruise_enabled
    if speed_units == "KPH":
      ret.cruiseState.speed = max(cp_chassis.vl["DI_state"]["DI_digitalSpeed"] * CV.KPH_TO_MS, 1e-3)
    elif speed_units == "MPH":
      ret.cruiseState.speed = max(cp_chassis.vl["DI_state"]["DI_digitalSpeed"] * CV.MPH_TO_MS, 1e-3)
    ret.cruiseState.available = cruise_state == "STANDBY" or ret.cruiseState.enabled
    ret.cruiseState.standstill = False
    ret.standstill = cruise_state == "STANDSTILL"
    ret.accFaulted = cruise_state == "FAULT"

    ret.gearShifter = GEAR_MAP[self.can_defines["DI_torque2"]["DI_gear"].get(int(cp_chassis.vl["DI_torque2"]["DI_gear"]), "DI_GEAR_INVALID")]

    doors = ["DOOR_STATE_FL", "DOOR_STATE_FR", "DOOR_STATE_RL", "DOOR_STATE_RR", "DOOR_STATE_FrontTrunk", "BOOT_STATE"]
    ret.doorOpen = any(self.can_defines["GTW_carState"][door].get(int(cp_chassis.vl["GTW_carState"][door]), "OPEN") == "OPEN" for door in doors)

    if self.CP.carFingerprint == CAR.TESLA_MODEL_X_HW1:
      ret.leftBlinker = cp_chassis.vl["STW_ACTN_RQ"]["TurnIndLvr_Stat"] == 1
      ret.rightBlinker = cp_chassis.vl["STW_ACTN_RQ"]["TurnIndLvr_Stat"] == 2
    else:
      ret.leftBlinker = cp_chassis.vl["GTW_carState"]["BC_indicatorLStatus"] == 1
      ret.rightBlinker = cp_chassis.vl["GTW_carState"]["BC_indicatorRStatus"] == 1

    if self.CP.flags & TeslaFlags.NO_SDM1:
      ret.seatbeltUnlatched = cp_chassis.vl["RCM_status"]["RCM_buckleDriverStatus"] != 1
    else:
      ret.seatbeltUnlatched = cp_chassis.vl["SDM1"]["SDM_bcklDrivStatus"] != 1

    ret.stockAeb = cp_ap_pt.vl["DAS_control"]["DAS_aebEvent"] == 1

    ret.stockLkas = cp_ap_party.vl["DAS_steeringControl"]["DAS_steeringControlType"] == 2

    self.das_control = copy.copy(cp_ap_pt.vl["DAS_control"])

    return ret, ret_iq

  def update(self, can_parsers) -> tuple[structs.CarState, structs.IQCarState]:
    if self.legacy:
      return self.update_legacy(can_parsers)

    cp_party = can_parsers[Bus.party]
    cp_ap_party = can_parsers[Bus.ap_party]
    ret = structs.CarState()
    ret_iq = structs.IQCarState()
    scale_speed = 1.01
    length = 0.11

  # Vehicle speed
    ret.vEgoRaw = cp_party.vl["DI_speed"]["DI_vehicleSpeed"] * CV.KPH_TO_MS
    ret.vEgo, ret.aEgo = self.update_speed_kf(ret.vEgoRaw)

    # Displayed speed
    ui_speed_units = self.can_define.dv["DI_speed"]["DI_uiSpeedUnits"].get(int(cp_party.vl["DI_speed"]["DI_uiSpeedUnits"]), None)
    if ui_speed_units == "DI_SPEED_KPH":
      ret.vEgoCluster = cp_party.vl["DI_speed"]["DI_uiSpeed"] * CV.KPH_TO_MS
    elif ui_speed_units == "DI_SPEED_MPH":
      ret.vEgoCluster = cp_party.vl["DI_speed"]["DI_uiSpeed"] * CV.MPH_TO_MS

    # Gas pedal
    ret.gasPressed = cp_party.vl["DI_systemStatus"]["DI_accelPedalPos"] > 0

    # Brake pedal
    ret.brake = 0
    ret.brakePressed = cp_party.vl["ESP_status"]["ESP_driverBrakeApply"] == 2

    # Steering wheel
    epas_status = cp_party.vl["EPAS3S_sysStatus"]
    self.hands_on_level = epas_status["EPAS3S_handsOnLevel"]
    ret.steeringAngleDeg = -epas_status["EPAS3S_internalSAS"]
    ret.steeringRateDeg = -cp_ap_party.vl["SCCM_steeringAngleSensor"]["SCCM_steeringAngleSpeed"]
    ret.steeringTorque = -epas_status["EPAS3S_torsionBarTorque"]
    ret.steeringTorqueEps = -epas_status["EPAS3S_steeringRackForce"] * length / self.CP.steerRatio

    # stock handsOnLevel uses >0.5 for 0.25s, but is too slow
    ret.steeringPressed = self.update_steering_pressed(abs(ret.steeringTorque) > STEER_THRESHOLD, 5)

    eac_status = self.can_define.dv["EPAS3S_sysStatus"]["EPAS3S_eacStatus"].get(int(epas_status["EPAS3S_eacStatus"]), None)
    ret.steerFaultPermanent = eac_status == "EAC_FAULT"
    ret.steerFaultTemporary = eac_status == "EAC_INHIBITED"

    # FSD disengages using union of handsOnLevel (slow overrides) and high angle rate faults (fast overrides, high speed)
    eac_error_code = self.can_define.dv["EPAS3S_sysStatus"]["EPAS3S_eacErrorCode"].get(int(epas_status["EPAS3S_eacErrorCode"]), None)
    ret.steeringDisengage = self.hands_on_level >= 3 or (eac_status == "EAC_INHIBITED" and
                                                         eac_error_code == "EAC_ERROR_HIGH_ANGLE_RATE_SAFETY")

    # Cruise state
    cruise_state = self.can_define.dv["DI_state"]["DI_cruiseState"].get(int(cp_party.vl["DI_state"]["DI_cruiseState"]), None)
    speed_units = self.can_define.dv["DI_state"]["DI_speedUnits"].get(int(cp_party.vl["DI_state"]["DI_speedUnits"]), None)

    summon_state = self.can_define.dv["DI_state"]["DI_autoparkState"].get(int(cp_party.vl["DI_state"]["DI_autoparkState"]), None)
    cruise_enabled = cruise_state in ("ENABLED", "STANDSTILL", "OVERRIDE", "PRE_FAULT", "PRE_CANCEL")
    self.cruise_override = cruise_state in ("OVERRIDE")
    self.update_summon_state(summon_state, cruise_enabled)

    # Match panda safety cruise engaged logic
    ret.cruiseState.enabled = cruise_enabled and not self.summon
    if speed_units == "KPH":
      ret.cruiseState.speedCluster = cp_party.vl["DI_state"]["DI_digitalSpeed"] * CV.KPH_TO_MS
    elif speed_units == "MPH":
      ret.cruiseState.speedCluster = cp_party.vl["DI_state"]["DI_digitalSpeed"] * CV.MPH_TO_MS
    ret.cruiseState.speed = max(ret.cruiseState.speedCluster / scale_speed, 1e-3)
    ret.cruiseState.available = cruise_state == "STANDBY" or ret.cruiseState.enabled
    ret.cruiseState.standstill = False  # This needs to be false, since we can resume from stop without sending anything special
    ret.standstill = cp_party.vl["ESP_B"]["ESP_vehicleStandstillSts"] == 1
    ret.accFaulted = cruise_state == "FAULT"

    # Gear
    ret.gearShifter = GEAR_MAP[self.can_define.dv["DI_systemStatus"]["DI_gear"].get(int(cp_party.vl["DI_systemStatus"]["DI_gear"]), "DI_GEAR_INVALID")]

    # HW4 gen2 doesn't send UI_warning, and moved DAS_status from 0x39b to 0x399
    if self.CP.flags & TeslaFlags.HW4_GEN2:
      # Doors
      ret.doorOpen = cp_party.vl["VehicleStatus"]["allDoorsClosed"] == 0

      # Blindspot
      ret.leftBlindspot = cp_ap_party.vl["DAS_statusGen2"]["DAS_blindSpotRearLeft"] != 0
      ret.rightBlindspot = cp_ap_party.vl["DAS_statusGen2"]["DAS_blindSpotRearRight"] != 0

      # Blinkers and seatbelt, only available with the VEHICLE bus tapped
      if self.CP.flags & TeslaFlags.HW4_GEN2_VEHICLE_BUS:
        cp_vehicle = can_parsers[Bus.adas]
        ret.leftBlinker = cp_vehicle.vl["VCFRONT_lighting"]["VCFRONT_indicatorLeftRequest"] != 0
        ret.rightBlinker = cp_vehicle.vl["VCFRONT_lighting"]["VCFRONT_indicatorRightRequest"] != 0
        ret.seatbeltUnlatched = cp_vehicle.vl["SeatBeltStatus"]["driverBuckleStatus"] != 1
    else:
      # Doors
      ret.doorOpen = cp_party.vl["UI_warning"]["anyDoorOpen"] == 1

      # Blinkers
      ret.leftBlinker = cp_party.vl["UI_warning"]["leftBlinkerBlinking"] in (1, 2)
      ret.rightBlinker = cp_party.vl["UI_warning"]["rightBlinkerBlinking"] in (1, 2)

      # Seatbelt
      ret.seatbeltUnlatched = cp_party.vl["UI_warning"]["buckleStatus"] != 1

      # Blindspot
      ret.leftBlindspot = cp_ap_party.vl["DAS_status"]["DAS_blindSpotRearLeft"] != 0
      ret.rightBlindspot = cp_ap_party.vl["DAS_status"]["DAS_blindSpotRearRight"] != 0

    # AEB
    ret.stockAeb = cp_ap_party.vl["DAS_control"]["DAS_aebEvent"] == 1

    # stalkless cars route the steering wheel cancel request through the AP computer
    cancel_request = int(cp_ap_party.vl["DAS_control"]["DAS_accState"]) == ACC_CANCEL_GENERIC_SILENT
    ret.buttonEvents = create_button_events(cancel_request, self.prev_cancel_request, {True: ButtonType.cancel}, unpressed_btn=False)
    self.prev_cancel_request = cancel_request

    # LKAS
    steer_control_type = int(cp_ap_party.vl["DAS_steeringControl"]["DAS_steeringControlType"])
    if self.CP.flags & TeslaFlags.LEGACY_DAS_STEERING:
      steer_control_type >>= 1  # legacy firmware uses a 2-bit field, one bit up from the 3-bit signal
    ret.stockLkas = steer_control_type == 2  # LANE_KEEP_ASSIST

    # Stock Autosteer should be disengaged (includes FSD)
    # TODO: find for TESLA_MODEL_X and HW2.5 vehicles
    # a CANParser registers a message the first time it's read and then reports the bus invalid
    # until that message arrives, so each branch may only touch what its own hardware sends
    if self.CP.flags & TeslaFlags.HW4_GEN2:
      ret.invalidLkasSetting = stock_autosteer_invalid(self.CP, self.CP_IQ, 0,
                                                       int(cp_party.vl["DAS_settings"]["DAS_autosteerEnabled"]))
    else:
      ret.invalidLkasSetting = stock_autosteer_invalid(self.CP, self.CP_IQ,
                                                       int(cp_ap_party.vl["DAS_status"]["DAS_autopilotState"]))

    # Buttons # ToDo: add Gap adjust button

    # Messages needed by carcontroller
    self.das_control = copy.copy(cp_ap_party.vl["DAS_control"])

    # Raw stock DAS_bodyControls bytes (bus 2), used to ride the blinker on the vehicle bus.
    # FIXME: gate by FingerPrint
    if TESLA_BLINKERS and Bus.cam in can_parsers:
      self.das_body_controls_dat = bytes(can_parsers[Bus.cam].dat.get(0x3E9, b""))

    IQCarState.update(self, ret, ret_iq, can_parsers)
    if ret.odometer > 0.0 and self._odometer_store is not None:
      ret.odometer = self._odometer_store.record(ret.odometer) or 0.0

    return ret, ret_iq

  @staticmethod
  def get_can_parsers(CP, CP_IQ):
    if CP.carFingerprint in LEGACY_CARS:
      canbus = get_legacy_canbus(CP.carFingerprint)
      return {
        Bus.party: CANParser(DBC[CP.carFingerprint][Bus.party], [], canbus.party),
        Bus.ap_party: CANParser(DBC[CP.carFingerprint][Bus.party], [], canbus.autopilot_party),
        Bus.pt: CANParser(DBC[CP.carFingerprint][Bus.pt], [], canbus.powertrain),
        Bus.ap_pt: CANParser(DBC[CP.carFingerprint][Bus.pt], [], canbus.autopilot_powertrain),
        Bus.chassis: CANParser(DBC[CP.carFingerprint][Bus.chassis], [], canbus.chassis),
      }

    parsers = {
      Bus.party: CANParser(DBC[CP.carFingerprint][Bus.party], [], CANBUS.party),
      Bus.ap_party: CANParser(DBC[CP.carFingerprint][Bus.party], [], CANBUS.autopilot_party),
      **IQCarState.get_parser(CP, CP_IQ),
    }
    if CP.flags & TeslaFlags.HW4_GEN2_VEHICLE_BUS and Bus.adas not in parsers:
      parsers[Bus.adas] = CANParser(DBC[CP.carFingerprint][Bus.adas], [], CANBUS.vehicle)
    # Stock DAS_bodyControls from the AP bus (bus 2) for the nav blinker.
    if TESLA_BLINKERS and CP_IQ.flags & TeslaFlagsIQ.HAS_VEHICLE_BUS and Bus.adas in DBC[CP.carFingerprint]:
      parsers[Bus.cam] = CANParser(DBC[CP.carFingerprint][Bus.adas], [("DAS_bodyControls", 2)], CANBUS.autopilot_party)
    return parsers
