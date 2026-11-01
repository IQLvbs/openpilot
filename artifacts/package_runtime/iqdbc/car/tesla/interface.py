from iqdbc.car import Bus, get_safety_config, structs
from iqdbc.car.interfaces import CarInterfaceBase
from iqdbc.car.tesla.carcontroller import CarController
from iqdbc.car.tesla.carstate import CarState
from iqdbc.car.tesla.values import (TeslaSafetyFlags, TeslaFlags, TeslaLegacySafetyFlags, CANBUS, CAR, DBC, Ecu, LEGACY_CARS,
                                    LEGACY_HW1_CARS, LEGACY_HW2_CARS, is_legacy_das_steering)
from iqdbc.car.tesla.radar_interface import RadarInterface, RADAR_START_ADDR

from iqdbc.lvbs.car.tesla.values import TeslaFlagsIQ, TeslaSafetyFlagsIQ


class CarInterface(CarInterfaceBase):
  CarState = CarState
  CarController = CarController
  RadarInterface = RadarInterface

  @staticmethod
  def _get_params(ret: structs.CarParams, candidate, fingerprint, car_fw, alpha_long, is_release, docs) -> structs.CarParams:
    if candidate in LEGACY_CARS:
      return CarInterface._get_params_legacy(ret, candidate, fingerprint)

    ret.brand = "tesla"

    ret.safetyConfigs = [get_safety_config(structs.CarParams.SafetyModel.tesla)]

    ret.steerLimitTimer = 0.4
    ret.steerActuatorDelay = 0.1
    ret.steerAtStandstill = True

    ret.steerControlType = structs.CarParams.SteerControlType.angle

    # 2026+ Model Y (Juniper) runs a second gen HW4 stack: DAS_status is at 0x399 instead of 0x39b,
    # UI_warning (blinkers, buckle switch & doors) isn't sent at all, and DAS_settings is on the
    # party bus (it's only sent sporadically on the autopilot party bus, too rarely to parse)
    if 0x399 in fingerprint[CANBUS.autopilot_party]:
      ret.flags |= TeslaFlags.HW4_GEN2.value
      ret.safetyConfigs[0].safetyParam |= TeslaSafetyFlags.HW4_GEN2.value

      # VCFRONT_lighting (blinkers) and SeatBeltStatus are only on the VEHICLE bus
      if 0x3F5 in fingerprint[CANBUS.vehicle]:
        ret.flags |= TeslaFlags.HW4_GEN2_VEHICLE_BUS.value

    # Model X and HW 2.5 vehicles are missing DAS_settings
    elif 0x293 not in fingerprint[CANBUS.autopilot_party]:
      ret.flags |= TeslaFlags.MISSING_DAS_SETTINGS.value

    # Radar support is intended to work for:
    # - Tesla Model 3 vehicles built approximately mid-2017 through early-2021
    # - Tesla Model Y vehicles built approximately mid-2020 through early-2021
    # - Vehicles equipped with the Continental ARS4-B radar (used on HW2 / HW2.5 / early HW3)
    # - Radar CAN lines must be tapped and connected to CAN bus 1 (normally not used for tesla vehicles)
    # On HW4 gen2 the VEHICLE bus is tapped into bus 1 and carries an unrelated RADAR_START_ADDR
    ret.radarUnavailable = (bool(ret.flags & TeslaFlags.HW4_GEN2) or RADAR_START_ADDR not in fingerprint[1]
                            or Bus.radar not in DBC[candidate])

    ret.alphaLongitudinalAvailable = True
    if alpha_long:
      ret.openpilotLongitudinalControl = True
      ret.safetyConfigs[0].safetyParam |= TeslaSafetyFlags.LONG_CONTROL.value

    legacy_das = any(fw.ecu == Ecu.eps and is_legacy_das_steering(candidate, fw.fwVersion) for fw in car_fw)
    if legacy_das:
      ret.flags |= TeslaFlags.LEGACY_DAS_STEERING.value
      ret.safetyConfigs[0].safetyParam |= TeslaSafetyFlags.LEGACY_DAS_STEERING.value

    ret.dashcamOnly = candidate in (CAR.TESLA_MODEL_X,)  # dashcam only, pending find invalidLkasSetting signal

    return ret

  @staticmethod
  def _get_params_legacy(ret: structs.CarParams, candidate, fingerprint) -> structs.CarParams:
    ret.brand = "tesla"

    if not any(0x201 in messages for messages in fingerprint.values()):
      ret.flags |= TeslaFlags.NO_SDM1.value

    safety_model = structs.CarParams.SafetyModel.teslaLegacy
    if candidate in LEGACY_HW1_CARS:
      ret.safetyConfigs = [
        get_safety_config(safety_model, int(TeslaLegacySafetyFlags.HW1)),
      ]
    else:
      hw = TeslaLegacySafetyFlags.HW2 if candidate in LEGACY_HW2_CARS else TeslaLegacySafetyFlags.HW3
      ret.safetyConfigs = [
        get_safety_config(safety_model, int(hw)),
        get_safety_config(safety_model, int(hw | TeslaLegacySafetyFlags.EXTERNAL_PANDA)),
      ]

    ret.steerLimitTimer = 0.4
    ret.steerActuatorDelay = 0.1
    ret.steerAtStandstill = True

    ret.steerControlType = structs.CarParams.SteerControlType.angle
    ret.radarUnavailable = candidate == CAR.TESLA_MODEL_S_HW2

    ret.alphaLongitudinalAvailable = True
    ret.openpilotLongitudinalControl = True
    ret.safetyConfigs[0].safetyParam |= TeslaLegacySafetyFlags.LONG_CONTROL.value

    return ret

  @staticmethod
  def _get_params_iq(stock_cp: structs.CarParams, ret: structs.IQCarParams, candidate, fingerprint: dict[int, dict[int, int]],
                     car_fw: list[structs.CarParams.CarFw], alpha_long: bool, is_release_iq: bool, docs: bool) -> structs.IQCarParams:

    if candidate in LEGACY_CARS:
      return ret

    stock_cp.enableBsm = True

    if candidate == CAR.TESLA_MODEL_X:
      stock_cp.dashcamOnly = False

    # Vehicle-bus messages can be slow enough to miss the initial capture window.
    # Accept either the established 0x3DF marker or the absolute odometer frame.
    vehicle_bus_seen = any(address in fingerprint[CANBUS.vehicle] for address in (0x3DF, 0x3B6))
    if vehicle_bus_seen:
      ret.flags |= TeslaFlagsIQ.HAS_VEHICLE_BUS.value
      ret.iqSafetyFlags |= TeslaSafetyFlagsIQ.HAS_VEHICLE_BUS

    return ret
