import numpy as np
from iqdbc.can import CANPacker
from iqdbc.car import Bus, structs
from iqdbc.car.lateral import apply_steer_angle_limits_vm
from iqdbc.car.interfaces import CarControllerBase
from iqdbc.car.tesla import TESLA_BLINKERS
from iqdbc.car.tesla.teslacan import TeslaCAN
from iqdbc.car.tesla.teslacan_legacy import TeslaCANLegacy
from iqdbc.car.tesla.values import CAR, CarControllerParams, LEGACY_CARS, LEGACY_HW1_CARS, get_legacy_canbus
from iqdbc.car.vehicle_model import VehicleModel
from iqpilot.selfdrive.car.enhanced_stock_longitudinal_control import get_set_speed_kph_from_params
from iqdbc.lvbs.car.tesla.torque_blend import TorqueBlendController
from iqdbc.lvbs.car.tesla.values import TeslaFlagsIQ


def get_safety_CP():
  # We use the TESLA_MODEL_Y platform for lateral limiting to match safety
  # A Model 3 at 40 m/s using the Model Y limits sees a <0.3% difference in max angle (from curvature factor)
  from iqdbc.car.tesla.interface import CarInterface
  return CarInterface.get_non_essential_params("TESLA_MODEL_Y")


class CarController(CarControllerBase):
  def __init__(self, dbc_names, CP, CP_IQ):
    CarControllerBase.__init__(self, dbc_names, CP, CP_IQ)
    self.coop_steer = TorqueBlendController()
    self.apply_angle_last = 0
    self.packer = CANPacker(dbc_names[Bus.party])
    self.tesla_can = TeslaCAN(CP, self.packer)
    self.apply_accel_last = 0.0

    # Vehicle model used for lateral limiting
    self.VM = VehicleModel(get_safety_CP())

    self.has_vehicle_bus = bool(CP_IQ.flags & TeslaFlagsIQ.HAS_VEHICLE_BUS)
    self.body_controls_counter_last = -1
    self.blinker_request_prev = False
    self.blinker_cancel_frame = 0

    self.legacy = CP.carFingerprint in LEGACY_CARS
    if self.legacy:
      from iqdbc.car.tesla.interface import CarInterface
      canbus = get_legacy_canbus(CP.carFingerprint)
      packers = {canbus.party: CANPacker(dbc_names[Bus.party]), canbus.powertrain: CANPacker(dbc_names[Bus.pt])}
      self.tesla_can_legacy = TeslaCANLegacy(packers, canbus)
      self.VM = VehicleModel(CarInterface.get_non_essential_params(CAR.TESLA_MODEL_S_HW3))

  def update_legacy(self, CC, CS):
    actuators = CC.actuators
    can_sends = []

    lat_active = CC.latActive and CS.hands_on_level < 3

    if self.frame % CarControllerParams.STEER_STEP == 0:
      self.apply_angle_last = apply_steer_angle_limits_vm(actuators.steeringAngleDeg, self.apply_angle_last, CS.out.vEgoRaw, CS.out.steeringAngleDeg,
                                                          lat_active, CarControllerParams, self.VM)
      counter = (self.frame // CarControllerParams.STEER_STEP) % 16
      can_sends.append(self.tesla_can_legacy.create_steering_control(counter, self.apply_angle_last, lat_active))

    if self.frame % 10 == 0 and self.CP.carFingerprint not in LEGACY_HW1_CARS:
      counter = (self.frame // 10) % 16
      can_sends.append(self.tesla_can_legacy.create_steering_allowed(counter))

    if self.frame % 4 == 0:
      state = 13 if CC.cruiseControl.cancel else 4
      accel = float(np.clip(actuators.accel, CarControllerParams.ACCEL_MIN, CarControllerParams.ACCEL_MAX))
      counter = (self.frame // 4) % 8
      can_sends.append(self.tesla_can_legacy.create_longitudinal_command(state, accel, counter, CS.out.vEgo, CC.longActive, CS.out.gasPressed))

    new_actuators = actuators.as_builder()
    new_actuators.steeringAngleDeg = self.apply_angle_last

    self.frame += 1
    return new_actuators, can_sends

  def update(self, CC, CC_IQ, CS, now_nanos):
    if self.legacy:
      return self.update_legacy(CC, CS)

    actuators = CC.actuators
    can_sends = []

    # Tesla EPS enforces disabling steering on heavy lateral override force.
    # When enabling in a tight curve, we wait until user reduces steering force to start steering.
    # Canceling is done on rising edge and is handled generically with CC.cruiseControl.cancel
    lat_active = CC.latActive and CS.hands_on_level < 3

    if self.frame % CarControllerParams.STEER_STEP == 0:
      # Angular rate limit based on speed
      self.apply_angle_last = apply_steer_angle_limits_vm(actuators.steeringAngleDeg, self.apply_angle_last, CS.out.vEgoRaw, CS.out.steeringAngleDeg,
                                                          lat_active, CarControllerParams, self.VM)

      can_sends.append(self.tesla_can.create_steering_control(*self.coop_steer.update(self.apply_angle_last, lat_active, self.CP_IQ, CS, self.VM)))

    if self.frame % 10 == 0:
      can_sends.append(self.tesla_can.create_steering_allowed())

    # Longitudinal control
    if self.CP.openpilotLongitudinalControl:
      if self.frame % 4 == 0:
        state = 13 if CC.cruiseControl.cancel else 4  # 4=ACC_ON, 13=ACC_CANCEL_GENERIC_SILENT
        accel = float(np.clip(actuators.accel, CarControllerParams.ACCEL_MIN, CarControllerParams.ACCEL_MAX))
        if not CC.longActive:
          accel = 0.
        self.apply_accel_last = accel
        cntr = (self.frame // 4) % 8
        set_speed_kph = get_set_speed_kph_from_params(CC_IQ.params)
        can_sends.append(self.tesla_can.create_longitudinal_command(state, accel, cntr, CS.out.vEgo, CC.longActive,
                                                                    CS.cruise_override, set_speed_kph=set_speed_kph,
                                                                    comfort_mode=bool(getattr(CC, "longComfortMode", False)),
                                                                    stopping=getattr(actuators, "longControlState", None) ==
                                                                    structs.CarControl.Actuators.LongControlState.stopping,
                                                                    cancel=CC.cruiseControl.cancel))

    else:
      # Increment counter so cancel is prioritized even without openpilot longitudinal
      if CC.cruiseControl.cancel:
        cntr = (CS.das_control["DAS_controlCounter"] + 1) % 8
        can_sends.append(self.tesla_can.create_longitudinal_command(13, 0, cntr, CS.out.vEgo, False, True))

    # Nav blinker control via DAS_bodyControls on the vehicle bus, phase-locked to the car's
    # counter. Cancel on the trailing edge since the body controller latches the signal.
    stock_dat = getattr(CS, 'das_body_controls_dat', b"")
    # FIXME: gate by FingerPrint
    if TESLA_BLINKERS and self.has_vehicle_bus and len(stock_dat) >= 8:
      left_blinker = CC.leftBlinker
      right_blinker = CC.rightBlinker

      driver_opposes = (left_blinker and CS.out.rightBlinker) or (right_blinker and CS.out.leftBlinker)
      if driver_opposes:
        left_blinker = right_blinker = False

      nav_requesting = left_blinker or right_blinker

      if self.blinker_request_prev and not nav_requesting and not driver_opposes:
        self.blinker_cancel_frame = self.frame + 150  # ~1.5 s
      self.blinker_request_prev = nav_requesting
      cancel = not nav_requesting and not driver_opposes and self.frame < self.blinker_cancel_frame

      body_counter = stock_dat[6] >> 4
      if body_counter != self.body_controls_counter_last:
        can_sends.append(self.tesla_can.create_body_controls(stock_dat, left_blinker, right_blinker, cancel))
      self.body_controls_counter_last = body_counter

    # TODO: HUD control
    new_actuators = actuators.as_builder()
    new_actuators.steeringAngleDeg = self.apply_angle_last
    new_actuators.accel = self.apply_accel_last
    new_actuators.curvature = float(self.coop_steer.debug_angle_desired_limited)  # debug
    new_actuators.torque = float(self.coop_steer.override_angle_accu)  # debug

    self.frame += 1
    return new_actuators, can_sends
