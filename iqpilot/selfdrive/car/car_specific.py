from iqpilot.cereal import car, log
from iqdbc.car import DT_CTRL, structs
from iqdbc.car.car_helpers import interfaces
from iqdbc.car.interfaces import MAX_CTRL_SPEED
from iqdbc.car.toyota.values import ToyotaFlags
from iqdbc.car.volkswagen.values import VolkswagenFlags

from iqpilot.selfdrive.selfdrived.events import Events

ButtonType = structs.CarState.ButtonEvent.Type
GearShifter = structs.CarState.GearShifter
EventName = log.OnroadEvent.EventName
NetworkLocation = structs.CarParams.NetworkLocation

HARD_DECEL_ACCEL = -4.0
HARD_DECEL_FAULT_WINDOW_FRAMES = int(1.5 / DT_CTRL)
VW_TSK_FLAGS = VolkswagenFlags.MEB | VolkswagenFlags.MQB_EVO


class CarSpecificEvents:
  def __init__(self, CP: structs.CarParams):
    self.CP = CP

    self.steering_unpressed = 0
    self.low_speed_alert = False
    self.no_steer_warning = False
    self.silent_steer_warning = True
    self.hard_decel_age = HARD_DECEL_FAULT_WINDOW_FRAMES + 1
    self.acc_fault_from_hard_decel = False

  def _acc_fault_from_hard_decel(self, CS: car.CarState, CS_prev: car.CarState) -> bool:
    if self.CP.brand != 'volkswagen' or not (getattr(self.CP, 'flags', 0) & VW_TSK_FLAGS):
      return False
    if CS.brakePressed and getattr(CS, 'aEgo', 0.) <= HARD_DECEL_ACCEL:
      self.hard_decel_age = 0
    else:
      self.hard_decel_age += 1
    faulted = CS.accFaulted or getattr(CS, 'cruiseFaultLateralMode', False)
    was_faulted = CS_prev.accFaulted or getattr(CS_prev, 'cruiseFaultLateralMode', False)
    if not faulted:
      self.acc_fault_from_hard_decel = False
    elif not was_faulted:
      self.acc_fault_from_hard_decel = self.hard_decel_age <= HARD_DECEL_FAULT_WINDOW_FRAMES
    return self.acc_fault_from_hard_decel

  def update(self, CS: car.CarState, CS_prev: car.CarState, CC: car.CarControl):
    if self.CP.brand in ('body', 'mock'):
      return Events()

    events = self.create_common_events(CS, CS_prev)

    if self.CP.brand == 'chrysler':
      if self.CP.minSteerSpeed > 0. and CS.vEgo < (self.CP.minSteerSpeed + 0.5):
        self.low_speed_alert = True
      elif CS.vEgo > (self.CP.minSteerSpeed + 1.):
        self.low_speed_alert = False
      if self.low_speed_alert:
        events.add(EventName.belowSteerSpeed)

    elif self.CP.brand == 'honda':
      if self.CP.pcmCruise and CS.vEgo < self.CP.minEnableSpeed:
        events.add(EventName.belowEngageSpeed)

      if self.CP.pcmCruise:
        if CS.cruiseState.enabled and not CS_prev.cruiseState.enabled:
          events.add(EventName.pcmEnable)
        elif not CS.cruiseState.enabled and (CC.actuators.accel >= 0. or not self.CP.openpilotLongitudinalControl):
          if CS.vEgo < self.CP.minEnableSpeed + 2.:
            events.add(EventName.speedTooLow)
          else:
            events.add(EventName.cruiseDisabled)
      if self.CP.minEnableSpeed > 0 and CS.vEgo < 0.001:
        events.add(EventName.manualRestart)

    elif self.CP.brand == 'toyota':
      if self.CP.openpilotLongitudinalControl:
        if CS.cruiseState.standstill and not CS.brakePressed and (CC.cruiseControl.resume or self.CP.flags & ToyotaFlags.HYBRID.value):
          events.add(EventName.resumeRequired)
        if CS.vEgo < self.CP.minEnableSpeed:
          events.add(EventName.belowEngageSpeed)
          if CC.actuators.accel > 0.3:
            events.add(EventName.speedTooLow)
          if CS.vEgo < 0.001:
            events.add(EventName.manualRestart)

    elif self.CP.brand == 'gm':
      if CS.vEgo < self.CP.minEnableSpeed and not (CS.standstill and CS.brake >= 20 and
                                                   self.CP.networkLocation == NetworkLocation.fwdCamera):
        events.add(EventName.belowEngageSpeed)
      if CS.cruiseState.standstill:
        events.add(EventName.resumeRequired)

    elif self.CP.brand == 'volkswagen':
      if self.CP.openpilotLongitudinalControl:
        if CS.vEgo < self.CP.minEnableSpeed + 0.5:
          events.add(EventName.belowEngageSpeed)
        if CC.enabled and CS.vEgo < self.CP.minEnableSpeed:
          events.add(EventName.speedTooLow)

    return events

  def create_common_events(self, CS: structs.CarState, CS_prev: car.CarState):
    events = Events()

    CI = interfaces[self.CP.carFingerprint]
    # TODO: cleanup the honda-specific logic
    pcm_enable = self.CP.pcmCruise and self.CP.brand != 'honda'
    # TODO: on some hyundai cars, the cancel button is also the pause/resume button,
    # so only use it for cancel when running openpilot longitudinal
    allow_button_cancel = self.CP.brand != 'hyundai'

    if CS.doorOpen:
      events.add(EventName.doorOpen)
    if CS.seatbeltUnlatched:
      events.add(EventName.seatbeltNotLatched)
    if CS.gearShifter != GearShifter.drive and CS.gearShifter not in CI.DRIVABLE_GEARS:
      events.add(EventName.wrongGear)
    if CS.gearShifter == GearShifter.reverse:
      events.add(EventName.reverseGear)
    if not CS.cruiseState.available and not getattr(CS, 'cruiseFaultLateralMode', False):
      events.add(EventName.wrongCarMode)
    if CS.espDisabled:
      events.add(EventName.espDisabled)
    if CS.espActive:
      events.add(EventName.espActive)
    if CS.stockFcw:
      events.add(EventName.stockFcw)
    if CS.stockAeb:
      events.add(EventName.stockAeb)
    if CS.stockLkas:
      events.add(EventName.stockLkas)
    if CS.vEgo > MAX_CTRL_SPEED:
      events.add(EventName.speedTooHigh)
    if CS.cruiseState.nonAdaptive:
      events.add(EventName.wrongCruiseMode)
    if CS.brakeHoldActive and self.CP.openpilotLongitudinalControl:
      events.add(EventName.brakeHold)
    if CS.parkingBrake:
      events.add(EventName.parkBrake)
    hard_decel_fault = self._acc_fault_from_hard_decel(CS, CS_prev)
    if getattr(CS, 'cruiseFaultLateralMode', False):
      if not getattr(CS_prev, 'cruiseFaultLateralMode', False):
        events.add(EventName.accFaultedHardDecelLateral if hard_decel_fault else EventName.cruiseFaultLateralAllowed)
    elif CS.accFaulted:
      events.add(EventName.accFaultedHardDecel if hard_decel_fault else EventName.accFaulted)
    if CS.steeringPressed:
      events.add(EventName.steerOverride)
    if CS.steeringDisengage and not CS_prev.steeringDisengage:
      events.add(EventName.steerDisengage)
    if CS.brakePressed and CS.standstill:
      events.add(EventName.preEnableStandstill)
    if CS.gasPressed:
      events.add(EventName.gasPressedOverride)
    if CS.vehicleSensorsInvalid:
      events.add(EventName.vehicleSensorsInvalid)
    if CS.invalidLkasSetting:
      events.add(EventName.invalidLkasSetting)
    if CS.lowSpeedAlert:
      events.add(EventName.belowSteerSpeed)
    if CS.buttonEnable:
      events.add(EventName.buttonEnable)

    for b in CS.buttonEvents:
      if b.type == ButtonType.cancel and (allow_button_cancel or not self.CP.pcmCruise):
        events.add(EventName.buttonCancel)

    self.steering_unpressed = 0 if CS.steeringPressed else self.steering_unpressed + 1
    if CS.steerFaultTemporary:
      if CS.steeringPressed and (not CS_prev.steerFaultTemporary or self.no_steer_warning):
        self.no_steer_warning = True
      else:
        self.no_steer_warning = False

        if self.silent_steer_warning or CS.standstill or self.steering_unpressed < int(1.5 / DT_CTRL):
          self.silent_steer_warning = True
          events.add(EventName.steerTempUnavailableSilent)
        else:
          events.add(EventName.steerTempUnavailable)
    else:
      self.no_steer_warning = False
      self.silent_steer_warning = False
    if CS.steerFaultPermanent:
      events.add(EventName.steerUnavailable)

    if pcm_enable:
      if CS.cruiseState.enabled and not CS_prev.cruiseState.enabled and not CS.blockPcmEnable:
        events.add(EventName.pcmEnable)
      elif not CS.cruiseState.enabled:
        events.add(EventName.pcmDisable)

    return events
