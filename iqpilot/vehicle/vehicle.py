"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos
"""
from iqpilot.cereal import log
from iqdbc.car import structs
from iqdbc.car.chrysler.values import RAM_DT
from iqpilot.selfdrive.selfdrived.events import Events
from iqpilot.selfdrive.selfdrived.iq_events import IQEvents

_E = log.OnroadEvent.EventName
_GEAR = structs.CarState.GearShifter
_RAM_STEER_FLOOR_MS = 14.5
_EPS_TIMER_EXPIRY_S = 340.0


class VehicleEvents:
  def __init__(self, CP: structs.CarParams, CP_IQ: structs.IQCarParams):
    self.CP = CP
    self.CP_IQ = CP_IQ
    self._steer_floor_held = False

  def update(self, CS: structs.CarState, events: Events, CS_IQ=None) -> IQEvents:
    handler = self._BRAND_HANDLERS.get(self.CP.brand)
    if handler is not None:
      handler(self, CS, events, CS_IQ)
    return IQEvents()

  def _ram_steer_floor(self, CS: structs.CarState, events: Events, _CS_IQ) -> None:
    if self.CP.carFingerprint in RAM_DT:
      events.remove(_E.belowSteerSpeed)
      if CS.vEgo >= self.CP.minEnableSpeed:
        self._steer_floor_held = False
      if self.CP.minEnableSpeed >= _RAM_STEER_FLOOR_MS and CS.gearShifter != _GEAR.drive:
        self._steer_floor_held = True
    if self._steer_floor_held:
      events.add(_E.belowSteerSpeed)

  def _toyota_interceptor_resume(self, CS: structs.CarState, events: Events, _CS_IQ) -> None:
    if not self.CP.openpilotLongitudinalControl:
      return
    stopped_and_coasting = CS.cruiseState.standstill and not CS.brakePressed
    if stopped_and_coasting and self.CP_IQ.enableGasInterceptor:
      events.remove(_E.resumeRequired)

  def _volkswagen_eps_timer(self, CS: structs.CarState, events: Events, CS_IQ) -> None:
    if CS_IQ is None:
      return
    if CS_IQ.hcaTimerAlert:
      events.add(_E.steerTimeLimit)
    session_s = max(CS_IQ.epsSessionSec, CS_IQ.hcaSessionSec)
    if CS.steerFaultTemporary and session_s >= _EPS_TIMER_EXPIRY_S:
      events.remove(_E.steerTempUnavailableSilent)
      events.add(_E.steerTempUnavailable)

  _BRAND_HANDLERS = {
    'chrysler': _ram_steer_floor,
    'toyota': _toyota_interceptor_resume,
    'volkswagen': _volkswagen_eps_timer,
  }
