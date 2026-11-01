"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos
"""

from types import SimpleNamespace

from iqpilot.cereal import custom
from iqdbc.car import structs
from iqdbc.car.hyundai.values import HyundaiFlags, HyundaiFlagsIQ
from iqpilot.sab.behavior import SteeringAssistanceBehavior
from iqpilot.selfdrive.selfdrived.iq_events import IQEvents
from iqpilot.selfdrive.selfdrived.events import Events
from iqpilot.cereal import log


ButtonType = structs.CarState.ButtonEvent.Type
EventName = log.OnroadEvent.EventName
EventNameIQ = custom.IQOnroadEvent.EventName
GuidanceState = custom.AlwaysOnLateral.AlwaysOnLateralState


class MockParams:
  def __init__(self, main_cruise_allowed: bool = False, aol_enabled: bool = True,
               pause_on_steering_override: bool = False):
    self.main_cruise_allowed = main_cruise_allowed
    self.aol_enabled = aol_enabled
    self.pause_on_steering_override = pause_on_steering_override

  def get_bool(self, key: str) -> bool:
    return {
      "AolEnabled": self.aol_enabled,
      "AolMainCruiseAllowed": self.main_cruise_allowed,
      "AolUnifiedEngagementMode": False,
      "AolPauseOnSteeringOverride": self.pause_on_steering_override,
      "JoystickDebugMode": False,
    }.get(key, False)

  def get(self, key: str, return_default: bool = False):
    if key == "AolSteeringMode":
      return 0 if return_default else b"0"
    return None

  def remove(self, key: str) -> None:
    return None


def make_selfdrive(cp_flags: int, brand: str = "hyundai", main_cruise_allowed: bool = False,
                   aol_enabled: bool = True, cp_iq_flags: int = 0,
                   pause_on_steering_override: bool = False):
  cp = SimpleNamespace(
    brand=brand,
    flags=cp_flags,
    passive=False,
    notCar=False,
    safetyModel=structs.CarParams.SafetyModel.noOutput,
  )
  cp_iq = SimpleNamespace(flags=cp_iq_flags)
  return SimpleNamespace(
    CP=cp,
    CP_IQ=cp_iq,
    params=MockParams(main_cruise_allowed, aol_enabled, pause_on_steering_override),
    state_machine=SimpleNamespace(soft_disable_timer=0, current_alert_types=[]),
    events=Events(),
    events_iq=IQEvents(),
    CS_prev=SimpleNamespace(
      gasPressed=False,
      cruiseState=SimpleNamespace(available=False),
      lateralAvailable=False,
    ),
    enabled=False,
    enabled_prev=False,
    initialized=True,
  )


def make_car_state():
  return SimpleNamespace(
    started=True,
    standstill=False,
    doorOpen=False,
    seatbeltUnlatched=False,
    gearShifter=structs.CarState.GearShifter.drive,
    vEgo=0.0,
    gasPressed=False,
    brakePressed=False,
    cruiseState=SimpleNamespace(available=False),
    lateralAvailable=False,
    buttonEvents=[structs.CarState.ButtonEvent(pressed=True, type=ButtonType.lkas)],
  )


def make_vw_car_state(cruise_available: bool, cruise_fault_lateral: bool = False, acc_faulted: bool = False):
  return SimpleNamespace(
    started=True,
    standstill=False,
    doorOpen=False,
    seatbeltUnlatched=False,
    gearShifter=structs.CarState.GearShifter.drive,
    vEgo=0.0,
    gasPressed=False,
    brakePressed=False,
    cruiseState=SimpleNamespace(available=cruise_available),
    lateralAvailable=cruise_available or cruise_fault_lateral,
    cruiseFaultLateralMode=cruise_fault_lateral,
    accFaulted=acc_faulted,
    buttonEvents=[],
  )


def test_hyundai_lkas_button_can_arm_guidance_before_lateral_available():
  selfdrive = make_selfdrive(0, cp_iq_flags=HyundaiFlagsIQ.HAS_LFA_BUTTON)
  guidance = SteeringAssistanceBehavior(selfdrive)
  car_state = make_car_state()
  car_state.buttonEvents = [structs.CarState.ButtonEvent(pressed=True, type=ButtonType.lfaButton)]

  guidance.update_events(car_state)

  assert selfdrive.events_iq.has(EventNameIQ.alcEngaged)


def test_hyundai_lkas_button_stays_inactive_without_platform_support():
  selfdrive = make_selfdrive(0)
  guidance = SteeringAssistanceBehavior(selfdrive)

  guidance.update_events(make_car_state())

  assert not selfdrive.events_iq.has(EventNameIQ.alcEngaged)


def test_main_cruise_drop_cuts_guidance_even_if_lateral_signal_stays_true():
  selfdrive = make_selfdrive(0, brand="volkswagen", main_cruise_allowed=True)
  selfdrive.CS_prev = make_vw_car_state(cruise_available=True)
  guidance = SteeringAssistanceBehavior(selfdrive)
  guidance.enabled = True

  guidance.update_events(make_vw_car_state(cruise_available=False))

  assert selfdrive.events_iq.has(EventNameIQ.alcDisengaged)


def test_faulted_lateral_mode_does_not_force_disable_guidance():
  selfdrive = make_selfdrive(0, brand="volkswagen", main_cruise_allowed=True)
  selfdrive.CS_prev = make_vw_car_state(cruise_available=True)
  guidance = SteeringAssistanceBehavior(selfdrive)
  guidance.enabled = True

  guidance.update_events(make_vw_car_state(cruise_available=False, cruise_fault_lateral=True))

  assert not selfdrive.events_iq.has(EventNameIQ.alcDisengaged)


def test_guidance_keeps_steering_when_long_drops_for_faulted_cruise():
  guidance, selfdrive = make_engaged_guidance(False)
  selfdrive.enabled_prev = True
  faulted = make_vw_car_state(cruise_available=False, cruise_fault_lateral=True)

  selfdrive.events.clear()
  selfdrive.events_iq.clear()
  selfdrive.events.add(EventName.pcmDisable)
  guidance.update(faulted)

  assert not selfdrive.events.has(EventName.pcmDisable)
  assert not selfdrive.events_iq.has(EventNameIQ.alcDisengaged)
  assert guidance.state_machine.state == GuidanceState.enabled
  assert guidance.active


def test_main_switch_rising_edge_arms_guidance_during_faulted_cruise():
  selfdrive = make_selfdrive(0, brand="volkswagen", main_cruise_allowed=True)
  selfdrive.CS_prev = make_vw_car_state(cruise_available=False, cruise_fault_lateral=False)
  guidance = SteeringAssistanceBehavior(selfdrive)

  guidance.update_events(make_vw_car_state(cruise_available=False, cruise_fault_lateral=True))

  assert selfdrive.events_iq.has(EventNameIQ.alcEngaged)


def test_main_cruise_rising_edge_does_not_engage_when_toggle_is_off():
  selfdrive = make_selfdrive(0, brand="volkswagen", main_cruise_allowed=True, aol_enabled=False)
  selfdrive.CS_prev = make_vw_car_state(cruise_available=False)
  guidance = SteeringAssistanceBehavior(selfdrive)

  guidance.update(make_vw_car_state(cruise_available=True))

  assert not selfdrive.events_iq.has(EventNameIQ.alcEngaged)
  assert not guidance.active
  assert not guidance.enabled
  assert guidance.state_machine.state == custom.AlwaysOnLateral.AlwaysOnLateralState.disabled


def run_cycle(guidance, selfdrive, car_state, steering_pressed: bool):
  selfdrive.events.clear()
  selfdrive.events_iq.clear()
  if steering_pressed:
    selfdrive.events.add(EventName.steerOverride)
  guidance.update(car_state)
  selfdrive.CS_prev = car_state


def make_engaged_guidance(pause_on_steering_override: bool):
  selfdrive = make_selfdrive(0, brand="volkswagen", pause_on_steering_override=pause_on_steering_override)
  selfdrive.CS_prev = make_vw_car_state(cruise_available=True)
  guidance = SteeringAssistanceBehavior(selfdrive)
  guidance.enabled = True
  guidance.state_machine.state = GuidanceState.enabled
  return guidance, selfdrive


def test_steering_override_parks_guidance_when_enabled():
  guidance, selfdrive = make_engaged_guidance(True)

  run_cycle(guidance, selfdrive, make_vw_car_state(cruise_available=True), steering_pressed=True)

  assert guidance.state_machine.state == GuidanceState.paused
  assert guidance.enabled
  assert not guidance.active


def test_guidance_resumes_once_steering_is_released():
  guidance, selfdrive = make_engaged_guidance(True)

  run_cycle(guidance, selfdrive, make_vw_car_state(cruise_available=True), steering_pressed=True)
  run_cycle(guidance, selfdrive, make_vw_car_state(cruise_available=True), steering_pressed=False)

  assert guidance.state_machine.state == GuidanceState.enabled
  assert guidance.active


def test_steering_override_keeps_torque_when_option_is_off():
  guidance, selfdrive = make_engaged_guidance(False)

  run_cycle(guidance, selfdrive, make_vw_car_state(cruise_available=True), steering_pressed=True)

  assert guidance.state_machine.state == GuidanceState.overriding
  assert guidance.active


def test_main_cruise_rising_edge_engages_when_toggle_is_on():
  selfdrive = make_selfdrive(0, brand="volkswagen", main_cruise_allowed=True, aol_enabled=True)
  selfdrive.CS_prev = make_vw_car_state(cruise_available=False)
  guidance = SteeringAssistanceBehavior(selfdrive)

  guidance.update(make_vw_car_state(cruise_available=True))

  assert selfdrive.events_iq.has(EventNameIQ.alcEngaged)
  assert guidance.active


HARD_STOP_TSK_FAULT = (
  [dict(cruise_available=True)] * 50
  + [dict(cruise_available=False, acc_faulted=True)] * 5
  + [dict(cruise_available=False, cruise_fault_lateral=True, acc_faulted=True)] * 400
  + [dict(cruise_available=True, cruise_fault_lateral=True)] * 20
  + [dict(cruise_available=True)] * 50
)


def replay_cruise_states(guidance, selfdrive, cycles):
  seen = []
  for kwargs in cycles:
    car_state = make_vw_car_state(**kwargs)
    selfdrive.events.clear()
    selfdrive.events_iq.clear()
    guidance.update_events(car_state)
    seen.append((selfdrive.events_iq.has(EventNameIQ.alcEngaged), selfdrive.events_iq.has(EventNameIQ.alcDisengaged)))
    selfdrive.CS_prev = car_state
  return seen


def test_hard_stop_cruise_fault_does_not_engage_guidance_the_driver_turned_off():
  selfdrive = make_selfdrive(0, brand="volkswagen", main_cruise_allowed=True)
  selfdrive.CS_prev = make_vw_car_state(cruise_available=True)
  guidance = SteeringAssistanceBehavior(selfdrive)

  seen = replay_cruise_states(guidance, selfdrive, HARD_STOP_TSK_FAULT)

  assert not any(engaged for engaged, _ in seen)


def test_hard_stop_cruise_fault_does_not_cycle_guidance_the_driver_left_on():
  selfdrive = make_selfdrive(0, brand="volkswagen", main_cruise_allowed=True)
  selfdrive.CS_prev = make_vw_car_state(cruise_available=True)
  guidance = SteeringAssistanceBehavior(selfdrive)
  guidance.enabled = True

  seen = replay_cruise_states(guidance, selfdrive, HARD_STOP_TSK_FAULT)

  assert seen == [(False, False)] * len(HARD_STOP_TSK_FAULT)


def test_main_switch_cycled_without_a_fault_still_engages_guidance():
  selfdrive = make_selfdrive(0, brand="volkswagen", main_cruise_allowed=True)
  selfdrive.CS_prev = make_vw_car_state(cruise_available=True)
  guidance = SteeringAssistanceBehavior(selfdrive)
  cycles = [dict(cruise_available=True)] * 10 + [dict(cruise_available=False)] * 200 + [dict(cruise_available=True)]

  seen = replay_cruise_states(guidance, selfdrive, cycles)

  assert [engaged for engaged, _ in seen].index(True) == len(cycles) - 1


def test_main_switch_turned_off_during_a_stuck_fault_still_releases_guidance():
  selfdrive = make_selfdrive(0, brand="volkswagen", main_cruise_allowed=True)
  selfdrive.CS_prev = make_vw_car_state(cruise_available=True)
  guidance = SteeringAssistanceBehavior(selfdrive)
  guidance.enabled = True
  cycles = [dict(cruise_available=True)] * 10 + [dict(cruise_available=False, acc_faulted=True)] * 3000

  seen = replay_cruise_states(guidance, selfdrive, cycles)

  assert any(disengaged for _, disengaged in seen)


def run_update_cycles(guidance, selfdrive, cycles, fault_event):
  engaged_events = 0
  for kwargs in cycles:
    car_state = make_vw_car_state(**kwargs)
    selfdrive.events.clear()
    selfdrive.events_iq.clear()
    if kwargs.get("acc_faulted") and fault_event:
      selfdrive.events.add(EventName.accFaulted)
    guidance.update(car_state)
    engaged_events += selfdrive.events_iq.has(EventNameIQ.alcEngaged)
    selfdrive.CS_prev = car_state
  return engaged_events


GATED_FAULT_STOP = (
  [dict(cruise_available=True)] * 50
  + [dict(cruise_available=False, acc_faulted=True)] * 400
  + [dict(cruise_available=True)] * 50
)


def test_guidance_cut_by_a_cruise_fault_resumes_once_when_the_fault_clears():
  guidance, selfdrive = make_engaged_guidance(False)
  selfdrive.params.main_cruise_allowed = True
  guidance.main_enabled_toggle = True

  engaged_events = run_update_cycles(guidance, selfdrive, GATED_FAULT_STOP, fault_event=True)

  assert engaged_events == 1
  assert guidance.state_machine.state == GuidanceState.enabled


def test_guidance_the_driver_turned_off_stays_off_through_a_cruise_fault():
  selfdrive = make_selfdrive(0, brand="volkswagen", main_cruise_allowed=True)
  selfdrive.CS_prev = make_vw_car_state(cruise_available=True)
  guidance = SteeringAssistanceBehavior(selfdrive)

  engaged_events = run_update_cycles(guidance, selfdrive, GATED_FAULT_STOP, fault_event=True)

  assert engaged_events == 0
  assert guidance.state_machine.state == GuidanceState.disabled
