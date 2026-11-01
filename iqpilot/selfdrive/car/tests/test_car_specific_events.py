from types import SimpleNamespace

from iqpilot.cereal import log
from iqdbc.car import structs
from iqdbc.car.volkswagen.values import VolkswagenFlags

from iqpilot.selfdrive.car.car_specific import CarSpecificEvents
from iqpilot.selfdrive.selfdrived.events import ET, Events

EventName = log.OnroadEvent.EventName
GearShifter = structs.CarState.GearShifter


def make_car_state(**overrides):
  base = dict(
    doorOpen=False,
    seatbeltUnlatched=False,
    gearShifter=GearShifter.drive,
    cruiseState=SimpleNamespace(available=True, enabled=False, nonAdaptive=False),
    cruiseFaultLateralMode=False,
    espDisabled=False,
    espActive=False,
    stockFcw=False,
    stockAeb=False,
    stockLkas=False,
    vEgo=15.0,
    brakeHoldActive=False,
    parkingBrake=False,
    accFaulted=False,
    steeringPressed=False,
    steeringDisengage=False,
    brakePressed=False,
    standstill=False,
    gasPressed=False,
    vehicleSensorsInvalid=False,
    invalidLkasSetting=False,
    lowSpeedAlert=False,
    buttonEnable=False,
    buttonEvents=[],
    steerFaultTemporary=False,
    steerFaultPermanent=False,
    blockPcmEnable=False,
  )
  base.update(overrides)
  return SimpleNamespace(**base)


def test_pcm_disable_emitted_during_cruise_fault_lateral_mode():
  cp = SimpleNamespace(
    brand="volkswagen",
    openpilotLongitudinalControl=False,
    minEnableSpeed=0.0,
    pcmCruise=True,
    carFingerprint="MOCK",
  )
  events = CarSpecificEvents(cp)

  cs = make_car_state(cruiseFaultLateralMode=True)
  cs_prev = make_car_state(cruiseState=SimpleNamespace(available=True, enabled=True, nonAdaptive=False))

  out = events.update(cs, cs_prev, SimpleNamespace(actuators=SimpleNamespace(accel=0.0)))

  assert out.has(EventName.pcmDisable)
  assert out.has(EventName.cruiseFaultLateralAllowed)


def test_cruise_fault_lateral_mode_replaces_acc_faulted_alert():
  cp = SimpleNamespace(
    brand="volkswagen",
    openpilotLongitudinalControl=False,
    minEnableSpeed=0.0,
    pcmCruise=True,
    carFingerprint="MOCK",
  )
  events = CarSpecificEvents(cp)

  cs = make_car_state(accFaulted=True, cruiseFaultLateralMode=True)
  cs_prev = make_car_state()

  out = events.update(cs, cs_prev, SimpleNamespace(actuators=SimpleNamespace(accel=0.0)))

  assert out.has(EventName.cruiseFaultLateralAllowed)
  assert not out.has(EventName.accFaulted)


def test_acc_faulted_still_emitted_without_cruise_fault_lateral_mode():
  cp = SimpleNamespace(
    brand="volkswagen",
    openpilotLongitudinalControl=False,
    minEnableSpeed=0.0,
    pcmCruise=True,
    carFingerprint="MOCK",
  )
  events = CarSpecificEvents(cp)

  cs = make_car_state(accFaulted=True)
  cs_prev = make_car_state()

  out = events.update(cs, cs_prev, SimpleNamespace(actuators=SimpleNamespace(accel=0.0)))

  assert out.has(EventName.accFaulted)
  assert not out.has(EventName.cruiseFaultLateralAllowed)


def test_pcm_disable_still_emitted_without_cruise_fault_lateral_mode():
  cp = SimpleNamespace(
    brand="volkswagen",
    openpilotLongitudinalControl=False,
    minEnableSpeed=0.0,
    pcmCruise=True,
    carFingerprint="MOCK",
  )
  events = CarSpecificEvents(cp)

  cs = make_car_state()
  cs_prev = make_car_state(cruiseState=SimpleNamespace(available=True, enabled=True, nonAdaptive=False))

  out = events.update(cs, cs_prev, SimpleNamespace(actuators=SimpleNamespace(accel=0.0)))

  assert out.has(EventName.pcmDisable)


HARD_DECEL = -6.5
HARD_DECEL_TEXT = "Temporary ACC fault caused by hard braking"


def make_vw_events(flags):
  cp = SimpleNamespace(
    brand="volkswagen",
    openpilotLongitudinalControl=False,
    minEnableSpeed=0.0,
    pcmCruise=True,
    carFingerprint="MOCK",
    flags=int(flags),
  )
  return CarSpecificEvents(cp)


def run_frames(events, frames):
  out = None
  prev = make_car_state()
  for overrides in frames:
    cs = make_car_state(**overrides)
    out = events.update(cs, prev, SimpleNamespace(actuators=SimpleNamespace(accel=0.0)))
    prev = cs
  return out


HARD_BRAKE = dict(brakePressed=True, aEgo=HARD_DECEL)
BRAKE_RELEASED = dict(brakePressed=False, aEgo=0.0)


def test_meb_fault_after_hard_driver_braking_raises_the_dedicated_event():
  events = make_vw_events(VolkswagenFlags.MEB)
  out = run_frames(events, [HARD_BRAKE] * 20 + [dict(accFaulted=True, **HARD_BRAKE)])

  assert out.has(EventName.accFaultedHardDecel)
  assert not out.has(EventName.accFaulted)


def test_mqb_evo_fault_after_hard_driver_braking_raises_the_dedicated_event():
  events = make_vw_events(VolkswagenFlags.MQB_EVO)
  out = run_frames(events, [HARD_BRAKE] * 20 + [dict(accFaulted=True, **HARD_BRAKE)])

  assert out.has(EventName.accFaultedHardDecel)


def test_dedicated_event_stays_through_the_whole_fault_and_clears_after():
  events = make_vw_events(VolkswagenFlags.MEB)
  out = run_frames(events, [HARD_BRAKE] * 5 + [dict(accFaulted=True, **BRAKE_RELEASED)] * 300)
  assert out.has(EventName.accFaultedHardDecel)

  out = run_frames(events, [BRAKE_RELEASED] * 5)
  assert not out.has(EventName.accFaultedHardDecel)
  assert not out.has(EventName.accFaulted)


def test_lateral_mode_variant_replaces_the_generic_lateral_notice():
  events = make_vw_events(VolkswagenFlags.MEB)
  out = run_frames(events, [HARD_BRAKE] * 5 + [dict(accFaulted=True, cruiseFaultLateralMode=True, **HARD_BRAKE)])

  assert out.has(EventName.accFaultedHardDecelLateral)
  assert not out.has(EventName.cruiseFaultLateralAllowed)
  assert not out.has(EventName.accFaultedHardDecel)
  assert not out.has(EventName.accFaulted)


def test_fault_without_hard_braking_keeps_the_generic_events():
  events = make_vw_events(VolkswagenFlags.MEB)
  out = run_frames(events, [BRAKE_RELEASED] * 20 + [dict(accFaulted=True, **BRAKE_RELEASED)])
  assert out.has(EventName.accFaulted)
  assert not out.has(EventName.accFaultedHardDecel)

  events = make_vw_events(VolkswagenFlags.MEB)
  out = run_frames(events, [dict(brakePressed=True, aEgo=-3.0)] * 20 + [dict(accFaulted=True, brakePressed=True, aEgo=-3.0)])
  assert out.has(EventName.accFaulted)


def test_fault_long_after_the_braking_keeps_the_generic_event():
  events = make_vw_events(VolkswagenFlags.MEB)
  out = run_frames(events, [HARD_BRAKE] * 5 + [BRAKE_RELEASED] * 200 + [dict(accFaulted=True, **BRAKE_RELEASED)])

  assert out.has(EventName.accFaulted)
  assert not out.has(EventName.accFaultedHardDecel)


def test_hard_braking_without_a_fault_raises_nothing():
  events = make_vw_events(VolkswagenFlags.MEB)
  out = run_frames(events, [HARD_BRAKE] * 100)

  assert not out.has(EventName.accFaulted)
  assert not out.has(EventName.accFaultedHardDecel)


def test_other_volkswagen_platforms_keep_the_generic_event():
  events = make_vw_events(0)
  out = run_frames(events, [HARD_BRAKE] * 20 + [dict(accFaulted=True, **HARD_BRAKE)])

  assert out.has(EventName.accFaulted)
  assert not out.has(EventName.accFaultedHardDecel)


def alert_texts(event_name, event_type):
  events = Events()
  events.add(event_name)
  return [(alert.alert_text_1, alert.alert_text_2) for alert in events.create_alerts([event_type], [])]


def test_dedicated_alerts_carry_the_requested_text():
  assert alert_texts(EventName.accFaultedHardDecel, ET.PERMANENT) == [("IQ.Pilot Temporarily Unavailable", HARD_DECEL_TEXT)]
  assert alert_texts(EventName.accFaultedHardDecelLateral, ET.PERMANENT) == [("IQ.Pilot Temporarily Unavailable", HARD_DECEL_TEXT)]
  assert alert_texts(EventName.accFaultedHardDecel, ET.NO_ENTRY)[0][1] == HARD_DECEL_TEXT


def test_dedicated_event_disables_exactly_like_the_generic_one():
  generic = {t for t in (ET.IMMEDIATE_DISABLE, ET.PERMANENT, ET.NO_ENTRY) if alert_texts(EventName.accFaulted, t)}
  dedicated = {t for t in (ET.IMMEDIATE_DISABLE, ET.PERMANENT, ET.NO_ENTRY) if alert_texts(EventName.accFaultedHardDecel, t)}

  assert dedicated == generic
  assert not alert_texts(EventName.accFaultedHardDecelLateral, ET.IMMEDIATE_DISABLE)
