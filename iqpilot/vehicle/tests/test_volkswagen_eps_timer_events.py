# Copyright © IQ.Lvbs, part of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos
from types import SimpleNamespace

from iqpilot.cereal import log
from iqpilot.selfdrive.selfdrived.events import Events
from iqpilot.vehicle.vehicle import VehicleEvents

EventName = log.OnroadEvent.EventName


def make(brand="volkswagen"):
  return VehicleEvents(SimpleNamespace(brand=brand), SimpleNamespace())


def iq_state(**overrides):
  base = dict(hcaTimerAlert=False, epsSessionSec=0.0, hcaSessionSec=0.0)
  base.update(overrides)
  return SimpleNamespace(**base)


def names(events):
  return set(events.names)


def test_timer_alert_raises_steer_time_limit():
  events = Events()
  make().update(SimpleNamespace(steerFaultTemporary=False), events, iq_state(hcaTimerAlert=True))
  assert EventName.steerTimeLimit in names(events)


def test_no_alert_when_timer_quiet():
  events = Events()
  make().update(SimpleNamespace(steerFaultTemporary=False), events, iq_state(epsSessionSec=200.0))
  assert EventName.steerTimeLimit not in names(events)


def test_expiry_rejection_is_always_audible():
  events = Events()
  events.add(EventName.steerTempUnavailableSilent)
  make().update(SimpleNamespace(steerFaultTemporary=True), events, iq_state(epsSessionSec=373.0))
  assert EventName.steerTempUnavailable in names(events)
  assert EventName.steerTempUnavailableSilent not in names(events)


def test_expiry_rejection_without_silencing_stays_audible():
  events = Events()
  events.add(EventName.steerTempUnavailable)
  make().update(SimpleNamespace(steerFaultTemporary=True), events, iq_state(hcaSessionSec=360.0))
  assert EventName.steerTempUnavailable in names(events)


def test_early_rejection_keeps_existing_behaviour():
  events = Events()
  events.add(EventName.steerTempUnavailableSilent)
  make().update(SimpleNamespace(steerFaultTemporary=True), events, iq_state(epsSessionSec=40.0, hcaSessionSec=40.0))
  assert EventName.steerTempUnavailableSilent in names(events)
  assert EventName.steerTempUnavailable not in names(events)


def test_other_brands_untouched():
  events = Events()
  events.add(EventName.steerTempUnavailable)
  make("honda").update(SimpleNamespace(steerFaultTemporary=True), events, iq_state(epsSessionSec=400.0, hcaTimerAlert=True))
  assert EventName.steerTimeLimit not in names(events)
