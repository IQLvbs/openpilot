"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos
"""
import pytest

from iqpilot.cereal import custom
from iqpilot.common.realtime import DT_CTRL
from iqpilot.selfdrive.selfdrived.events import ET
from iqpilot.selfdrive.selfdrived.state import SOFT_DISABLE_TIME
from iqpilot.sab.behavior import (GuidanceStateMachine, PAUSE_WITH_IQ_EVENTS,
                                                 PAUSE_WITH_STOCK_EVENTS)

State = custom.AlwaysOnLateral.AlwaysOnLateralState
EventNameIQ = custom.IQOnroadEvent.EventName

SOFT_DISABLE_FRAMES = int(SOFT_DISABLE_TIME / DT_CTRL)

ENABLE = ET.ENABLE
NO_ENTRY = ET.NO_ENTRY
SOFT = ET.SOFT_DISABLE
USER = ET.USER_DISABLE
IMMEDIATE = ET.IMMEDIATE_DISABLE
OVERRIDE = ET.OVERRIDE_LATERAL
SILENT = "silent-disable"
PAUSE_OK = "pause-eligible"


class SignalBag:
  def __init__(self, types=(), names=()):
    self._types = set(types)
    self._names = set(names)

  def contains(self, event_type):
    return event_type in self._types

  def has(self, name):
    return name in self._names

  def contains_in_list(self, names):
    return any(n in self._names for n in names)


class Host:
  class _SSM:
    def __init__(self):
      self.current_alert_types = []
      self.soft_disable_timer = 0

  def __init__(self, signals, selfdrive_enabled=False):
    types = {s for s in signals if s in (ENABLE, NO_ENTRY, SOFT, USER, IMMEDIATE, OVERRIDE)}
    names = set()
    if SILENT in signals:
      names.add(EventNameIQ.alcDisengagedSilent)
    if PAUSE_OK in signals:
      names.add(PAUSE_WITH_IQ_EVENTS[0])

    self.enabled = selfdrive_enabled
    self.state_machine = self._SSM()
    self.events = SignalBag(types)
    self.events_iq = SignalBag((), names)


class Sab:
  def __init__(self, host, pause_on_lateral_override=False):
    self.selfdrive = host
    self.pause_on_lateral_override = pause_on_lateral_override


def machine_at(state, signals, selfdrive_enabled=False, pause_on_lateral_override=False):
  host = Host(signals, selfdrive_enabled)
  m = GuidanceStateMachine(Sab(host, pause_on_lateral_override))
  m.state = state
  return m, host


TRANSITIONS = [
  ("idle stays idle", State.disabled, (), State.disabled),
  ("engage", State.disabled, (ENABLE,), State.enabled),
  ("engage while overriding", State.disabled, (ENABLE, OVERRIDE), State.overriding),
  ("blocked entry", State.disabled, (ENABLE, NO_ENTRY), State.disabled),
  ("blocked entry parks when pause-eligible", State.disabled, (ENABLE, NO_ENTRY, PAUSE_OK), State.paused),
  ("cruise steady", State.enabled, (), State.enabled),
  ("driver off switch", State.enabled, (USER,), State.disabled),
  ("driver off switch, silent -> pause", State.enabled, (USER, SILENT), State.paused),
  ("hard fault", State.enabled, (IMMEDIATE,), State.disabled),
  ("grace period entry", State.enabled, (SOFT,), State.softDisabling),
  ("hands on wheel", State.enabled, (OVERRIDE,), State.overriding),
  ("user beats soft", State.enabled, (USER, SOFT), State.disabled),
  ("hard beats soft", State.enabled, (IMMEDIATE, SOFT), State.disabled),
  ("condition cleared", State.softDisabling, (), State.enabled),
  ("user during grace", State.softDisabling, (USER,), State.disabled),
  ("hard during grace", State.softDisabling, (IMMEDIATE,), State.disabled),
  ("stays parked", State.paused, (), State.paused),
  ("blocked resume", State.paused, (ENABLE, NO_ENTRY), State.paused),
  ("resume", State.paused, (ENABLE,), State.enabled),
  ("resume into override", State.paused, (ENABLE, OVERRIDE), State.overriding),
  ("user kill while parked", State.paused, (USER,), State.disabled),
  ("silent user kill re-parks", State.paused, (USER, SILENT), State.paused),
  ("hard fault while parked", State.paused, (IMMEDIATE,), State.disabled),
  ("override released", State.overriding, (), State.enabled),
  ("override held", State.overriding, (OVERRIDE,), State.overriding),
  ("override to grace", State.overriding, (SOFT,), State.softDisabling),
  ("override user kill", State.overriding, (USER,), State.disabled),
  ("override hard fault", State.overriding, (IMMEDIATE,), State.disabled),
]

@pytest.mark.parametrize("label,start,signals,expected", TRANSITIONS, ids=[t[0] for t in TRANSITIONS])
def test_transition(label, start, signals, expected):
  m, _ = machine_at(start, signals)
  m.update()
  assert m.state == expected


@pytest.mark.parametrize("start,signals,expected_enabled,expected_active", [
  (State.disabled, (), False, False),
  (State.disabled, (ENABLE,), True, True),
  (State.disabled, (ENABLE, NO_ENTRY, PAUSE_OK), True, False),
  (State.enabled, (), True, True),
  (State.enabled, (SOFT,), True, True),
  (State.enabled, (USER,), False, False),
  (State.overriding, (OVERRIDE,), True, True),
])
def test_update_outputs(start, signals, expected_enabled, expected_active):
  m, _ = machine_at(start, signals)
  enabled, active = m.update()
  assert (enabled, active) == (expected_enabled, expected_active)

class TestSoftDisableTimer:
  def test_grace_period_arms_timer_when_solo(self):
    m, host = machine_at(State.enabled, (SOFT,))
    m.update()
    assert m.state == State.softDisabling
    assert host.state_machine.soft_disable_timer == SOFT_DISABLE_FRAMES
    assert ET.SOFT_DISABLE in host.state_machine.current_alert_types

  def test_grace_period_skips_timer_when_selfdrive_owns_it(self):
    m, host = machine_at(State.enabled, (SOFT,), selfdrive_enabled=True)
    m.update()
    assert m.state == State.softDisabling
    assert host.state_machine.soft_disable_timer == 0

  def test_expiry_disables(self):
    m, host = machine_at(State.softDisabling, (SOFT,))
    host.state_machine.soft_disable_timer = 0
    m.update()
    assert m.state == State.disabled

  def test_countdown_keeps_grace(self):
    m, host = machine_at(State.softDisabling, (SOFT,))
    host.state_machine.soft_disable_timer = 5
    m.update()
    assert m.state == State.softDisabling

class TestAlertQueueing:
  def test_alerts_only_queued_when_solo(self):
    m, host = machine_at(State.disabled, (ENABLE,), selfdrive_enabled=True)
    m.update()
    assert host.state_machine.current_alert_types == []

  def test_engage_alert_queued(self):
    m, host = machine_at(State.disabled, (ENABLE,))
    m.update()
    assert ET.ENABLE in host.state_machine.current_alert_types
    assert ET.WARNING in host.state_machine.current_alert_types

  def test_no_entry_alert_queued(self):
    m, host = machine_at(State.disabled, (ENABLE, NO_ENTRY))
    m.update()
    assert ET.NO_ENTRY in host.state_machine.current_alert_types

  def test_user_disable_alert_always_queued(self):
    m, host = machine_at(State.enabled, (USER,), selfdrive_enabled=True)
    m.update()
    assert ET.USER_DISABLE in host.state_machine.current_alert_types

  def test_override_alert_repeats_while_held(self):
    m, host = machine_at(State.overriding, (OVERRIDE,), selfdrive_enabled=True)
    m.update()
    assert ET.OVERRIDE_LATERAL in host.state_machine.current_alert_types

class TestPauseEligibility:
  @pytest.mark.parametrize("event_name", PAUSE_WITH_IQ_EVENTS)
  def test_each_iq_pause_event_parks(self, event_name):
    host = Host((ENABLE, NO_ENTRY))
    host.events_iq = SignalBag((), {event_name})
    m = GuidanceStateMachine(Sab(host))
    m.state = State.disabled
    m.update()
    assert m.state == State.paused

  @pytest.mark.parametrize("event_name", PAUSE_WITH_STOCK_EVENTS)
  def test_each_stock_pause_event_parks(self, event_name):
    host = Host((ENABLE, NO_ENTRY))
    host.events = SignalBag({ENABLE, NO_ENTRY}, {event_name})
    m = GuidanceStateMachine(Sab(host))
    m.state = State.disabled
    m.update()
    assert m.state == State.paused

OVERRIDE_PAUSE_TRANSITIONS = [
  ("hands on wheel parks", State.enabled, (OVERRIDE,), State.paused),
  ("override held stays parked", State.overriding, (OVERRIDE,), State.paused),
  ("engage while overriding parks", State.disabled, (ENABLE, OVERRIDE), State.paused),
  ("parked resume waits for release", State.paused, (ENABLE, OVERRIDE), State.paused),
  ("release resumes", State.paused, (ENABLE,), State.enabled),
  ("hands off keeps steering", State.enabled, (), State.enabled),
  ("user kill still kills", State.enabled, (OVERRIDE, USER), State.disabled),
  ("hard fault still faults", State.enabled, (OVERRIDE, IMMEDIATE), State.disabled),
  ("grace beats override", State.enabled, (OVERRIDE, SOFT), State.softDisabling),
]

@pytest.mark.parametrize("label,start,signals,expected", OVERRIDE_PAUSE_TRANSITIONS,
                         ids=[t[0] for t in OVERRIDE_PAUSE_TRANSITIONS])
def test_override_pause_transition(label, start, signals, expected):
  m, _ = machine_at(start, signals, pause_on_lateral_override=True)
  m.update()
  assert m.state == expected


class TestOverridePauseOutputs:
  def test_torque_stops_while_overriding(self):
    m, _ = machine_at(State.enabled, (OVERRIDE,), pause_on_lateral_override=True)
    enabled, active = m.update()
    assert (enabled, active) == (True, False)

  def test_torque_returns_on_release(self):
    m, _ = machine_at(State.paused, (ENABLE,), pause_on_lateral_override=True)
    enabled, active = m.update()
    assert (enabled, active) == (True, True)

  def test_override_alert_not_queued_while_parked(self):
    m, host = machine_at(State.enabled, (OVERRIDE,), pause_on_lateral_override=True)
    m.update()
    assert ET.OVERRIDE_LATERAL not in host.state_machine.current_alert_types
