import pytest

from iqpilot.cereal import car, log, messaging
from iqpilot.cereal.services import SERVICE_LIST
from iqpilot.common.realtime import DT_CTRL
from iqpilot.selfdrive.selfdrived.events import Events, ET
from iqpilot.selfdrive.selfdrived.helpers import CRITICAL_SERVICES, unresponsive_critical_services
from iqpilot.selfdrive.selfdrived.state import StateMachine, State, SOFT_DISABLE_TIME

EventName = log.OnroadEvent.EventName


class Drive:
  # selfdrived's SubMaster fed at 100Hz, each critical service at its own rate
  def __init__(self, not_running=(), ignore_alive=(), extra=()):
    self.sm = messaging.SubMaster([*CRITICAL_SERVICES, 'managerState', *extra], ignore_alive=list(ignore_alive))
    self.frame = 0
    manager = messaging.new_message('managerState')
    manager.managerState.processes = [{'name': n, 'running': False, 'shouldBeRunning': True} for n in not_running]
    self.sm.update_msgs(0., [manager.as_reader()])
    for _ in range(100):
      self.step()

  def step(self, silent=()):
    self.frame += 1
    msgs = []
    for s in self.sm.services:
      if s == 'managerState':
        continue
      period = round(1 / (SERVICE_LIST[s].frequency * DT_CTRL)) if SERVICE_LIST[s].frequency > 0 else 0
      if period and self.frame % period == 0 and s not in silent:
        msgs.append(messaging.new_message(s).as_reader())
    self.sm.update_msgs(self.frame * DT_CTRL, msgs)
    return unresponsive_critical_services(self.sm)

  def gap(self, service, gap_s):
    flagged = [self.step(silent=(service,)) for _ in range(int(round(gap_s / DT_CTRL)))]
    return flagged, [self.step() for _ in range(5)][-1]  # one 20Hz period to resume


@pytest.mark.parametrize('service', [s for s in CRITICAL_SERVICES if SERVICE_LIST[s].frequency == 20])
def test_normal_jitter_is_not_flagged(service):
  flagged, _ = Drive().gap(service, 0.45)
  assert not any(flagged)


# planner/radar stalls recorded on 0000000d--ecb4786fa1
@pytest.mark.parametrize('gap_s', (0.506, 0.695, 0.983, 1.167))
@pytest.mark.parametrize('service', ('radarState', 'longitudinalPlan'))
def test_stalls_are_flagged_until_the_output_returns(service, gap_s):
  flagged, after = Drive().gap(service, gap_s)
  first = next(i for i, f in enumerate(flagged) if f)
  assert (first + 1) * DT_CTRL == pytest.approx(10 / SERVICE_LIST[service].frequency, abs=6 * DT_CTRL)
  assert all(f == [service] for f in flagged[first:])
  assert after == []


def test_controls_are_flagged_after_a_tenth_of_a_second():
  flagged, after = Drive().gap('controlsState', 0.15)
  assert not any(flagged[:9]) and flagged[-1] == ['controlsState'] and after == []


def test_dead_model_output_is_flagged_and_clears_on_recovery():
  # 14.5 s model outage on 00000007--f9364b6c2f while iqmodeld restarted
  flagged, after = Drive().gap('modelV2', 14.454)
  assert all(f == ['modelV2'] for f in flagged[int(0.6 / DT_CTRL):])
  assert after == []


def test_output_that_never_arrived_is_flagged():
  sm = messaging.SubMaster([*CRITICAL_SERVICES])
  sm.update_msgs(1., [])
  assert unresponsive_critical_services(sm) == list(CRITICAL_SERVICES)


def test_ignored_and_non_critical_services_are_not_flagged():
  drive = Drive(ignore_alive=['radarState'], extra=['cameraOdometry'])
  for _ in range(int(5 / DT_CTRL)):
    assert drive.step(silent=('radarState', 'cameraOdometry')) == []


@pytest.mark.parametrize('event', (EventName.commIssue, EventName.commIssueAvgFreq, EventName.selfdrivedLagging))
def test_generic_comm_issues_are_soft_warnings(event):
  events = Events()
  events.add(event)
  msg = events.to_msg()[0]
  assert not (msg.softDisable or msg.immediateDisable or msg.noEntry or msg.warning)
  assert events.create_alerts([ET.SOFT_DISABLE, ET.NO_ENTRY, ET.WARNING], [None, None, Drive().sm, False, 100, None]) == []


def test_unresponsive_planner_refuses_engagement_with_a_reason():
  drive = Drive()
  for _ in range(int(0.6 / DT_CTRL)):
    drive.step(silent=('longitudinalPlan',))
  events = Events()
  events.add(EventName.processNotRunning)
  machine = StateMachine()
  events.add(EventName.buttonEnable)
  machine.update(events)
  assert machine.state == State.disabled
  entry, = events.create_alerts([ET.NO_ENTRY], [None, None, drive.sm, False, 100, None])
  assert "Planner Not Responding" in (entry.alert_text_1, entry.alert_text_2)


def test_unresponsive_model_warns_disengages_and_blocks_reengagement():
  drive = Drive()
  for _ in range(int(1 / DT_CTRL)):
    drive.step(silent=('modelV2',))
  sm = drive.sm
  events = Events()
  events.add(EventName.processNotRunning)
  machine = StateMachine()
  machine.state = State.enabled
  machine.update(events)
  assert machine.state == State.softDisabling
  alerts = events.create_alerts(machine.current_alert_types, [None, None, sm, False, machine.soft_disable_timer, None])
  assert any(a.alert_text_2 == "Driving Model Not Responding" and a.alert_size == log.SelfdriveState.AlertSize.full and
             a.audible_alert != car.CarControl.HUDControl.AudibleAlert.none for a in alerts)

  for _ in range(int(SOFT_DISABLE_TIME / DT_CTRL) + 1):
    machine.update(events)
  assert machine.state == State.disabled
  events.add(EventName.buttonEnable)
  machine.update(events)
  assert machine.state == State.disabled
  alerts = events.create_alerts(machine.current_alert_types, [None, None, sm, False, 0, None])
  assert any("Driving Model Not Responding" in (a.alert_text_1, a.alert_text_2) for a in alerts)


def test_dead_process_keeps_process_not_running_wording():
  sm = Drive(not_running=['plannerd']).sm
  events = Events()
  events.add(EventName.processNotRunning)
  soft, = events.create_alerts([ET.SOFT_DISABLE], [None, None, sm, False, 100, None])
  entry, = events.create_alerts([ET.NO_ENTRY], [None, None, sm, False, 100, None])
  assert soft.alert_text_2 == "Process Not Running"
  assert "plannerd" in (entry.alert_text_1, entry.alert_text_2)
