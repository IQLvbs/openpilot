from types import SimpleNamespace

import numpy as np
import pytest

from iqpilot.cereal import log, messaging
from iqpilot.selfdrive.controls.lib import longitudinal_planner as lp


class _StrictStruct:
  # Device pycapnp rejects NumPy scalars; a different desktop runtime accepts them.
  def __setattr__(self, name, value):
    assert not isinstance(value, np.generic), f"{name} got numpy {type(value).__name__}"
    object.__setattr__(self, name, value)


def _planner_with_numpy_stop_plan():
  planner = object.__new__(lp.LongitudinalPlanner)
  planner.stop_position = np.float64(1.5)
  planner.custom_stop_distance = SimpleNamespace(valid=np.bool_(True), requested_distance=np.float64(3.0))
  planner.stop_plan = SimpleNamespace(
    trajectory=np.zeros(3), state="tracking", remaining=np.float64(4.25), shortfall=np.float64(0.0097),
    target=SimpleNamespace(event_id=np.int64(7), source="lead", position=np.float64(10.0), uncertainty=np.float64(0.2)),
  )
  return planner


def test_stop_offset_publishes_plain_values_from_numpy_planner_state():
  stop = _StrictStruct()
  _planner_with_numpy_stop_plan()._fill_stop_offset(stop)
  assert (stop.distance, stop.shortfall, stop.baselineDistance, stop.eventId) == (4.25, 0.0097, 8.5, 7)
  assert stop.valid is True and stop.configurationValid is True


def test_stop_offset_fills_a_real_longitudinal_plan():
  msg = messaging.new_message('longitudinalPlan')
  _planner_with_numpy_stop_plan()._fill_stop_offset(msg.longitudinalPlan.stopOffset)
  with log.Event.from_bytes(msg.to_bytes()) as received:
    stop = received.longitudinalPlan.stopOffset
    assert stop.source == "lead"
    assert stop.shortfall == pytest.approx(0.0097)
    assert stop.distance == pytest.approx(4.25)
    assert stop.baselineDistance == pytest.approx(8.5)
    assert stop.eventId == 7
    assert list(stop.supportedSources) == ["lead"]


def test_stop_offset_without_a_target_remains_invalid():
  planner = _planner_with_numpy_stop_plan()
  planner.stop_plan = lp.StopOffsetPlan()
  msg = messaging.new_message('longitudinalPlan')
  planner._fill_stop_offset(msg.longitudinalPlan.stopOffset)
  with log.Event.from_bytes(msg.to_bytes()) as received:
    stop = received.longitudinalPlan.stopOffset
    assert not stop.valid
    assert stop.eventId == 0
    assert stop.distance == 0.0
    assert stop.shortfall == 0.0


def test_publish_casts_the_reported_numpy_shortfall_at_the_message_boundary(monkeypatch):
  planner = _planner_with_numpy_stop_plan()
  planner.stop_position = 1.5
  planner.custom_stop_distance = SimpleNamespace(valid=True, requested_distance=3.0)
  planner.stop_plan.remaining = 4.25
  planner.stop_plan.shortfall = np.float64(0.009767158603821702)
  planner.stop_plan.target = SimpleNamespace(event_id=7, source="lead", position=10.0, uncertainty=0.2)
  planner.mpc = SimpleNamespace(solve_time=0.0, source=lp.LongitudinalPlanSource.lead0,
                                lead_xv_0=np.zeros((2, 2)), lead_xv_1=np.zeros((2, 2)))
  planner.v_desired_trajectory = planner.a_desired_trajectory = planner.j_desired_trajectory = np.zeros(2)
  planner.output_a_target, planner.output_should_stop, planner.allow_throttle, planner.fcw = 0.0, False, True, False
  planner.publish_longitudinal_plan_iq = lambda *args: None
  stop = _StrictStruct()
  envelope = SimpleNamespace(logMonoTime=1_000_000_000, longitudinalPlan=SimpleNamespace(stopOffset=stop))
  monkeypatch.setattr(lp.messaging, 'new_message', lambda service: envelope)

  class Inputs(dict):
    logMonoTime = {'modelV2': 1_000_000_000}

    def all_checks(self, service_list):
      return True

  sm = Inputs(radarState=SimpleNamespace(leadOne=SimpleNamespace(status=True, dRel=10.0),
                                        leadTwo=SimpleNamespace(status=False)))
  sent = []
  planner.publish(sm, SimpleNamespace(send=lambda service, message: sent.append((service, message))))
  assert stop.shortfall == 0.009767158603821702
  assert type(stop.shortfall) is float
  assert sent == [('longitudinalPlan', envelope)]
