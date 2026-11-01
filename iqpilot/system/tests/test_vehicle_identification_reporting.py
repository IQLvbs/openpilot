# Copyright (c) 2026 IQ.Lvbs. All rights reserved.
import pytest

from iqpilot.system import sentry


@pytest.fixture
def reporting(monkeypatch):
  events, local, flushes = [], [], []
  monkeypatch.setattr(sentry.cloudlog, 'info', local.append)
  monkeypatch.setattr(sentry, 'set_user', lambda: None)
  monkeypatch.setattr(sentry.sentry_sdk, 'flush', lambda: flushes.append(True))

  def capture(**event):
    events.append((event, dict(sentry.sentry_sdk.get_current_scope()._tags)))

  monkeypatch.setattr(sentry.sentry_sdk, 'capture_message', capture)
  return events, local, flushes


def test_disabled_reporting_keeps_only_local_identification(reporting, monkeypatch):
  monkeypatch.setenv('IQPILOT_ENABLE_SENTRY', '0')
  events, local, flushes = reporting
  sentry.record_vehicle_identification('HONDA', 'honda')
  assert local == [{'vehicle_identification': 'HONDA', 'brand': 'honda', 'matched': True}]
  assert events == flushes == []


@pytest.mark.parametrize('candidate,brand,message,level', [
  ('HONDA', 'honda', 'Fingerprinted HONDA', 'info'),
  ('MOCK', 'mock', "car doesn't match any fingerprints", 'error'),
])
def test_opted_in_events_do_not_leak_vehicle_tags(reporting, monkeypatch, candidate, brand, message, level):
  monkeypatch.setenv('IQPILOT_ENABLE_SENTRY', '1')
  events, _, flushes = reporting
  before = dict(sentry.sentry_sdk.get_current_scope()._tags)
  sentry.record_vehicle_identification(candidate, brand)
  assert events == [({'message': message, 'level': level}, {**before, 'carFingerprint': candidate, 'carName': brand})]
  assert dict(sentry.sentry_sdk.get_current_scope()._tags) == before
  assert flushes == [True]


def test_reporting_failure_does_not_interrupt_vehicle_setup(reporting, monkeypatch):
  monkeypatch.setenv('IQPILOT_ENABLE_SENTRY', '1')
  failures = []
  monkeypatch.setattr(sentry.cloudlog, 'exception', failures.append)

  def fail(**_):
    raise RuntimeError('transport unavailable')

  monkeypatch.setattr(sentry.sentry_sdk, 'capture_message', fail)
  sentry.record_vehicle_identification('HONDA', 'honda')
  assert failures == ['vehicle identification reporting failed']
