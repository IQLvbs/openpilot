from types import SimpleNamespace

import pytest

from iqdbc.car import structs
from iqpilot.cereal import car, custom
from iqpilot.selfdrive.car.card import Car


@pytest.mark.parametrize('age_ns,envelope_valid,policy_valid,expected', [
  (99_000_000, True, True, True),
  (101_000_000, True, True, False),
  (0, False, True, False),
  (0, True, False, False),
])
def test_card_delivers_only_fresh_valid_comfort_to_the_carcontroller(monkeypatch, age_ns, envelope_valid, policy_valid, expected):
  now = 10_000_000_000
  monkeypatch.setattr('iqpilot.selfdrive.car.card.time.monotonic_ns', lambda: now)
  monkeypatch.setattr('iqpilot.selfdrive.car.card.REPLAY', False)
  received = []

  def apply(cc, iq, timestamp, model):
    received.append(iq.longitudinalComfort)
    return structs.CarControl.Actuators(), []

  instance = Car.__new__(Car)
  instance.initialized_prev = True
  instance._needs_iq_lead_data = False
  instance._pair_actuator_feedback = False
  instance._last_sendcan_mono_ns = None
  instance._perf_ring = SimpleNamespace(push=lambda sample: None)
  instance.CI = SimpleNamespace(apply=apply)
  instance.pm = SimpleNamespace(send=lambda *args: None)
  instance.sm = type('Inputs', (dict,), {
    'frame': 1, 'recv_frame': {'carControl': 1}, 'all_alive': lambda self, services: True,
    'logMonoTime': {'iqCarControl': now - age_ns}, 'valid': {'iqCarControl': envelope_valid, 'modelV2': False},
  })()
  cc_iq = custom.IQCarControl.new_message(longitudinalComfort={'valid': policy_valid, 'jerkUp': 0.8, 'jerkDown': 5.0})
  instance.controls_update(car.CarState.new_message(canValid=True), car.CarControl.new_message(), cc_iq)
  assert len(received) == 1
  assert received[0].valid == expected
  assert received[0].jerkDown > received[0].jerkUp
