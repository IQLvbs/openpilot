from types import SimpleNamespace

import pytest

from iqdbc.car import structs
from iqpilot.cereal import custom, log
from iqpilot.selfdrive.controls.iq_controls_layer import IQControlsLayer
from iqpilot.selfdrive.controls.lib.longcontrol import LongControl
from iqpilot.selfdrive.controls.tests.lateral_control_scenarios import build_controls, load_controls_module, set_frame


def test_each_control_tick_delivers_the_current_policy_without_recomposing_slow_iq_state():
  layer = IQControlsLayer.__new__(IQControlsLayer)
  layer._needs_iq_lead_data = False
  layer._cached_iq_carcontrol = None
  compositions = []

  def compose(sm):
    compositions.append(True)
    return custom.IQCarControl.new_message(angleOffsetDeg=len(compositions))

  layer._compose_iq_carcontrol = compose
  sm = SimpleNamespace(updated={'iqState': False, 'iqPlan': False, 'radarState': False})
  sm = type('Inputs', (dict,), {'updated': sm.updated})(carState=SimpleNamespace(canValid=True))
  messages = []
  pm = SimpleNamespace(send=lambda service, message: messages.append(message.to_bytes()))
  for jerk in (0.8, 1.5, 3.0, 5.0):
    layer.publish_iq_state(sm, pm, structs.IQCarControl.LongitudinalComfort(valid=True, jerkUp=jerk))
  layer.publish_iq_state(sm, pm, structs.IQCarControl.LongitudinalComfort())
  sm.updated['iqPlan'] = True
  layer.publish_iq_state(sm, pm, structs.IQCarControl.LongitudinalComfort(valid=True, jerkUp=0.8))
  assert len(messages) == 6
  assert len(compositions) == 2
  observed = []
  for wire in messages:
    with log.Event.from_bytes(wire) as message:
      observed.append((message.iqCarControl.longitudinalComfort.valid, message.iqCarControl.longitudinalComfort.jerkUp,
                       message.iqCarControl.angleOffsetDeg))
  assert [round(jerk, 1) for _, jerk, _ in observed] == [0.8, 1.5, 3.0, 5.0, 0.0, 0.8]
  assert not observed[4][0]
  assert observed[-2][2] < observed[-1][2]


def test_controlsd_applies_the_gate_and_passes_plan_emergencies_to_the_policy(monkeypatch):
  controls = build_controls(load_controls_module(monkeypatch), 'pid')
  controls.CP = controls.CP.as_builder()
  controls.CP.openpilotLongitudinalControl = True
  controls.LoC = LongControl(controls.CP, controls.CP_IQ)
  set_frame(controls, 100)
  controls.sm['longitudinalPlan'].aTarget = 1.0
  legacy, _ = controls.state_control()
  assert not controls.longitudinal_comfort.control.valid
  controls.enable_long_comfort_mode = True
  controls.sm['carState'].aEgo = 0.0
  engaged, _ = controls.state_control()
  assert 0.0 < engaged.actuators.accel < legacy.actuators.accel
  controls.sm['longitudinalPlan'].fcw = True
  controls.sm['longitudinalPlan'].aTarget = -2.0
  emergency, _ = controls.state_control()
  assert emergency.actuators.accel < -2.0
  assert controls.longitudinal_comfort.control.bypassJerkLimit


@pytest.mark.parametrize('version,valid,expected_distance,expected_event', [
  (1, True, 2.25, 17), (0, True, None, 0), (2, True, None, 0), (1, False, None, 0),
])
def test_controlsd_delivers_stop_target_and_comfort_together(monkeypatch, version, valid, expected_distance, expected_event):
  controls = build_controls(load_controls_module(monkeypatch), 'pid')
  controls.CP = controls.CP.as_builder()
  controls.CP.openpilotLongitudinalControl = True
  controls.enable_long_comfort_mode = True
  set_frame(controls, 100)
  plan = controls.sm['longitudinalPlan']
  plan.stopOffset = SimpleNamespace(version=version, valid=valid, distance=2.25, eventId=17)
  plan.longitudinalPlanSource = 'lead0'
  plan.fcw = True
  calls = []
  monkeypatch.setattr(controls.LoC, "update", lambda *args, **kwargs: calls.append(kwargs) or -1.0)
  command, _ = controls.state_control()
  assert command.actuators.accel == -1.0
  assert len(calls) == 1
  assert calls[0]['stop_distance'] == expected_distance
  assert calls[0]['stop_event_id'] == expected_event
  assert calls[0]['comfort'] is controls.longitudinal_comfort
  assert calls[0]['comfort_enabled'] is True
  assert calls[0]['fcw'] is True
  assert calls[0]['plan_source'] == 'lead0'
