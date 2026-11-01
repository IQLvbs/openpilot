"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos
"""
import pytest

from iqpilot.selfdrive.controls.lib.iq_dynamic.imahelper import IQModeEngine
from iqpilot.selfdrive.controls.lib.longitudinal_planner import (
  MODE_BLEND_BRAKE_PASS, MODE_BLEND_JERK_DOWN, MODE_BLEND_JERK_UP, limit_mode_transition,
)
from iqpilot.selfdrive.controls.lib.tests.test_longitudinal_planner_accel_state import CRUISE, build_planner, build_sm

DT = 0.05


def drive(engine, requests):
  modes = []
  for mode, urgency, emergency in requests:
    engine.request(mode, urgency, emergency)
    engine.update()
    modes.append(engine.get_mode())
  return modes


def count_switches(modes):
  return sum(1 for prev, cur in zip(modes, modes[1:], strict=False) if prev != cur)


def test_flickering_requests_do_not_chatter():
  pattern = [('blended', 0.95, False)] * 15 + [('acc', 1.0, False)] * 15
  modes = drive(IQModeEngine(), pattern * 20)

  assert modes[-1] == 'blended'
  assert count_switches(modes) == 1


def test_sustained_acc_request_leaves_blended_after_dwell():
  engine = IQModeEngine()
  drive(engine, [('blended', 1.0, False)] * 20)
  assert engine.get_mode() == 'blended'

  modes = drive(engine, [('acc', 0.7, False)] * 100)
  first_acc = modes.index('acc')

  assert first_acc * DT >= 2.0
  assert modes[-1] == 'acc'


def test_blended_entry_is_prompt():
  modes = drive(IQModeEngine(), [('acc', 0.7, False)] * 20 + [('blended', 0.9, False)] * 20)

  assert modes[20:].index('blended') * DT <= 0.5


def test_emergency_switches_immediately_and_holds():
  engine = IQModeEngine()
  drive(engine, [('acc', 0.7, False)] * 20)
  modes = drive(engine, [('blended', 1.0, True)] + [('acc', 1.0, False)] * 30)

  assert modes[0] == 'blended'
  assert all(mode == 'blended' for mode in modes)


@pytest.mark.parametrize("a_prev,a_target,expected", [
  (0.0, 1.5, MODE_BLEND_JERK_UP * DT),
  (1.5, 0.0, 1.5 - MODE_BLEND_JERK_DOWN * DT),
  (0.2, MODE_BLEND_BRAKE_PASS - 0.5, MODE_BLEND_BRAKE_PASS - 0.5),
  (0.3, 0.31, 0.31),
])
def test_limit_mode_transition(a_prev, a_target, expected):
  assert limit_mode_transition(a_target, a_prev, DT) == pytest.approx(expected)


def test_planner_output_does_not_step_on_mode_switch():
  v_ego = 12.0
  planner = build_planner(init_v=v_ego)
  e2e = {'active': True}
  planner.is_e2e = lambda sm: e2e['active']
  sm = build_sm(v_ego, 150.0, v_ego + 5.0, CRUISE)

  for _ in range(100):
    planner.update(sm)
  blended_output = float(planner.output_a_target)

  e2e['active'] = False
  outputs = [blended_output]
  for _ in range(60):
    planner.update(sm)
    outputs.append(float(planner.output_a_target))

  steps = [cur - prev for prev, cur in zip(outputs, outputs[1:], strict=False)]
  assert max(steps) <= MODE_BLEND_JERK_UP * planner.dt + 1e-6
  assert outputs[-1] > blended_output + 0.5
