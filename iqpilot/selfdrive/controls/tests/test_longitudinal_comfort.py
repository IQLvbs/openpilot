from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from iqdbc.car import structs
from iqpilot.common.realtime import DT_CTRL
from iqpilot.selfdrive.controls.lib.longitudinal_comfort import LongitudinalComfort

PHYSICAL = (-3.5, 2.0)


class ComfortLoop:
  def __init__(self, native=False):
    capability = structs.IQCarParams.LongitudinalActuation()
    if native:
      capability.jerkControl = capability.JerkControl.nativeAsymmetric
      capability.comfortBand = capability.ComfortBand.asymmetric
      capability.maxJerkUp = capability.maxJerkDown = 5.0
    self.policy = LongitudinalComfort(capability)
    self.cs = SimpleNamespace(vEgo=20.0, aEgo=0.0, vCruise=100.0, gasPressed=False, brakePressed=False)

  def tick(self, *, a_target=0.0, command=0.0, previous_command=0.0, **kwargs):
    inputs = dict(enabled=True, active=True, car_state=self.cs, a_target=a_target, should_stop=False, stopping=False,
                  previous_command=previous_command, physical_limits=PHYSICAL)
    inputs.update(kwargs)
    bounds = self.policy.update(**inputs)
    self.policy.finish(a_target, command, self.cs.aEgo)
    return bounds, replace(self.policy.control)

  def advance(self, seconds, **kwargs):
    return [self.tick(**kwargs) for _ in range(round(seconds / DT_CTRL))]


def test_gate_off_has_only_physical_bounds_and_forgets_the_ramp():
  loop = ComfortLoop()
  for bounds, control in loop.advance(5.0, enabled=False, fcw=True, a_target=-2.0):
    assert bounds == PHYSICAL
    assert not control.valid
  first, _ = loop.tick()
  loop.advance(2.0)
  loop.tick(enabled=False)
  restarted, _ = loop.tick()
  assert restarted == first
  assert first[0] > PHYSICAL[0] and first[1] < PHYSICAL[1]


def test_four_second_ramp_rises_slowly_then_removes_software_limits():
  loop = ComfortLoop()
  samples = loop.advance(4.01)
  upper = [bounds[1] for bounds, _ in samples[:-1]]
  assert upper[0] / DT_CTRL == pytest.approx(0.8)
  assert np.diff(upper) / DT_CTRL / DT_CTRL == pytest.approx(1.05)
  assert samples[-2][0][1] < 5.0 * DT_CTRL
  assert samples[-1][0] == PHYSICAL


def test_native_actuation_uses_the_same_ramp_without_software_bounds():
  loop = ComfortLoop(native=True)
  samples = loop.advance(4.01)
  assert all(bounds == PHYSICAL for bounds, _ in samples)
  assert samples[0][1].jerkUp < samples[200][1].jerkUp < samples[-1][1].jerkUp
  assert samples[-1][1].jerkUp == samples[-1][1].jerkDown == 5.0


@pytest.mark.parametrize('measured', [0.6, -0.6, 50.0, -50.0, float('nan'), float('inf')])
def test_engagement_seeds_from_finite_measured_accel_within_physical_limits(measured):
  loop = ComfortLoop()
  loop.cs.aEgo = measured
  bounds, _ = loop.tick(previous_command=-1.0)
  seed = np.clip(measured, *PHYSICAL) if np.isfinite(measured) else 0.0
  assert bounds[0] <= seed <= bounds[1]
  assert bounds[0] >= max(PHYSICAL[0], seed - 0.8 * DT_CTRL)
  assert bounds[1] <= min(PHYSICAL[1], seed + 0.8 * DT_CTRL)


def test_speed_edits_reset_once_per_episode_and_a_held_button_cannot_pin_the_ramp():
  loop = ComfortLoop()
  loop.advance(5.0)
  loop.cs.vCruise += 0.5
  first, _ = loop.tick(previous_command=0.4)
  assert first[0] < 0.4 < first[1] < 0.42
  for i in range(600):
    if i % 10 == 0:
      loop.cs.vCruise += 0.5
    bounds, _ = loop.tick(previous_command=0.4)
    if i >= 400:
      assert bounds == PHYSICAL
  loop.advance(0.6)
  loop.cs.vCruise -= 1.0
  next_episode, _ = loop.tick(previous_command=-0.3)
  assert -0.32 < next_episode[0] < -0.3 < next_episode[1] < -0.28


@pytest.mark.parametrize('invalid', [0.0, -1.0, 255.0, float('nan'), float('inf')])
def test_unset_cruise_and_subthreshold_changes_do_not_restart_a_ramp(invalid):
  loop = ComfortLoop()
  loop.cs.vCruise = invalid
  loop.advance(5.0)
  loop.cs.vCruise = 100.0
  assert loop.tick()[0] == PHYSICAL  # first initialized speed is a baseline
  loop.cs.vCruise += 0.49
  assert loop.tick()[0] == PHYSICAL
  loop.cs.vCruise = invalid
  assert loop.tick()[0] == PHYSICAL
  loop.cs.vCruise = 100.5
  assert loop.tick()[0][1] < 0.02


@pytest.mark.parametrize('override', ['gas', 'event', 'brake', 'inactive'])
def test_override_suspends_constraints_and_release_restarts_from_measured_accel(override):
  loop = ComfortLoop()
  loop.advance(5.0)
  loop.cs.gasPressed = override == 'gas'
  loop.cs.brakePressed = override == 'brake'
  bounds, _ = loop.tick(driver_override=override == 'event', active=override != 'inactive')
  assert bounds == PHYSICAL
  loop.cs.gasPressed = loop.cs.brakePressed = False
  loop.cs.aEgo = 0.5
  bounds, _ = loop.tick(previous_command=-0.5)
  assert 0.48 < bounds[0] < 0.5 < bounds[1] < 0.52


def test_native_gas_override_softens_jerk_so_the_car_does_not_cut_power_abruptly():
  loop = ComfortLoop(native=True)
  loop.advance(5.0)
  loop.cs.gasPressed = True
  _, control = loop.tick()
  assert control.valid and control.jerkUp == control.jerkDown == pytest.approx(0.5)


def test_plan_ripple_while_following_a_lead_keeps_the_comfort_band_open():
  loop = ComfortLoop(native=True)
  loop.advance(5.0)
  ripple = [0.02 * np.sin(i / 3) for i in range(200)]
  controls = [loop.tick(a_target=a, command=a, previous_command=a, plan_source='lead0')[1] for a in ripple]
  assert not any(c.bypassJerkDownLimit for c in controls)
  assert controls[-1].comfortEnabled and controls[-1].accelDeviationAbove > 0


@pytest.mark.parametrize('source', ['lead0', 'lead1', 'lead2'])
def test_lead_target_reductions_bypass_even_while_still_positive_and_until_caught_up(source):
  loop = ComfortLoop()
  loop.tick(a_target=0.6, previous_command=0.1)
  bounds, control = loop.tick(a_target=0.5, previous_command=0.1, plan_source=source)
  assert control.bypassJerkDownLimit and bounds[0] == PHYSICAL[0]
  assert bounds[1] < PHYSICAL[1]  # upward comfort still applies
  loop.advance(0.3, a_target=0.5, previous_command=0.1, plan_source=source)
  assert loop.tick(a_target=0.5, previous_command=0.1)[1].bypassJerkDownLimit
  # A constant target still needs unrestricted braking while the command catches up.
  held = loop.advance(0.8, a_target=0.4, previous_command=0.8, plan_source=source)
  assert all(c.bypassJerkDownLimit and b[0] == PHYSICAL[0] for b, c in held)
  assert not loop.tick(a_target=0.4, previous_command=0.3, plan_source=source)[1].bypassJerkDownLimit


def test_lead_braking_hold_expires_half_a_second_after_the_last_reduction():
  loop = ComfortLoop()
  loop.tick(a_target=0.8)
  loop.tick(a_target=0.6, plan_source='lead0')
  assert all(c.bypassJerkDownLimit for _, c in loop.advance(0.49, a_target=0.6))
  assert not loop.tick(a_target=0.6)[1].bypassJerkDownLimit


@pytest.mark.parametrize('source', ['cruise', 'e2e', 'lead0'])
def test_hard_braking_bypasses_downward_comfort_independently_of_source(source):
  loop = ComfortLoop(native=True)
  _, control = loop.tick(a_target=-1.5, plan_source=source)
  assert control.bypassJerkDownLimit
  assert control.jerkDown == 5.0 and control.jerkUp < 1.0
  assert control.accelDeviationAbove == control.accelDeviationBelow == 0


@pytest.mark.parametrize('native', [False, True])
def test_fcw_bypasses_both_jerk_limits_and_zeros_bands(native):
  loop = ComfortLoop(native)
  bounds, control = loop.tick(fcw=True, a_target=-1.0)
  assert bounds == PHYSICAL
  assert control.bypassJerkLimit
  assert control.jerkUp == control.jerkDown == 5.0
  assert control.accelDeviationAbove == control.accelDeviationBelow == 0


@pytest.mark.parametrize('reason', ['slow', 'stop', 'stopping'])
def test_stop_suppression_discards_the_ramp_and_waits_for_sustained_departure(reason):
  loop = ComfortLoop(native=True)
  loop.tick()
  loop.cs.vEgo = 1.9 if reason == 'slow' else 20.0
  bounds, control = loop.tick(should_stop=reason == 'stop', stopping=reason == 'stopping')
  assert bounds == PHYSICAL
  assert control.jerkUp == control.jerkDown == 5.0
  assert control.accelDeviationAbove == control.accelDeviationBelow == 0
  loop.cs.vEgo = 2.6
  loop.advance(0.3)
  loop.cs.vEgo = 2.5
  assert loop.tick()[1].accelDeviationAbove == 0
  loop.cs.vEgo = 2.6
  assert all(c.accelDeviationAbove == 0 for _, c in loop.advance(0.49))
  assert loop.tick()[1].comfortEnabled
  assert loop.tick()[1].jerkUp == 5.0


@pytest.mark.parametrize('target,command', [(0.8, 0.8), (-0.8, -0.8), (0.01, -0.8), (-0.01, 0.8), (0.3, -0.8)])
def test_band_allows_more_deviation_in_the_demand_direction(target, command):
  loop = ComfortLoop(native=True)
  _, control = loop.tick(a_target=target, command=command)
  sign = target if abs(target) > 0.10 else command
  in_direction, against = (control.accelDeviationAbove, control.accelDeviationBelow) if sign > 0 else (
    control.accelDeviationBelow, control.accelDeviationAbove)
  assert 0 < against < in_direction < 0.18
  assert against == pytest.approx(0.02)
  assert in_direction == pytest.approx(0.10)


def test_steady_cruise_allows_at_least_a_third_of_a_meter_per_second_squared_of_deviation():
  loop = ComfortLoop(native=True)
  # Several seconds of small cruise corrections must keep a broad band.
  controls = [loop.tick(a_target=0.05, command=0.08)[1] for _ in range(500)]
  assert all(0.35 <= c.accelDeviationAbove <= 0.375 and 0.35 <= c.accelDeviationBelow <= 0.375 for c in controls)


@pytest.mark.parametrize('demand', [0.8, -0.8])
def test_heavy_demand_immediately_closes_the_opposing_side_to_at_most_two_hundredths(demand):
  loop = ComfortLoop(native=True)
  loop.advance(3.0)
  _, control = loop.tick(a_target=demand, command=demand)
  in_direction, against = (control.accelDeviationAbove, control.accelDeviationBelow) if demand > 0 else (
    control.accelDeviationBelow, control.accelDeviationAbove)
  assert 0 <= against <= 0.02 + 1e-9
  assert against < in_direction <= 0.10 + 1e-9


def test_bands_tighten_immediately_but_widen_only_after_sustained_low_measured_accel():
  loop = ComfortLoop(native=True)
  _, wide = loop.tick()
  loop.cs.aEgo = 0.3
  _, tight = loop.tick(a_target=0.8, command=0.8)
  assert tight.accelDeviationAbove < wide.accelDeviationAbove
  assert tight.accelDeviationBelow < wide.accelDeviationBelow
  assert all(c.accelDeviationAbove == tight.accelDeviationAbove for _, c in loop.advance(0.8))
  loop.cs.aEgo = 0.0
  assert all(c.accelDeviationAbove == tight.accelDeviationAbove for _, c in loop.advance(0.49))
  _, widening = loop.tick()
  assert 0 < widening.accelDeviationAbove - tight.accelDeviationAbove <= 0.24 * DT_CTRL + 1e-9
  recovered = loop.advance(1.5)[-1][1]
  assert recovered.accelDeviationAbove >= 0.35
  assert recovered.accelDeviationBelow >= 0.35
