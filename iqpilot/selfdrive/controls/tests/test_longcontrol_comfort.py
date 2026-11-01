from types import SimpleNamespace

import pytest

from iqdbc.car import gen_empty_fingerprint, structs
from iqdbc.car.hyundai.interface import CarInterface
from iqpilot.common.params import Params
from iqpilot.common.realtime import DT_CTRL
from iqpilot.selfdrive.controls.lib.longcontrol import LongControl, LongCtrlState
from iqpilot.selfdrive.controls.lib.longitudinal_comfort import LongitudinalComfort

PHYSICAL = (-3.5, 2.0)


def control_loop(native=False):
  cp = CarInterface.get_non_essential_params('HYUNDAI_SONATA')
  cp_iq = CarInterface.get_non_essential_params_iq(cp, 'HYUNDAI_SONATA')
  if native:
    cp_iq.longitudinalActuation.jerkControl = 'nativeAsymmetric'
    cp_iq.longitudinalActuation.comfortBand = 'asymmetric'
  else:
    # Exercise the shared software fallback independently of Hyundai's adapter.
    cp_iq.longitudinalActuation = structs.IQCarParams.LongitudinalActuation()
  return LongControl(cp, cp_iq), LongitudinalComfort(cp_iq.longitudinalActuation)


def car_state(**kwargs):
  values = dict(vEgo=20.0, vCruise=100.0, aEgo=0.0, gasPressed=False, brakePressed=False, standstill=False,
                cruiseState=SimpleNamespace(standstill=False))
  values.update(kwargs)
  return SimpleNamespace(**values)


def update(control, comfort, *, cs=None, a_target=1.0, limits=PHYSICAL, active=True, should_stop=False, **kwargs):
  return control.update(active, cs or car_state(), a_target, should_stop, limits, v_target_now=20.0,
                        comfort=comfort, **kwargs)


def test_gate_off_is_identical_to_the_original_longcontrol_path():
  legacy, _ = control_loop()
  gated, comfort = control_loop()
  for i in range(600):
    inputs = dict(cs=car_state(aEgo=0.2, vEgo=1.0 if i % 30 < 4 else 20.0), a_target=(i % 40 - 20) / 10,
                  should_stop=i % 30 < 4, active=i % 50 != 0, gas_override=i % 20 < 3)
    assert update(gated, comfort, comfort_enabled=False, **inputs) == update(legacy, None, **inputs)
    assert (gated.pid.neg_limit, gated.pid.pos_limit) == PHYSICAL
    assert gated.pid.i == legacy.pid.i


def test_software_ramp_limits_the_pid_itself_and_does_not_wind_up_the_integrator():
  control, comfort = control_loop()
  first = update(control, comfort, comfort_enabled=True)
  assert 0 < first <= 0.8 * DT_CTRL
  assert control.pid.pos_limit == first
  assert control.pid.i == 0  # sustained saturation cannot accumulate positive error
  for _ in range(30):
    previous = control.last_output_accel
    output = update(control, comfort, comfort_enabled=True)
    assert output <= previous + comfort.control.jerkUp * DT_CTRL + 1e-9
    assert control.pid.pos_limit == output
    assert control.pid.i == 0


@pytest.mark.parametrize('native', [False, True])
def test_normal_control_has_no_extra_software_bound(native):
  control, comfort = control_loop(native)
  for _ in range(401):
    update(control, comfort, comfort_enabled=True)
  previous = control.last_output_accel
  output = update(control, comfort, comfort_enabled=True, a_target=-1.4)
  assert (control.pid.neg_limit, control.pid.pos_limit) == PHYSICAL
  assert previous - output > 5.0 * DT_CTRL


@pytest.mark.parametrize('limits', [(-3.5, -0.5), (1.0, 2.0)])
def test_physical_bounds_win_when_the_comfort_intersection_is_empty(limits):
  control, comfort = control_loop()
  update(control, comfort, comfort_enabled=True)
  output = update(control, comfort, comfort_enabled=True, limits=limits)
  assert (control.pid.neg_limit, control.pid.pos_limit) == limits
  assert limits[0] <= output <= limits[1]
  # The next ramp interval surrounds the final physically clipped output.
  update(control, comfort, comfort_enabled=True)
  assert control.pid.neg_limit == pytest.approx(output - comfort.control.jerkDown * DT_CTRL)
  assert control.pid.pos_limit == pytest.approx(min(PHYSICAL[1], output + comfort.control.jerkUp * DT_CTRL))


def test_lead_braking_removes_only_the_downward_pid_bound_and_fcw_removes_both():
  control, comfort = control_loop()
  update(control, comfort, comfort_enabled=True, a_target=1.0)
  lead_output = update(control, comfort, comfort_enabled=True, a_target=0.8, plan_source='lead0')
  assert control.pid.neg_limit == PHYSICAL[0]
  assert control.pid.pos_limit < PHYSICAL[1]
  assert lead_output > 0  # a reduction in positive target is also lead braking
  emergency_output = update(control, comfort, comfort_enabled=True, a_target=-2.0, fcw=True)
  assert (control.pid.neg_limit, control.pid.pos_limit) == PHYSICAL
  assert emergency_output < -2.0


def test_gas_override_while_long_active_suspends_the_pid_bounds():
  control, comfort = control_loop()
  output = update(control, comfort, comfort_enabled=True, cs=car_state(gasPressed=True), gas_override=True)
  assert output > 0.8 * DT_CTRL
  assert (control.pid.neg_limit, control.pid.pos_limit) == PHYSICAL
  released = update(control, comfort, comfort_enabled=True, cs=car_state(aEgo=0.4))
  assert 0.4 < released <= 0.4 + 0.8 * DT_CTRL + 1e-9


@pytest.mark.parametrize('smooth', [False, True])
def test_stopping_and_departure_match_legacy_without_comfort_delays(smooth):
  Params().put_bool('IQForceStops', smooth)
  legacy, _ = control_loop()
  control, comfort = control_loop()
  for speed, stopping in [(2.8, True), (0.5, True), (0.0, True), (0.0, False), (1.0, False), (2.6, False)]:
    cs = car_state(vEgo=speed, standstill=speed == 0.0)
    inputs = dict(cs=cs, a_target=-0.5 if stopping else 0.8, should_stop=stopping)
    assert update(control, comfort, comfort_enabled=True, **inputs) == update(legacy, None, **inputs)
    assert (control.pid.neg_limit, control.pid.pos_limit) == PHYSICAL
  assert control.long_control_state == LongCtrlState.pid


def test_native_actuation_never_adds_a_second_pid_slew_limit():
  control, comfort = control_loop(native=True)
  output = update(control, comfort, comfort_enabled=True)
  assert comfort.control.jerkUp == 0.8
  assert (control.pid.neg_limit, control.pid.pos_limit) == PHYSICAL
  assert output > 0.8 * DT_CTRL


@pytest.mark.parametrize('candidate', ['HYUNDAI_SONATA', 'KIA_EV6'])
def test_hyundai_uses_native_jerk_from_the_first_pid_tick_without_a_software_limit(candidate):
  fingerprint = gen_empty_fingerprint()
  cp = CarInterface.get_params(candidate, fingerprint, [], True, False, False)
  cp_iq = CarInterface.get_params_iq(cp, candidate, fingerprint, [], True, False, False)
  control = LongControl(cp, cp_iq)
  comfort = LongitudinalComfort(cp_iq.longitudinalActuation)
  output = update(control, comfort, comfort_enabled=True)
  assert comfort.control.valid and 0 < comfort.control.jerkUp <= 0.8
  assert (control.pid.neg_limit, control.pid.pos_limit) == PHYSICAL
  assert output > 0.8 * DT_CTRL


def test_native_band_uses_the_final_pid_command_when_it_exceeds_the_target():
  control, comfort = control_loop(native=True)
  output = update(control, comfort, comfort_enabled=True, a_target=0.05, cs=car_state(aEgo=-1.0))
  assert output > 0.8 > 0.05
  assert 0 < comfort.control.accelDeviationBelow < comfort.control.accelDeviationAbove < 0.18
