from dataclasses import replace

import pytest

from iqdbc.can import CANParser
from iqdbc.car import Bus, structs
from iqdbc.car.volkswagen.interface import CarInterface
from iqdbc.car.volkswagen.values import CAR, DBC, VolkswagenFlags

NATIVE_CARS = (CAR.VOLKSWAGEN_GOLF_MK7, CAR.VOLKSWAGEN_ID4_MK1, CAR.VOLKSWAGEN_GOLF_MK8)


def make_car(candidate):
  fingerprint = {bus: {} for bus in range(7)}
  cp = CarInterface.get_params(candidate, fingerprint, [], alpha_long=True, is_release=False, docs=False)
  cp_iq = CarInterface.get_params_iq(cp, candidate, fingerprint, [], alpha_long=True, is_release_iq=False, docs=False)
  car = CarInterface(cp, cp_iq)
  car.update([(0, [])])
  car.CS.out = structs.CarState()
  car.CS.out.cruiseState.available = True
  car.CS.acc_type = 1
  return car


def policy(**kwargs):
  values = dict(valid=True, jerkUp=0.8, jerkDown=3.1, comfortEnabled=True, accelDeviationAbove=0.18, accelDeviationBelow=0.1)
  values.update(kwargs)
  return structs.IQCarControl.LongitudinalComfort(**values)


def apply(car, data=None, *, gate=True, fcw=False, gas=False, active=True, hold=False):
  cs = car.CS
  cs.out.vEgo = cs.out.vEgoRaw = 0.0 if hold else 20.0
  cs.out.gasPressed = gas
  cs.esp_hold_confirmation = hold
  cc = structs.CarControl()
  cc.enabled = cc.longActive = active
  cc.longComfortMode = gate
  cc.actuators.accel = -0.5 if hold else 0.6
  cc.actuators.longControlState = structs.CarControl.Actuators.LongControlState.stopping if hold else (
    structs.CarControl.Actuators.LongControlState.pid)
  if fcw:
    cc.hudControl.visualAlert = structs.CarControl.HUDControl.VisualAlert.fcw
  iq = structs.IQCarControl()
  if data is not None:
    iq.longitudinalComfort = data
  car.CC.frame += -car.CC.frame % car.CC.CCP.ACC_CONTROL_STEP
  now = (car.CC.frame + 1) * 10_000_000
  output, messages = car.apply(cc.as_reader(), iq, now)
  name = 'ACC_18' if car.CP.flags & (VolkswagenFlags.MEB | VolkswagenFlags.MQB_EVO) else 'ACC_06'
  parser = CANParser(DBC[car.CP.carFingerprint][Bus.pt], [(name, 0)], car.CC.CAN.pt)
  parser.update([(now, messages)])
  assert parser.vl[name]['ACC_Status_ACC'] > 0
  return output, messages, parser.vl[name]


@pytest.mark.parametrize('candidate', list(CAR))
def test_only_mqb_meb_and_mqb_evo_advertise_native_longitudinal_actuation(candidate):
  cp = CarInterface.get_non_essential_params(candidate)
  cp_iq = CarInterface.get_non_essential_params_iq(cp, candidate)
  actuation = cp_iq.longitudinalActuation
  native = not cp.flags & (VolkswagenFlags.PQ | VolkswagenFlags.MLB)
  assert actuation.jerkControl == ('nativeAsymmetric' if native else 'software')
  assert actuation.comfortBand == ('asymmetric' if native else 'none')
  if native:
    assert actuation.maxJerkUp == actuation.maxJerkDown == 5.0


@pytest.mark.parametrize('candidate', NATIVE_CARS)
@pytest.mark.parametrize('state', [{}, {'gas': True}, {'fcw': True}, {'hold': True}, {'active': False}])
def test_gate_off_can_is_byte_identical_with_or_without_policy_data(candidate, state):
  legacy = make_car(candidate)
  gated = make_car(candidate)
  assert apply(legacy, gate=False, **state)[1] == apply(gated, policy(), gate=False, **state)[1]


@pytest.mark.parametrize('candidate', NATIVE_CARS)
def test_invalid_or_missing_comfort_data_keeps_legacy_can(candidate):
  legacy = make_car(candidate)
  invalid = make_car(candidate)
  missing = make_car(candidate)
  expected = apply(legacy, gate=False)[1]
  assert apply(invalid, policy(valid=False))[1] == expected
  assert apply(missing)[1] == expected
  _, _, fields = apply(make_car(candidate), gate=False)
  assert fields['ACC_pos_Sollbeschl_Grad_02'] == fields['ACC_neg_Sollbeschl_Grad_02'] == 4.0
  if candidate == CAR.VOLKSWAGEN_GOLF_MK7:
    assert fields['ACC_zul_Regelabw_unten'] <= 0.2 < fields['ACC_zul_Regelabw_unten'] + 0.024
    assert fields['ACC_zul_Regelabw_oben'] <= 0.2 < fields['ACC_zul_Regelabw_oben'] + 0.0625
  else:
    assert fields['ACC_zul_Regelabw_unten'] == fields['ACC_zul_Regelabw_oben'] == 0.0


@pytest.mark.parametrize('candidate', NATIVE_CARS)
def test_native_can_encodes_independent_policy_jerks_and_conservative_physical_bands(candidate):
  data = policy()
  _, _, fields = apply(make_car(candidate), data)
  assert fields['ACC_pos_Sollbeschl_Grad_02'] == pytest.approx(data.jerkUp, abs=0.025)
  assert fields['ACC_neg_Sollbeschl_Grad_02'] == pytest.approx(data.jerkDown, abs=0.025)
  assert fields['ACC_pos_Sollbeschl_Grad_02'] < fields['ACC_neg_Sollbeschl_Grad_02']
  # The port's physical sign convention: above -> unten, below -> oben.
  for name, maximum, step in [('ACC_zul_Regelabw_unten', data.accelDeviationAbove, 0.024),
                             ('ACC_zul_Regelabw_oben', data.accelDeviationBelow, 0.0625)]:
    assert 0 < fields[name] <= maximum < fields[name] + step


@pytest.mark.parametrize('candidate', NATIVE_CARS)
def test_steady_cruise_uses_a_wider_band_within_each_stock_physical_envelope(candidate):
  _, _, fields = apply(make_car(candidate), policy(accelDeviationAbove=0.375, accelDeviationBelow=0.375))
  assert 0.18 < fields['ACC_zul_Regelabw_unten'] <= 0.192
  assert 0.35 <= fields['ACC_zul_Regelabw_oben'] <= 0.5
  _, _, capped = apply(make_car(candidate), policy(accelDeviationAbove=1.0, accelDeviationBelow=1.0))
  assert capped['ACC_zul_Regelabw_unten'] <= 0.192
  assert capped['ACC_zul_Regelabw_oben'] <= 0.5


@pytest.mark.parametrize('candidate', NATIVE_CARS)
@pytest.mark.parametrize('bypass', ['hud_fcw', 'policy_fcw', 'braking'])
def test_native_emergency_and_braking_bypasses_take_precedence(candidate, bypass):
  data = policy(bypassJerkLimit=bypass == 'policy_fcw', bypassJerkDownLimit=bypass == 'braking')
  _, _, fields = apply(make_car(candidate), data, fcw=bypass == 'hud_fcw')
  assert fields['ACC_neg_Sollbeschl_Grad_02'] == 5.0
  assert fields['ACC_pos_Sollbeschl_Grad_02'] == (pytest.approx(data.jerkUp) if bypass == 'braking' else 5.0)
  assert fields['ACC_zul_Regelabw_unten'] == fields['ACC_zul_Regelabw_oben'] == 0.0


@pytest.mark.parametrize('candidate', NATIVE_CARS[1:])
def test_meb_and_evo_full_hold_still_zero_jerks_and_bands(candidate):
  _, _, fields = apply(make_car(candidate), policy(), hold=True)
  assert fields['ACC_pos_Sollbeschl_Grad_02'] == fields['ACC_neg_Sollbeschl_Grad_02'] == 0.0
  assert fields['ACC_zul_Regelabw_unten'] == fields['ACC_zul_Regelabw_oben'] == 0.0


@pytest.mark.parametrize('candidate', NATIVE_CARS)
def test_nonfinite_comfort_data_uses_legacy_can(candidate):
  assert apply(make_car(candidate), policy(jerkUp=float('nan')))[1] == apply(make_car(candidate), gate=False)[1]


@pytest.mark.parametrize('candidate', NATIVE_CARS)
def test_native_band_can_be_disabled_without_disabling_the_jerk_ramp(candidate):
  _, _, fields = apply(make_car(candidate), replace(policy(), comfortEnabled=False))
  assert fields['ACC_pos_Sollbeschl_Grad_02'] < 1.0
  assert fields['ACC_zul_Regelabw_unten'] == fields['ACC_zul_Regelabw_oben'] == 0.0


@pytest.mark.parametrize('candidate', NATIVE_CARS)
def test_values_just_below_a_dbc_step_never_round_up(candidate):
  data = policy(accelDeviationAbove=0.096 - 1e-9, accelDeviationBelow=0.125 - 1e-9)
  _, _, fields = apply(make_car(candidate), data)
  assert fields['ACC_zul_Regelabw_unten'] <= data.accelDeviationAbove
  assert fields['ACC_zul_Regelabw_oben'] <= data.accelDeviationBelow
