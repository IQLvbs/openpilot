from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from iqdbc.can import CANParser
from iqdbc.car import Bus, gen_empty_fingerprint, structs
from iqdbc.car.hyundai.carcontroller import HyundaiJerk
from iqdbc.car.hyundai.interface import CarInterface
from iqdbc.car.hyundai.values import CAR, DBC, HyundaiFlags
from iqpilot.common.params import Params

PATHS = ('radar', 'camera', 'canfd', 'canfd_camera', 'canfd_hda2')


@pytest.fixture(autouse=True)
def isolated_params(monkeypatch, tmp_path):
  monkeypatch.setenv('PARAMS_ROOT', str(tmp_path))


def make_car(path):
  canfd, camera = path.startswith('canfd'), path in ('camera', 'canfd_camera')
  candidate = CAR.KIA_EV6 if canfd else CAR.HYUNDAI_SONATA
  Params().put_int('HyundaiCameraSCC', int(camera))
  fingerprint = gen_empty_fingerprint()
  if camera and not canfd:
    fingerprint[2][0x389] = 8
  cp = CarInterface.get_params(candidate, fingerprint, [], True, False, False)
  if canfd and not camera:
    cp.flags &= ~HyundaiFlags.CAMERA_SCC.value
  if path == 'canfd_hda2':
    cp.flags |= HyundaiFlags.CANFD_HDA2.value
  cp_iq = CarInterface.get_params_iq(cp, candidate, fingerprint, [], True, False, False)
  car = CarInterface(cp, cp_iq)
  car.update([(0, [])])
  car.CS.out = structs.CarState(vEgo=20.0, vEgoRaw=20.0)
  car.CS.out.cruiseState.available = True
  car.CS.out.gearShifter = structs.CarState.GearShifter.drive
  if camera:
    names = ('SCC_CONTROL',) if canfd else ('SCC11', 'SCC12', 'SCC13', 'SCC14')
    for name in names:
      setattr(car.CS, name.lower(), dict(car.CS.cp_cam.vl[name]))
  return car


def policy(**kwargs):
  values = dict(valid=True, comfortEnabled=True, jerkUp=0.89, jerkDown=3.19,
                accelDeviationAbove=0.18, accelDeviationBelow=0.06)
  values.update(kwargs)
  return structs.IQCarControl.LongitudinalComfort(**values)


def apply(car, data=None, *, gate=True, gas=False, fcw=False, active=True, stopping=False, accel=0.6):
  cc = structs.CarControl()
  cc.enabled = active
  cc.longActive = active and not gas
  cc.longComfortMode = gate
  cc.cruiseControl.override = gas
  cc.actuators.accel = accel
  cc.actuators.jerk = 0.3
  cc.actuators.longControlState = 'off' if not active else 'stopping' if stopping else 'pid'
  if fcw:
    cc.hudControl.visualAlert = structs.CarControl.HUDControl.VisualAlert.fcw
  cc.hudControl.setSpeed = 25.0
  car.CS.out.gasPressed = gas
  iq = structs.IQCarControl()
  if data is not None:
    iq.longitudinalComfort = data
  car.CC.frame += -car.CC.frame % 2
  now = (car.CC.frame + 1) * 10_000_000
  _, messages = car.apply(cc.as_reader(), iq, now)
  canfd = car.CP.flags & HyundaiFlags.CANFD
  name, bus = ('SCC_CONTROL', car.CC.CAN.ECAN) if canfd else ('SCC14', 0)
  parser = CANParser(DBC[car.CP.carFingerprint][Bus.pt], [(name, 0)], bus)
  assert any(addr == parser.dbc.name_to_msg[name].address and src == bus for addr, _, src in messages)
  parser.update([(now, messages)])
  return messages, dict(parser.vl[name])


@pytest.mark.parametrize('candidate', list(CAR))
@pytest.mark.parametrize('openpilot_long', [False, True])
def test_all_hyundai_platforms_use_native_jerk_and_only_classic_scc14_senders_advertise_bands(candidate, openpilot_long):
  fingerprint = gen_empty_fingerprint()
  cp = CarInterface.get_params(candidate, fingerprint, [], openpilot_long, False, False)
  cp_iq = CarInterface.get_params_iq(cp, candidate, fingerprint, [], openpilot_long, False, False)
  capability = cp_iq.longitudinalActuation
  assert capability.jerkControl == 'nativeAsymmetric'
  assert capability.maxJerkUp == capability.maxJerkDown == 5.0
  has_band = cp.openpilotLongitudinalControl and not cp.flags & (HyundaiFlags.CANFD | HyundaiFlags.CAMERA_SCC)
  assert capability.comfortBand == ('asymmetric' if has_band else 'none')


@pytest.mark.parametrize('scc14_bus,scc14_size', [(None, None), (0, 8), (2, 8), (2, 16)])
def test_classic_camera_band_capability_requires_an_scc14_on_its_forwarded_bus(scc14_bus, scc14_size):
  Params().put_int('HyundaiCameraSCC', 1)
  fingerprint = gen_empty_fingerprint()
  if scc14_bus is not None:
    fingerprint[scc14_bus][0x389] = scc14_size
  cp = CarInterface.get_params(CAR.HYUNDAI_SONATA, fingerprint, [], True, False, False)
  capability = CarInterface.get_params_iq(cp, CAR.HYUNDAI_SONATA, fingerprint, [], True, False, False).longitudinalActuation
  assert capability.comfortBand == ('asymmetric' if (scc14_bus, scc14_size) == (2, 8) else 'none')


@pytest.mark.parametrize('path', PATHS)
@pytest.mark.parametrize('state', [{}, {'gas': True}, {'fcw': True}, {'active': False}, {'stopping': True}])
def test_gate_off_can_is_byte_identical_even_when_policy_requests_different_limits(path, state):
  legacy, gated = make_car(path), make_car(path)
  for accel in (0.0, 0.8, -0.7, -3.3):
    assert apply(legacy, gate=False, accel=accel, **state)[0] == apply(gated, policy(), gate=False, accel=accel, **state)[0]


@pytest.mark.parametrize('canfd', [False, True])
@pytest.mark.parametrize('accel', [-3.3, 0.0, 1.5])
@pytest.mark.parametrize('mpc_jerk', [-1.0, 0.0, 2.0])
def test_gate_off_retains_carrotpilot_jerk_and_band_requirements(canfd, accel, mpc_jerk):
  cc = structs.CarControl()
  cc.actuators.longControlState = 'pid'
  cc.actuators.jerk = mpc_jerk
  jerk = HyundaiJerk()
  jerk.make_jerk(SimpleNamespace(flags=HyundaiFlags.CANFD if canfd else 0), SimpleNamespace(out=SimpleNamespace(aEgo=0.0)),
                 accel, cc.actuators, cc, structs.IQCarControl(longitudinalComfort=policy()))
  assert jerk.jerk_u == pytest.approx(np.clip(2 * mpc_jerk, 0.5, 5.0))
  minimum_down = np.clip(1.2 + 2 * max(0.0, -accel - 2.8), 1.2, 5.0) if canfd else 1.0
  assert jerk.jerk_l == pytest.approx(max(minimum_down, np.clip(-4 * mpc_jerk, 1.2 if canfd else 1.0, 5.0)))
  assert jerk.cb_upper == pytest.approx(0.0 if canfd else np.clip(0.9 + 0.2 * accel, 0.0, 1.2))
  assert jerk.cb_lower == pytest.approx(0.0 if canfd else np.clip(0.8 + 0.2 * accel, 0.0, 1.2))


@pytest.mark.parametrize('path', PATHS)
@pytest.mark.parametrize('data', [None, policy(valid=False)] + [
  policy(**{field: value}) for field in ('jerkUp', 'jerkDown', 'accelDeviationAbove', 'accelDeviationBelow')
  for value in (float('nan'), float('inf'), -0.1)
] + [policy(jerkUp=0.0), policy(jerkDown=0.0)])
def test_missing_invalid_or_nonfinite_data_restores_legacy_can_after_a_valid_tick(path, data):
  legacy, gated = make_car(path), make_car(path)
  apply(legacy, policy())
  apply(gated, policy())
  assert apply(gated, data)[0] == apply(legacy, gate=False)[0]


@pytest.mark.parametrize('path', PATHS)
def test_native_can_uses_independent_jerk_limits_without_rounding_above_the_policy(path):
  data = policy()
  _, fields = apply(make_car(path), data)
  for name, maximum in [('JerkUpperLimit', data.jerkUp), ('JerkLowerLimit', data.jerkDown)]:
    assert 0 < fields[name] <= maximum < fields[name] + 0.1
  assert fields['JerkUpperLimit'] < fields['JerkLowerLimit']


@pytest.mark.parametrize('path', PATHS)
def test_native_jerk_stays_within_the_ports_five_meter_per_second_cubed_ceiling(path):
  _, fields = apply(make_car(path), policy(jerkUp=8.0, jerkDown=9.0, comfortEnabled=False))
  assert fields['JerkUpperLimit'] == fields['JerkLowerLimit'] == 5.0


@pytest.mark.parametrize('path', PATHS)
def test_comfort_mode_stops_exactly_like_comfort_mode_off(path):
  # The shared policy sends maximum jerk while stopping; the port's own soft
  # stopping jerk must win so the final stop isn't harsher with comfort on.
  legacy, gated = make_car(path), make_car(path)
  for accel in (-0.2, -0.6, -1.2):
    data = policy(jerkUp=5.0, jerkDown=5.0, comfortEnabled=False)
    assert apply(legacy, gate=False, accel=accel, stopping=True)[0] == apply(gated, data, accel=accel, stopping=True)[0]


@pytest.mark.parametrize('path', ['radar', 'camera'])
def test_scc14_upper_is_physical_above_and_lower_is_physical_below_with_conservative_steps(path):
  data = policy()
  _, fields = apply(make_car(path), data)
  for name, maximum in [('ComfortBandUpper', data.accelDeviationAbove), ('ComfortBandLower', data.accelDeviationBelow)]:
    assert 0 < fields[name] <= maximum < fields[name] + 0.02
  assert fields['ComfortBandUpper'] > fields['ComfortBandLower']


@pytest.mark.parametrize('path', ['radar', 'camera'])
def test_classic_steady_band_reaches_at_least_point_two_two_within_the_active_stock_envelope(path):
  _, fields = apply(make_car(path), policy(accelDeviationAbove=0.375, accelDeviationBelow=0.375))
  assert 0.22 <= fields['ComfortBandUpper'] <= 0.24
  assert 0.22 <= fields['ComfortBandLower'] <= 0.24


@pytest.mark.parametrize('path', ['radar', 'camera'])
def test_a_band_just_below_a_dbc_step_never_rounds_up(path):
  data = policy(accelDeviationAbove=0.18 - 1e-9, accelDeviationBelow=0.06 - 1e-9)
  _, fields = apply(make_car(path), data)
  assert fields['ComfortBandUpper'] <= data.accelDeviationAbove
  assert fields['ComfortBandLower'] <= data.accelDeviationBelow


@pytest.mark.parametrize('path', PATHS)
@pytest.mark.parametrize('bypass', ['hud_fcw', 'policy_fcw', 'lead_braking'])
def test_emergencies_restore_downward_authority_and_zero_bands(path, bypass):
  data = policy(bypassJerkLimit=bypass == 'policy_fcw', bypassJerkDownLimit=bypass == 'lead_braking')
  _, fields = apply(make_car(path), data, fcw=bypass == 'hud_fcw')
  assert fields['JerkLowerLimit'] == 5.0
  assert fields['JerkUpperLimit'] == (pytest.approx(0.8) if bypass == 'lead_braking' else 5.0)
  prefix = 'AccelLimitBand' if path.startswith('canfd') else 'ComfortBand'
  assert fields[prefix + 'Upper'] == fields[prefix + 'Lower'] == 0.0


@pytest.mark.parametrize('path', PATHS)
@pytest.mark.parametrize('stopping', [False, True])
def test_gas_override_uses_published_soft_jerk_even_with_a_pending_stop(path, stopping):
  data = policy(jerkUp=0.5, jerkDown=0.5, comfortEnabled=False)
  _, fields = apply(make_car(path), data, gas=True, stopping=stopping)
  assert fields['JerkUpperLimit'] == fields['JerkLowerLimit'] == 0.5
  prefix = 'AccelLimitBand' if path.startswith('canfd') else 'ComfortBand'
  assert fields[prefix + 'Upper'] == fields[prefix + 'Lower'] == 0.0


@pytest.mark.parametrize('path', PATHS)
def test_disabling_the_band_keeps_native_jerk_and_canfd_drive_mode(path):
  _, fields = apply(make_car(path), replace(policy(), comfortEnabled=False))
  assert fields['JerkUpperLimit'] <= 0.89
  assert fields['JerkLowerLimit'] <= 3.19
  prefix = 'AccelLimitBand' if path.startswith('canfd') else 'ComfortBand'
  assert fields[prefix + 'Upper'] == fields[prefix + 'Lower'] == 0.0
  if path.startswith('canfd'):
    assert fields['DriveMode'] == 0
