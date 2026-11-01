import pytest

from iqdbc.car import gen_empty_fingerprint, structs
from iqdbc.car.car_helpers import interfaces


def test_ports_without_native_adapters_keep_software_jerk_and_no_native_comfort_band():
  for candidate, interface in interfaces.items():
    cp = interface.get_non_essential_params(candidate)
    if cp.brand in ('volkswagen', 'hyundai'):
      continue
    capability = interface.get_non_essential_params_iq(cp, candidate).longitudinalActuation
    assert capability.jerkControl == 'software', candidate
    assert capability.comfortBand == 'none', candidate


@pytest.mark.parametrize('candidate', ['TOYOTA_RAV4_TSS2', 'TESLA_MODEL_3', 'VOLKSWAGEN_PASSAT_NMS'])
def test_non_migrated_car_side_shaping_ignores_the_new_native_policy(candidate, monkeypatch, tmp_path):
  pq_blend = candidate == 'VOLKSWAGEN_PASSAT_NMS'
  if pq_blend:
    from iqpilot.common.params import Params
    monkeypatch.setenv('PARAMS_ROOT', str(tmp_path))
    Params().put_bool('IQDynamicBlendStockRadar', True)
  interface = interfaces[candidate]
  cars = []
  for _ in range(2):
    fingerprint = gen_empty_fingerprint()
    cp = interface.get_params(candidate, fingerprint, [], True, False, False)
    cp_iq = interface.get_params_iq(cp, candidate, fingerprint, [], True, False, False)
    car = interface(cp, cp_iq)
    car.update([(0, [])])
    car.CS.out.vEgo = car.CS.out.vEgoRaw = 20.0
    car.CS.out.cruiseState.available = True
    if pq_blend:
      car.CS.acc_radar_sta_adr = 1
      car.CS.acc_radar_sollbeschl = -0.6
      car.CS.acc_radar_aendgrad = 2.0
      car.CS.acc_radar_regelabw = 0.1
    cars.append(car)
  legacy_iq, policy_iq = structs.IQCarControl(), structs.IQCarControl()
  legacy_iq.useRadarAccel = policy_iq.useRadarAccel = pq_blend
  policy_iq.longitudinalComfort = structs.IQCarControl.LongitudinalComfort(
    valid=True, jerkUp=0.8, jerkDown=0.8, comfortEnabled=True, accelDeviationAbove=0.18, accelDeviationBelow=0.1)
  previous_toyota_reference = 0.0
  for frame in range(60):
    cc = structs.CarControl()
    cc.enabled = cc.longActive = True
    cc.longComfortMode = True  # Tesla's existing comfort_mode keeps this gate.
    cc.actuators.longControlState = structs.CarControl.Actuators.LongControlState.pid
    cc.actuators.accel = 1.0 if frame < 30 else -1.0
    outputs = [car.apply(cc.as_reader(), iq, (frame + 1) * 10_000_000)
               for car, iq in zip(cars, (legacy_iq, policy_iq), strict=True)]
    assert outputs[0][0].accel == outputs[1][0].accel
    assert outputs[0][1] == outputs[1][1]
    if pq_blend:
      assert outputs[1][0].accel == pytest.approx(cars[1].CS.acc_radar_sollbeschl)
    if candidate.startswith('TOYOTA'):
      # Toyota retains its own 4 m/s^3 reference shaping after LongControl.
      assert abs(cars[1].CC.prev_accel - previous_toyota_reference) <= 4.0 * 0.03 + 1e-9
      previous_toyota_reference = cars[1].CC.prev_accel
