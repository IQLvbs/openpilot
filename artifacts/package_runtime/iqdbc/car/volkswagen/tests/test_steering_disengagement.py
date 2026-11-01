"""Full controller/CAN check for ordinary disengagement and intentional AOL."""
import pytest

from iqdbc.can import CANParser
from iqdbc.car import Bus, structs
from iqdbc.car.volkswagen.interface import CarInterface
from iqdbc.car.volkswagen.values import CAR, DBC, VolkswagenFlags


@pytest.fixture(params=[CAR.VOLKSWAGEN_GOLF_MK7, CAR.VOLKSWAGEN_PASSAT_B7, CAR.AUDI_A4_MK4])
def car(request):
  candidate = request.param
  fingerprint = {bus: {} for bus in range(7)}
  cp = CarInterface.get_params(candidate, fingerprint, [], alpha_long=False, is_release=False, docs=False)
  cp_iq = CarInterface.get_params_iq(cp, candidate, fingerprint, [], alpha_long=False, is_release_iq=False, docs=False)
  interface = CarInterface(cp, cp_iq)
  interface.update([(0, [])])
  interface.CS.out = structs.CarState()
  interface.CS.out.vEgo = interface.CS.out.vEgoRaw = 20.0
  interface.CS.out.cruiseState.available = True
  return interface


def apply(car, *, enabled, torque, aol=False, aol_available=None, lat_active=True):
  cc = structs.CarControl()
  cc.enabled = enabled
  cc.latActive = lat_active
  cc.actuators.torque = torque
  iq = structs.IQCarControl()
  iq.aol.enabled = iq.aol.active = aol
  iq.aol.available = aol if aol_available is None else aol_available
  iq.aol.state = "enabled" if aol else "disabled"
  car.CC.frame += -car.CC.frame % car.CC.CCP.STEER_STEP
  now = (car.CC.frame + 1) * 10_000_000
  _, messages = car.apply(cc.as_reader(), iq, now)
  pq = bool(car.CP.flags & VolkswagenFlags.PQ)
  name = "HCA_1" if pq else "HCA_01"
  parser = CANParser(DBC[car.CP.carFingerprint][Bus.pt], [(name, 0)], car.CC._pt_tx_bus)
  address = car.CC.packer_pt.dbc.name_to_msg[name].address
  assert any(addr == address and bus == car.CC._pt_tx_bus for addr, _, bus in messages)
  parser.update([(now, messages)])
  fields = parser.vl[name]
  return fields["HCA_Status" if pq else "HCA_01_Status_HCA"], fields["LM_Offset" if pq else "HCA_01_LM_Offset"]


@pytest.mark.parametrize("torque", [-1.0, 1.0])
@pytest.mark.parametrize("aol_available,aol_active", [(False, False), (True, False), (False, True)])
def test_ordinary_disengagement_stops_next_hca_frame(car, torque, aol_available, aol_active):
  for _ in range(20):
    status, magnitude = apply(car, enabled=True, torque=torque)
  assert status in (5, 7) and magnitude > 0
  for _ in range(12):
    assert apply(car, enabled=False, torque=torque, aol_available=aol_available, aol=aol_active) == (3, 0)
  status, magnitude = apply(car, enabled=True, torque=torque)
  assert status in (5, 7) and magnitude > 0


@pytest.mark.parametrize("torque", [-1.0, 1.0])
def test_intentional_aol_keeps_steering_with_cruise_disabled(car, torque):
  for _ in range(20):
    status, magnitude = apply(car, enabled=False, torque=torque, aol=True)
  assert status in (5, 7) and magnitude > 0
  assert apply(car, enabled=False, torque=torque, aol=True, lat_active=False) == (3, 0)
  status, magnitude = apply(car, enabled=False, torque=torque, aol=True)
  assert status in (5, 7) and magnitude > 0
  # The AOL state itself also withdraws permission, even if latActive lingers.
  assert apply(car, enabled=False, torque=torque, aol_available=True) == (3, 0)
