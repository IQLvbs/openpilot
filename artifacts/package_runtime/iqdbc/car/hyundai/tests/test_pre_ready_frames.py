import pytest

from iqdbc.can import CANPacker
from iqdbc.car import Bus, gen_empty_fingerprint
from iqdbc.car.hyundai.hyundaicanfd import CanBus
from iqdbc.car.hyundai.interface import CarInterface
from iqdbc.car.hyundai.values import CAR, DBC, HyundaiFlags
from iqpilot.common.params import Params

HDA2_EV = HyundaiFlags.CANFD | HyundaiFlags.CANFD_HDA2 | HyundaiFlags.EV
DT_NS = 10_000_000


def hda2_ev6(openpilot_long, monkeypatch, tmp_path):
  monkeypatch.setenv("PARAMS_ROOT", str(tmp_path))
  Params().put_bool("ControlsReady", False)
  fingerprint = gen_empty_fingerprint()
  cp = CarInterface.get_params(CAR.KIA_EV6, fingerprint, [], openpilot_long, False, False)
  cp.flags = int(HDA2_EV)
  cp.openpilotLongitudinalControl = openpilot_long
  cp_iq = CarInterface.get_params_iq(cp, CAR.KIA_EV6, fingerprint, [], openpilot_long, False, False)
  return CarInterface(cp, cp_iq)


def drive(interface, frames_before_ready, frames_after_ready, steps_before=60, steps_after=200):
  t = 0
  for _ in range(steps_before):
    t += DT_NS
    interface.update([(t, frames_before_ready)])
  Params().put_bool("ControlsReady", True)
  for _ in range(steps_after):
    t += DT_NS
    interface.update([(t, frames_after_ready)])


def frame(packer, name, bus):
  addr, dat, _ = packer.make_can_msg(name, bus, {})
  return addr, dat, bus


@pytest.mark.parametrize("openpilot_long", [True, False])
def test_frames_that_stop_at_controls_ready_are_never_registered(openpilot_long, monkeypatch, tmp_path):
  interface = hda2_ev6(openpilot_long, monkeypatch, tmp_path)
  can = CanBus(interface.CP)
  packer = CANPacker(DBC[CAR.KIA_EV6][Bus.pt])
  scc_on_ecan = frame(packer, "SCC_CONTROL", can.ECAN)
  lane_on_acan = frame(packer, "CAM_0x2a4", can.ACAN)
  lane_on_cam = frame(packer, "CAM_0x2a4", can.CAM)

  before = [scc_on_ecan, lane_on_acan, lane_on_cam]
  after = [lane_on_cam] if openpilot_long else [scc_on_ecan, lane_on_cam]
  drive(interface, before, after)

  parsers = interface.can_parsers
  assert lane_on_cam[0] in parsers[Bus.cam].addresses
  assert lane_on_acan[0] not in parsers[Bus.alt].addresses
  assert (scc_on_ecan[0] in parsers[Bus.pt].addresses) == (not openpilot_long)
  assert interface.CS.cam_0x2a4 is not None
