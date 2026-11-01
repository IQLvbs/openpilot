import pytest

from iqdbc.car import gen_empty_fingerprint, structs
from iqdbc.car.hyundai.hyundaicanfd import CanBus
from iqdbc.car.hyundai.interface import CarInterface
from iqdbc.car.hyundai.values import CAR, HyundaiFlags
from iqpilot.common.params import Params

CLUSTER = 0x1E0
RADAR_FLAGS = HyundaiFlags.CANFD | HyundaiFlags.RADAR_SCC | HyundaiFlags.CANFD_ALT_GEARS
HDA2_EV = HyundaiFlags.CANFD | HyundaiFlags.CANFD_HDA2 | HyundaiFlags.EV


def make(monkeypatch, tmp_path, platform, flags, openpilot_long=True):
  monkeypatch.setenv("PARAMS_ROOT", str(tmp_path))
  Params().put_bool("ControlsReady", True)
  fingerprint = gen_empty_fingerprint()
  cp = CarInterface.get_params(platform, fingerprint, [], openpilot_long, False, False)
  cp.flags = int(flags)
  cp.openpilotLongitudinalControl = openpilot_long
  cp_iq = CarInterface.get_params_iq(cp, platform, fingerprint, [], openpilot_long, False, False)
  return CarInterface(cp, cp_iq)


def sent_frames(interface, frames=100):
  cc = structs.CarControl().as_reader()
  cc_iq = structs.IQCarControl()
  out = []
  t = 0
  for _ in range(frames):
    t += 10_000_000
    interface.update([(t, [])])
    _, sends = interface.apply(cc, cc_iq, t)
    out.extend(sends)
  return out


def per_bus(sends, addr):
  return {bus: sum(1 for s in sends if s[0] == addr and s[2] == bus) for bus in {s[2] for s in sends if s[0] == addr}}


def test_radar_scc_car_feeds_the_camera_alongside_the_car(monkeypatch, tmp_path):
  interface = make(monkeypatch, tmp_path, CAR.GENESIS_GV70_1ST_GEN, RADAR_FLAGS)
  can = CanBus(interface.CP)
  sends = sent_frames(interface)

  assert per_bus(sends, 0x140) == {can.CAM: 100}
  assert per_bus(sends, 0x160) == {can.ECAN: 50, can.CAM: 50}
  assert per_bus(sends, 0x1A0) == {can.ECAN: 50, can.CAM: 50}
  ecan = [s for s in sends if s[0] == 0x1A0 and s[2] == can.ECAN]
  cam = [s for s in sends if s[0] == 0x1A0 and s[2] == can.CAM]
  assert [s[1] for s in ecan] == [s[1] for s in cam]


def test_hda2_longitudinal_car_feeds_the_camera_the_adas_status(monkeypatch, tmp_path):
  interface = make(monkeypatch, tmp_path, CAR.KIA_EV6, HDA2_EV)
  can = CanBus(interface.CP)

  assert per_bus(sent_frames(interface), 0x51) == {can.ACAN: 100, can.CAM: 100}


@pytest.mark.parametrize("platform,flags", [(CAR.GENESIS_GV70_1ST_GEN, RADAR_FLAGS), (CAR.KIA_EV6, HDA2_EV)])
def test_stock_longitudinal_cars_send_nothing_to_the_camera(monkeypatch, tmp_path, platform, flags):
  interface = make(monkeypatch, tmp_path, platform, flags, openpilot_long=False)
  can = CanBus(interface.CP)

  assert not [s for s in sent_frames(interface) if s[2] == can.CAM and s[0] in (0x140, 0x160, 0x1A0, 0x51)]


def test_radar_scc_cluster_message_ignores_a_faulted_camera_frame(monkeypatch, tmp_path):
  interface = make(monkeypatch, tmp_path, CAR.GENESIS_GV70_1ST_GEN, RADAR_FLAGS)
  can = CanBus(interface.CP)
  interface.CS.lfahda_cluster = {"COUNTER": 3, "HDA_OptUsmSta": 7, "LFA_OptUsmSta": 0, "HDA_CntrlModSta": 0}
  clusters = [s for s in sent_frames(interface) if s[0] == CLUSTER and s[2] == can.ECAN]

  assert clusters
  assert {s[1][3] & 0x7 for s in clusters} == {2}
