import pytest

from iqdbc.car import gen_empty_fingerprint, structs
from iqdbc.car.hyundai.hyundaicanfd import CanBus
from iqdbc.car.hyundai.interface import CarInterface
from iqdbc.car.hyundai.values import CAR, HyundaiFlags, HyundaiSafetyFlags
from iqdbc.safety.tests.libsafety import libsafety_py
from iqpilot.common.params import Params

ADRV_0x51 = 0x51
LFAHDA_CLUSTER = 0x1E0
ADRV_0x160 = 0x160
CAMERA_BUS = 2
HDA2_EV = HyundaiFlags.CANFD | HyundaiFlags.CANFD_HDA2 | HyundaiFlags.EV


def ev6(monkeypatch, tmp_path, flags=HDA2_EV, openpilot_long=True):
  monkeypatch.setenv("PARAMS_ROOT", str(tmp_path))
  Params().put_bool("ControlsReady", True)
  fingerprint = gen_empty_fingerprint()
  cp = CarInterface.get_params(CAR.KIA_EV6, fingerprint, [], openpilot_long, False, False)
  cp.flags = int(flags)
  cp.openpilotLongitudinalControl = openpilot_long
  cp_iq = CarInterface.get_params_iq(cp, CAR.KIA_EV6, fingerprint, [], openpilot_long, False, False)
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


def test_hda2_long_replaces_the_silenced_adas_heartbeat_and_cluster_status(monkeypatch, tmp_path):
  interface = ev6(monkeypatch, tmp_path)
  can = CanBus(interface.CP)
  sends = sent_frames(interface)

  heartbeat = [s for s in sends if s[0] == ADRV_0x51]
  assert len(heartbeat) == 200
  assert {s[2] for s in heartbeat} == {can.ACAN, can.CAM}
  assert sum(1 for s in heartbeat if s[2] == can.ACAN) == 100

  cluster = [s for s in sends if s[0] == LFAHDA_CLUSTER]
  assert len(cluster) == 20
  assert {s[2] for s in cluster} == {can.ECAN}


@pytest.mark.parametrize("flags,openpilot_long", [
  (HDA2_EV, False),
  (HDA2_EV | HyundaiFlags.CAMERA_SCC, True),
])
def test_cars_that_keep_their_adas_ecu_send_neither(flags, openpilot_long, monkeypatch, tmp_path):
  interface = ev6(monkeypatch, tmp_path, flags, openpilot_long)
  addresses = {s[0] for s in sent_frames(interface)}

  assert ADRV_0x51 not in addresses
  if not openpilot_long:
    assert LFAHDA_CLUSTER not in addresses


def test_panda_safety_transmits_the_replacements_for_this_ev6(monkeypatch, tmp_path):
  interface = ev6(monkeypatch, tmp_path)
  sends = [s for s in sent_frames(interface) if s[0] in (ADRV_0x51, LFAHDA_CLUSTER, ADRV_0x160)]
  safety = libsafety_py.libsafety
  param = HyundaiSafetyFlags.EV_GAS | HyundaiSafetyFlags.LONG | HyundaiSafetyFlags.CANFD_LKA_STEERING
  assert param == 21
  assert safety.set_safety_hooks(structs.CarParams.SafetyModel.hyundaiCanfd, param) == 0
  safety.init_tests()

  assert sends
  for addr, dat, bus in sends:
    assert safety.safety_tx_hook(libsafety_py.make_CANPacket(addr, bus, dat)), (hex(addr), bus)


def test_hda2_long_sends_the_cameras_healthy_fca_frame_on_a_can(monkeypatch, tmp_path):
  interface = ev6(monkeypatch, tmp_path)
  can = CanBus(interface.CP)
  frames = [s for s in sent_frames(interface) if s[0] == ADRV_0x160 and s[2] == can.ACAN]

  assert len(frames) == 50
  assert {bytes(s[1])[3:11] for s in frames} == {bytes.fromhex("0300000002fffc01")}


def test_the_cameras_own_fca_frame_no_longer_reaches_the_car_while_it_is_replaced(monkeypatch, tmp_path):
  interface = ev6(monkeypatch, tmp_path)
  can = CanBus(interface.CP)
  safety = libsafety_py.libsafety
  param = HyundaiSafetyFlags.EV_GAS | HyundaiSafetyFlags.LONG | HyundaiSafetyFlags.CANFD_LKA_STEERING
  assert safety.set_safety_hooks(structs.CarParams.SafetyModel.hyundaiCanfd, param) == 0
  safety.init_tests()
  safety.set_timer(10_000_000)

  assert safety.safety_fwd_hook(CAMERA_BUS, ADRV_0x160) != -1
  frame = next(s for s in sent_frames(interface) if s[0] == ADRV_0x160 and s[2] == can.ACAN)
  assert safety.safety_tx_hook(libsafety_py.make_CANPacket(frame[0], frame[2], frame[1]))
  assert safety.safety_fwd_hook(CAMERA_BUS, ADRV_0x160) == -1
