from types import SimpleNamespace

import pytest

from iqdbc.can import CANPacker
from iqdbc.car.hyundai import hyundaicanfd
from iqdbc.car.hyundai.values import CAR, DBC, HyundaiFlags, HyundaiSafetyFlags
from iqdbc.car.structs import CarParams
from iqdbc.safety.tests.hyundai_common import canfd_accel
from iqdbc.safety.tests.libsafety import libsafety_py

DBC_NAME = DBC[CAR.KIA_EV6]["pt"]
CAN = SimpleNamespace(ECAN=0, ACAN=1, CAM=2)

RADAR_0x140 = [
  bytes.fromhex("e423ad4180000000"),
  bytes.fromhex("91ebae4180000000"),
  bytes.fromhex("42acaf4180000000"),
  bytes.fromhex("3448b04180000000"),
]
RADAR_0x160 = [
  bytes.fromhex("e218d70300000002fffc010000000000"),
  bytes.fromhex("f861d80300000002fffc010000000000"),
  bytes.fromhex("dcc9d90300000002fffc010000000000"),
]
ADAS_0x51 = [
  bytes.fromhex("b41c13080100e2fe000000000020ffff00000000000000000000000000000000"),
  bytes.fromhex("01db15080100e2fe000000000020ffff00000000000000000000000000000000"),
]
ADAS_0x51_CONSTANT_BYTES = (3, 6, 7, 13, 14, 15)


def params(flags):
  return SimpleNamespace(flags=int(flags))


def pack(name, values, counter):
  _, dat, _ = CANPacker(DBC_NAME).make_can_msg(name, CAN.CAM, values, rx_counter=counter)
  return dat


@pytest.mark.parametrize("frame", RADAR_0x140)
def test_radar_heartbeat_reproduces_the_recorded_frames(frame):
  assert pack("RADAR_0x140", hyundaicanfd.RADAR_0x140_VALUES, frame[2]) == frame


@pytest.mark.parametrize("frame", RADAR_0x160)
def test_radar_liveness_reproduces_the_recorded_frames(frame):
  assert pack("ADRV_0x160", hyundaicanfd.RADAR_0x160_VALUES, frame[2]) == frame


@pytest.mark.parametrize("frame", ADAS_0x51)
def test_adas_status_matches_the_constant_bytes_of_the_recorded_frames(frame):
  ours = pack("ADRV_0x51", hyundaicanfd.ADRV_0x51_CAMERA_VALUES, frame[2])
  assert len(ours) == len(frame) == 32
  for index in ADAS_0x51_CONSTANT_BYTES:
    assert ours[index] == frame[index]


def run(flags, frames=100):
  packer = CANPacker(DBC_NAME)
  return [msg for frame in range(frames) for msg in hyundaicanfd.create_camera_keepalive(params(flags), packer, CAN, frame)]


def test_radar_scc_car_feeds_the_camera_what_the_silenced_radar_sent():
  sends = run(HyundaiFlags.CANFD | HyundaiFlags.RADAR_SCC)
  assert {s[2] for s in sends} == {CAN.CAM}
  counts = {addr: sum(1 for s in sends if s[0] == addr) for addr in {s[0] for s in sends}}
  assert counts == {0x140: 100, 0x160: 50}


def test_hda2_car_feeds_the_camera_the_adas_status_every_frame():
  sends = run(HyundaiFlags.CANFD | HyundaiFlags.CANFD_HDA2 | HyundaiFlags.EV)
  assert [s[0] for s in sends] == [0x51] * 100
  assert {s[2] for s in sends} == {CAN.CAM}


def test_camera_scc_car_keeps_its_own_stock_traffic():
  assert run(HyundaiFlags.CANFD | HyundaiFlags.CAMERA_SCC) == []


def tx_allowed(param, packet):
  safety = libsafety_py.libsafety
  assert safety.set_safety_hooks(CarParams.SafetyModel.hyundaiCanfd, param) == 0
  safety.init_tests()
  return bool(safety.safety_tx_hook(packet))


def camera_bus_frames():
  return {
    0x140: libsafety_py.make_CANPacket(0x140, 2, bytes(8)),
    0x160: libsafety_py.make_CANPacket(0x160, 2, bytes(16)),
    0x1A0: canfd_accel(0, bus=2),
  }


@pytest.mark.parametrize("addr", [0x140, 0x160, 0x1A0])
def test_panda_lets_a_radar_scc_car_transmit_to_the_camera_only_under_longitudinal(addr):
  frame = camera_bus_frames()[addr]
  assert tx_allowed(HyundaiSafetyFlags.LONG, frame)
  assert not tx_allowed(0, frame)


def test_panda_lets_an_hda2_longitudinal_car_transmit_the_adas_status_to_the_camera():
  hda2_long = HyundaiSafetyFlags.CANFD_LKA_STEERING | HyundaiSafetyFlags.LONG
  assert tx_allowed(hda2_long, libsafety_py.make_CANPacket(0x51, 2, bytes(32)))


def test_panda_keeps_the_camera_bus_closed_to_other_frames_on_a_radar_scc_car():
  assert not tx_allowed(HyundaiSafetyFlags.LONG, libsafety_py.make_CANPacket(0x1EA, 2, bytes(32)))
  assert not tx_allowed(HyundaiSafetyFlags.LONG, libsafety_py.make_CANPacket(0x51, 2, bytes(32)))
