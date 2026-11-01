import pytest

from iqdbc.can import CANPacker
from iqdbc.car import Bus, gen_empty_fingerprint, get_safety_config, structs
from iqdbc.car.hyundai.interface import CarInterface
from iqdbc.car.hyundai.values import CAR, CAMERA_SCC_CAR, DBC, HyundaiFlags
from iqpilot.common.params import Params

CAMERA_FRAME = 0x2A4
CANFD_CARS = tuple(sorted(CAR.with_flags(HyundaiFlags.CANFD)))


def build(candidate, hda2, camera_scc, alpha_long, monkeypatch, tmp_path):
  monkeypatch.setenv("PARAMS_ROOT", str(tmp_path))
  params = Params()
  params.put_bool("ControlsReady", True)
  params.put_int("HyundaiCameraSCC", camera_scc)
  fingerprint = gen_empty_fingerprint()
  if hda2:
    fingerprint[2 if camera_scc == 0 else 1][0x50] = 16
    fingerprint[2][CAMERA_FRAME] = 24
  cp = CarInterface.get_params(candidate, fingerprint, [], alpha_long, False, False)
  cp_iq = CarInterface.get_params_iq(cp, candidate, fingerprint, [], alpha_long, False, False)
  return CarInterface(cp, cp_iq)


def register(interface, seen=(Bus.cam, Bus.alt)):
  for bus in seen:
    interface.can_parsers[bus].seen_addresses.add(CAMERA_FRAME)
  interface.CS.controls_ready_count = 123
  interface.CS.monitor_fingerprint(interface.can_parsers, True)


@pytest.mark.parametrize("candidate", CANFD_CARS, ids=str)
@pytest.mark.parametrize("alpha_long", (False, True))
@pytest.mark.parametrize("hda2,camera_scc,expected", (
  (True, 0, Bus.cam), (True, 1, Bus.alt), (True, 2, Bus.alt),
  (False, 0, Bus.alt), (False, 1, Bus.alt),
))
def test_source_preference_across_canfd_platforms(candidate, alpha_long, hda2, camera_scc, expected, monkeypatch, tmp_path):
  interface = build(candidate, hda2, camera_scc, alpha_long, monkeypatch, tmp_path)
  register(interface)
  # Factory camera-SCC platforms retain their original A-CAN preference too.
  if candidate in CAMERA_SCC_CAR:
    expected = Bus.alt
  for bus in (Bus.cam, Bus.alt):
    assert (CAMERA_FRAME in interface.can_parsers[bus].addresses) == (bus == expected)
  parser = interface.can_parsers[expected]
  assert interface.CS.cam_0x2a4 is parser.vl["CAM_0x2a4"]
  state = parser.message_states[CAMERA_FRAME]
  assert not state.ignore_alive
  assert not state.ignore_checksum


@pytest.mark.parametrize("hda2,camera_scc", ((True, 0), (True, 1), (False, 0)))
@pytest.mark.parametrize("seen", ((Bus.cam,), (Bus.alt,), ()))
def test_single_source_and_absent_frame(hda2, camera_scc, seen, monkeypatch, tmp_path):
  interface = build(CAR.KIA_EV6, hda2, camera_scc, False, monkeypatch, tmp_path)
  register(interface, seen)
  for bus in (Bus.cam, Bus.alt):
    assert (CAMERA_FRAME in interface.can_parsers[bus].addresses) == (bus in seen)


def test_startup_copy_disappears_but_camera_stays_valid(monkeypatch, tmp_path):
  interface = build(CAR.KIA_EV6, True, 0, False, monkeypatch, tmp_path)
  register(interface)
  camera = interface.can_parsers[Bus.cam]
  alternate = interface.can_parsers[Bus.alt]
  packer = CANPacker(DBC[CAR.KIA_EV6][Bus.pt])
  for frame in range(200):
    nanos = (frame + 1) * 50_000_000
    camera_msg = packer.make_can_msg("CAM_0x2a4", camera.bus, {"COUNTER": frame % 256})
    frames = [camera_msg]
    if frame < 20:
      frames.append((camera_msg[0], camera_msg[1], alternate.bus))
    for parser in (camera, alternate):
      parser.update([(nanos, frames)])
      assert parser.can_valid
  # A real loss on the selected source must still fail validation.
  for frame in range(50):
    camera.update([(nanos + (frame + 1) * 50_000_000, [])])
    valid = camera.can_valid
  assert not valid


def test_missing_alternate_parser_uses_camera(monkeypatch, tmp_path):
  interface = build(CAR.KIA_EV6, True, 0, False, monkeypatch, tmp_path)
  del interface.can_parsers[Bus.alt]
  register(interface, (Bus.cam,))
  assert CAMERA_FRAME in interface.can_parsers[Bus.cam].addresses


@pytest.mark.parametrize("camera_scc", (0, 1))
def test_alternate_steering_camera_frame_is_unchanged(camera_scc, monkeypatch, tmp_path):
  interface = build(CAR.KIA_EV6, True, camera_scc, False, monkeypatch, tmp_path)
  interface.CP.flags |= HyundaiFlags.CANFD_HDA2_ALT_STEERING.value
  for bus in (Bus.cam, Bus.alt):
    interface.can_parsers[bus].seen_addresses.add(0x362)
  register(interface, ())
  assert 0x362 in interface.can_parsers[Bus.cam].addresses
  assert 0x362 not in interface.can_parsers[Bus.alt].addresses
  assert interface.CS.cam_0x362 is interface.can_parsers[Bus.cam].vl["CAM_0x362"]


def test_standard_hda2_with_second_panda(monkeypatch, tmp_path):
  interface = build(CAR.KIA_EV6, True, 0, False, monkeypatch, tmp_path)
  safety = interface.CP.safetyConfigs[0]
  interface.CP.safetyConfigs = [get_safety_config(structs.CarParams.SafetyModel.noOutput),
                                get_safety_config(safety.safetyModel, safety.safetyParam)]
  interface = CarInterface(interface.CP, interface.CP_IQ)
  assert interface.can_parsers[Bus.cam].bus == 6
  assert interface.can_parsers[Bus.alt].bus == 4
  register(interface)
  assert CAMERA_FRAME in interface.can_parsers[Bus.cam].addresses
  assert CAMERA_FRAME not in interface.can_parsers[Bus.alt].addresses
