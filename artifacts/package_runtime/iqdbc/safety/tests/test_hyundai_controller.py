import pytest

from iqdbc.car import gen_empty_fingerprint, structs
from iqdbc.car.hyundai.interface import CarInterface
from iqdbc.car.hyundai.values import CAR, HyundaiFlags
from iqdbc.safety.tests.libsafety import libsafety_py


@pytest.mark.parametrize("candidate", list(CAR), ids=lambda candidate: candidate.value)
@pytest.mark.parametrize("alpha_long", (False, True), ids=("stock_long", "openpilot_long"))
def test_controller_frames_match_configured_safety(candidate, alpha_long, monkeypatch, tmp_path):
  """Run real controller output through the safety configuration selected for every HKG platform."""
  monkeypatch.setenv("PARAMS_ROOT", str(tmp_path / candidate.value))
  fingerprint = gen_empty_fingerprint()
  cp = CarInterface.get_params(candidate, fingerprint, [], alpha_long, False, False)
  cp_iq = CarInterface.get_params_iq(cp, candidate, fingerprint, [], alpha_long, False, False)
  interface = CarInterface(cp, cp_iq)
  interface.update([])

  safety_config = cp.safetyConfigs[-1]
  safety = libsafety_py.libsafety
  assert safety.set_safety_hooks(safety_config.safetyModel.raw, safety_config.safetyParam) == 0
  safety.init_tests()
  safety.set_controls_allowed(True)

  control = structs.CarControl.new_message()
  control.enabled = True
  control.latActive = True
  control.longActive = alpha_long
  control.actuators.torque = 0.01
  control.actuators.accel = 0.0

  for frame in range(20):
    _, can_sends = interface.apply(control.as_reader(), structs.IQCarControl())
    assert isinstance(can_sends, list)
    for address, data, bus in can_sends:
      packet = libsafety_py.make_CANPacket(address, bus, data)
      rejection = f"{candidate.value} frame {frame}: safety rejected address={address:#x} bus={bus} data={data.hex()}"
      assert safety.safety_tx_hook(packet), rejection


def seed_camera_frames(CS):
  CS.lfa = {"STEER_REQ": 1, "LKA_MODE": 2, "LKA_ICON": 1, "TORQUE_REQUEST": 0, "LKA_ACTIVE": 0,
            "HAS_LANE_SAFETY": 0, "VALUE63": 0, "VALUE64": 0, "COUNTER": 0}
  CS.mdps = {"LKA_ACTIVE": 0, "LFA2_ACTIVE": 0, "STEERING_COL_TORQUE": 0.0}
  CS.steer_touch_2af = {"TOUCH_DETECT": 0, "TOUCH1": 0, "TOUCH2": 0, "CHECKSUM_": 0}


def test_ev6_camera_scc_has_one_lfa_sender(monkeypatch, tmp_path):
  monkeypatch.setenv("PARAMS_ROOT", str(tmp_path))
  fingerprint = gen_empty_fingerprint()
  cp = CarInterface.get_params(CAR.KIA_EV6, fingerprint, [], False, False, False)
  cp.flags &= ~HyundaiFlags.CANFD_HDA2.value
  cp.flags |= HyundaiFlags.CANFD_CAMERA_SCC.value
  cp_iq = CarInterface.get_params_iq(cp, CAR.KIA_EV6, fingerprint, [], False, False, False)
  interface = CarInterface(cp, cp_iq)
  interface.update([])

  control = structs.CarControl.new_message()
  control.enabled = True
  control.latActive = True
  control.actuators.torque = 0.01

  seed_camera_frames(interface.CS)

  for _ in range(20):
    _, can_sends = interface.apply(control.as_reader(), structs.IQCarControl())
    assert [(address, bus) for address, _, bus in can_sends if address == 0x12A] == [(0x12A, 0)]
    assert [(address, bus) for address, _, bus in can_sends if address == 0xEA] == [(0xEA, 2)]
    assert all(bus == 2 for address, _, bus in can_sends if address == 0x2AF)
