import pytest

from iqdbc.car.hyundai.values import HyundaiSafetyFlags
from iqdbc.car.structs import CarParams
from iqdbc.safety.tests.hyundai_common import TESTER_PRESENT, canfd_accel, canfd_steer, packet
from iqdbc.safety.tests.libsafety import libsafety_py


@pytest.fixture
def safety():
  return libsafety_py.libsafety


@pytest.mark.parametrize("param", (
  0,
  HyundaiSafetyFlags.EV_GAS,
  HyundaiSafetyFlags.HYBRID_GAS,
  HyundaiSafetyFlags.LONG,
  HyundaiSafetyFlags.CAMERA_SCC,
  HyundaiSafetyFlags.CANFD_LKA_STEERING,
  HyundaiSafetyFlags.CANFD_ALT_BUTTONS,
  HyundaiSafetyFlags.CANFD_LKA_STEERING | HyundaiSafetyFlags.CANFD_LKA_STEERING_ALT,
  HyundaiSafetyFlags.CANFD_LKA_STEERING | HyundaiSafetyFlags.LONG,
  HyundaiSafetyFlags.CANFD_LKA_STEERING | HyundaiSafetyFlags.LONG | HyundaiSafetyFlags.CANFD_ALT_BUTTONS,
))
def test_canfd_safety_configurations_initialize(safety, param):
  assert safety.set_safety_hooks(CarParams.SafetyModel.hyundaiCanfd, param) == 0
  safety.init_tests()
  assert safety.get_current_safety_param() == param


@pytest.mark.parametrize(("param", "addr", "length"), (
  (0, 0x12A, 16),
  (HyundaiSafetyFlags.CANFD_LKA_STEERING, 0x50, 16),
  (HyundaiSafetyFlags.CANFD_LKA_STEERING | HyundaiSafetyFlags.CANFD_LKA_STEERING_ALT, 0x110, 32),
  (HyundaiSafetyFlags.CANFD_LKA_STEERING | HyundaiSafetyFlags.LONG, 0x12A, 16),
))
def test_canfd_steering_limits(safety, param, addr, length):
  safety.set_safety_hooks(CarParams.SafetyModel.hyundaiCanfd, param)
  safety.init_tests()

  safety.set_controls_allowed(False)
  assert safety.safety_tx_hook(canfd_steer(addr, length, 0, False))
  assert not safety.safety_tx_hook(canfd_steer(addr, length, 1))

  safety.set_controls_allowed(True)
  assert safety.safety_tx_hook(canfd_steer(addr, length, 2))
  safety.set_desired_torque_last(270)
  safety.set_rt_torque_last(270)
  assert safety.safety_tx_hook(canfd_steer(addr, length, 270))
  assert not safety.safety_tx_hook(canfd_steer(addr, length, 271))
  assert not safety.safety_tx_hook(canfd_steer(addr, length, -271))


def test_canfd_tx_whitelist_and_buttons(safety):
  safety.set_safety_hooks(CarParams.SafetyModel.hyundaiCanfd, 0)
  safety.init_tests()
  assert not safety.safety_tx_hook(packet(0x123, 0, 8))
  assert not safety.safety_tx_hook(packet(0x12A, 1, 16))

  safety.set_controls_allowed(False)
  assert not safety.safety_tx_hook(packet(0x1CF, 0, 8, {2: 1}))
  assert not safety.safety_tx_hook(packet(0x1CF, 0, 8, {2: 2}))
  assert not safety.safety_tx_hook(packet(0x1CF, 0, 8, {2: 4}))

  safety.set_controls_allowed(True)
  assert safety.safety_tx_hook(packet(0x1CF, 0, 8, {2: 1}))
  assert not safety.safety_tx_hook(packet(0x1CF, 0, 8, {2: 2}))
  assert safety.safety_tx_hook(packet(0x1CF, 0, 8, {2: 4}))

  safety.set_controls_allowed(False)
  safety.set_cruise_engaged_prev(True)
  assert safety.safety_tx_hook(packet(0x1CF, 0, 8, {2: 4}))


def test_canfd_camera_scc_button_passthrough(safety):
  safety.set_safety_hooks(CarParams.SafetyModel.hyundaiCanfd, HyundaiSafetyFlags.CAMERA_SCC)
  safety.init_tests()

  safety.set_controls_allowed(False)
  assert safety.safety_tx_hook(packet(0x1CF, 2, 8))
  assert not safety.safety_tx_hook(packet(0x1CF, 0, 8))
  assert not safety.safety_tx_hook(packet(0x1CF, 2, 8, {2: 1}))
  assert not safety.safety_tx_hook(packet(0x1CF, 2, 8, {2: 2}))
  assert not safety.safety_tx_hook(packet(0x1CF, 2, 8, {2: 4}))

  safety.set_controls_allowed(True)
  assert safety.safety_tx_hook(packet(0x1CF, 2, 8, {2: 1}))
  assert not safety.safety_tx_hook(packet(0x1CF, 2, 8, {2: 2}))
  assert safety.safety_tx_hook(packet(0x1CF, 2, 8, {2: 4}))


def test_canfd_camera_scc_single_owner_forwarding(safety):
  safety.set_safety_hooks(CarParams.SafetyModel.hyundaiCanfd, HyundaiSafetyFlags.CAMERA_SCC)
  safety.init_tests()
  safety.set_timer(1_000_000)

  assert safety.safety_fwd_hook(2, 0x12A) == -1
  assert safety.safety_fwd_hook(2, 0x1E0) == -1
  assert safety.safety_fwd_hook(2, 0x1A0) == 0
  assert safety.safety_fwd_hook(0, 0xEA) == -1
  assert safety.safety_fwd_hook(0, 0x2AF) == -1
  assert safety.safety_tx_hook(packet(0xEA, 2, 24))
  assert safety.safety_tx_hook(packet(0x2AF, 2, 8))
  assert not safety.safety_tx_hook(packet(0xEA, 0, 24))
  assert not safety.safety_tx_hook(packet(0x2AF, 0, 8))


def test_canfd_camera_scc_longitudinal_single_owner_forwarding(safety):
  param = HyundaiSafetyFlags.CAMERA_SCC | HyundaiSafetyFlags.LONG
  safety.set_safety_hooks(CarParams.SafetyModel.hyundaiCanfd, param)
  safety.init_tests()
  safety.set_timer(1_000_000)

  assert safety.safety_fwd_hook(2, 0x12A) == -1
  assert safety.safety_fwd_hook(2, 0x1A0) == -1
  assert safety.safety_fwd_hook(0, 0x175) == -1
  assert safety.safety_fwd_hook(0, 0xEA) == -1
  assert safety.safety_fwd_hook(0, 0x2AF) == -1
  assert safety.safety_tx_hook(packet(0xEA, 2, 24))
  assert safety.safety_tx_hook(packet(0x2AF, 2, 8))


def test_canfd_camera_scc_forwards_every_frame_openpilot_does_not_replace(safety):
  param = HyundaiSafetyFlags.CAMERA_SCC | HyundaiSafetyFlags.LONG
  safety.set_safety_hooks(CarParams.SafetyModel.hyundaiCanfd, param)
  safety.init_tests()
  safety.set_timer(1_000_000)

  # openpilot sends these, so the stock frames must stay off the destination bus
  for addr in (0x12A, 0x1A0, 0x1E0):
    assert safety.safety_fwd_hook(2, addr) == -1
  for addr in (0x175, 0x1CF):
    assert safety.safety_fwd_hook(0, addr) == -1

  # openpilot doctors these for the camera, so the stock frame must not also reach it
  assert safety.safety_fwd_hook(0, 0xEA) == -1
  assert safety.safety_fwd_hook(0, 0x2AF) == -1

  # openpilot never sends these on a camera-SCC car, so blocking them strands the ECUs
  assert safety.safety_fwd_hook(2, 0x160) == 0
  assert safety.safety_fwd_hook(0, 0x1AA) == 2
  assert safety.safety_fwd_hook(2, 0x11A) == 0
  assert safety.safety_fwd_hook(2, 0x1B5) == 0
  assert safety.safety_fwd_hook(2, 0x1FA) == 0
  assert safety.safety_fwd_hook(2, 0x2BA) == 0


def test_canfd_radar_scc_keeps_stock_camera_feedback(safety):
  safety.set_safety_hooks(CarParams.SafetyModel.hyundaiCanfd, 0)
  safety.init_tests()
  safety.set_timer(1_000_000)

  assert safety.safety_fwd_hook(0, 0xEA) == 2
  assert safety.safety_fwd_hook(0, 0x2AF) == 2
  assert not safety.safety_tx_hook(packet(0xEA, 2, 24))
  assert not safety.safety_tx_hook(packet(0x2AF, 2, 8))


def test_canfd_stock_longitudinal_only_allows_cancel(safety):
  safety.set_safety_hooks(CarParams.SafetyModel.hyundaiCanfd, 0)
  safety.init_tests()
  assert safety.safety_tx_hook(canfd_accel(0, acc_mode=4))
  assert not safety.safety_tx_hook(canfd_accel(1, acc_mode=4))
  assert not safety.safety_tx_hook(canfd_accel(0, acc_mode=0))


def test_canfd_longitudinal_accel_limits(safety):
  safety.set_safety_hooks(CarParams.SafetyModel.hyundaiCanfd, HyundaiSafetyFlags.LONG)
  safety.init_tests()

  safety.set_controls_allowed(False)
  assert safety.safety_tx_hook(canfd_accel(0))
  assert not safety.safety_tx_hook(canfd_accel(1))

  safety.set_controls_allowed(True)
  for accel in (-400, 0, 250):
    assert safety.safety_tx_hook(canfd_accel(accel))
  for accel in (-401, 251):
    assert not safety.safety_tx_hook(canfd_accel(accel))


def test_canfd_hda2_diagnostic_payload_is_restricted(safety):
  param = HyundaiSafetyFlags.CANFD_LKA_STEERING | HyundaiSafetyFlags.LONG
  safety.set_safety_hooks(CarParams.SafetyModel.hyundaiCanfd, param)
  safety.init_tests()
  assert safety.safety_tx_hook(packet(0x730, 1, 8, dict(enumerate(TESTER_PRESENT))))
  assert not safety.safety_tx_hook(packet(0x730, 1, 8, {0: 3, 1: 0x22}))
