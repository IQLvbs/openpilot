import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from iqdbc.can import CANPacker, CANParser
from iqdbc.car import gen_empty_fingerprint
from iqdbc.car.hyundai import hyundaicanfd
from iqdbc.car.hyundai.hyundaicanfd import CanBus, HANDS_ON_SPOOF_FRAMES, HANDS_ON_SPOOF_PERIOD, \
                                            HANDS_ON_SPOOF_TORQUE, _hide_replaced_adas_service_warning, \
                                            apply_accel_jerk_limit, create_steering_messages_camera_scc, \
                                            hyundai_crc8, longitudinal_interlock_active
from iqdbc.car.hyundai.interface import CarInterface
from iqdbc.car.hyundai.stopping import CanfdStopping, StopPhase
from iqdbc.car.hyundai.values import CAR, DBC, HyundaiFlags
from iqdbc.car.common.conversions import Conversions as CV

DBC_NAME = DBC[CAR.KIA_EV6]["pt"]
SAFETY_HEADER = Path(__file__).resolve().parents[3] / "safety" / "modes" / "hyundai_canfd.h"


class StubParams:
  def get_int(self, key):
    return 0

  def get_bool(self, key):
    return False

  def put_int(self, key, value):
    pass

  def put_bool(self, key, value):
    pass


@pytest.fixture(autouse=True)
def stub_params(monkeypatch):
  monkeypatch.setattr("iqdbc.car.hyundai.interface.Params", StubParams)


def ev6_params(camera_scc=False):
  fingerprint = gen_empty_fingerprint()
  for addr in (0x35, 0xA0, 0xEA, 0x125, 0x130, 0x175, 0x1CF):
    fingerprint[0][addr] = 32
  for addr in (0x12A, 0x1A0, 0x1E0):
    fingerprint[2][addr] = 32
  CP = CarInterface.get_params(CAR.KIA_EV6, fingerprint, [], False, False, False)
  if camera_scc:
    CP.flags |= HyundaiFlags.CAMERA_SCC.value
  return CP


class FakeCarState:
  def __init__(self, *, steer_req=1, alerts_1=0, has_touch=True, has_adrv=True):
    self.mdps = {"LKA_ACTIVE": 0, "LKA_FAULT": 0, "LFA2_ACTIVE": 0, "LFA2_FAULT": 0,
                 "STEERING_OUT_TORQUE": 0.0, "STEERING_COL_TORQUE": 0.0,
                 "STEERING_ANGLE": 0.0, "STEERING_ANGLE_2": 0.0,
                 "NEW_SIGNAL_1": 0, "NEW_SIGNAL_2": 0, "NEW_SIGNAL_3": 0,
                 "NEW_SIGNAL_4": 0, "NEW_SIGNAL_5": 0, "NEW_SIGNAL_6": 0}
    self.lfa = {"STEER_REQ": steer_req, "LKA_MODE": 2, "LKA_ICON": 1, "TORQUE_REQUEST": 0,
                "LKA_ACTIVE": 0, "VALUE63": 0, "VALUE64": 0, "HAS_LANE_SAFETY": 0,
                "LKAS_ANGLE_CMD": 0.0, "LKAS_ANGLE_ACTIVE": 0, "LKAS_ANGLE_MAX_TORQUE": 0,
                "NEW_SIGNAL_1": 0, "DampingGain": 0, "LKA_WARNING": 0, "FCA_SYSWARN": 0,
                "VALUE231": 0, "VALUE239": 0, "VALUE247": 0, "VALUE255": 0, "COUNTER": 0}
    self.lfa_alt = None
    self.steer_touch_2af = {"TOUCH_DETECT": 0, "TOUCH1": 0, "TOUCH2": 0, "CHECKSUM_": 0} if has_touch else None
    self.adrv_0x161 = {"ALERTS_1": alerts_1} if has_adrv else None


def decode(msgs, name, bus, dbc=DBC_NAME):
  parser = CANParser(dbc, [(name, 1)], bus)
  frames = [(addr, dat, src) for addr, dat, src in msgs]
  assert frames, f"{name} not transmitted"
  parser.update([(0, frames)])
  return dict(parser.vl[name])


def steering_msgs(CP, CS, frame=0, lat_active=True, apply_steer=0, angle_control=False):
  packer = CANPacker(DBC_NAME)
  CAN = CanBus(CP)
  CC = SimpleNamespace(latActive=lat_active, enabled=lat_active)
  return create_steering_messages_camera_scc(frame, packer, CP, CAN, CC, lat_active, apply_steer,
                                             CS, 0.0, 0, angle_control), CAN


def test_ev6_stays_camera_scc_so_the_scc_rx_check_stays_on_the_camera_bus():
  CP = ev6_params()
  assert CP.flags & HyundaiFlags.CANFD_CAMERA_SCC.value
  assert not CP.flags & HyundaiFlags.CANFD_HDA2.value


def test_fingerprint_cannot_distinguish_the_camera_bus():
  fingerprint = gen_empty_fingerprint()
  for addr in (0x35, 0xA0, 0xEA, 0x125, 0x130, 0x175, 0x1CF, 0x12A, 0x1A0, 0x1E0):
    fingerprint[0][addr] = 32
    fingerprint[2][addr] = 32
  CP = CarInterface.get_params(CAR.KIA_EV6, fingerprint, [], False, False, False)
  assert CP.flags & HyundaiFlags.CANFD_CAMERA_SCC.value


@pytest.mark.parametrize("platform,button_message,copy_enabled", [
  (CAR.HYUNDAI_TUCSON_4TH_GEN, "CRUISE_BUTTONS_ALT", False),
  (CAR.HYUNDAI_TUCSON_4TH_GEN, "CRUISE_BUTTONS", True),
  (CAR.KIA_EV6, "CRUISE_BUTTONS_ALT", True),
  (CAR.HYUNDAI_IONIQ_5, "CRUISE_BUTTONS_ALT", True),
])
@pytest.mark.parametrize("enabled,main_mode,acc_mode,lfa_symbol,info_display,stopping", [
  (False, False, 0, 1, 0, False),
  (True, False, 0, 1, 0, False),
  (True, True, 0, 1, 0, False),
  (True, True, 4, 1, 0, False),
  (True, True, 1, 0, 0, False),
  (True, True, 1, 1, 4, False),
  (True, True, 1, 1, 4, True),
])
def test_tucson_avoids_only_the_unblocked_camera_button_copy(
    platform, button_message, copy_enabled, enabled, main_mode, acc_mode, lfa_symbol, info_display, stopping, monkeypatch):
  monkeypatch.setattr(hyundaicanfd, "Params", StubParams)
  CP = SimpleNamespace(carFingerprint=platform, flags=HyundaiFlags.CAMERA_SCC.value)
  CAN = SimpleNamespace(ECAN=0, CAM=2)
  CC = SimpleNamespace(enabled=enabled, latActive=True)
  CS = SimpleNamespace(
    modelV2=None, lfahda_cluster={"HDA_CntrlModSta": 0, "HDA_LFA_SymSta": lfa_symbol},
    cruise_buttons_msg={"COUNTER": 42, "CRUISE_BUTTONS": 0}, cruise_btns_msg_canfd=button_message,
    MainMode_ACC=main_mode, ACCMode=acc_mode, scc_control={"InfoDisplay": info_display},
    adrv_0x161=None, adrv_0x1ea=None, ccnc_0x162=None, adrv_0x200={"COUNTER": 17, "TauGapSet": 4},
    out=SimpleNamespace(vEgo=20.0),
  )
  hud = SimpleNamespace(leadDistanceBars=2)
  button_address = 0x1AA if button_message == "CRUISE_BUTTONS_ALT" else 0x1CF

  def collect():
    packer = CANPacker(DBC_NAME)
    return [hyundaicanfd.create_ccnc_messages(CP, packer, CAN, frame, CC, CS, hud,
                                            0.0, False, False, False, stopping, False)
            for frame in range(1000)]

  candidate = collect()
  with monkeypatch.context() as m:
    m.setattr(hyundaicanfd, "CAMERA_BUTTON_COPY_DISABLED", set())
    original = collect()
  for frame, (before, after) in enumerate(zip(original, candidate, strict=True)):
    buttons = [msg for msg in after if msg[0] == button_address and msg[2] == CAN.CAM]
    assert len(buttons) == int(copy_enabled and frame % 2 == 0)
    expected = before if copy_enabled else [msg for msg in before if msg[0] != button_address or msg[2] != CAN.CAM]
    assert after == expected
  assert any(msg[0] == 0x200 for frame in candidate for msg in frame)


def test_camera_scc_steering_feeds_mdps_back_to_the_camera():
  CP = ev6_params(camera_scc=True)
  CS = FakeCarState(steer_req=1)
  msgs, CAN = steering_msgs(CP, CS)
  mdps = [m for m in msgs if m[0] == 0xEA]
  assert mdps and mdps[0][2] == CAN.CAM
  assert decode(mdps, "MDPS", CAN.CAM)["LKA_ACTIVE"] == 1

  CS = FakeCarState(steer_req=0)
  msgs, CAN = steering_msgs(CP, CS)
  mdps = [m for m in msgs if m[0] == 0xEA]
  assert decode(mdps, "MDPS", CAN.CAM)["LKA_ACTIVE"] == 0


def test_generic_steering_path_sends_nothing_to_the_camera():
  CP = ev6_params()
  packer = CANPacker(DBC_NAME)
  CAN = CanBus(CP)
  msgs = hyundaicanfd.create_steering_messages(packer, CP, CAN, True, True, 0, 0.0, 0, False)
  assert all(src != CAN.CAM for _, _, src in msgs)


@pytest.mark.parametrize("frame,spoofed", [(0, True), (HANDS_ON_SPOOF_FRAMES - 1, True),
                                           (HANDS_ON_SPOOF_FRAMES, False), (HANDS_ON_SPOOF_PERIOD, True)])
def test_hands_on_wheel_spoof_window(frame, spoofed):
  CP = ev6_params(camera_scc=True)
  CS = FakeCarState()
  CS.mdps["STEERING_COL_TORQUE"] = 0.0
  msgs, CAN = steering_msgs(CP, CS, frame=frame)
  torque = decode([m for m in msgs if m[0] == 0xEA], "MDPS", CAN.CAM)["STEERING_COL_TORQUE"]
  assert (torque == pytest.approx(HANDS_ON_SPOOF_TORQUE, abs=1.0)) is spoofed


def test_steer_touch_spoof_is_sent_on_the_camera_bus():
  CP = ev6_params(camera_scc=True)
  CS = FakeCarState()
  msgs, CAN = steering_msgs(CP, CS, frame=0)
  touch = [m for m in msgs if m[0] == 0x2AF]
  assert touch and touch[0][2] == CAN.CAM
  values = decode(touch, "STEER_TOUCH_2AF", CAN.CAM)
  assert values["TOUCH_DETECT"] == 3
  assert values["TOUCH1"] == 50 and values["TOUCH2"] == 50

  msgs, CAN = steering_msgs(CP, CS, frame=HANDS_ON_SPOOF_FRAMES)
  values = decode([m for m in msgs if m[0] == 0x2AF], "STEER_TOUCH_2AF", CAN.CAM)
  assert values["TOUCH_DETECT"] == 0


def test_steer_touch_is_only_sent_every_ten_frames():
  CP = ev6_params(camera_scc=True)
  CS = FakeCarState()
  assert any(m[0] == 0x2AF for m in steering_msgs(CP, CS, frame=10)[0])
  assert not any(m[0] == 0x2AF for m in steering_msgs(CP, CS, frame=11)[0])


def test_missing_camera_messages_are_tolerated():
  CP = ev6_params(camera_scc=True)
  CS = FakeCarState(has_touch=False, has_adrv=False)
  CS.mdps = None
  CS.lfa = None
  msgs, _ = steering_msgs(CP, CS)
  assert msgs == []


def test_lfa_torque_command_follows_lat_active():
  CP = ev6_params(camera_scc=True)
  CS = FakeCarState()
  msgs, CAN = steering_msgs(CP, CS, lat_active=True, apply_steer=120)
  values = decode([m for m in msgs if m[0] == 0x12A], "LFA", CAN.ECAN)
  assert values["STEER_REQ"] == 1
  assert values["TORQUE_REQUEST"] == 120
  assert values["LKA_ICON"] == 2

  msgs, CAN = steering_msgs(CP, CS, lat_active=False, apply_steer=0)
  values = decode([m for m in msgs if m[0] == 0x12A], "LFA", CAN.ECAN)
  assert values["STEER_REQ"] == 0
  assert values["LKA_ICON"] == 1
  assert values["DampingGain"] == 100


@pytest.mark.parametrize("alerts_1,overridden", [(0, True), (10, True), (11, False), (26, False), (27, True)])
def test_angle_control_defers_to_emergency_steering(alerts_1, overridden):
  CP = ev6_params(camera_scc=True)
  CP.flags |= HyundaiFlags.ANGLE_CONTROL.value
  CS = FakeCarState(alerts_1=alerts_1)
  CS.lfa["LKA_ACTIVE"] = 0
  msgs, CAN = steering_msgs(CP, CS, lat_active=True, angle_control=True)
  values = decode([m for m in msgs if m[0] == 0x12A], "LFA", CAN.ECAN)
  assert (values["LKA_ACTIVE"] == 3) is overridden


def test_hyundai_crc8_matches_reference_vector():
  assert hyundai_crc8(b"") == 0x00
  assert hyundai_crc8(bytes(7)) == 0x6A
  assert hyundai_crc8(b"\x01\x02\x03\x04\x05\x06\x07") == 0xB7
  assert hyundai_crc8(b"\xff" * 7) == 0x53


def test_hidden_adas_service_warning():
  values = {"FAULT_LCA": 1, "FAULT_HDA": 1, "FAULT_DAS": 1, "FAULT_LSS": 1}
  _hide_replaced_adas_service_warning(values)
  assert values == {"FAULT_LCA": 0, "FAULT_HDA": 0, "FAULT_DAS": 0, "FAULT_LSS": 1}


def test_unrelated_das_fault_is_preserved():
  values = {"FAULT_LCA": 0, "FAULT_HDA": 0, "FAULT_DAS": 1}
  _hide_replaced_adas_service_warning(values)
  assert values["FAULT_DAS"] == 1


def test_accel_jerk_limit_is_asymmetric():
  assert apply_accel_jerk_limit(2.0, 0.0, 5.0, 1.0, dt=0.02) == pytest.approx(0.1)
  assert apply_accel_jerk_limit(-2.0, 0.0, 5.0, 1.0, dt=0.02) == pytest.approx(-0.02)
  assert apply_accel_jerk_limit(0.05, 0.0, 5.0, 1.0, dt=0.02) == pytest.approx(0.05)
  assert apply_accel_jerk_limit(1.0, 0.0, -3.0, -3.0, dt=0.02) == pytest.approx(0.0)


@pytest.mark.parametrize("brake_hold,parking,expected", [(False, False, False), (True, False, True),
                                                         (False, True, True), (True, True, True)])
def test_longitudinal_interlock(brake_hold, parking, expected):
  CS = SimpleNamespace(out=SimpleNamespace(brakeHoldActive=brake_hold, parkingBrake=parking))
  assert longitudinal_interlock_active(CS) is bool(expected)


def _safety_tx_entries():
  text = SAFETY_HEADER.read_text()
  block = text.split("HYUNDAI_CANFD_HDA1_TX_MSGS_COMMON")[1].split("const CanMsg HYUNDAI_CANFD_HDA1_TX_MSGS")[0]
  return {(int(addr, 0), int(bus)) for addr, bus in re.findall(r"\{(0x[0-9A-Fa-f]+|\d+),\s*(\d+),", block)}


def test_safety_allows_camera_facing_feedback():
  entries = _safety_tx_entries()
  assert (0x0EA, 2) in entries
  assert (0x2AF, 2) in entries


class FakeLongCarState:
  def __init__(self, *, available=True, brake_hold=False, parking=False, scc_control=True):
    wheels = SimpleNamespace(fl=0.0, fr=0.0, rl=0.0, rr=0.0)
    self.out = SimpleNamespace(
      vEgo=0.0, vEgoRaw=0.0, aEgo=0.0, wheelSpeeds=wheels, canValid=True,
      brakePressed=False, gasPressed=False, gearShifter="drive",
      brakeHoldActive=brake_hold, parkingBrake=parking,
      cruiseState=SimpleNamespace(available=available, standstill=False),
    )
    self.softHoldActive = 0
    self.paddle_button_prev = 0
    self.canfdSccHoldActive = False
    self.scc_control = {
      "ACCMode": 0, "MainMode_ACC": 0, "StopReq": 0, "aReqValue": 0.0, "aReqRaw": 0.0,
      "VSetDis": 0, "JerkLowerLimit": 0.0, "JerkUpperLimit": 0.0, "DISTANCE_SETTING": 0,
      "DriveMode": 0, "ACC_ObjDist": 0.0, "ACC_ObjRelSpd": 0.0, "ACC_ObjLatPos": 0.0,
      "HUD_LEAD_INFO": 0, "DriverAlert": 0, "TARGET_DISTANCE": 0.0, "InfoDisplay": 0,
      "TakeOverReq": 0, "SysFailState": 0, "AccelLimitBandUpper": 0.0, "AccelLimitBandLower": 0.0,
      "ZEROS_7": 0, "ZEROS_2": 0, "ZEROS": 0, "NSCCOper": 0, "NSCCOnOff": 0,
      "CRUSE_INFO_SET_2": 0, "ObjDistLevel": 0, "COUNTER": 0,
    } if scc_control else None


def scc2(CS, enabled=True, accel=1.0, stopping=False, gas_override=False, stop_controller=None):
  CP = ev6_params(camera_scc=True)
  packer = CANPacker(DBC_NAME)
  CAN = CanBus(CP)
  jerk = SimpleNamespace(carrot_cruise=0, carrot_cruise_accel=0.0, jerk_u=5.0, jerk_l=5.0, comfort_active=False)
  hud = SimpleNamespace(leadDistanceBars=2, leadVisible=False, leadRelSpeed=0.0, leadRadar=0, leadDistance=0.0)
  msg, last = hyundaicanfd.create_acc_control_scc2(packer, CAN, enabled, 0.0, accel, stopping, gas_override,
                                                   30 * CV.MS_TO_KPH, hud, jerk, CS, stop_controller)
  return msg, last, CAN


def test_scc2_returns_the_ramped_accel():
  msg, last, CAN = scc2(FakeLongCarState())
  assert msg is not None
  values = decode([msg], "SCC_CONTROL", CAN.ECAN)
  assert values["ACCMode"] == 1
  assert values["SysFailState"] == 0
  assert values["DriverAlert"] == 0
  assert last == pytest.approx(values["aReqValue"], abs=0.02)
  assert 0.0 < last < 1.0


def test_scc2_resets_the_stop_controller_without_camera_scc_frame():
  controller = CanfdStopping()
  controller.phase = StopPhase.request
  msg, last, _ = scc2(FakeLongCarState(scc_control=False), stop_controller=controller)
  assert msg is None and last == 0.0
  assert controller.phase == StopPhase.idle


@pytest.mark.parametrize("kwargs", [{"available": False}, {"brake_hold": True}, {"parking": True}])
def test_scc2_disengages_on_interlocks(kwargs):
  msg, _, CAN = scc2(FakeLongCarState(**kwargs), stopping=True)
  values = decode([msg], "SCC_CONTROL", CAN.ECAN)
  assert values["ACCMode"] == 0
  assert values["StopReq"] == 0


def test_scc2_requests_stop_when_enabled():
  msg, _, CAN = scc2(FakeLongCarState(), stopping=True)
  assert decode([msg], "SCC_CONTROL", CAN.ECAN)["StopReq"] == 1


def test_stop_controller_holds_then_releases():
  controller = CanfdStopping()
  args = dict(active=True, requested=True, held=False, accel=-1.0, previous_value=0.0, jerk_u=5.0, jerk_l=5.0)
  command = controller.update(speed=0.0, **args)
  assert command.stop_req == 1
  assert controller.phase == StopPhase.request

  for _ in range(int(3.0 / 0.02) + 2):
    command = controller.update(speed=0.3, **args)
  assert controller.phase in (StopPhase.release, StopPhase.fallback)
  assert command.stop_req == 0
  assert command.raw <= -0.5


def test_stop_controller_retries_once_then_falls_back():
  controller = CanfdStopping()
  args = dict(active=True, requested=True, held=False, accel=-1.0, previous_value=0.0, jerk_u=5.0, jerk_l=5.0)
  for _ in range(int(10.0 / 0.02)):
    controller.update(speed=0.3, **args)
  assert controller.retried
  assert controller.phase == StopPhase.fallback


def test_stop_controller_resets_when_inactive():
  controller = CanfdStopping()
  controller.update(active=True, requested=True, speed=0.0, held=True, accel=-1.0,
                    previous_value=0.0, jerk_u=5.0, jerk_l=5.0)
  assert controller.phase == StopPhase.held
  assert controller.update(active=False, requested=True, speed=0.0, held=True, accel=-1.0,
                           previous_value=0.0, jerk_u=5.0, jerk_l=5.0) is None
  assert controller.phase == StopPhase.idle


def test_stopping_experiment_is_inert_without_a_controller():
  values = {"ACCMode": 1, "StopReq": 1, "aReqRaw": -1.0, "aReqValue": -1.0}
  hyundaicanfd.apply_stopping_experiment(values, FakeLongCarState(), None, -1.0, 0.0, 5.0, 5.0)
  assert values == {"ACCMode": 1, "StopReq": 1, "aReqRaw": -1.0, "aReqValue": -1.0}


def test_stopping_experiment_blocks_on_pedal():
  CS = FakeLongCarState()
  CS.out.brakePressed = True
  values = {"ACCMode": 1, "StopReq": 1, "aReqRaw": -1.0, "aReqValue": -1.0,
            "JerkUpperLimit": 5.0, "JerkLowerLimit": 5.0,
            "AccelLimitBandUpper": 0.0, "AccelLimitBandLower": 0.0}
  hyundaicanfd.apply_stopping_experiment(values, CS, CanfdStopping(), -1.0, 0.0, 5.0, 5.0)
  assert values["StopReq"] == 0
  assert values["aReqValue"] == 0.0
