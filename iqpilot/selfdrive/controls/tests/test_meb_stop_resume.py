# Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
import sys
from types import ModuleType

import pytest

from iqdbc.can import CANPacker, CANParser
from iqdbc.car import Bus, structs
from iqdbc.car.volkswagen import mebcan
from iqdbc.car.volkswagen.interface import CarInterface
from iqdbc.car.volkswagen.values import CAR, DBC
from iqpilot.common.realtime import DT_CTRL
from iqpilot.selfdrive.controls.lib.longcontrol import LongControl, LongCtrlState

ESC_HOLDING = 3
ACCEL_LIMITS = (-3.5, 2.0)
STOPPED_TIME = 3.0
RESUME_DEADLINE = 0.5
VERIFIED_IMPORT = "iqpilot.system.proprietary_runtime._verified_import"


class InertAlc:
  def __getattr__(self, name):
    return lambda *args, **kwargs: None

  def angle_lateral_control_enabled(self, controller, cs):
    return False

  def vw_driver_override_threshold_cnm(self, cs, platform, allowance):
    return allowance

  def create_vehicle_odometer_store(self, cp, params):
    return InertAlc()


@pytest.fixture(autouse=True)
def lateral_free_alc(monkeypatch):
  alc = InertAlc()
  stub = ModuleType(VERIFIED_IMPORT)
  stub.import_verified_module = lambda *args: alc
  monkeypatch.setitem(sys.modules, VERIFIED_IMPORT, stub)


class MebStopAndGo:
  def __init__(self, platform):
    fingerprint = {bus: {} for bus in range(7)}
    cp = CarInterface.get_params(platform, fingerprint, [], alpha_long=True, is_release=False, docs=False)
    cp_iq = CarInterface.get_params_iq(cp, platform, fingerprint, [], alpha_long=True, is_release_iq=False, docs=False)
    assert cp.openpilotLongitudinalControl and not cp.pcmCruise
    self.car = CarInterface(cp, cp_iq)
    self.long_control = LongControl(cp, cp_iq)
    self.long_control.smooth.enabled = False
    self.packer = CANPacker(DBC[platform][Bus.pt])
    self.acc_parser = CANParser(DBC[platform][Bus.pt], [("ACC_18", 0)], self.car.CC.CAN.pt)
    self.frame = 0

  def step(self, should_stop, a_target):
    self.frame += 1
    now_nanos = self.frame * int(DT_CTRL * 1e9)
    esc = self.packer.make_can_msg("ESC_50", self.car.CC.CAN.pt, {"Motion_State": ESC_HOLDING, "COUNTER": self.frame % 16})
    car_state, _ = self.car.update([(now_nanos, [esc])])

    accel = self.long_control.update(True, car_state, a_target, should_stop, ACCEL_LIMITS, v_target_now=0.0)
    cc = structs.CarControl()
    cc.enabled = cc.longActive = True
    cc.actuators.accel = float(accel)
    cc.actuators.longControlState = self.long_control.long_control_state
    _, messages = self.car.apply(cc.as_reader(), structs.IQCarControl(), now_nanos)
    self.acc_parser.update([(now_nanos, messages)])
    return car_state, self.acc_parser.vl["ACC_18"]


@pytest.mark.parametrize("platform", [CAR.VOLKSWAGEN_ID4_MK1, CAR.SKODA_ENYAQ_MK1])
def test_held_stop_resumes_when_the_plan_moves_off(platform):
  drive = MebStopAndGo(platform)
  for _ in range(int(STOPPED_TIME / DT_CTRL)):
    car_state, acc = drive.step(True, -0.5)
  assert drive.car.CS.esp_hold_confirmation
  assert drive.long_control.long_control_state == LongCtrlState.stopping
  assert acc["ACC_Anforderung_HMS"] == mebcan.ACC_HMS_HOLD
  assert not acc["ACC_Anfahren"]

  resumed_after = None
  for i in range(int(2 * RESUME_DEADLINE / DT_CTRL)):
    car_state, acc = drive.step(False, 0.5)
    assert not car_state.cruiseState.standstill
    if acc["ACC_Anfahren"]:
      resumed_after = (i + 1) * DT_CTRL
      break

  assert resumed_after is not None and resumed_after <= RESUME_DEADLINE
  assert acc["ACC_Anforderung_HMS"] == mebcan.ACC_HMS_RELEASE
  assert acc["ACC_Sollbeschleunigung_02"] > 0.0


@pytest.mark.parametrize("platform", [CAR.VOLKSWAGEN_ID4_MK1])
def test_one_frame_stop_dropout_never_launches(platform):
  drive = MebStopAndGo(platform)
  for _ in range(int(STOPPED_TIME / DT_CTRL)):
    drive.step(True, -0.5)

  for i in range(int(STOPPED_TIME / DT_CTRL)):
    _, acc = drive.step(i % 2 == 0, 0.5)
    assert not acc["ACC_Anfahren"]
    assert acc["ACC_Anforderung_HMS"] == mebcan.ACC_HMS_HOLD
