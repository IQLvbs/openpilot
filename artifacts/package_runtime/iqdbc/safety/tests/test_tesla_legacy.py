"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos
"""
import unittest
import numpy as np

from iqdbc.car.lateral import get_max_angle_delta_vm, get_max_angle_vm
from iqdbc.car.tesla.values import CarControllerParams, TeslaLegacySafetyFlags
from iqdbc.car.structs import CarParams
from iqdbc.car.vehicle_model import VehicleModel
from iqdbc.can import CANDefine
from iqdbc.safety.tests.libsafety import libsafety_py
import iqdbc.safety.tests.common as common
from iqdbc.safety.tests.common import CANPackerSafety, away_round, round_speed

MSG_DAS_steeringControl = 0x488
MSG_APS_eacMonitor = 0x27d
MSG_DAS_Control_HW1 = 0x2b9
MSG_DAS_Control_PT = 0x2bf


def round_angle(apply_angle, can_offset=0):
  apply_angle_can = (apply_angle + 1638.35) / 0.1 + can_offset
  rnd_offset = 1e-5 if apply_angle >= 0 else -1e-5
  return away_round(apply_angle_can + rnd_offset) * 0.1 - 1638.35


def get_legacy_vm():
  from iqdbc.car.tesla.interface import CarInterface
  return VehicleModel(CarInterface.get_non_essential_params("TESLA_MODEL_S_HW3"))


def test_legacy_safety_params_match_vehicle_model():
  from iqdbc.car.tesla.interface import CarInterface
  from iqdbc.car.vehicle_model import calc_slip_factor
  CP = CarInterface.get_non_essential_params("TESLA_MODEL_S_HW3")
  assert abs(calc_slip_factor(VehicleModel(CP)) - (-0.0005666493436310427)) < 1e-12
  assert CP.steerRatio == 15.
  assert abs(CP.wheelbase - 2.96) < 1e-6


class TeslaLegacyAngleBase(common.AngleSteeringSafetyTest):
  STANDSTILL_THRESHOLD = 0.1
  GAS_PRESSED_THRESHOLD = 3

  STEER_ANGLE_MAX = 360
  DEG_TO_CAN = 10

  ANGLE_RATE_BP = None
  ANGLE_RATE_UP = None
  ANGLE_RATE_DOWN = None

  LATERAL_FREQUENCY = 50

  cnt_epas = 0
  cnt_angle_cmd = 0

  def _get_steer_cmd_angle_max(self, speed):
    return get_max_angle_vm(max(speed, 1), self.VM, CarControllerParams)

  def _angle_cmd_msg(self, angle: float, state: bool | int, increment_timer: bool = True, bus: int = 0):
    values = {"DAS_steeringAngleRequest": angle, "DAS_steeringControlType": state}
    if increment_timer:
      self.safety.set_timer(self.cnt_angle_cmd * int(1e6 / self.LATERAL_FREQUENCY))
      self.__class__.cnt_angle_cmd += 1
    return self.packer.make_can_msg_safety("DAS_steeringControl", bus, values)

  def _angle_meas_msg(self, angle: float, hands_on_level: int = 0, eac_status: int = 1, eac_error_code: int = 0):
    values = {
      "EPAS_internalSAS": angle,
      "EPAS_handsOnLevel": hands_on_level,
      "EPAS_eacStatus": eac_status,
      "EPAS_eacErrorCode": eac_error_code,
    }
    return self.packer.make_can_msg_safety("EPAS_sysStatus", 0, values)

  def test_angle_cmd_when_enabled(self):
    pass

  def test_steering_wheel_disengage(self):
    for hands_on_level in range(4):
      for eac_status in range(8):
        for eac_error_code in range(16):
          self.safety.set_controls_allowed(True)

          should_disengage = hands_on_level >= 3 or (eac_status == 0 and eac_error_code == 9)
          self.assertTrue(self._rx(self._angle_meas_msg(0, hands_on_level=hands_on_level,
                                                        eac_status=eac_status, eac_error_code=eac_error_code)))
          self.assertNotEqual(should_disengage, self.safety.get_controls_allowed())
          self.assertEqual(should_disengage, self.safety.get_steering_disengage_prev())

          self.assertTrue(self._rx(self._angle_meas_msg(0, hands_on_level=0, eac_status=1, eac_error_code=0)))
          self.assertNotEqual(should_disengage, self.safety.get_controls_allowed())
          self.assertFalse(self.safety.get_steering_disengage_prev())

  def test_steering_control_type(self):
    self.safety.set_controls_allowed(True)
    for steer_control_type in range(4):
      should_tx = steer_control_type in (self.steer_control_types["NONE"],
                                         self.steer_control_types["ANGLE_CONTROL"])
      self.assertEqual(should_tx, self._tx(self._angle_cmd_msg(0, state=steer_control_type)))

  def test_stock_lkas_passthrough(self):
    no_lkas_msg = self._angle_cmd_msg(0, state=False)
    no_lkas_msg_cam = self._angle_cmd_msg(0, state=True, bus=2)
    lkas_msg_cam = self._angle_cmd_msg(0, state=self.steer_control_types['LANE_KEEP_ASSIST'], bus=2)

    self.assertEqual(1, self._rx(no_lkas_msg_cam))
    self.assertEqual(-1, self.safety.safety_fwd_hook(2, no_lkas_msg_cam.addr))
    self.assertTrue(self._tx(no_lkas_msg))

    self.assertEqual(1, self._rx(lkas_msg_cam))
    self.assertEqual(0, self.safety.safety_fwd_hook(2, lkas_msg_cam.addr))
    self.assertFalse(self._tx(no_lkas_msg))

  def test_stock_lkas_ignored_while_lateral_active(self):
    self.safety.set_controls_allowed(True)
    lkas_msg_cam = self._angle_cmd_msg(0, state=self.steer_control_types['LANE_KEEP_ASSIST'], bus=2)
    self.assertEqual(1, self._rx(lkas_msg_cam))
    self.assertEqual(-1, self.safety.safety_fwd_hook(2, lkas_msg_cam.addr))
    self.assertTrue(self._tx(self._angle_cmd_msg(0, state=False)))

  def test_lateral_accel_limit(self):
    for speed in np.linspace(0, 40, 50):
      speed = max(speed, 1)
      speed = round_speed(away_round(speed / 0.01 * 3.6) * 0.01 / 3.6)
      for sign in (-1, 1):
        self.safety.set_controls_allowed(True)
        self._reset_speed_measurement(speed + 1)

        angle_unit_offset = -1 if sign == -1 else 0
        max_angle = round_angle(get_max_angle_vm(speed, self.VM, CarControllerParams), angle_unit_offset + 1) * sign
        max_angle = np.clip(max_angle, -self.STEER_ANGLE_MAX, self.STEER_ANGLE_MAX)
        self.safety.set_desired_angle_last(round(max_angle * self.DEG_TO_CAN))

        self.assertTrue(self._tx(self._angle_cmd_msg(max_angle, True)))

        max_angle_raw = round_angle(get_max_angle_vm(speed, self.VM, CarControllerParams), angle_unit_offset + 2) * sign
        max_angle = np.clip(max_angle_raw, -self.STEER_ANGLE_MAX, self.STEER_ANGLE_MAX)
        self._tx(self._angle_cmd_msg(max_angle, True))

        should_tx = abs(max_angle_raw) >= self.STEER_ANGLE_MAX
        self.assertEqual(should_tx, self._tx(self._angle_cmd_msg(max_angle, True)))

  def test_lateral_jerk_limit(self):
    for speed in np.linspace(0, 40, 50):
      speed = max(speed, 1)
      speed = round_speed(away_round(speed / 0.01 * 3.6) * 0.01 / 3.6)
      for sign in (-1, 1):
        self.safety.set_controls_allowed(True)
        self._reset_speed_measurement(speed + 1)
        self._tx(self._angle_cmd_msg(0, True))

        angle_unit_offset = 1 if sign == -1 else 0

        max_angle_delta = round_angle(get_max_angle_delta_vm(speed, self.VM, CarControllerParams),
                                      angle_unit_offset) * sign
        self.assertTrue(self._tx(self._angle_cmd_msg(max_angle_delta, True)))
        self.assertTrue(self._tx(self._angle_cmd_msg(max_angle_delta, True)))
        self.assertTrue(self._tx(self._angle_cmd_msg(0, True)))

        max_angle_delta = round_angle(get_max_angle_delta_vm(speed, self.VM, CarControllerParams),
                                      angle_unit_offset + 1) * sign
        self.assertFalse(self._tx(self._angle_cmd_msg(max_angle_delta, True)))

        self.safety.set_desired_angle_last(round(max_angle_delta * self.DEG_TO_CAN))
        self.assertTrue(self._tx(self._angle_cmd_msg(max_angle_delta, True)))

        self.assertFalse(self._tx(self._angle_cmd_msg(0, True)))
        self.assertTrue(self._tx(self._angle_cmd_msg(0, True)))


class TeslaLegacyLongitudinalBase(common.LongitudinalAccelSafetyTest):
  MAX_ACCEL = 2.0
  MIN_ACCEL = -3.48
  INACTIVE_ACCEL = 0.0

  def _long_control_msg(self, set_speed, acc_state=0, jerk_limits=(0, 0), accel_limits=(0, 0), aeb_event=0, bus=0):
    values = {
      "DAS_setSpeed": set_speed,
      "DAS_accState": acc_state,
      "DAS_aebEvent": aeb_event,
      "DAS_jerkMin": jerk_limits[0],
      "DAS_jerkMax": jerk_limits[1],
      "DAS_accelMin": accel_limits[0],
      "DAS_accelMax": accel_limits[1],
    }
    return self.packer_long.make_can_msg_safety("DAS_control", bus, values)

  def _accel_msg(self, accel: float):
    return self._long_control_msg(10, accel_limits=(accel, max(accel, 0)))

  def test_no_aeb(self):
    for aeb_event in range(4):
      self.assertEqual(self._tx(self._long_control_msg(10, aeb_event=aeb_event)), aeb_event == 0)

  def test_stock_aeb_passthrough(self):
    no_aeb_msg = self._long_control_msg(10, aeb_event=0)
    no_aeb_msg_cam = self._long_control_msg(10, aeb_event=0, bus=2)
    aeb_msg_cam = self._long_control_msg(10, aeb_event=1, bus=2)

    self.assertEqual(1, self._rx(no_aeb_msg_cam))
    self.assertEqual(-1, self.safety.safety_fwd_hook(2, no_aeb_msg_cam.addr))
    self.assertTrue(self._tx(no_aeb_msg))

    self.assertEqual(1, self._rx(aeb_msg_cam))
    self.assertEqual(0, self.safety.safety_fwd_hook(2, aeb_msg_cam.addr))
    self.assertFalse(self._tx(no_aeb_msg))

  def test_prevent_reverse(self):
    self.safety.set_controls_allowed(True)

    self.assertTrue(self._tx(self._long_control_msg(set_speed=10, accel_limits=(1.1, 0.8))))
    self.assertTrue(self._tx(self._long_control_msg(set_speed=0, accel_limits=(1.1, 0.8))))

    self.assertTrue(self._tx(self._long_control_msg(set_speed=10, accel_limits=(0, 0))))
    self.assertTrue(self._tx(self._long_control_msg(set_speed=0, accel_limits=(0, 0))))

    self.assertTrue(self._tx(self._long_control_msg(set_speed=10, accel_limits=(-0.8, 1.3))))
    self.assertTrue(self._tx(self._long_control_msg(set_speed=0, accel_limits=(0.8, -1.3))))
    self.assertTrue(self._tx(self._long_control_msg(set_speed=0, accel_limits=(0, -1.3))))

    self.assertFalse(self._tx(self._long_control_msg(set_speed=10, accel_limits=(-1.1, -0.6))))
    self.assertFalse(self._tx(self._long_control_msg(set_speed=0, accel_limits=(-0.6, -1.1))))
    self.assertFalse(self._tx(self._long_control_msg(set_speed=0, accel_limits=(-0.1, -0.1))))


class TeslaLegacyStockSideBase(common.CarSafetyTest):
  STANDSTILL_THRESHOLD = 0.1
  GAS_PRESSED_THRESHOLD = 3

  chassis_bus = 0

  def _user_brake_msg(self, brake):
    values = {"driverBrakeStatus": 2 if brake else 1}
    return self.packer_chassis.make_can_msg_safety("BrakeMessage", self.chassis_bus, values)

  def _speed_msg(self, speed):
    values = {"ESP_vehicleSpeed": speed * 3.6}
    return self.packer_chassis.make_can_msg_safety("ESP_B", self.chassis_bus, values)

  def _vehicle_moving_msg(self, speed: float):
    values = {"DI_cruiseState": 3 if speed <= self.STANDSTILL_THRESHOLD else 2, "DI_speedUnits": 1}
    return self.packer_chassis.make_can_msg_safety("DI_state", self.chassis_bus, values)

  def _user_gas_msg(self, gas):
    values = {"DI_pedalPos": gas}
    return self.packer_chassis.make_can_msg_safety("DI_torque1", self.chassis_bus, values)

  def _pcm_status_msg(self, enable):
    values = {"DI_cruiseState": 2 if enable else 0, "DI_speedUnits": 1}
    return self.packer_chassis.make_can_msg_safety("DI_state", self.chassis_bus, values)

  def test_rx_hook(self):
    for _ in range(5):
      msg = self._angle_cmd_msg(0, True, bus=2)
      self.safety.set_controls_allowed(True)
      self.assertTrue(self._rx(msg))
      self.assertTrue(self.safety.get_controls_allowed())

    for _ in range(5):
      msg = self._speed_msg(0)
      self.safety.set_controls_allowed(True)
      self.assertTrue(self._rx(msg))
      self.assertTrue(self.safety.get_controls_allowed())

  def test_vehicle_speed_measurements(self):
    self._common_measurement_test(self._speed_msg, 0, 285 / 3.6, 1,
                                  self.safety.get_vehicle_speed_min, self.safety.get_vehicle_speed_max)

  def test_prev_gas(self):
    pass

  def test_no_disengage_on_gas(self):
    pass


class TestTeslaHW2Safety(TeslaLegacyStockSideBase, TeslaLegacyAngleBase):
  chassis_bus = 0
  RELAY_MALFUNCTION_ADDRS = {0: (MSG_DAS_steeringControl, MSG_APS_eacMonitor)}
  FWD_BLACKLISTED_ADDRS = {2: [MSG_DAS_steeringControl, MSG_APS_eacMonitor]}
  TX_MSGS = [[MSG_DAS_steeringControl, 0], [MSG_APS_eacMonitor, 0]]
  SAFETY_PARAM = int(TeslaLegacySafetyFlags.HW2)

  def setUp(self):
    self.VM = get_legacy_vm()
    self.packer = CANPackerSafety("tesla_can")
    self.packer_chassis = CANPackerSafety("tesla_can")
    self.define = CANDefine("tesla_can")
    self.steer_control_types = {d: v for v, d in self.define.dv["DAS_steeringControl"]["DAS_steeringControlType"].items()}

    self.safety = libsafety_py.libsafety
    self.safety.set_safety_hooks(CarParams.SafetyModel.teslaLegacy, self.SAFETY_PARAM)
    self.safety.init_tests()


class TestTeslaHW3Safety(TeslaLegacyStockSideBase, TeslaLegacyAngleBase):
  chassis_bus = 1
  RELAY_MALFUNCTION_ADDRS = {0: (MSG_DAS_steeringControl, MSG_APS_eacMonitor)}
  FWD_BLACKLISTED_ADDRS = {2: [MSG_DAS_steeringControl, MSG_APS_eacMonitor]}
  TX_MSGS = [[MSG_DAS_steeringControl, 0], [MSG_APS_eacMonitor, 0]]
  SAFETY_PARAM = int(TeslaLegacySafetyFlags.HW3)

  def setUp(self):
    self.VM = get_legacy_vm()
    self.packer = CANPackerSafety("tesla_raven_party")
    self.packer_chassis = CANPackerSafety("tesla_can")
    self.define = CANDefine("tesla_raven_party")
    self.steer_control_types = {d: v for v, d in self.define.dv["DAS_steeringControl"]["DAS_steeringControlType"].items()}

    self.safety = libsafety_py.libsafety
    self.safety.set_safety_hooks(CarParams.SafetyModel.teslaLegacy, self.SAFETY_PARAM)
    self.safety.init_tests()


class TeslaLegacyExternalPandaBase(TeslaLegacyLongitudinalBase, common.CarSafetyTest):
  STANDSTILL_THRESHOLD = 0.1
  GAS_PRESSED_THRESHOLD = 3

  RELAY_MALFUNCTION_ADDRS = {0: (MSG_DAS_Control_PT,)}
  FWD_BLACKLISTED_ADDRS = {2: [MSG_DAS_Control_PT]}
  TX_MSGS = [[MSG_DAS_Control_PT, 0]]

  def setUp(self):
    self.packer = CANPackerSafety("tesla_powertrain")
    self.packer_long = self.packer
    self.safety = libsafety_py.libsafety
    self.safety.set_safety_hooks(CarParams.SafetyModel.teslaLegacy, self.SAFETY_PARAM)
    self.safety.init_tests()

  def _vehicle_moving_msg(self, speed: float):
    values = {"DI_cruiseState": 3 if speed <= self.STANDSTILL_THRESHOLD else 2, "DI_speedUnits": 1}
    return self.packer.make_can_msg_safety("DI_state", 0, values)

  def _user_brake_msg(self, brake):
    values = {"driverBrakeStatus": 2 if brake else 1}
    return self.packer.make_can_msg_safety("BrakeMessage", 0, values)

  def _user_gas_msg(self, gas):
    values = {"DI_pedalPos": gas}
    return self.packer.make_can_msg_safety("DI_torque1", 0, values)

  def _pcm_status_msg(self, enable):
    values = {"DI_cruiseState": 2 if enable else 0, "DI_speedUnits": 1}
    return self.packer.make_can_msg_safety("DI_state", 0, values)

  def _speed_msg(self, speed):
    return self._vehicle_moving_msg(speed)

  def test_rx_hook(self):
    for _ in range(5):
      msg = self._long_control_msg(0, bus=2)
      self.safety.set_controls_allowed(True)
      self.assertTrue(self._rx(msg))
      self.assertTrue(self.safety.get_controls_allowed())


class TestTeslaHW2ExternalPandaSafety(TeslaLegacyExternalPandaBase):
  SAFETY_PARAM = int(TeslaLegacySafetyFlags.HW2 | TeslaLegacySafetyFlags.EXTERNAL_PANDA)


class TestTeslaHW3ExternalPandaSafety(TeslaLegacyExternalPandaBase):
  SAFETY_PARAM = int(TeslaLegacySafetyFlags.HW3 | TeslaLegacySafetyFlags.EXTERNAL_PANDA)


class TestTeslaHW1Safety(TeslaLegacyLongitudinalBase, TeslaLegacyAngleBase, common.CarSafetyTest):
  RELAY_MALFUNCTION_ADDRS = {0: (MSG_DAS_steeringControl, MSG_DAS_Control_HW1)}
  FWD_BLACKLISTED_ADDRS = {2: [MSG_DAS_steeringControl, MSG_DAS_Control_HW1]}
  TX_MSGS = [[MSG_DAS_steeringControl, 0], [MSG_DAS_Control_HW1, 0]]

  STANDSTILL_THRESHOLD = 0.1
  GAS_PRESSED_THRESHOLD = 3

  def setUp(self):
    self.VM = get_legacy_vm()
    self.packer = CANPackerSafety("tesla_can")
    self.packer_long = self.packer
    self.define = CANDefine("tesla_can")
    self.steer_control_types = {d: v for v, d in self.define.dv["DAS_steeringControl"]["DAS_steeringControlType"].items()}

    self.safety = libsafety_py.libsafety
    self.safety.set_safety_hooks(CarParams.SafetyModel.teslaLegacy, int(TeslaLegacySafetyFlags.HW1))
    self.safety.init_tests()

  def _user_brake_msg(self, brake):
    values = {"driverBrakeStatus": 2 if brake else 1}
    return self.packer.make_can_msg_safety("BrakeMessage", 0, values)

  def _speed_msg(self, speed):
    values = {"ESP_vehicleSpeed": speed * 3.6}
    return self.packer.make_can_msg_safety("ESP_B", 0, values)

  def _vehicle_moving_msg(self, speed: float):
    values = {"DI_cruiseState": 3 if speed <= self.STANDSTILL_THRESHOLD else 2}
    return self.packer.make_can_msg_safety("DI_state", 0, values)

  def _user_gas_msg(self, gas):
    values = {"DI_pedalPos": gas}
    return self.packer.make_can_msg_safety("DI_torque1", 0, values)

  def _pcm_status_msg(self, enable):
    values = {"DI_cruiseState": 2 if enable else 0}
    return self.packer.make_can_msg_safety("DI_state", 0, values)

  def test_rx_hook(self):
    for msg_type in ("angle", "long", "speed"):
      for _ in range(5):
        if msg_type == "angle":
          msg = self._angle_cmd_msg(0, True, bus=2)
        elif msg_type == "long":
          msg = self._long_control_msg(0, bus=2)
        else:
          msg = self._speed_msg(0)

        self.safety.set_controls_allowed(True)
        self.assertTrue(self._rx(msg))
        self.assertTrue(self.safety.get_controls_allowed())

  def test_vehicle_speed_measurements(self):
    self._common_measurement_test(self._speed_msg, 0, 285 / 3.6, 1,
                                  self.safety.get_vehicle_speed_min, self.safety.get_vehicle_speed_max)


if __name__ == "__main__":
  unittest.main()
