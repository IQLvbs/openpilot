from parameterized import parameterized

from iqpilot.cereal import car, log
from iqdbc.car.car_helpers import interfaces
from iqdbc.car.honda.values import CAR as HONDA
from iqdbc.car.toyota.values import CAR as TOYOTA
from iqdbc.car.nissan.values import CAR as NISSAN
from iqdbc.car.gm.values import CAR as GM
from iqdbc.car.volkswagen.values import CAR as VOLKSWAGEN
from iqdbc.car.vehicle_model import VehicleModel
from iqpilot.common.realtime import DT_CTRL
from iqpilot.selfdrive.car.helpers import convert_to_capnp
from iqpilot.selfdrive.controls.lib.latcontrol_pid import LatControlPID
from iqpilot.selfdrive.controls.lib.latcontrol_torque import LatControlTorque
from iqpilot.selfdrive.controls.lib.latcontrol_torque_pq import LatControlTorquePQ
from iqpilot.selfdrive.controls.lib.latcontrol_torque_v0 import LatControlTorqueV0, is_vw_mqb_torque
import iqpilot.selfdrive.controls.lib.latcontrol_torque_pq as latcontrol_torque_pq
from iqpilot.selfdrive.controls.lib.latcontrol_angle import LatControlAngle
from iqpilot.selfdrive.locationd.helpers import Pose
from iqpilot.common.mock.generators import generate_deviceMotion
from iqpilot.selfdrive.car import interfaces as iqpilot_interfaces


class TestLatControl:

  @staticmethod
  def build_pq_controller():
    car_name = VOLKSWAGEN.VOLKSWAGEN_PASSAT_NMS
    CarInterface = interfaces[car_name]
    CP = CarInterface.get_non_essential_params(car_name)
    CP_IQ = CarInterface.get_non_essential_params_iq(CP, car_name)
    CI = CarInterface.__new__(CarInterface)
    CI.CP = CP
    CI.CP_IQ = CP_IQ
    iqpilot_interfaces.apply_iq_car_config(CI)
    return CP, LatControlTorquePQ(CP.as_reader(), convert_to_capnp(CP_IQ).as_reader(), CI, DT_CTRL)

  @staticmethod
  def build_v0_controller():
    car_name = TOYOTA.TOYOTA_RAV4
    CarInterface = interfaces[car_name]
    CP = CarInterface.get_non_essential_params(car_name)
    CP_IQ = CarInterface.get_non_essential_params_iq(CP, car_name)
    CI = CarInterface(CP, CP_IQ)
    iqpilot_interfaces.apply_iq_car_config(CI)
    return CP, LatControlTorqueV0(CP.as_reader(), convert_to_capnp(CP_IQ).as_reader(), CI, DT_CTRL)

  @parameterized.expand([(HONDA.HONDA_CIVIC, LatControlPID), (TOYOTA.TOYOTA_RAV4, LatControlTorque), (TOYOTA.TOYOTA_RAV4, LatControlTorqueV0),
                         (NISSAN.NISSAN_LEAF, LatControlAngle), (GM.CHEVROLET_BOLT_EUV, LatControlTorque)])
  def test_saturation(self, car_name, controller):
    CarInterface = interfaces[car_name]
    CP = CarInterface.get_non_essential_params(car_name)
    CP_IQ = CarInterface.get_non_essential_params_iq(CP, car_name)
    CI = CarInterface(CP, CP_IQ)
    iqpilot_interfaces.apply_iq_car_config(CI)
    CP_IQ = convert_to_capnp(CP_IQ)
    VM = VehicleModel(CP)

    controller = controller(CP.as_reader(), CP_IQ.as_reader(), CI, DT_CTRL)

    CS = car.CarState.new_message()
    CS.vEgo = 30
    CS.steeringPressed = False

    params = log.VehicleParameters.new_message()

    lp = generate_deviceMotion()
    pose = Pose.from_live_pose(lp.deviceMotion)

    # Saturate for curvature limited and controller limited
    for _ in range(1000):
      _, _, lac_log = controller.update(True, CS, VM, params, False, 0, pose, True, 0.2)
    assert lac_log.saturated

    for _ in range(1000):
      _, _, lac_log = controller.update(True, CS, VM, params, False, 0, pose, False, 0.2)
    assert not lac_log.saturated

    for _ in range(1000):
      _, _, lac_log = controller.update(True, CS, VM, params, False, 1, pose, False, 0.2)
    assert lac_log.saturated

  def test_pq_controller_update(self):
    CP, controller = self.build_pq_controller()
    VM = VehicleModel(CP)

    CS = car.CarState.new_message()
    CS.vEgo = 30
    params = log.VehicleParameters.new_message()
    pose = Pose.from_live_pose(generate_deviceMotion().deviceMotion)

    _, _, lac_log = controller.update(True, CS, VM, params, False, 0.001, pose, False, 0.2)
    assert lac_log.active

  def test_v0_uses_current_lateral_acceleration_setpoint(self):
    CP, controller = self.build_v0_controller()
    VM = VehicleModel(CP)
    CS = car.CarState.new_message(vEgo=30)
    params = log.VehicleParameters.new_message()
    pose = Pose.from_live_pose(generate_deviceMotion().deviceMotion)

    _, _, lac_log = controller.update(True, CS, VM, params, False, 0.001, pose, False, 0.3)

    assert lac_log.version == 0
    assert abs(lac_log.desiredLateralAccel - 0.9) < 1e-6

  def test_v0_platform_selection(self):
    mqb_interface = interfaces[VOLKSWAGEN.VOLKSWAGEN_PASSAT_MK8]
    mqb_params = mqb_interface.get_non_essential_params(VOLKSWAGEN.VOLKSWAGEN_PASSAT_MK8)

    assert is_vw_mqb_torque(mqb_params)
    for car_name in (VOLKSWAGEN.VOLKSWAGEN_PASSAT_MK7, VOLKSWAGEN.VOLKSWAGEN_GOLF_MK8,
                     VOLKSWAGEN.VOLKSWAGEN_ID4_MK1, VOLKSWAGEN.AUDI_A4_MK4, TOYOTA.TOYOTA_RAV4):
      CarInterface = interfaces[car_name]
      assert not is_vw_mqb_torque(CarInterface.get_non_essential_params(car_name))

  def test_pq_controller_inactive_slew_reset(self):
    CP, controller = self.build_pq_controller()
    controller.lateral_acceleration_slew_limiter.enabled = True
    VM = VehicleModel(CP)
    CS = car.CarState.new_message(vEgo=30)
    params = log.VehicleParameters.new_message()
    pose = Pose.from_live_pose(generate_deviceMotion().deviceMotion)

    torque, angle, lac_log = controller.update(False, CS, VM, params, False, 0.001, pose, False, 0.2)
    assert torque == 0.0
    assert angle == 0.0
    assert not lac_log.active
    assert controller.lateral_acceleration_slew_limiter.a_lim == 0.9

  @staticmethod
  def settle_pq_live_torque(controller, factor, offset, valid, seconds=60.0):
    controller.set_live_torque_valid(valid)
    for _ in range(int(seconds / DT_CTRL)):
      controller.update_live_torque_params(factor, offset, 0.4)

  def test_pq_live_torque_falls_back_until_torqued_is_valid(self):
    _, controller = self.build_pq_controller()
    self.settle_pq_live_torque(controller, 3.0, 0.2, valid=False)
    assert abs(controller.torque_params.latAccelFactor - latcontrol_torque_pq.DEFAULT_LAT_ACCEL_FACTOR) < 1e-6
    assert abs(controller.torque_params.latAccelOffset - latcontrol_torque_pq.DEFAULT_LAT_ACCEL_OFFSET) < 1e-6
    assert abs(controller.torque_params.friction - latcontrol_torque_pq.DEFAULT_FRICTION) < 1e-6

  def test_pq_live_torque_uses_learned_factor_and_offset(self):
    _, controller = self.build_pq_controller()
    self.settle_pq_live_torque(controller, 2.35, 0.047, valid=True)
    expected_factor = 2.35 * latcontrol_torque_pq.LIVE_LAT_ACCEL_FACTOR_GAIN
    assert abs(controller.torque_params.latAccelFactor - expected_factor) < 1e-3
    assert abs(controller.torque_params.latAccelOffset - 0.047) < 1e-3
    assert abs(controller.torque_params.friction - latcontrol_torque_pq.DEFAULT_FRICTION) < 1e-6

  def test_pq_live_torque_changes_are_smoothed(self):
    _, controller = self.build_pq_controller()
    controller.set_live_torque_valid(True)
    controller.update_live_torque_params(3.0, 0.2, 0.4)
    step = controller.torque_params.latAccelFactor - latcontrol_torque_pq.DEFAULT_LAT_ACCEL_FACTOR
    assert 0.0 < step < 0.01

  def test_pq_live_torque_is_bounded(self):
    _, controller = self.build_pq_controller()
    self.settle_pq_live_torque(controller, 10.0, 2.0, valid=True)
    assert abs(controller.torque_params.latAccelFactor - latcontrol_torque_pq.LIVE_LAT_ACCEL_FACTOR_BOUNDS[1]) < 1e-3
    assert abs(controller.torque_params.latAccelOffset - latcontrol_torque_pq.LIVE_LAT_ACCEL_OFFSET_BOUNDS[1]) < 1e-3
    self.settle_pq_live_torque(controller, 0.1, -2.0, valid=True)
    assert abs(controller.torque_params.latAccelFactor - latcontrol_torque_pq.LIVE_LAT_ACCEL_FACTOR_BOUNDS[0]) < 1e-3
    assert abs(controller.torque_params.latAccelOffset - latcontrol_torque_pq.LIVE_LAT_ACCEL_OFFSET_BOUNDS[0]) < 1e-3

  def test_pq_live_torque_rejects_non_finite_estimates(self):
    _, controller = self.build_pq_controller()
    self.settle_pq_live_torque(controller, float('nan'), 0.2, valid=True)
    assert abs(controller.torque_params.latAccelFactor - latcontrol_torque_pq.DEFAULT_LAT_ACCEL_FACTOR) < 1e-6
    assert abs(controller.torque_params.latAccelOffset - latcontrol_torque_pq.DEFAULT_LAT_ACCEL_OFFSET) < 1e-6

  def test_pq_live_torque_freeze_keeps_defaults(self, monkeypatch):
    monkeypatch.setattr(latcontrol_torque_pq, "FREEZE_LIVE_TORQUE_PARAMS", True)
    _, controller = self.build_pq_controller()
    self.settle_pq_live_torque(controller, 3.0, 0.2, valid=True, seconds=1.0)
    assert abs(controller.torque_params.latAccelFactor - latcontrol_torque_pq.DEFAULT_LAT_ACCEL_FACTOR) < 1e-6
    assert abs(controller.torque_params.latAccelOffset - latcontrol_torque_pq.DEFAULT_LAT_ACCEL_OFFSET) < 1e-6

  def test_pq_integrator_gain_rises_at_highway_speed(self):
    _, controller = self.build_pq_controller()
    controller.pid.speed = 5.0
    assert abs(controller.pid.k_i - latcontrol_torque_pq.KI) < 1e-9
    controller.pid.speed = 30.0
    assert abs(controller.pid.k_i - latcontrol_torque_pq.KI_HIGHWAY) < 1e-9
