# Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
import math
import sys
from collections import deque
from types import ModuleType

import numpy as np
import pytest

from iqdbc.can import CANPacker, CANParser
from iqdbc.car import Bus, structs
from iqdbc.car.volkswagen.interface import CarInterface
from iqdbc.car.volkswagen.mebutils import CurvatureSmoother
from iqdbc.car.volkswagen.values import CAR, DBC

VERIFIED_IMPORT = "iqpilot.system.proprietary_runtime._verified_import"
PLATFORMS = [CAR.VOLKSWAGEN_ID4_MK1, CAR.VOLKSWAGEN_ID4_MK2, CAR.VOLKSWAGEN_GOLF_MK8]
DT = 0.01
POWER_RESOLUTION = 0.4
SMOOTHER_DT = 0.02
PASS_THROUGH_TAU = 1e-6
EPS_DELAY_FRAMES = 3
EPS_TAU = 0.15
EPS_GAIN = 0.98
VEHICLE_DELAY_FRAMES = 11


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


class SteeringRig:
  def __init__(self, platform, smoothing=True, curvature_controller=True, ratio=1.0, driver_torque=0):
    fingerprint = {bus: {} for bus in range(7)}
    cp = CarInterface.get_params(platform, fingerprint, [], alpha_long=False, is_release=False, docs=False)
    cp_iq = CarInterface.get_params_iq(cp, platform, fingerprint, [], alpha_long=False, is_release_iq=False, docs=False)
    self.car = CarInterface(cp, cp_iq)
    if not smoothing:
      self.car.CC.curvature_smoother.tau_v = [PASS_THROUGH_TAU] * len(self.car.CC.curvature_smoother.tau_v)
    self.packer = CANPacker(DBC[platform][Bus.pt])
    self.parser = CANParser(DBC[platform][Bus.pt], [("HCA_03", 0)], self.car.CC.CAN.pt)
    self.bus = self.car.CC.CAN.pt
    self.curvature_controller = curvature_controller
    self.ratio = ratio
    self.driver_torque = driver_torque
    self.eps = 0.
    self.command = 0.
    self.power = 0.
    self.commands = deque([0.] * (EPS_DELAY_FRAMES + 1), maxlen=EPS_DELAY_FRAMES + 1)
    self.eps_history = deque([0.] * (VEHICLE_DELAY_FRAMES + 1), maxlen=VEHICLE_DELAY_FRAMES + 1)
    self.frame = 0
    self.sent_frames = 0
    self.real = 0.

  def step(self, v_ego, desired, lat_active=True):
    self.frame += 1
    self.real = self.eps_history[0] / self.ratio
    yaw_rate = self.real * v_ego
    now = self.frame * 10_000_000
    counter = self.frame % 16
    kph = v_ego * 3.6
    messages = [
      self.packer.make_can_msg("QFK_01", self.bus, {"Curvature": abs(self.eps), "Curvature_VZ": 1 if self.eps > 0 else 0, "COUNTER": counter}),
      self.packer.make_can_msg("ESC_50", self.bus, {"Yaw_Rate": abs(math.degrees(yaw_rate)), "Yaw_Rate_Sign": 1 if yaw_rate > 0 else 0, "COUNTER": counter}),
      self.packer.make_can_msg("ESC_51", self.bus, {"VL_Radgeschw": kph, "VR_Radgeschw": kph, "HL_Radgeschw": kph, "HR_Radgeschw": kph, "COUNTER": counter}),
      self.packer.make_can_msg("LH_EPS_03", self.bus, {
        "EPS_Lenkmoment": abs(self.driver_torque), "EPS_VZ_Lenkmoment": 1 if self.driver_torque < 0 else 0, "COUNTER": counter,
      }),
    ]
    self.car.update([(now, messages)])
    cc = structs.CarControl()
    cc.enabled = cc.latActive = lat_active
    cc.actuators.curvature = float(desired)
    cc.currentCurvature = float(self.eps / self.ratio)
    cc.curvatureControllerActive = self.curvature_controller
    cc.angularVelocity = [0., 0., float(yaw_rate)]
    _, sends = self.car.apply(cc.as_reader(), structs.IQCarControl(), now)
    hca = [s for s in sends if s[0] == 0x303]
    if hca:
      self.parser.update([(now, hca)])
      self.power = self.parser.vl["HCA_03"]["Power"]
      self.sent_frames += 1
    self.command = float(self.car.CC.apply_curvature_last)
    self.commands.append(self.command)
    self.eps += (1 - math.exp(-DT / EPS_TAU)) * (EPS_GAIN * self.commands[0] - self.eps)
    self.eps_history.append(self.eps)
    return self.command


@pytest.mark.parametrize("platform", PLATFORMS)
def test_steering_power_ramps_to_full_power_without_overshoot(platform):
  rig = SteeringRig(platform)
  step = rig.car.CC.CCP.STEERING_POWER_STEP
  assert rig.car.CC.CCP.STEERING_POWER_MAX == 100
  powers = []
  for _ in range(400):
    sent_before = rig.sent_frames
    rig.step(20., 0.)
    if rig.sent_frames > sent_before:
      powers.append(rig.power)

  assert powers[-1] == 100.
  assert max(powers) == 100.
  assert all(later - earlier <= step + POWER_RESOLUTION for earlier, later in zip(powers, powers[1:], strict=False))
  assert powers.index(100.) * 2 * DT < 1.0


@pytest.mark.parametrize("platform", PLATFORMS)
def test_driver_torque_still_fades_power_to_the_minimum(platform):
  rig = SteeringRig(platform)
  for _ in range(400):
    rig.step(20., 0.)
  assert rig.power == 100.

  rig.driver_torque = rig.car.CC.CCP.STEER_DRIVER_MAX + 20
  for _ in range(400):
    rig.step(20., 0.)
  assert rig.power == rig.car.CC.CCP.STEERING_POWER_MIN


@pytest.mark.parametrize("platform", PLATFORMS)
def test_power_ramps_down_before_hca_releases(platform):
  rig = SteeringRig(platform)
  for _ in range(400):
    rig.step(20., 0.)
  powers = []
  for _ in range(100):
    sent_before = rig.sent_frames
    rig.step(20., 0., lat_active=False)
    if rig.sent_frames > sent_before:
      powers.append(rig.power)

  assert powers[0] < 100.
  assert powers[-1] == 0.
  assert all(earlier - later <= rig.car.CC.CCP.STEERING_POWER_STEP + POWER_RESOLUTION for earlier, later in zip(powers, powers[1:], strict=False))


def make_smoother(speed=10.):
  smoother = CurvatureSmoother([3., 10., 25.], [0.05, 0.0325, 0.03], SMOOTHER_DT)
  smoother.reset(0.)
  return smoother, speed


def two_stage_delay(tau):
  return 2 * SMOOTHER_DT / (math.exp(SMOOTHER_DT / tau) - 1.)


def test_smoother_step_response_is_monotonic_with_the_expected_group_delay():
  smoother, speed = make_smoother(10.)
  response = [smoother.update(1., speed) for _ in range(300)]

  assert all(later >= earlier for earlier, later in zip(response, response[1:], strict=False))
  assert max(response) <= 1.
  assert response[-1] == pytest.approx(1., abs=1e-6)
  mean_delay = sum(1. - y for y in response) * SMOOTHER_DT
  assert mean_delay == pytest.approx(two_stage_delay(0.0325), rel=0.01)


@pytest.mark.parametrize("speed,tau", [(3., 0.05), (10., 0.0325), (25., 0.03), (1., 0.05), (40., 0.03)])
def test_smoother_time_constant_follows_speed(speed, tau):
  smoother, _ = make_smoother()
  response = [smoother.update(1., speed) for _ in range(300)]
  mean_delay = sum(1. - y for y in response) * SMOOTHER_DT
  assert mean_delay == pytest.approx(two_stage_delay(tau), rel=0.01)


@pytest.mark.parametrize("platform", PLATFORMS)
def test_filter_delay_is_nearly_flat_across_road_speeds(platform):
  rig = SteeringRig(platform)
  ccp = rig.car.CC.CCP
  delays = [two_stage_delay(float(np.interp(v, ccp.CURVATURE_SMOOTH_SPEED_BP, ccp.CURVATURE_SMOOTH_TAU_V))) for v in range(10, 41)]

  assert max(delays) / min(delays) < 1.25
  assert max(delays) < 0.06
  assert two_stage_delay(ccp.CURVATURE_SMOOTH_TAU_V[0]) < 0.1


def test_smoother_passes_slow_changes_and_cuts_plan_noise():
  def gain(frequency):
    smoother, speed = make_smoother(10.)
    samples = [smoother.update(math.sin(2 * math.pi * frequency * i * 0.02), speed) for i in range(1500)]
    return max(abs(s) for s in samples[1000:])

  assert gain(0.2) > 0.97
  assert gain(5.) < 0.6
  assert gain(10.) < 0.25


def test_smoother_reset_holds_the_value():
  smoother, speed = make_smoother()
  smoother.reset(0.004)
  assert smoother.update(0.004, speed) == pytest.approx(0.004, abs=1e-12)


@pytest.mark.parametrize("platform", PLATFORMS)
def test_filter_is_active_by_default_and_converges_to_the_plan(platform):
  desired = 0.0004
  commands = {}
  for smoothing in (False, True):
    rig = SteeringRig(platform, smoothing=smoothing, curvature_controller=False)
    for _ in range(100):
      rig.step(15., 0., lat_active=False)
    for _ in range(2):
      rig.step(15., desired)
    commands[smoothing] = rig.command
    for _ in range(200):
      rig.step(15., desired)
    assert rig.command == pytest.approx(desired, rel=0.03)

  assert commands[False] == pytest.approx(desired, rel=1e-6)
  assert commands[True] < 0.6 * desired


@pytest.mark.parametrize("platform", PLATFORMS)
def test_reengaging_does_not_ramp_from_the_stale_filter_state(platform):
  rig = SteeringRig(platform, curvature_controller=False)
  for _ in range(200):
    rig.step(15., 0.003)
  for _ in range(100):
    rig.step(15., 0.003, lat_active=False)
  rig.step(15., -0.0004)
  rig.step(15., -0.0004)

  assert abs(rig.command) < 0.0015


@pytest.mark.parametrize("platform", PLATFORMS)
def test_quick_reengage_keeps_the_held_curvature(platform):
  rig = SteeringRig(platform, curvature_controller=False)
  held = 0.003
  for _ in range(300):
    rig.step(15., held)
  for _ in range(6):
    rig.step(15., held, lat_active=False)
  before = rig.command
  commands = [rig.step(15., held) for _ in range(10)]

  assert before == pytest.approx(held, rel=0.1)
  assert all(c == pytest.approx(held, rel=0.1) for c in commands)


def noisy_plan(frames, amplitude=0.0025, noise=0.0004, seed=3):
  rng = np.random.default_rng(seed)
  plan = []
  held = 0.
  for i in range(frames):
    if i % 5 == 0:
      held = rng.normal(0., noise)
    plan.append(amplitude * math.sin(2 * math.pi * 0.25 * i * DT) + held)
  return plan


def drive(platform, smoothing, ratio, v_ego=12., frames=1500):
  rig = SteeringRig(platform, smoothing=smoothing, ratio=ratio)
  plan = noisy_plan(frames)
  commands, real, reference = [], [], []
  for i in range(frames):
    commands.append(rig.step(v_ego, plan[i]))
    real.append(rig.real)
    reference.append(math.sin(2 * math.pi * 0.25 * (i * DT - 0.3)) * 0.0025)
  return np.array(commands), np.array(real), np.array(reference), v_ego


@pytest.mark.parametrize("platform", [CAR.VOLKSWAGEN_ID4_MK1, CAR.VOLKSWAGEN_GOLF_MK8])
@pytest.mark.parametrize("ratio", [0.86, 1.0, 1.25])
def test_smoothing_cuts_command_activity_without_hurting_tracking(platform, ratio):
  off_command, off_real, reference, v_ego = drive(platform, False, ratio)
  on_command, on_real, _, _ = drive(platform, True, ratio)

  def rms(values):
    return float(np.sqrt(np.sum(values ** 2) / len(values)))

  def activity(command):
    return rms(np.diff(command[200:]))

  def tracking_error(real):
    return rms((real - reference)[300:] * v_ego ** 2)

  assert activity(on_command) < 0.6 * activity(off_command)
  assert tracking_error(on_real) < tracking_error(off_real) + 0.08
  assert tracking_error(on_real) < 0.25


@pytest.mark.parametrize("ratio", [0.86, 1.0, 1.25])
def test_steady_curve_is_reached_through_the_filter(ratio):
  for v_ego in (4., 12., 28.):
    rig = SteeringRig(CAR.VOLKSWAGEN_ID4_MK1, ratio=ratio)
    desired = 1.5 / v_ego ** 2
    for _ in range(600):
      rig.step(v_ego, desired)
    assert rig.real == pytest.approx(desired, rel=0.05)
