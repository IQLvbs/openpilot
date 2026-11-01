# Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
import math
from types import SimpleNamespace

import numpy as np
import pytest

from iqdbc.car.lateral import CurvatureSteeringLimits, apply_std_curvature_limits
from iqdbc.car.volkswagen.interface import CarInterface
from iqdbc.car.volkswagen.mebutils import CurvatureSmoother, LatControlCurvature
from iqdbc.car.volkswagen.values import CAR, CarControllerParams

DT = 0.01
STEER_STEP = 2
EPS_TAU = 0.25
EPS_DELAY = 0.01
YAW_DELAY = 0.06
POSE_DELAY = 0.03
POSE_PERIOD_STEPS = 5
HIGHWAY_SPEED = 29.0
HIGHWAY_EPS_SCALE = 1.37
TEST_FREQUENCIES = np.arange(0.3, 1.55, 0.1)
PLAN_AMPLITUDE = 2e-5
PEAK_POWER_LIMIT = 1.7
CROSSWIND_CURVATURE = 0.3e-3
CROSSWIND_OFFSET_LIMIT = 1.25


def platform_params(platform):
  fingerprint = {bus: {} for bus in range(7)}
  cp = CarInterface.get_params(platform, fingerprint, [], alpha_long=False, is_release=False, docs=False)
  return CarControllerParams(cp)


def eps_scale(speed):
  return float(np.interp(speed, [10.5, 15.5, 21.5, 29.0], [0.96, 1.04, 1.15, HIGHWAY_EPS_SCALE]))


def run_loop(ccp, plan, speed, disturbance=None):
  controller = LatControlCurvature(ccp.CURVATURE_PID, ccp.CURVATURE_LIMITS.CURVATURE_MAX, 1 / (DT * STEER_STEP))
  smoother = CurvatureSmoother(ccp.CURVATURE_SMOOTH_SPEED_BP, ccp.CURVATURE_SMOOTH_TAU_V, DT * STEER_STEP)
  limits = CurvatureSteeringLimits(ccp.CURVATURE_LIMITS.CURVATURE_MAX)
  eps_queue = [0.0] * (int(round(EPS_DELAY / DT)) + 1)
  yaw_queue = [0.0] * (int(round(YAW_DELAY / DT)) + 1)
  pose_queue = [0.0] * (int(round(POSE_DELAY / DT)) + 1)
  eps_curvature = 0.0
  applied = 0.0
  measured = 0.0
  vehicle = np.zeros(len(plan))
  for i, desired in enumerate(plan):
    if i % STEER_STEP == 0:
      if i % POSE_PERIOD_STEPS == 0:
        measured = pose_queue[0]
      cs = SimpleNamespace(vEgo=speed, steeringPressed=False, steeringCurvature=eps_curvature)
      cc = SimpleNamespace(currentCurvature=measured, rollCompensation=0.0, angularVelocity=[0.0, 0.0, measured * speed], steerLimited=False)
      command = controller.update(cs, cc, desired) + (eps_curvature - measured)
      command = smoother.update(command, speed)
      applied = apply_std_curvature_limits(command, applied, speed, eps_curvature, False, STEER_STEP, True, limits)
    eps_queue.append(applied)
    eps_curvature += DT * (eps_queue.pop(0) - eps_curvature) / EPS_TAU
    yaw_queue.append(eps_curvature)
    true_curvature = yaw_queue.pop(0) / eps_scale(speed)
    if disturbance is not None:
      true_curvature += disturbance[i]
    pose_queue.append(true_curvature)
    pose_queue.pop(0)
    vehicle[i] = true_curvature
  return vehicle


def closed_loop_peak_power(ccp):
  seconds = 60.0
  t = np.arange(0, seconds, DT)
  phases = np.random.default_rng(7).uniform(0, 2 * math.pi, len(TEST_FREQUENCIES))
  plan = sum(PLAN_AMPLITUDE * np.sin(2 * math.pi * f * t + p) for f, p in zip(TEST_FREQUENCIES, phases))
  vehicle = run_loop(ccp, plan, HIGHWAY_SPEED)
  settled = slice(int(10 / DT), None)
  window = t[settled] - t[settled][0]
  powers = []
  for f in TEST_FREQUENCIES:
    basis = np.exp(-2j * math.pi * f * window)
    ratio = abs(np.sum(vehicle[settled] * basis)) / abs(np.sum(plan[settled] * basis))
    powers.append(ratio ** 2)
  return max(powers)


def crosswind_peak_offset(ccp):
  steps = int(30 / DT)
  t = np.arange(steps) * DT
  disturbance = np.where(t > 5, CROSSWIND_CURVATURE, 0.0)
  vehicle = run_loop(ccp, np.zeros(steps), HIGHWAY_SPEED, disturbance)
  offset = np.cumsum(np.cumsum(vehicle * HIGHWAY_SPEED ** 2)) * DT * DT
  return float(np.abs(offset).max())


@pytest.mark.parametrize("platform", [CAR.VOLKSWAGEN_ID4_MK1, CAR.VOLKSWAGEN_ID4_MK2, CAR.VOLKSWAGEN_GOLF_MK8])
def test_highway_loop_does_not_amplify_plan_wiggle(platform):
  assert closed_loop_peak_power(platform_params(platform)) <= PEAK_POWER_LIMIT


@pytest.mark.parametrize("platform", [CAR.VOLKSWAGEN_ID4_MK1, CAR.VOLKSWAGEN_ID4_MK2, CAR.VOLKSWAGEN_GOLF_MK8])
def test_highway_loop_still_rejects_a_crosswind_step(platform):
  assert crosswind_peak_offset(platform_params(platform)) <= CROSSWIND_OFFSET_LIMIT
