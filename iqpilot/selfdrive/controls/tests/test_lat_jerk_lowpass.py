"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""
import numpy as np

from iqpilot.cereal import car, log
from iqdbc.car.car_helpers import interfaces
from iqdbc.car.toyota.values import CAR as TOYOTA
from iqdbc.car.vehicle_model import VehicleModel
from iqpilot.common.realtime import DT_CTRL
from iqpilot.selfdrive.controls.lib.latcontrol_torque import LatControlTorque
from iqpilot.selfdrive.car.helpers import convert_to_capnp
from iqpilot.selfdrive.car import interfaces as iqpilot_interfaces
from iqpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N

CAR_NAME = TOYOTA.TOYOTA_COROLLA_TSS2


def _brain():
  CI_cls = interfaces[CAR_NAME]
  CP = CI_cls.get_non_essential_params(CAR_NAME)
  CP_IQ = CI_cls.get_non_essential_params_iq(CP, CAR_NAME)
  CI = CI_cls(CP, CP_IQ)
  iqpilot_interfaces.apply_iq_car_config(CI)
  ctrl = LatControlTorque(CP.as_reader(), convert_to_capnp(CP_IQ).as_reader(), CI, DT_CTRL)
  return ctrl.nnff_assist, VehicleModel(CP)


def _model(rng):
  # same-sign accel ramp (so sign_locked_min yields a real jerk) whose slope jitters frame to
  # frame the way a spatial big model's path does — this is what drives jerk_ahead to swing.
  n = max(CONTROL_N, 33)
  slope = abs(0.5 + rng.normal(0, 0.25))
  m = log.ModelDataV2.new_message()
  m.acceleration.y = (slope * np.arange(n) * 0.1).tolist()
  m.orientation.x = [0.0] * n
  m.orientation.y = [0.0] * n
  return m


def _cs():
  cs = car.CarState.new_message()
  cs.vEgo = 25.0
  cs.steeringRateDeg = 0.0
  return cs


def _run(lp_on):
  brain, VM = _brain()
  rng = np.random.default_rng(7)
  cs = _cs()
  out = []
  for _ in range(400):
    brain.update_model_v2(_model(rng))
    if not lp_on:
      brain._jerk_lp.update = lambda x: x  # bypass low-pass == pre-fix behavior
    brain.update_calculations(cs, VM, 0.0)
    out.append(brain.jerk_ahead)
  return np.array(out)


def test_lowpass_cuts_jerk_command_swing():
  old = _run(lp_on=False)
  new = _run(lp_on=True)
  # the path must actually exercise the jerk feed-forward (guard against a vacuous test)
  assert np.abs(np.diff(old)).mean() > 0.02, "input did not exercise jerk_ahead"
  old_swing = np.abs(np.diff(old)).mean()
  new_swing = np.abs(np.diff(new)).mean()
  # low-pass must cut the frame-to-frame jerk swing (the wheel oscillation) by a large margin
  assert new_swing < 0.3 * old_swing, (old_swing, new_swing)


def test_gain_zero_matches_stock():
  brain, VM = _brain()
  brain._jerk_param_ok = False
  brain._jerk_gain = 0.0
  rng = np.random.default_rng(1)
  cs = _cs()
  for _ in range(60):
    brain.update_model_v2(_model(rng))
    brain.update_calculations(cs, VM, 0.0)
  assert brain.jerk_ahead == 0.0  # no model-jerk term == sunny/stock feedforward
