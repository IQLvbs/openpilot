# Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
from types import SimpleNamespace

import numpy as np
import pytest

from iqpilot.common.realtime import DT_MDL
from iqpilot.selfdrive.controls.lib.longcontrol import LongControl, LongCtrlState
from iqpilot.selfdrive.controls.lib.longitudinal_planner import CRUISE_STOP_SPEED, J_CRUISE_BP, J_CRUISE_VALS, get_cruise_accel
from iqpilot.selfdrive.controls.lib.smooth_stops import SMOOTH_STOP_MAX_SPEED

ACCEL_LIMITS = (-3.5, 2.0)
PLANNER_CP = SimpleNamespace(steerRatio=15.3, wheelbase=2.64)


def make_control(monkeypatch):
  cp = SimpleNamespace(
    carFingerprint="VOLKSWAGEN_GOLF_MK7",
    stopAccel=-0.55,
    longitudinalTuning=SimpleNamespace(kpBP=[0.], kpV=[0.1], kiBP=[0.], kiV=[0.0]),
  )
  cp_iq = SimpleNamespace(stoppingDecelRateOverride=0.0, enableGasInterceptor=False)
  control = LongControl(cp, cp_iq)
  monkeypatch.setattr(control.smooth, "read_params", lambda: None)
  control.smooth.enabled = True
  return control


def car_state(v_ego):
  return SimpleNamespace(vEgo=v_ego, aEgo=0.0, brakePressed=False, standstill=False,
                         cruiseState=SimpleNamespace(standstill=False))


@pytest.mark.parametrize("e2e", [False, True])
def test_forced_decel_at_speed_ramps_without_requesting_a_stop(e2e):
  accel, should_stop = get_cruise_accel(e2e, 0.0, 30.8, 0.0, 0.0, 0.0, PLANNER_CP, DT_MDL, -0.3, True)
  assert should_stop is False
  assert accel >= -np.interp(30.8, J_CRUISE_BP, J_CRUISE_VALS) * DT_MDL - 1e-6


@pytest.mark.parametrize("e2e", [False, True])
def test_forced_decel_requests_a_stop_once_nearly_stopped(e2e):
  _, should_stop = get_cruise_accel(e2e, 0.0, CRUISE_STOP_SPEED / 2, -1.2, -1.2, 0.0, PLANNER_CP, DT_MDL, -0.3, True)
  assert should_stop is True


def test_smooth_stop_never_brakes_hard_at_highway_speed(monkeypatch):
  control = make_control(monkeypatch)
  control.long_control_state = LongCtrlState.pid
  for _ in range(10):
    accel = control.update(True, car_state(30.8), -0.39, True, ACCEL_LIMITS, lead_distance=40.0, has_lead=True,
                           v_target_now=30.8)
  assert accel > -1.0


def test_smooth_stop_still_settles_at_stopping_speed(monkeypatch):
  control = make_control(monkeypatch)
  control.long_control_state = LongCtrlState.pid
  calls = []
  monkeypatch.setattr(control.smooth, "settle", lambda *args, stop_distance: calls.append((args, stop_distance)) or -0.4)
  v_ego = SMOOTH_STOP_MAX_SPEED / 3
  accel = control.update(True, car_state(v_ego), -0.5, True, ACCEL_LIMITS, lead_distance=0.0, has_lead=False,
                         v_target_now=v_ego)
  assert calls == [((-0.5, v_ego, 0.0, False, 0.0), None)]
  assert accel == -0.4


def test_smooth_stop_is_skipped_above_its_speed_range(monkeypatch):
  control = make_control(monkeypatch)
  control.long_control_state = LongCtrlState.pid
  calls = []
  monkeypatch.setattr(control.smooth, "settle", lambda *args: calls.append(args) or -3.5)
  control.update(True, car_state(SMOOTH_STOP_MAX_SPEED + 1.0), -0.5, True, ACCEL_LIMITS, lead_distance=40.0,
                 has_lead=True, v_target_now=SMOOTH_STOP_MAX_SPEED + 1.0)
  assert calls == []
