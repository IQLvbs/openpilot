from types import SimpleNamespace

import numpy as np
import pytest

from iqpilot.common.filter_simple import FirstOrderFilter
from iqpilot.common.realtime import DT_CTRL
from iqpilot.selfdrive.controls.lib.longcontrol import (A_EGO_FILTER_TS, ACCEL_ERROR_KP, FILTERED_A_EGO_CARS,
                                                       SPEED_ERROR_PID_CARS, LongControl, LongCtrlState)


@pytest.mark.parametrize("current_state", [LongCtrlState.off, LongCtrlState.stopping, LongCtrlState.pid])
@pytest.mark.parametrize("active", [False, True])
@pytest.mark.parametrize("stop_requested,braking,cruise_hold,interceptor,expected_at_rest", [
  (False, False, False, False, LongCtrlState.pid),
  (False, False, False, True, LongCtrlState.pid),
  (False, False, True, False, LongCtrlState.stopping),
  (False, False, True, True, LongCtrlState.pid),
  (False, True, False, False, LongCtrlState.stopping),
  (False, True, False, True, LongCtrlState.stopping),
  (False, True, True, False, LongCtrlState.stopping),
  (False, True, True, True, LongCtrlState.stopping),
  (True, False, False, False, LongCtrlState.stopping),
  (True, False, False, True, LongCtrlState.stopping),
  (True, False, True, False, LongCtrlState.stopping),
  (True, False, True, True, LongCtrlState.stopping),
  (True, True, False, False, LongCtrlState.stopping),
  (True, True, False, True, LongCtrlState.stopping),
  (True, True, True, False, LongCtrlState.stopping),
  (True, True, True, True, LongCtrlState.stopping),
])
def test_stop_and_release_policy(current_state, active, stop_requested, braking, cruise_hold, interceptor, expected_at_rest):
  control = object.__new__(LongControl)
  control.CP_IQ = SimpleNamespace(enableGasInterceptor=interceptor)
  control.long_control_state = current_state
  control.smooth = SimpleNamespace(enabled=False)
  car_state = SimpleNamespace(brakePressed=braking, cruiseState=SimpleNamespace(standstill=cruise_hold))

  control._update_state(active, car_state, stop_requested)

  expected = expected_at_rest
  if not active:
    expected = LongCtrlState.off
  elif current_state == LongCtrlState.pid and not stop_requested:
    expected = LongCtrlState.pid
  assert control.long_control_state == expected


@pytest.mark.parametrize("current_state", [LongCtrlState.off, LongCtrlState.stopping, LongCtrlState.pid])
@pytest.mark.parametrize("settled", [False, True])
def test_smooth_stop_enters_hold_only_after_settling(current_state, settled):
  requests = []
  control = object.__new__(LongControl)
  control.CP_IQ = SimpleNamespace(enableGasInterceptor=False)
  control.long_control_state = current_state
  control.smooth = SimpleNamespace(enabled=True, want_hold=lambda *args: requests.append(args) or settled)
  car_state = SimpleNamespace(vEgo=0.4, standstill=False, brakePressed=False, cruiseState=SimpleNamespace(standstill=False))

  control._update_state(True, car_state, True)

  if current_state == LongCtrlState.stopping:
    assert requests == []
    assert control.long_control_state == LongCtrlState.stopping
  else:
    assert requests == [(True, 0.4, False)]
    assert control.long_control_state == (LongCtrlState.stopping if settled else LongCtrlState.pid)


def test_gas_override_preserves_negative_accel_command():
  pid_calls = []
  control = object.__new__(LongControl)
  control.CP = SimpleNamespace(stopAccel=-0.55)
  control.CP_IQ = SimpleNamespace(enableGasInterceptor=False)
  control.long_control_state = LongCtrlState.pid
  control.pid = SimpleNamespace(
    update=lambda error, **kwargs: pid_calls.append((error, kwargs)) or -0.5,
    reset=lambda: None,
  )
  control.last_output_accel = -0.4
  control.stop_event_id = 0
  control.stopping_decel_rate = 1.0
  control.speed_error_pid = False
  control.filter_a_ego = False
  control.a_ego_filter = FirstOrderFilter(0.0, A_EGO_FILTER_TS, DT_CTRL, initialized=False)
  control.smooth = SimpleNamespace(enabled=False, update=lambda: None, reset=lambda: None)
  car_state = SimpleNamespace(
    vEgo=15.0,
    aEgo=0.0,
    brakePressed=False,
    standstill=False,
    cruiseState=SimpleNamespace(standstill=False),
  )

  output = control.update(True, car_state, -0.5, False, (-3.5, 2.0), gas_override=True, v_target_now=15.0)

  assert output == -0.5
  assert pid_calls == [(-0.5, {"speed": 15.0, "feedforward": -0.5, "freeze_integrator": True})]


def pid_control(fingerprint):
  pid_calls = []
  control = LongControl.__new__(LongControl)
  control.CP = SimpleNamespace(stopAccel=-0.55)
  control.CP_IQ = SimpleNamespace(enableGasInterceptor=False)
  control.long_control_state = LongCtrlState.pid
  control.pid = SimpleNamespace(
    update=lambda error, **kwargs: pid_calls.append((error, kwargs)) or kwargs["feedforward"] + error,
    reset=lambda: None,
  )
  control.last_output_accel = 0.0
  control.stop_event_id = 0
  control.stopping_decel_rate = 1.0
  control.speed_error_pid = fingerprint in SPEED_ERROR_PID_CARS
  control.filter_a_ego = fingerprint in FILTERED_A_EGO_CARS
  control.a_ego_filter = FirstOrderFilter(0.0, A_EGO_FILTER_TS, DT_CTRL, initialized=False)
  control.smooth = SimpleNamespace(enabled=False, update=lambda: None, reset=lambda: None)
  return control, pid_calls


def cruising(v_ego, a_ego):
  return SimpleNamespace(vEgo=v_ego, aEgo=a_ego, brakePressed=False, standstill=False,
                         cruiseState=SimpleNamespace(standstill=False))


def noisy_cruise_commands(control, samples=500):
  rng = np.random.default_rng(0)
  return np.array([control.update(True, cruising(13.9, a_ego), 0.0, False, (-3.5, 2.0), v_target_now=13.9)
                   for a_ego in rng.normal(0.0, 0.15, samples)])


@pytest.mark.parametrize("fingerprint", ["KIA_EV6", "HYUNDAI_TUCSON_4TH_GEN"])
def test_speed_error_pid_corrects_speed_not_measured_accel(fingerprint):
  control, pid_calls = pid_control(fingerprint)

  control.update(True, cruising(13.8, 0.4), 0.05, False, (-3.5, 2.0), v_target_now=13.9)

  error, kwargs = pid_calls[0]
  assert error == pytest.approx(0.1)
  assert kwargs["feedforward"] == 0.05


@pytest.mark.parametrize("fingerprint", ["KIA_EV6", "HYUNDAI_TUCSON_4TH_GEN"])
def test_speed_error_command_ignores_accelerometer_noise_at_steady_speed(fingerprint):
  control, _ = pid_control(fingerprint)
  assert np.ptp(noisy_cruise_commands(control)) == 0.0


def test_elantra_keeps_the_accel_loop_on_smoothed_accel():
  control, pid_calls = pid_control("HYUNDAI_ELANTRA_2021")

  control.update(True, cruising(13.8, 0.4), 0.05, False, (-3.5, 2.0), v_target_now=20.0)

  assert pid_calls[0][0] == pytest.approx(0.05 - 0.4)


def test_elantra_command_attenuates_accelerometer_noise():
  raw, _ = pid_control("HYUNDAI_ELANTRA_2021")
  raw.filter_a_ego = False
  smoothed, _ = pid_control("HYUNDAI_ELANTRA_2021")

  raw_jerk = np.mean(np.abs(np.diff(noisy_cruise_commands(raw))))
  smoothed_jerk = np.mean(np.abs(np.diff(noisy_cruise_commands(smoothed))))

  assert smoothed_jerk < raw_jerk / 4


@pytest.mark.parametrize("fingerprint", ["HYUNDAI_SONATA", "KIA_K5_2021", "HYUNDAI_IONIQ_5", "HONDA_CIVIC"])
def test_other_cars_keep_the_original_accel_loop(fingerprint):
  control, pid_calls = pid_control(fingerprint)

  control.update(True, cruising(13.8, 0.4), 0.05, False, (-3.5, 2.0), v_target_now=13.9)
  control.update(True, cruising(13.8, -0.2), 0.05, False, (-3.5, 2.0), v_target_now=13.9)

  assert [call[0] for call in pid_calls] == pytest.approx([0.05 - 0.4, 0.05 + 0.2])


@pytest.mark.parametrize("fingerprint,speed_error,filtered", [
  ("KIA_EV6", True, False),
  ("HYUNDAI_TUCSON_4TH_GEN", True, False),
  ("HYUNDAI_ELANTRA_2021", False, True),
  ("HYUNDAI_SONATA", False, False),
  ("KIA_EV6_PE", False, False),
  ("HYUNDAI_IONIQ_5", False, False),
  ("HYUNDAI_ELANTRA_HEV_2021", False, False),
])
def test_interface_fingerprint_selects_the_loop(fingerprint, speed_error, filtered):
  from iqdbc.car.hyundai.interface import CarInterface

  CP = CarInterface.get_non_essential_params(fingerprint)
  CP_IQ = CarInterface.get_non_essential_params_iq(CP, fingerprint)
  control = LongControl(CP, CP_IQ)

  assert control.speed_error_pid == speed_error
  assert control.filter_a_ego == filtered


def real_control(fingerprint):
  from iqdbc.car.hyundai.interface import CarInterface

  CP = CarInterface.get_non_essential_params(fingerprint)
  CP_IQ = CarInterface.get_non_essential_params_iq(CP, fingerprint)
  return LongControl(CP, CP_IQ)


def test_elantra_corrects_a_fraction_of_the_accel_error():
  control = real_control("HYUNDAI_ELANTRA_2021")
  control.long_control_state = LongCtrlState.pid
  control.filter_a_ego = False

  output = control.update(True, cruising(13.9, -0.3), 0.2, False, (-3.5, 2.0), v_target_now=13.9)

  assert output == pytest.approx(0.2 + ACCEL_ERROR_KP["HYUNDAI_ELANTRA_2021"] * (0.2 + 0.3))


@pytest.mark.parametrize("fingerprint", ["KIA_EV6", "HYUNDAI_SONATA", "HYUNDAI_ELANTRA_HEV_2021", "HYUNDAI_IONIQ_5"])
def test_other_hyundais_keep_full_pid_gain(fingerprint):
  control = real_control(fingerprint)

  assert control.pid.k_p == 1.0
