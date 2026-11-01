# Copyright © IQ.Lvbs, part of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
import pytest

from iqpilot.cereal import car, custom
from iqpilot.common.constants import CV
from iqpilot.selfdrive.car.cruise import VCruiseHelper, V_CRUISE_UNSET, V_CRUISE_MIN, V_CRUISE_MAX, IMPERIAL_INCREMENT
from iqpilot.selfdrive.car import card


@pytest.fixture
def helper():
  cp = car.CarParams(brand="volkswagen", pcmCruise=False, openpilotLongitudinalControl=True)
  helper = VCruiseHelper(cp, custom.IQCarParams(pcmCruiseSpeed=True))
  helper.set_speed_to_limit = False
  helper.iq_set_speed_mode = 0
  return helper


def state(speed_kph=50.0, button=None, pressed=False, available=True):
  cs = car.CarState(vEgo=speed_kph * CV.KPH_TO_MS, cruiseState={"available": available})
  if button is not None:
    cs.buttonEvents = [car.CarState.ButtonEvent(type=button, pressed=pressed)]
  return cs


def test_main_keeps_speed_unset_even_with_speed_limit(helper):
  helper.set_speed_to_limit = True
  helper.speed_limit_state = custom.IQPlan.SpeedLimit.AssistState.active
  helper.has_speed_limit = True
  helper.speed_limit_final_last_kph = 80.0
  for _ in range(20):
    helper.update_v_cruise(state(), False, False)
    assert helper.v_cruise_kph == helper.v_cruise_cluster_kph == V_CRUISE_UNSET


def test_card_waits_for_engagement_before_initializing(helper, monkeypatch, mocker):
  daemon = object.__new__(card.Car)
  daemon.can_sock = None
  daemon.can_rcv_cum_timeout_counter = 0
  daemon.CI = mocker.MagicMock()
  daemon.CI.update.return_value = (state(), custom.IQCarState())
  daemon.RI = mocker.MagicMock()
  daemon.sm = mocker.MagicMock()
  daemon.sm.updated = {"iqPlan": False}
  control = car.CarControl(enabled=False)
  daemon.sm.__getitem__.return_value = control
  daemon.inject_joystick_buttons = mocker.MagicMock()
  daemon.v_cruise_helper = helper
  daemon.is_metric = False
  daemon.experimental_mode = False
  daemon.iq_dynamic_mode = False
  daemon.CC_prev = car.CarControl(enabled=False)
  daemon.CS_prev = state(button="setCruise")
  monkeypatch.setattr(card.messaging, "drain_sock_raw", lambda *args, **kwargs: [])
  monkeypatch.setattr(card, "can_capnp_to_list", lambda messages: [])
  monkeypatch.setattr(card, "convert_to_capnp", lambda value: value)
  cs, _, _ = daemon.state_update()
  assert cs.vCruise == cs.vCruiseCluster == V_CRUISE_UNSET
  control.enabled = True
  cs, _, _ = daemon.state_update()
  assert cs.vCruise == cs.vCruiseCluster == 50.0


@pytest.mark.parametrize("experimental,dynamic,speed,expected", [
  (False, False, 0.0, V_CRUISE_MIN),
  (False, False, 20.0, 20.0),
  (False, False, 80.0, 80.0),
  (True, False, 20.0, round(65 * CV.MPH_TO_KPH, 1)),
  (True, False, 120.0, 120.0),
  (True, True, 20.0, 20.0),
  (False, False, 220.0, V_CRUISE_MAX),
])
def test_set_initializes_from_current_speed_and_mode(helper, experimental, dynamic, speed, expected):
  cs = state(speed, "setCruise")
  helper.update_v_cruise(cs, False, False)
  assert not helper.v_cruise_initialized
  helper.initialize_v_cruise(cs, experimental, dynamic)
  assert helper.v_cruise_kph == helper.v_cruise_cluster_kph == expected


@pytest.mark.parametrize("button,sign", [("accelCruise", 1), ("decelCruise", -1)])
@pytest.mark.parametrize("metric", [False, True])
def test_first_standby_adjustment_uses_current_speed(helper, button, sign, metric):
  helper.update_v_cruise(state(button=button, pressed=True), False, metric)
  assert not helper.v_cruise_initialized
  helper.update_v_cruise(state(button=button), False, metric)
  expected = round(50.0 + sign * (1.0 if metric else IMPERIAL_INCREMENT), 1)
  assert helper.v_cruise_kph == helper.v_cruise_cluster_kph == expected
  helper.initialize_v_cruise(state(button="setCruise"), True, False)
  assert helper.v_cruise_kph == expected


def test_main_off_clears_preset(helper):
  helper.initialize_v_cruise(state(), False, False)
  helper.update_v_cruise(state(available=False), False, False)
  helper.update_v_cruise(state(), False, False)
  assert helper.v_cruise_kph == helper.v_cruise_cluster_kph == V_CRUISE_UNSET


def test_speed_limit_does_not_replace_standby_preset(helper):
  helper.initialize_v_cruise(state(), False, False)
  helper.set_speed_to_limit = True
  helper.speed_limit_state = custom.IQPlan.SpeedLimit.AssistState.active
  helper.has_speed_limit = True
  helper.speed_limit_final_last_kph = 80.0
  helper.update_v_cruise(state(), False, False)
  assert helper.v_cruise_kph == 50.0
  helper.update_v_cruise(state(), True, False)
  assert helper.v_cruise_kph == 80.0


@pytest.mark.parametrize("use_current,expected", [(False, round(72 * CV.MPH_TO_KPH, 1)), (True, 50.0)])
def test_custom_experimental_set_speed_is_preserved(helper, use_current, expected):
  helper.iq_set_speed_mode = 1
  helper.iq_set_speed_use_current = use_current
  helper.iq_set_speed_mph = 72
  helper.initialize_v_cruise(state(), True, False)
  assert helper.v_cruise_kph == expected
