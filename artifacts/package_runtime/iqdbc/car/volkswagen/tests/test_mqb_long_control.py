"""Exercise the port through IQdbc's controller, schemas, and encoded CAN messages."""
import math

import numpy as np
import pytest

from iqdbc.can import CANPacker, CANParser
from iqdbc.car import structs
from iqdbc.car.volkswagen.interface import CarInterface
from iqdbc.car.volkswagen.mqbcan import MqbLongStateMachine
from iqdbc.car.volkswagen.values import CAR

LongCtrlState = structs.CarControl.Actuators.LongControlState


@pytest.fixture
def mqb_car():
  fingerprint = {bus: {} for bus in range(7)}
  cp = CarInterface.get_params(CAR.VOLKSWAGEN_GOLF_MK7, fingerprint, [], alpha_long=True, is_release=False, docs=False)
  cp_iq = CarInterface.get_params_iq(cp, CAR.VOLKSWAGEN_GOLF_MK7, fingerprint, [], alpha_long=True, is_release_iq=False, docs=False)
  car = CarInterface(cp, cp_iq)
  car.update([(0, [])])
  car.CS.acc_type = 1
  car.CS.rolling_backward = car.CS.rolling_forward = False
  car.CS.out = structs.CarState()
  car.CS.out.cruiseState.available = True
  return car


def apply_long(car, *, accel=-1.0, speed=0.0, planned_speed=0.0, pitch=0.0, active=True, gas=False, brake=False,
               comfort=False, fcw=False, lead_distance=0.0, policy=None):
  cs = car.CS
  cs.out.vEgo = speed
  cs.out.standstill = speed == 0.0
  cs.out.gasPressed, cs.out.brakePressed = gas, brake
  cc = structs.CarControl()
  cc.enabled = cc.longActive = active
  cc.cruiseControl.override = gas
  cc.actuators.accel = accel
  cc.actuators.speed = planned_speed
  cc.actuators.longControlState = LongCtrlState.stopping if accel < 0 else LongCtrlState.pid
  cc.orientationNED = [0.0, pitch, 0.0]
  cc.longComfortMode = comfort
  cc.hudControl.leadVisible = lead_distance > 0
  cc.hudControl.leadDistance = lead_distance
  if fcw:
    cc.hudControl.visualAlert = structs.CarControl.HUDControl.VisualAlert.fcw
  # Each sample represents the next longitudinal-control tick.
  car.CC.frame += -car.CC.frame % car.CC.CCP.ACC_CONTROL_STEP
  now_nanos = (car.CC.frame + 1) * 10_000_000
  iq = structs.IQCarControl()
  if policy is not None:
    iq.longitudinalComfort = policy
  output, messages = car.apply(cc.as_reader(), iq, now_nanos)
  parser = CANParser("vw_mqb", [("ACC_06", 0), ("ACC_07", 0)], car.CC.CAN.pt)
  parser.update([(now_nanos, messages)])
  assert {0x122, 0x12E} <= {address for address, _, _ in messages}
  assert cc.actuators.accel == pytest.approx(accel)
  return output, parser.vl["ACC_06"], parser.vl["ACC_07"]


@pytest.mark.parametrize("speed", [0.0, 1.0, MqbLongStateMachine.ESP_OVERRIDE_SPEED - 0.01])
@pytest.mark.parametrize("accel", [-1.0, 0.5])
def test_hold_acquisition_keeps_tsk_starting_and_requests_esp_stop(mqb_car, speed, accel):
  output, tsk, esp = apply_long(mqb_car, accel=accel, speed=speed, planned_speed=10.0)
  assert output.accel == pytest.approx(accel)
  assert tsk["ACC_Sollbeschleunigung_02"] == pytest.approx(accel)
  assert (tsk["ACC_Anfahren"], tsk["ACC_Anhalten"]) == (1, 0)
  assert (esp["ACC_Anfahren"], esp["ACC_Anhalten"], esp["ACC_Anforderung_HMS"]) == (0, 1, 1)
  assert esp["ACC_Anhalteweg"] == pytest.approx(0.3)
  assert tsk["ACC_neg_Sollbeschl_Grad_02"] == 4.0
  assert tsk["ACC_pos_Sollbeschl_Grad_02"] == 4.0


@pytest.mark.parametrize("speed", [
  # vEgo is Float32: the nearest float to 9.5 kph rounds below the threshold.
  float(np.nextafter(np.float32(MqbLongStateMachine.ESP_OVERRIDE_SPEED), np.float32(np.inf))),
  MqbLongStateMachine.ESP_OVERRIDE_SPEED + 0.01,
])
def test_override_boundary_keeps_requested_tsk_state(mqb_car, speed):
  _, tsk, esp = apply_long(mqb_car, speed=speed)
  assert (tsk["ACC_Anfahren"], tsk["ACC_Anhalten"]) == (0, 1)
  assert (esp["ACC_Anfahren"], esp["ACC_Anhalten"]) == (0, 1)


@pytest.mark.parametrize("speed", [0.0, 1.0, MqbLongStateMachine.ESP_OVERRIDE_SPEED + 0.01])
def test_hold_grant_survives_esp_stopping_signal_clearing(mqb_car, speed):
  apply_long(mqb_car, speed=speed)
  mqb_car.CS.esp_stopping = True
  for _ in range(3):
    output, tsk, esp = apply_long(mqb_car, speed=speed)
    assert mqb_car.CC.mqb_long_state.can_stop_forever
    assert output.accel == -1.0
    for msg in (tsk, esp):
      assert (msg["ACC_Anfahren"], msg["ACC_Anhalten"]) == (1, 0)
      assert msg["ACC_Sollbeschleunigung_02"] == -1.0
  mqb_car.CS.esp_stopping = False
  # Once the ESP procedure ends below the override threshold, retain the grant
  # and let TSK resume the requested stopping state.
  _, tsk, esp = apply_long(mqb_car)
  assert mqb_car.CC.mqb_long_state.can_stop_forever
  assert (tsk["ACC_Anfahren"], tsk["ACC_Anhalten"]) == (0, 1)
  assert (esp["ACC_Anfahren"], esp["ACC_Anhalten"]) == (1, 0)


@pytest.mark.parametrize("active,brake", [(False, False), (True, True)])
def test_inactive_or_driver_braking_disables_both_can_requests(mqb_car, active, brake):
  mqb_car.CS.esp_hold_confirmation = True
  output, tsk, esp = apply_long(mqb_car, active=active, brake=brake)
  assert output.accel == 0.0
  for msg in (tsk, esp):
    assert msg["ACC_Sollbeschleunigung_02"] == pytest.approx(3.01)
    assert (msg["ACC_Anfahren"], msg["ACC_Anhalten"]) == (0, 0)
  assert esp["ACC_Anforderung_HMS"] == 0
  assert tsk["ACC_StartStopp_Info"] == 0


@pytest.mark.parametrize("acc_type", [0, 2])
def test_other_acc_types_bypass_extended_standstill(mqb_car, acc_type):
  mqb_car.CS.acc_type = acc_type
  output, tsk, esp = apply_long(mqb_car, speed=0.1, pitch=math.atan(0.12))
  assert output.accel == -1.0
  assert tsk["ACC_Typ"] == acc_type
  for msg in (tsk, esp):
    assert (msg["ACC_Anfahren"], msg["ACC_Anhalten"]) == (0, 1)


@pytest.mark.parametrize("accel", [-1.0, 0.5])
def test_iq_gas_override_keeps_zero_command_on_a_hill(mqb_car, accel):
  output, tsk, esp = apply_long(mqb_car, accel=accel, gas=True, pitch=math.atan(0.12), planned_speed=2.0)
  assert output.accel == 0.0
  assert tsk["ACC_Status_ACC"] == 4
  assert tsk["ACC_Sollbeschleunigung_02"] == 0.0
  assert esp["ACC_Sollbeschleunigung_02"] == 0.0
  assert not mqb_car.CC.mqb_long_state.start_commit_active


def test_iq_actuator_speed_drives_hill_start_commitment(mqb_car):
  output, tsk, esp = apply_long(mqb_car, accel=0.4, pitch=math.atan(0.12), planned_speed=2.0)
  assert mqb_car.CC.mqb_long_state.start_commit_active
  assert output.accel == pytest.approx(1.2)
  assert output.speed == 2.0
  for msg in (tsk, esp):
    assert (msg["ACC_Anfahren"], msg["ACC_Anhalten"]) == (1, 0)
    assert msg["ACC_Sollbeschleunigung_02"] == pytest.approx(1.2)


def test_esp_hold_torque_signals_round_trip():
  packer = CANPacker("vw_mqb")
  parser = CANParser("vw_mqb", [("ESP_15", 0)], 0)
  message = packer.make_can_msg("ESP_15", 0, {"ESP_Index_Haltemoment": 1, "ESP_Haltemoment": 1230})
  parser.update([(10_000_000, [message])])
  assert parser.vl["ESP_15"]["ESP_Index_Haltemoment"] == 1
  assert parser.vl["ESP_15"]["ESP_Haltemoment"] == 1230


def comfort_fields(tsk):
  return (tsk["ACC_zul_Regelabw_unten"], tsk["ACC_zul_Regelabw_oben"],
          tsk["ACC_neg_Sollbeschl_Grad_02"], tsk["ACC_pos_Sollbeschl_Grad_02"])


@pytest.mark.parametrize("kwargs", [{}, {"fcw": True}, {"gas": True}, {"lead_distance": 40.0}])
def test_comfort_mode_off_keeps_fixed_comfort_band_and_jerk(mqb_car, kwargs):
  for _ in range(20):
    _, tsk, _ = apply_long(mqb_car, accel=0.5, speed=20.0, **kwargs)
  # The DBC encodes the band in 0.024 / 0.0625 steps, so the fixed 0.2 lands one step below.
  assert comfort_fields(tsk) == pytest.approx((0.2, 0.2, 4.0, 4.0), abs=0.0625)


def test_comfort_mode_fcw_forces_fastest_reaction(mqb_car):
  policy = structs.IQCarControl.LongitudinalComfort(valid=True, jerkUp=0.8, jerkDown=0.8, comfortEnabled=True,
                                                  accelDeviationAbove=0.18, accelDeviationBelow=0.18)
  for _ in range(20):
    apply_long(mqb_car, accel=0.5, speed=20.0, comfort=True, lead_distance=40.0, policy=policy)
  _, tsk, _ = apply_long(mqb_car, accel=-3.0, speed=20.0, comfort=True, lead_distance=40.0, fcw=True, policy=policy)
  assert comfort_fields(tsk) == pytest.approx((0.0, 0.0, 5.0, 5.0))


def test_comfort_mode_driver_gas_override_suspends_jerk_constraints(mqb_car):
  policy = structs.IQCarControl.LongitudinalComfort(valid=True, jerkUp=5.0, jerkDown=5.0)
  _, tsk, _ = apply_long(mqb_car, accel=0.5, speed=20.0, comfort=True, gas=True, policy=policy)
  assert comfort_fields(tsk) == pytest.approx((0.0, 0.0, 5.0, 5.0))
