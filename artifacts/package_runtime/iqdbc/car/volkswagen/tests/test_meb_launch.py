"""Copyright (c) IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved."""

from types import SimpleNamespace

import pytest

from iqdbc.can import CANPacker, CANParser
from iqdbc.car import structs
from iqdbc.car.volkswagen import mebcan
from iqdbc.car.volkswagen.values import VolkswagenFlags


State = structs.CarControl.Actuators.LongControlState


@pytest.mark.parametrize("active,state,accel,hold,speed,override,expected", [
  (True, State.pid, -0.435, False, 0.248, False, False),
  (True, State.pid, -0.3, True, 0.0, False, False),
  (True, State.pid, 0.0, True, 0.0, False, False),
  (True, State.pid, 0.3, True, 0.0, False, True),
  (True, State.pid, 0.3, False, 0.2, False, True),
  (True, State.pid, 0.3, False, 0.25, False, False),
  (True, State.stopping, 0.3, True, 0.0, False, False),
  (True, State.off, 0.3, True, 0.0, False, False),
  (False, State.pid, 0.3, True, 0.0, False, False),
  (True, State.pid, 0.3, True, 0.0, True, False),
])
def test_launch_requires_active_propulsion_intent(active, state, accel, hold, speed, override, expected):
  assert mebcan.acc_starting(active, state, accel, hold, speed, override) == expected


@pytest.mark.parametrize("dbc,flags", [
  ("vw_meb", VolkswagenFlags.MEB),
  ("vw_meb_2024", VolkswagenFlags.MEB | VolkswagenFlags.MEB_GEN2),
  ("vw_mqbevo", VolkswagenFlags.MQB_EVO),
])
def test_stop_hold_launch_can_sequence(dbc, flags):
  packer = CANPacker(dbc)
  parser = CANParser(dbc, [("ACC_18", 50)], 0)
  previous, ramp = 0, 0
  samples = [(State.pid, -0.435, False, 0.248, 0)] * 47 + [
    (State.stopping, -0.395, False, 0.048, 1),
    (State.stopping, -0.55, True, 0.0, 1),
    (State.pid, 0.3, True, 0.0, 4),
    (State.pid, 0.3, False, 0.3, 5),
    (State.pid, 0.3, False, 1.0, 5),
    (State.pid, 0.3, False, 1.5, 0),
  ]
  for frame, (state, accel, hold, speed, expected_hms) in enumerate(samples):
    starting = mebcan.acc_starting(True, state, accel, hold, speed, False)
    stopping = state == State.stopping
    hms, ramp = mebcan.acc_hold_type(True, False, True, starting, stopping, hold, speed, previous, ramp)
    previous = hms
    messages = mebcan.create_acc_accel_control(
      packer, 0, SimpleNamespace(flags=flags), 1, True, 4.0, 4.0, 0.0, 0.0,
      accel, mebcan.ACC_CTRL_ACTIVE, hms, stopping, starting, hold, speed * 3.6, False, False,
    )
    parser.update([frame * 20_000_000, messages])
    values = parser.vl["ACC_18"]
    assert values["ACC_Anforderung_HMS"] == expected_hms
    assert values["ACC_Anfahren"] == (expected_hms == 4)
    assert values["ACC_Anhalten"] == (stopping and not hold)
    assert values["ACC_Sollbeschleunigung_02"] == pytest.approx(3.01 if stopping and hold else accel, abs=0.005)
    assert values["ACC_neg_Sollbeschl_Grad_02"] == (0 if stopping and hold else 4)
