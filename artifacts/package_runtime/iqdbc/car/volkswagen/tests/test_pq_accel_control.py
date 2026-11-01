"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""

import pytest

from iqdbc.can import CANPacker
from iqdbc.car.volkswagen import pqcan


@pytest.mark.parametrize("accel, acc_control, expected", [
  (-1.0, 1, False),
  (0.0, 1, False),
  (0.05, 1, False),
  (0.051, 1, True),
  (1.0, 1, True),
  (1.0, 0, False),
])
def test_positive_acceleration_prevents_fuel_cutoff(accel, acc_control, expected):
  packer = CANPacker("vw_pq")
  messages = pqcan.create_acc_accel_control(
    packer, 0, 1, accel, acc_control, False, False, False, 0.2, 0.3, False,
  )
  assert len(messages) == 1
  _, data, _ = messages[0]
  assert bool(data[1] & 0x80) is expected
