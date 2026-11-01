"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""

import pytest

from iqdbc.car.volkswagen.carcontroller import accel_during_driver_override


@pytest.mark.parametrize("accel", [-3.5, -0.5, 0.0, 0.5, 2.0])
def test_opted_in_driver_override_sends_neutral_accel(accel):
  assert accel_during_driver_override(accel, True, True) == 0.0


@pytest.mark.parametrize("gas_pressed,keep_long_active", [(False, False), (False, True), (True, False)])
def test_other_longitudinal_paths_preserve_accel(gas_pressed, keep_long_active):
  assert accel_during_driver_override(-0.7, gas_pressed, keep_long_active) == -0.7
