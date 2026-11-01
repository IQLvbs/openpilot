# Copyright (c) 2026 IQ.Lvbs. All rights reserved.
import math

import numpy as np
import pytest

from iqpilot.cereal import car, log


@pytest.mark.parametrize("direction", [-1.0, 1.0])
@pytest.mark.parametrize("speed", [2.0, 6.0, 15.0, 30.0])
@pytest.mark.parametrize("old,current,actual,expected", [
  (0.4, 1.2, 0.3, 0.4),
  (2.0, 0.8, 1.2, 0.8),
  (1.2, -0.4, 0.7, 0.0),
  (1.2, 0.0, 0.3, 0.0),
])
def test_standard_feedback_reference(controller_factory, direction, speed, old, current, actual, expected):
  controller, vm, cp = controller_factory()
  controller.turn_guards_enabled = False
  controller.lat_accel_request_buffer.extend([direction * old] * controller.lat_accel_request_buffer_len)
  angle = math.degrees(vm.get_steer_from_curvature(-direction * actual / speed ** 2, speed, 0.0))
  state = car.CarState.new_message(vEgo=speed, steeringAngleDeg=angle)
  params = log.VehicleParameters.new_message(stiffnessFactor=1.0, steerRatio=cp.steerRatio)
  output, _, trace = controller.update(True, state, vm, params, True, direction * current / speed ** 2,
                                       None, False, 0.3)
  assert trace.actualLateralAccel == pytest.approx(direction * actual, abs=2e-6)
  assert trace.desiredLateralAccel == pytest.approx(direction * expected, abs=2e-6)
  assert trace.p == pytest.approx(controller.pid.k_p * direction * (expected - actual), abs=2e-6)
  assert np.isfinite(output)
  assert controller.pid.i == 0.0
