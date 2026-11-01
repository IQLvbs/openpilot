# Copyright (c) 2026 IQ.Lvbs. All rights reserved.
import functools
import math
from dataclasses import dataclass

CAR_CONTROL_TIMEOUT_NS = 60_000_000
STEER_TORQUE_SATURATION_THRESHOLD = 1e-2
PAIRING_FIELDS = ("requestedTorque", "requestedSteeringAngleDeg", "carControlMonoTime", "actuationMonoTime", "latActive")


def car_output_supports_pairing(schema=None) -> bool:
  if schema is None:
    return _loaded_car_output_supports_pairing()
  return set(PAIRING_FIELDS) <= set(schema.fieldnames)


@functools.cache
def _loaded_car_output_supports_pairing() -> bool:
  from iqpilot.cereal import car
  return car_output_supports_pairing(car.CarOutput.schema)


@dataclass(frozen=True)
class AppliedActuators:
  requested_torque: float = 0.0
  requested_steering_angle_deg: float = 0.0
  car_control_mono_time: int = 0
  actuation_mono_time: int = 0
  lat_active: bool = False

  @classmethod
  def capture(cls, command, car_control_mono_time: int, actuation_mono_time: int):
    actuators = command.actuators
    return cls(float(actuators.torque), float(actuators.steeringAngleDeg), car_control_mono_time, actuation_mono_time,
               bool(command.latActive))

  def write_to(self, output) -> None:
    output.requestedTorque = self.requested_torque
    output.requestedSteeringAngleDeg = self.requested_steering_angle_deg
    output.carControlMonoTime = self.car_control_mono_time
    output.actuationMonoTime = self.actuation_mono_time
    output.latActive = self.lat_active


@dataclass(frozen=True)
class SteeringActuatorFeedback:
  limited: bool
  # requested minus applied; None for angle control, where no clamp direction is known
  torque_difference: float | None = None
  applied_torque: float | None = None


def steering_feedback_from_output(output, *, feedback_valid: bool, output_mono_time: int, now_mono_time: int,
                                  angle_control: bool, angle_threshold: float) -> SteeringActuatorFeedback:
  request_time, apply_time = output.carControlMonoTime, output.actuationMonoTime
  if not feedback_valid or not output.latActive or not 0 < request_time <= apply_time <= output_mono_time <= now_mono_time:
    return SteeringActuatorFeedback(True)
  if now_mono_time - request_time > CAR_CONTROL_TIMEOUT_NS or now_mono_time - apply_time > CAR_CONTROL_TIMEOUT_NS:
    return SteeringActuatorFeedback(True)

  if angle_control:
    requested, applied = output.requestedSteeringAngleDeg, output.actuatorsOutput.steeringAngleDeg
    threshold = angle_threshold
  else:
    requested, applied = output.requestedTorque, output.actuatorsOutput.torque
    threshold = STEER_TORQUE_SATURATION_THRESHOLD
  if not (math.isfinite(requested) and math.isfinite(applied)):
    return SteeringActuatorFeedback(True)
  difference = requested - applied
  return SteeringActuatorFeedback(abs(difference) > threshold, None if angle_control else difference,
                                  None if angle_control else applied)


def steering_limited_from_feedback(output, **kwargs) -> bool:
  return steering_feedback_from_output(output, **kwargs).limited
