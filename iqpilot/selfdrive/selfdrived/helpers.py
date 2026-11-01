import math
from collections import deque
from enum import StrEnum, auto

from iqpilot.cereal import car, messaging
from iqpilot.common.realtime import DT_CTRL
from iqpilot.selfdrive.locationd.helpers import Pose
from iqdbc.car import ACCELERATION_DUE_TO_GRAVITY
from iqdbc.car.lateral import ISO_LATERAL_ACCEL
from iqdbc.car.interfaces import ACCEL_MIN, ACCEL_MAX

# Outputs the car can't be driven without, keyed to the name shown to the driver. Once one
# stops arriving (not alive: 0.5 s at 20Hz, 0.1 s for controls), engaging is refused and an
# engaged drive takes over, naming the output. Low average rates and every other service
# stay silent soft warnings (commIssue).
CRITICAL_SERVICES = {
  'modelV2': "Driving Model",
  'longitudinalPlan': "Planner",
  'radarState': "Lead Tracking",
  'deviceMotion': "Localizer",
  'controlsState': "Controls",
}

MIN_EXCESSIVE_ACTUATION_COUNT = int(0.25 / DT_CTRL)
MIN_LATERAL_ENGAGE_BUFFER = int(1 / DT_CTRL)

STEER_SHORTFALL_MIN_SPEED = 5.0
STEER_SHORTFALL_TURN_ACCEL = 1.0
STEER_SHORTFALL_MIN_ACCEL = 0.5
STEER_SHORTFALL_RATIO = 0.25
STEER_SHORTFALL_TIME = 1.0
STEER_SHORTFALL_DELAY_BOUNDS = (0.1, 1.0)


class ExcessiveActuationType(StrEnum):
  LONGITUDINAL = auto()
  LATERAL = auto()


class ExcessiveActuationCheck:
  def __init__(self):
    self._excessive_counter = 0
    self._engaged_counter = 0

  def update(self, sm: messaging.SubMaster, CS: car.CarState, calibrated_pose: Pose) -> ExcessiveActuationType | None:
    # CS.aEgo can be noisy to bumps in the road, transitioning from standstill, losing traction, etc.
    # longitudinal
    accel_calibrated = calibrated_pose.acceleration.x
    excessive_long_actuation = sm['carControl'].longActive and ((not CS.gasPressed and accel_calibrated > ACCEL_MAX * 2) or
                                                                accel_calibrated < ACCEL_MIN * 2)

    # lateral
    yaw_rate = calibrated_pose.angular_velocity.yaw
    roll = sm['vehicleParameters'].roll
    roll_compensated_lateral_accel = (CS.vEgo * yaw_rate) - (math.sin(roll) * ACCELERATION_DUE_TO_GRAVITY)

    # Prevent false positives after overriding
    excessive_lat_actuation = False
    self._engaged_counter = self._engaged_counter + 1 if sm['carControl'].latActive and not CS.steeringPressed else 0
    if self._engaged_counter > MIN_LATERAL_ENGAGE_BUFFER:
      if abs(roll_compensated_lateral_accel) > ISO_LATERAL_ACCEL * 2:
        excessive_lat_actuation = True

    # deviceMotion acceleration can be noisy due to bad mounting or aliased deviceMotion measurements
    livepose_valid = abs(CS.aEgo - accel_calibrated) < 2
    self._excessive_counter = self._excessive_counter + 1 if livepose_valid and (excessive_long_actuation or excessive_lat_actuation) else 0

    excessive_type = None
    if self._excessive_counter > MIN_EXCESSIVE_ACTUATION_COUNT:
      if excessive_long_actuation:
        excessive_type = ExcessiveActuationType.LONGITUDINAL
      else:
        excessive_type = ExcessiveActuationType.LATERAL

    return excessive_type


class SteerShortfallCheck:
  def __init__(self):
    self._desired_curvatures: deque[float] = deque(maxlen=int(STEER_SHORTFALL_DELAY_BOUNDS[1] / DT_CTRL) + 1)
    self._shortfall_frames = 0

  def update(self, desired_curvature: float, v_ego: float, actual_lateral_accel: float, response_delay: float, eligible: bool) -> bool:
    self._desired_curvatures.append(desired_curvature)
    delay = min(max(response_delay, STEER_SHORTFALL_DELAY_BOUNDS[0]), STEER_SHORTFALL_DELAY_BOUNDS[1])
    lag_frames = min(int(round(delay / DT_CTRL)), len(self._desired_curvatures) - 1)
    desired_lateral_accel = self._desired_curvatures[-1 - lag_frames] * v_ego ** 2

    shortfall = abs(desired_lateral_accel) - actual_lateral_accel * math.copysign(1.0, desired_lateral_accel)
    falling_short = (eligible and v_ego > STEER_SHORTFALL_MIN_SPEED and
                     abs(desired_lateral_accel) > STEER_SHORTFALL_TURN_ACCEL and
                     shortfall > max(STEER_SHORTFALL_MIN_ACCEL, STEER_SHORTFALL_RATIO * abs(desired_lateral_accel)))

    self._shortfall_frames = self._shortfall_frames + 1 if falling_short else 0
    return self._shortfall_frames >= int(STEER_SHORTFALL_TIME / DT_CTRL)


def unresponsive_critical_services(sm: messaging.SubMaster) -> list[str]:
  return [s for s in CRITICAL_SERVICES if s in sm.alive and s not in sm.ignore_alive and not sm.alive[s]]
