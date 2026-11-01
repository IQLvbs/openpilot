"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos
"""
import numpy as np
import iqpilot.cereal.messaging as messaging
from iqpilot.cereal import custom

TURN_NUDGE_TORQUE = 0.8
EXIT_NUDGE_TORQUE = 0.6
TURN_PULSE_FRAMES = 50
EXIT_PULSE_FRAMES = 75

IQP_NAV_TORQUE_INFLUENCE_ENABLED = False

class NavTorquePulseBrain:
  def __init__(self, lac_torque):
    self._controller = lac_torque
    self._nav_sm = messaging.SubMaster(["iqNavState"], poll="iqNavState")
    self._nav_key = ""
    self._nav_pulse_sign = 0.0
    self._nav_pulse_frames = 0

  def _lookup_nav_pulse(self):
    if not IQP_NAV_TORQUE_INFLUENCE_ENABLED:
      return "", 0.0, 0

    self._nav_sm.update(0)
    nav_state = self._nav_sm["iqNavState"]
    phase = getattr(nav_state, "maneuverPhase", custom.IQNavState.ManeuverPhase.none)
    maneuver_direction = getattr(nav_state, "maneuverDirection", custom.NavDirection.none)

    # left nudges negative, otherwise positive
    def turn(tag, direction):
      return f"turn{tag}:{direction}", -TURN_NUDGE_TORQUE if direction == 1 else TURN_NUDGE_TORQUE, TURN_PULSE_FRAMES

    def keep(tag, direction):
      return f"{tag}:{direction}", -EXIT_NUDGE_TORQUE if direction == 1 else EXIT_NUDGE_TORQUE, EXIT_PULSE_FRAMES

    if phase == custom.IQNavState.ManeuverPhase.turnActive:
      return turn("-phase", getattr(nav_state, "turnDesireDirection", 0))
    if phase == custom.IQNavState.ManeuverPhase.highwayCommit and maneuver_direction in (custom.NavDirection.left, custom.NavDirection.right):
      return keep("highway-phase", getattr(nav_state, "lanePositioningDirection", 0))
    if getattr(nav_state, "shouldSendTurnDesire", False):
      return turn("", getattr(nav_state, "turnDesireDirection", 0))
    if getattr(nav_state, "shouldSendLanePositioning", False):
      return keep("keep", getattr(nav_state, "lanePositioningDirection", 0))
    return "", 0.0, 0

  def nudge_output_torque(self, active: bool, car_state, output_torque: float) -> float:
    if not IQP_NAV_TORQUE_INFLUENCE_ENABLED:
      self._nav_pulse_frames = 0
      self._nav_key = ""
      return output_torque

    nav_key, pulse_sign, pulse_frames = self._lookup_nav_pulse()

    if not active or getattr(car_state, "steeringPressed", False):
      self._nav_pulse_frames = 0
      if not nav_key:
        self._nav_key = ""
      return output_torque

    if nav_key and nav_key != self._nav_key:
      self._nav_key = nav_key
      self._nav_pulse_sign = pulse_sign
      self._nav_pulse_frames = pulse_frames
    elif not nav_key and self._nav_pulse_frames == 0:
      self._nav_key = ""

    if self._nav_pulse_frames > 0:
      self._nav_pulse_frames -= 1
      steer_max = float(getattr(self._controller, "steer_max", 1.0))
      output_torque = float(np.clip(output_torque + self._nav_pulse_sign, -steer_max, steer_max))

    return output_torque
