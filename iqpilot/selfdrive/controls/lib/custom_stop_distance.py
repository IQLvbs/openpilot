"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos
"""
import math

from iqpilot.common.params import Params
from iqpilot.common.realtime import DT_MDL
from iqpilot.selfdrive.controls.lib.stop_offset import VALIDATION_MAX_OFFSET

CUSTOM_STOP_DISTANCE_PARAM = "IQCustomStopDistance"
MIN_DISTANCE_M = 0
MAX_DISTANCE_M = int(VALIDATION_MAX_OFFSET)


class CustomStopDistance:
  def __init__(self, distance=None, params=None):
    self.params = params if params is not None else Params()
    self.frame = 0
    self.distance = 0.0
    self.requested_distance = 0.0
    self.valid = True
    self.fixed = distance is not None
    if self.fixed:
      if not math.isfinite(distance) or not MIN_DISTANCE_M <= distance <= MAX_DISTANCE_M:
        raise ValueError("unsupported stopping offset")
      self.distance = float(distance)
      self.requested_distance = self.distance
    else:
      self.read_params()

  def read_params(self) -> None:
    stored = self.params.get(CUSTOM_STOP_DISTANCE_PARAM, return_default=True)
    try:
      requested = float(stored) if stored is not None else 0.0
      self.valid = math.isfinite(requested) and MIN_DISTANCE_M <= requested <= MAX_DISTANCE_M
    except (ValueError, TypeError):
      self.valid = False
      requested = 0.0
    # Do not rewrite a stored request to fit this build's supported range.
    self.requested_distance = requested if math.isfinite(requested) else 0.0
    self.distance = requested if self.valid else 0.0

  def update(self, *, active=False) -> None:
    if not self.fixed and not active and self.frame % int(3 / DT_MDL) == 0:
      self.read_params()
    self.frame += 1

  def apply_lead(self, lead_dict: dict) -> dict:
    # Stop targets belong to planning; never move a measured obstacle.
    return lead_dict
