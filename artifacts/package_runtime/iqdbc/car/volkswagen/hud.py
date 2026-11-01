# Copyright © IQ.Lvbs, part of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
from iqdbc.car import DT_CTRL


class AccHudState:
  DISPLAY_FRAMES = int(2.0 / DT_CTRL)
  LEAD_CLEAR_FRAMES = int(1.0 / DT_CTRL)

  def __init__(self):
    self.enabled_last = False
    self.distance_bars_last = None
    self.lead_armed = True
    self.lead_absent_frames = 0
    self.lead_until = 0
    self.priority_until = 0

  def update(self, frame, enabled, lead_visible, distance_bars):
    if enabled != self.enabled_last:
      self.priority_until = frame + self.DISPLAY_FRAMES
      self.lead_armed = True
      self.lead_until = 0

    if not enabled:
      self.lead_armed = True
      self.lead_until = 0
      self.lead_absent_frames = 0
    elif lead_visible:
      if self.lead_armed:
        self.lead_until = frame + self.DISPLAY_FRAMES
        self.priority_until = self.lead_until
        self.lead_armed = False
      self.lead_absent_frames = 0
    else:
      self.lead_absent_frames += 1
      if self.lead_absent_frames >= self.LEAD_CLEAR_FRAMES:
        self.lead_armed = True

    if enabled and self.distance_bars_last is not None and distance_bars != self.distance_bars_last:
      self.priority_until = frame + self.DISPLAY_FRAMES

    self.enabled_last = enabled
    self.distance_bars_last = distance_bars
    return enabled and lead_visible and frame < self.lead_until, frame < self.priority_until
