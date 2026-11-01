# Copyright © IQ.Lvbs, part of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
from iqdbc.car import DT_CTRL


class EpsSessionClock:
  def __init__(self, reset_gap_s: float):
    self.reset_gap_frames = round(reset_gap_s / DT_CTRL)
    self.age_frames = 0
    self.gap_frames = 0
    self.resets = 0
    self.open = False

  @property
  def age_s(self) -> float:
    return self.age_frames * DT_CTRL

  def update(self, eps_active: bool) -> float:
    if eps_active:
      self.open = True
      self.gap_frames = 0
    elif self.open:
      self.gap_frames += 1
      if self.gap_frames >= self.reset_gap_frames:
        self.open = False
        self.age_frames = 0
        self.gap_frames = 0
        self.resets += 1
    if self.open:
      self.age_frames += 1
    return self.age_s
