"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos
"""
from iqpilot.common.steer_delay import cached_steer_delay


class InferenceStateBase:
  def __init__(self):
    self.lat_delay = cached_steer_delay()
