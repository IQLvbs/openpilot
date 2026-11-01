#!/usr/bin/env python3
import numpy as np


class FanController:
  def update(self, cur_temp: float, ignition: bool, max_cool: bool = False) -> int:
    if max_cool:
      return 100

    fan_pwr_out = int(np.interp(cur_temp, [70.0, 85.0, 90.0], [0, 80, 100]))

    if not ignition:
      fan_pwr_out = min(fan_pwr_out, 30)

    return fan_pwr_out
