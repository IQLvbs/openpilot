"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos
"""
from dataclasses import dataclass

from iqpilot.common.params import Params
MIN_BUTTON_STEP = 1
MAX_BUTTON_STEP = 10
SNAP_TO_GRID_THRESHOLD = 5
STOCK_HOLD_MULTIPLIER = 5

@dataclass(frozen=True)
class LongIncrementConfig:
  enabled: bool
  tap_step: int
  hold_step: int

def _clamp_step(value) -> int:
  try:
    step = int(value)
  except (TypeError, ValueError):
    return MIN_BUTTON_STEP
  return min(max(step, MIN_BUTTON_STEP), MAX_BUTTON_STEP)

def read_long_increment_config(params: Params) -> LongIncrementConfig:
  return LongIncrementConfig(enabled=params.get_bool("LongIncrementsEnabled"), tap_step=_clamp_step(params.get("LongIncrementTapStep", return_default=True)), hold_step=_clamp_step(params.get("LongIncrementHoldStep", return_default=True)))

def resolve_button_step(config: LongIncrementConfig, held: bool, unit_step: float) -> tuple[bool, float]:
  if not config.enabled:
    return held, unit_step * (STOCK_HOLD_MULTIPLIER if held else 1)

  multiplier = config.hold_step if held else config.tap_step
  snap_to_grid = multiplier >= SNAP_TO_GRID_THRESHOLD
  return snap_to_grid, unit_step * multiplier
