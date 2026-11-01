"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""

from iqpilot.cereal import log


LONGITUDINAL_MODE_STOCK = 0
LONGITUDINAL_MODE_CHILL = 1
LONGITUDINAL_MODE_DYNAMIC = 2
LONGITUDINAL_MODE_PILOT = 3
IQ_LONGITUDINAL_MODES = (LONGITUDINAL_MODE_CHILL, LONGITUDINAL_MODE_DYNAMIC, LONGITUDINAL_MODE_PILOT)

PERSONALITY_AGGRESSIVE = log.LongitudinalPersonality.schema.enumerants["aggressive"]
PERSONALITY_STANDARD = log.LongitudinalPersonality.schema.enumerants["standard"]
PERSONALITY_RELAXED = log.LongitudinalPersonality.schema.enumerants["relaxed"]
PERSONALITY_VALUES = (PERSONALITY_AGGRESSIVE, PERSONALITY_STANDARD, PERSONALITY_RELAXED)


def get_longitudinal_mode(params) -> int:
  if not params.get_bool("AlphaLongitudinalEnabled"):
    return LONGITUDINAL_MODE_STOCK
  if not params.get_bool("ExperimentalMode"):
    return LONGITUDINAL_MODE_CHILL
  return LONGITUDINAL_MODE_DYNAMIC if params.get_bool("IQDynamicMode") else LONGITUDINAL_MODE_PILOT


def get_valid_personality(params) -> int:
  personality = params.get("LongitudinalPersonality", return_default=True)
  if personality not in PERSONALITY_VALUES:
    personality = min(max(personality, PERSONALITY_AGGRESSIVE), PERSONALITY_RELAXED)
    params.put("LongitudinalPersonality", personality)
  return personality


def set_valid_personality(params, personality: int) -> None:
  if personality not in PERSONALITY_VALUES:
    raise ValueError(f"invalid longitudinal personality: {personality}")
  params.put("LongitudinalPersonality", personality)


def apply_longitudinal_mode(params, mode: int) -> None:
  if mode == LONGITUDINAL_MODE_STOCK:
    params.put_bool("AlphaLongitudinalEnabled", False)
    params.put_bool("ExperimentalMode", False)
    params.put_bool("IQDynamicMode", False)
  elif mode == LONGITUDINAL_MODE_CHILL:
    params.put_bool("AlphaLongitudinalEnabled", True)
    params.put_bool("ExperimentalMode", False)
    params.put_bool("IQDynamicMode", False)
  elif mode == LONGITUDINAL_MODE_DYNAMIC:
    params.put_bool("AlphaLongitudinalEnabled", True)
    params.put_bool("ExperimentalMode", True)
    params.put_bool("IQDynamicMode", True)
  elif mode == LONGITUDINAL_MODE_PILOT:
    params.put_bool("AlphaLongitudinalEnabled", True)
    params.put_bool("ExperimentalMode", True)
    params.put_bool("IQDynamicMode", False)
  else:
    raise ValueError(f"invalid longitudinal mode: {mode}")


def longitudinal_mode_needs_cycle(previous: int, mode: int) -> bool:
  return (previous == LONGITUDINAL_MODE_STOCK) != (mode == LONGITUDINAL_MODE_STOCK)


def next_longitudinal_mode(current: int, onroad: bool, iq_modes_available: bool) -> int:
  if onroad:
    order = list(IQ_LONGITUDINAL_MODES)
  else:
    order = [LONGITUDINAL_MODE_STOCK] + (list(IQ_LONGITUDINAL_MODES) if iq_modes_available else [])
  if current not in order:
    return current if onroad else order[0]
  return order[(order.index(current) + 1) % len(order)]


def get_follow_distance_state(params) -> tuple[int | None, bool]:
  if get_longitudinal_mode(params) == LONGITUDINAL_MODE_STOCK:
    get_valid_personality(params)
    return None, False
  return get_valid_personality(params), True


def get_runtime_personality(params) -> int:
  return get_valid_personality(params)
