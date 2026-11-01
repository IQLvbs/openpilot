"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""

import pytest

from iqpilot.selfdrive.longitudinal_settings import (
  LONGITUDINAL_MODE_CHILL,
  LONGITUDINAL_MODE_DYNAMIC,
  LONGITUDINAL_MODE_PILOT,
  LONGITUDINAL_MODE_STOCK,
  PERSONALITY_AGGRESSIVE,
  PERSONALITY_RELAXED,
  PERSONALITY_STANDARD,
  apply_longitudinal_mode,
  get_follow_distance_state,
  get_longitudinal_mode,
  get_runtime_personality,
  longitudinal_mode_needs_cycle,
  next_longitudinal_mode,
  set_valid_personality,
)


class Params:
  def __init__(self, personality=PERSONALITY_STANDARD):
    self.values = {
      "AlphaLongitudinalEnabled": True,
      "ExperimentalMode": True,
      "IQDynamicMode": True,
      "LongitudinalPersonality": personality,
    }
    self.personality_writes = []

  def get(self, key, return_default=False):
    return self.values[key]

  def get_bool(self, key):
    return bool(self.values[key])

  def put(self, key, value):
    self.values[key] = value
    if key == "LongitudinalPersonality":
      self.personality_writes.append(value)

  def put_bool(self, key, value):
    self.values[key] = bool(value)


def test_personality_writer_rejects_stock_value():
  params = Params()

  with pytest.raises(ValueError):
    set_valid_personality(params, 3)

  assert params.personality_writes == []


def test_mode_paths_never_rewrite_a_valid_personality():
  params = Params(PERSONALITY_AGGRESSIVE)

  for mode in range(4):
    apply_longitudinal_mode(params, mode)

  assert params.personality_writes == []


def test_stock_mode_preserves_personality_and_dynamic_restores_it():
  params = Params(PERSONALITY_AGGRESSIVE)

  apply_longitudinal_mode(params, LONGITUDINAL_MODE_STOCK)

  assert get_follow_distance_state(params) == (None, False)
  assert get_runtime_personality(params) == PERSONALITY_AGGRESSIVE
  assert params.values["LongitudinalPersonality"] == PERSONALITY_AGGRESSIVE
  assert params.personality_writes == []

  apply_longitudinal_mode(params, LONGITUDINAL_MODE_DYNAMIC)

  assert get_longitudinal_mode(params) == LONGITUDINAL_MODE_DYNAMIC
  assert get_follow_distance_state(params) == (PERSONALITY_AGGRESSIVE, True)


def test_stock_mode_sanitizes_legacy_stock_personality_value():
  params = Params(3)

  apply_longitudinal_mode(params, LONGITUDINAL_MODE_STOCK)

  assert get_follow_distance_state(params) == (None, False)
  assert params.values["LongitudinalPersonality"] == PERSONALITY_RELAXED
  assert params.personality_writes == [PERSONALITY_RELAXED]


def test_chill_mode_keeps_personality_selectable():
  params = Params(PERSONALITY_AGGRESSIVE)

  apply_longitudinal_mode(params, LONGITUDINAL_MODE_CHILL)

  assert get_longitudinal_mode(params) == LONGITUDINAL_MODE_CHILL
  assert get_follow_distance_state(params) == (PERSONALITY_AGGRESSIVE, True)
  assert get_runtime_personality(params) == PERSONALITY_AGGRESSIVE
  assert params.values["LongitudinalPersonality"] == PERSONALITY_AGGRESSIVE
  assert params.personality_writes == []


def test_chill_mode_sanitizes_legacy_stock_personality_value():
  params = Params(3)

  apply_longitudinal_mode(params, LONGITUDINAL_MODE_CHILL)

  assert get_runtime_personality(params) == PERSONALITY_RELAXED
  assert get_follow_distance_state(params) == (PERSONALITY_RELAXED, True)
  assert params.personality_writes == [PERSONALITY_RELAXED]


def test_personality_survives_mode_round_trip():
  params = Params(PERSONALITY_STANDARD)

  for mode in (LONGITUDINAL_MODE_CHILL, LONGITUDINAL_MODE_PILOT, LONGITUDINAL_MODE_CHILL, LONGITUDINAL_MODE_DYNAMIC, LONGITUDINAL_MODE_CHILL):
    apply_longitudinal_mode(params, mode)
    assert get_runtime_personality(params) == PERSONALITY_STANDARD

  assert params.personality_writes == []


def test_iq_modes_enable_valid_personality_selection():
  params = Params(PERSONALITY_STANDARD)

  assert get_follow_distance_state(params) == (PERSONALITY_STANDARD, True)

  params.values["IQDynamicMode"] = False

  assert get_follow_distance_state(params) == (PERSONALITY_STANDARD, True)

  params.values["ExperimentalMode"] = False

  assert get_longitudinal_mode(params) == LONGITUDINAL_MODE_CHILL
  assert get_follow_distance_state(params) == (PERSONALITY_STANDARD, True)


def test_onroad_cycles_only_between_iq_modes():
  assert next_longitudinal_mode(LONGITUDINAL_MODE_CHILL, True, True) == LONGITUDINAL_MODE_DYNAMIC
  assert next_longitudinal_mode(LONGITUDINAL_MODE_DYNAMIC, True, True) == LONGITUDINAL_MODE_PILOT
  assert next_longitudinal_mode(LONGITUDINAL_MODE_PILOT, True, True) == LONGITUDINAL_MODE_CHILL


def test_onroad_stock_acc_is_locked():
  assert next_longitudinal_mode(LONGITUDINAL_MODE_STOCK, True, True) == LONGITUDINAL_MODE_STOCK
  assert next_longitudinal_mode(LONGITUDINAL_MODE_STOCK, True, False) == LONGITUDINAL_MODE_STOCK


def test_offroad_cycles_through_stock_acc():
  assert next_longitudinal_mode(LONGITUDINAL_MODE_PILOT, False, True) == LONGITUDINAL_MODE_STOCK
  assert next_longitudinal_mode(LONGITUDINAL_MODE_STOCK, False, True) == LONGITUDINAL_MODE_CHILL


def test_offroad_without_iq_modes_only_offers_stock_acc():
  assert next_longitudinal_mode(LONGITUDINAL_MODE_STOCK, False, False) == LONGITUDINAL_MODE_STOCK
  assert next_longitudinal_mode(LONGITUDINAL_MODE_PILOT, False, False) == LONGITUDINAL_MODE_STOCK


def test_cycle_only_when_crossing_stock_boundary():
  assert longitudinal_mode_needs_cycle(LONGITUDINAL_MODE_STOCK, LONGITUDINAL_MODE_CHILL)
  assert longitudinal_mode_needs_cycle(LONGITUDINAL_MODE_PILOT, LONGITUDINAL_MODE_STOCK)
  assert not longitudinal_mode_needs_cycle(LONGITUDINAL_MODE_CHILL, LONGITUDINAL_MODE_PILOT)
  assert not longitudinal_mode_needs_cycle(LONGITUDINAL_MODE_DYNAMIC, LONGITUDINAL_MODE_CHILL)
