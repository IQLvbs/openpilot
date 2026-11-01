"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""
from iqpilot.selfdrive.iqmodeld.modeld_selector import BIG_MAX_LAG_FRAMES, hold_big_plan

PLAN = {"frame_id": 100, "msgs": {}}


def test_a_gap_in_a_live_big_model_repeats_its_last_plan():
  # an eGPU running at 13 Hz left every third tick to the small model, and two models steered on alternate frames
  assert hold_big_plan(True, PLAN, 100, 101) is PLAN
  assert hold_big_plan(True, PLAN, 100, 100 + BIG_MAX_LAG_FRAMES) is PLAN


def test_a_stalled_big_model_hands_over_to_the_small_one():
  assert hold_big_plan(True, PLAN, 100, 101 + BIG_MAX_LAG_FRAMES) is None


def test_nothing_is_held_before_the_big_model_is_active_or_after_a_reset():
  assert hold_big_plan(False, PLAN, 100, 101) is None
  assert hold_big_plan(True, None, 100, 101) is None
  assert hold_big_plan(True, PLAN, -1, 101) is None
  assert hold_big_plan(True, PLAN, 100, 99) is None
