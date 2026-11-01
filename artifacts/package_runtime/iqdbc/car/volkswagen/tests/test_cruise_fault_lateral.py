# Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
from iqdbc.car.volkswagen.carstate import CarState


def fresh_state(lateral_available_prev):
  state = CarState.__new__(CarState)
  state.cruise_faulted_frames = 0
  state.cruise_fault_clear_frames = 0
  state.cruise_fault_lateral_active = False
  state.lateral_available_prev = lateral_available_prev
  return state


def run(state, frames):
  modes = []
  for allow, candidate, lateral_available in frames:
    state.update_cruise_fault_lateral(allow, candidate)
    modes.append(state.cruise_fault_lateral_active)
    state.lateral_available_prev = lateral_available or state.cruise_fault_lateral_active
  return modes


def test_fault_from_an_available_cruise_takes_over_lateral_on_the_same_frame():
  state = fresh_state(lateral_available_prev=True)
  assert run(state, [(True, True, False)] * 3) == [True, True, True]


def test_fault_without_an_available_cruise_still_waits_for_the_debounce():
  state = fresh_state(lateral_available_prev=False)
  modes = run(state, [(True, True, False)] * 6)
  assert modes == [False, False, False, False, True, True]


def test_lateral_mode_is_held_through_the_release_debounce_then_dropped():
  state = fresh_state(lateral_available_prev=True)
  run(state, [(True, True, False)] * 3)
  modes = run(state, [(True, False, True)] * 21)
  assert modes[:19] == [True] * 19
  assert modes[19] is False


def test_lateral_mode_never_arms_when_the_toggle_is_off():
  state = fresh_state(lateral_available_prev=True)
  assert run(state, [(False, True, False)] * 10) == [False] * 10


def test_toggle_turning_off_clears_an_active_lateral_mode():
  state = fresh_state(lateral_available_prev=True)
  run(state, [(True, True, False)] * 3)
  assert run(state, [(False, True, False)]) == [False]
