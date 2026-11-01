# Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
import pytest

from iqpilot.cereal import custom
from iqpilot.common.constants import CV

TurnDirection = custom.IQTurnSignalDirection


def test_turn_speed_gate_is_fixed_at_twenty_mph_whatever_the_param_says(monkeypatch):
  from iqpilot.selfdrive.controls.lib.helpers import lane_turn

  class FakeParams:
    def get(self, *_a, **_k):
      return "30.0"

    def get_bool(self, *_a, **_k):
      return True

  monkeypatch.setattr(lane_turn, "Params", FakeParams)
  planner = lane_turn.TurnSignalPlanner(None)
  assert planner.speed_gate == pytest.approx(20.0 * CV.MPH_TO_MS)

  planner.sample(blink_l=True, speed_mps=19.0 * CV.MPH_TO_MS)
  assert planner.output() == TurnDirection.turnLeft

  planner.sample(blink_l=True, speed_mps=21.0 * CV.MPH_TO_MS)
  assert planner.output() == TurnDirection.none
