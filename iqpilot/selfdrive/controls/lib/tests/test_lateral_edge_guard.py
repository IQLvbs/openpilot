"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from iqpilot.cereal import custom, log
import iqpilot.cereal.messaging as messaging
from iqpilot.common.realtime import DT_MDL
from iqpilot.selfdrive.controls.lib.helpers.lateral_edge_guard import (
  ADJACENT_LANE_LINE_PROB,
  BLOCK_DEBOUNCE_S,
  CLEAR_DEBOUNCE_S,
  LANE_CENTER_OFFSET_M,
  MAX_MEASURED_LANE_WIDTH_M,
  MAX_VALID_ROAD_EDGE_STD_M,
  MIN_ACTIVE_SPEED_MPS,
  MIN_MEASURED_LANE_WIDTH_M,
  REQUIRED_ROAD_EDGE_DISTANCE_M,
  UNAVAILABLE_HOLD_S,
  LateralEdgeGuard,
  RoadEdgeDataState,
  evaluate_road_edge,
)
from iqpilot.selfdrive.selfdrived.iq_events import EVENTS_IQ, ET
from iqpilot.selfdrive.selfdrived.selfdrived import SelfdriveD


@dataclass
class Edge:
  x: list[float]
  y: list[float]


@dataclass
class ModelData:
  roadEdges: list[Edge]
  roadEdgeStds: list[float]


@dataclass
class LaneModelData:
  roadEdges: list[Edge]
  roadEdgeStds: list[float]
  laneLines: list[Edge]
  laneLineProbs: list[float]


class CarState:
  def __init__(self, left_blindspot: bool = False) -> None:
    self.vEgo = MIN_ACTIVE_SPEED_MPS + 1.0
    self.leftBlinker = True
    self.rightBlinker = False
    self.leftBlindspot = left_blindspot
    self.rightBlindspot = False
    self.steeringPressed = True
    self.steeringTorque = 1.0
    self.brakePressed = False
    self.standstill = False


def edge_model(left_distance_m: float = 6.0, right_distance_m: float = 6.0,
               left_std_m: float = 0.0, right_std_m: float = 0.0) -> ModelData:
  xs = [5.0, 20.0, 40.0]
  return ModelData(
    [Edge(xs, [-left_distance_m] * len(xs)), Edge(xs, [right_distance_m] * len(xs))],
    [left_std_m, right_std_m],
  )


def lane_model(left_distance_m: float = 4.0, outer_prob: float = 0.0,
               ego_width_m: float = 3.5, ego_prob: float = 0.9) -> LaneModelData:
  xs = [5.0, 20.0, 40.0]
  base = edge_model(left_distance_m, left_distance_m)
  half = ego_width_m / 2.0
  lines = [Edge(xs, [-(half + 3.0)] * 3), Edge(xs, [-half] * 3),
           Edge(xs, [half] * 3), Edge(xs, [half + 3.0] * 3)]
  return LaneModelData(base.roadEdges, base.roadEdgeStds, lines,
                       [outer_prob, ego_prob, ego_prob, outer_prob])


def cycles(duration_s: float) -> int:
  return math.ceil(duration_s / DT_MDL)


def update_for(guard: LateralEdgeGuard, modeldata: ModelData | None, duration_s: float,
               speed_mps: float = MIN_ACTIVE_SPEED_MPS) -> None:
  for _ in range(cycles(duration_s)):
    guard.update(modeldata, speed_mps, DT_MDL)


def test_valid_geometry_blocks_and_clear_geometry_does_not_block() -> None:
  blocked = evaluate_road_edge(edge_model(4.0).roadEdges[0], 0.2, log.LaneChangeDirection.left)
  clear = evaluate_road_edge(edge_model(6.0).roadEdges[0], 0.2, log.LaneChangeDirection.left)
  assert blocked.state == RoadEdgeDataState.VALID
  assert blocked.should_block is True
  assert clear.state == RoadEdgeDataState.VALID
  assert clear.should_block is False


def test_unavailable_and_invalid_are_distinct() -> None:
  unavailable = evaluate_road_edge(Edge([5.0], []), 0.2, log.LaneChangeDirection.left)
  invalid = evaluate_road_edge(edge_model().roadEdges[0], MAX_VALID_ROAD_EDGE_STD_M + 0.01,
                               log.LaneChangeDirection.left)
  assert unavailable.state == RoadEdgeDataState.UNAVAILABLE
  assert unavailable.lateral_distance_m is None
  assert invalid.state == RoadEdgeDataState.INVALID
  assert invalid.should_block is None


def test_distance_threshold_on_either_side() -> None:
  epsilon_m = 0.001
  for direction, edge_index in ((log.LaneChangeDirection.left, 0), (log.LaneChangeDirection.right, 1)):
    below = edge_model(REQUIRED_ROAD_EDGE_DISTANCE_M - epsilon_m, REQUIRED_ROAD_EDGE_DISTANCE_M - epsilon_m)
    above = edge_model(REQUIRED_ROAD_EDGE_DISTANCE_M + epsilon_m, REQUIRED_ROAD_EDGE_DISTANCE_M + epsilon_m)
    assert evaluate_road_edge(below.roadEdges[edge_index], 0.0, direction).should_block is True
    assert evaluate_road_edge(above.roadEdges[edge_index], 0.0, direction).should_block is False


def test_disabled_guard_never_blocks() -> None:
  guard = LateralEdgeGuard(enabled=False)
  update_for(guard, edge_model(4.0), BLOCK_DEBOUNCE_S * 2)
  assert guard.block_for_direction(log.LaneChangeDirection.left) == custom.IQLateralEdgeBlock.none


def test_enabled_guard_blocks_after_debounce() -> None:
  guard = LateralEdgeGuard(enabled=True)
  update_for(guard, edge_model(4.0), BLOCK_DEBOUNCE_S)
  assert guard.block_for_direction(log.LaneChangeDirection.left) == custom.IQLateralEdgeBlock.left


def test_parameter_refresh_controls_guard() -> None:
  class EdgeGuardParams:
    enabled = True

    def get_bool(self, key: str) -> bool:
      assert key == "IQEdgeGuard"
      return self.enabled

  params = EdgeGuardParams()
  guard = LateralEdgeGuard(enabled=False)
  guard._params = params
  guard._param_refresh_frame = 0
  update_for(guard, edge_model(4.0), BLOCK_DEBOUNCE_S)
  assert guard.block_for_direction(log.LaneChangeDirection.left) == custom.IQLateralEdgeBlock.left

  params.enabled = False
  guard._param_refresh_frame = 50
  guard.update(edge_model(4.0), MIN_ACTIVE_SPEED_MPS, DT_MDL)
  assert guard.block_for_direction(log.LaneChangeDirection.left) == custom.IQLateralEdgeBlock.none


def test_clear_debounce_rejects_a_single_blocking_frame() -> None:
  guard = LateralEdgeGuard(enabled=True)
  update_for(guard, edge_model(4.0), BLOCK_DEBOUNCE_S)
  update_for(guard, edge_model(6.0), CLEAR_DEBOUNCE_S - DT_MDL)
  assert guard.block_for_direction(log.LaneChangeDirection.left) == custom.IQLateralEdgeBlock.left
  guard.update(edge_model(4.0), MIN_ACTIVE_SPEED_MPS, DT_MDL)
  update_for(guard, edge_model(6.0), CLEAR_DEBOUNCE_S)
  assert guard.block_for_direction(log.LaneChangeDirection.left) == custom.IQLateralEdgeBlock.none


def test_unavailable_holds_then_falls_back_to_not_blocking() -> None:
  guard = LateralEdgeGuard(enabled=True)
  update_for(guard, edge_model(4.0), BLOCK_DEBOUNCE_S)
  update_for(guard, None, UNAVAILABLE_HOLD_S - DT_MDL)
  assert guard.left_measurement.state == RoadEdgeDataState.UNAVAILABLE
  assert guard.block_for_direction(log.LaneChangeDirection.left) == custom.IQLateralEdgeBlock.left
  guard.update(None, MIN_ACTIVE_SPEED_MPS, DT_MDL)
  assert guard.block_for_direction(log.LaneChangeDirection.left) == custom.IQLateralEdgeBlock.none


def test_speed_gate_is_inactive_below_threshold() -> None:
  guard = LateralEdgeGuard(enabled=True)
  update_for(guard, edge_model(4.0), BLOCK_DEBOUNCE_S, MIN_ACTIVE_SPEED_MPS - 0.01)
  assert guard.block_for_direction(log.LaneChangeDirection.left) == custom.IQLateralEdgeBlock.none


def test_visible_outer_lane_line_overrides_edge_block() -> None:
  guard = LateralEdgeGuard(enabled=True)
  update_for(guard, lane_model(4.0, outer_prob=ADJACENT_LANE_LINE_PROB + 0.2), BLOCK_DEBOUNCE_S * 4)
  assert guard.block_for_direction(log.LaneChangeDirection.left) == custom.IQLateralEdgeBlock.none


def test_measured_lane_width_is_clamped_and_falls_back() -> None:
  assert LateralEdgeGuard._measured_lane_width(None) == LANE_CENTER_OFFSET_M
  assert LateralEdgeGuard._measured_lane_width(edge_model(4.0)) == LANE_CENTER_OFFSET_M
  assert LateralEdgeGuard._measured_lane_width(lane_model(4.0, ego_prob=0.1)) == LANE_CENTER_OFFSET_M
  assert LateralEdgeGuard._measured_lane_width(lane_model(4.0, ego_width_m=9.0)) == MAX_MEASURED_LANE_WIDTH_M
  assert LateralEdgeGuard._measured_lane_width(lane_model(4.0, ego_width_m=0.5)) == MIN_MEASURED_LANE_WIDTH_M
  assert LateralEdgeGuard._measured_lane_width(lane_model(4.0, ego_width_m=3.2)) == 3.2


def test_published_edge_block_maps_to_distinct_event_and_alert() -> None:
  message = messaging.new_message("iqDriveModelData")
  message.iqDriveModelData.lateralEdgeBlock = custom.IQLateralEdgeBlock.right

  class SubMaster:
    updated = {"iqDriveModelData": True}

    def __getitem__(self, service: str):
      assert service == "iqDriveModelData"
      return message.iqDriveModelData

  selfdrived = SelfdriveD.__new__(SelfdriveD)
  selfdrived.sm = SubMaster()
  selfdrived._cached_model_event_names = ()
  selfdrived._refresh_cached_model_events()

  event_name = custom.IQOnroadEvent.EventName.lateralEdgeBlocked
  assert selfdrived._cached_model_event_names == (event_name,)
  alert = EVENTS_IQ[event_name][ET.WARNING]
  assert alert.alert_text_1 == "Lane Change Blocked"
  assert alert.alert_text_2 == "Road edge detected"
