"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import IntEnum
from typing import Any

from iqpilot.cereal import custom, log
from iqpilot.common.constants import CV
from iqpilot.common.params import Params
from iqpilot.common.swaglog import cloudlog


MIN_ACTIVE_SPEED_MPS = 20.0 * CV.MPH_TO_MS
MAX_VALID_ROAD_EDGE_STD_M = 1.0
EDGE_CONFIDENCE_SIGMA = 1.0
ROAD_EDGE_LOOKAHEAD_MIN_M = 5.0
ROAD_EDGE_LOOKAHEAD_MAX_M = 40.0
LANE_CENTER_OFFSET_M = 3.5
VEHICLE_LATERAL_HALF_WIDTH_M = 1.90 / 2.0
EDGE_CLEARANCE_MARGIN_M = 0.25
ADJACENT_LANE_LINE_PROB = 0.5
EGO_LANE_LINE_PROB_MIN = 0.5
MIN_MEASURED_LANE_WIDTH_M = 2.5
MAX_MEASURED_LANE_WIDTH_M = 4.5
OUTER_LANE_LINE_INDEX = (0, 3)
EGO_LANE_LINE_INDEX = (1, 2)
REQUIRED_ROAD_EDGE_DISTANCE_M = LANE_CENTER_OFFSET_M + VEHICLE_LATERAL_HALF_WIDTH_M + EDGE_CLEARANCE_MARGIN_M
BLOCK_DEBOUNCE_S = 0.30
CLEAR_DEBOUNCE_S = 0.50
UNAVAILABLE_HOLD_S = 0.50
TIMER_EPSILON_S = 1e-9
PARAM_REFRESH_FRAMES = 50

LaneChangeDirection = log.LaneChangeDirection
LateralEdgeBlock = custom.IQLateralEdgeBlock


class RoadEdgeDataState(IntEnum):
  VALID = 0
  UNAVAILABLE = 1
  INVALID = 2


@dataclass(frozen=True, slots=True)
class RoadEdgeMeasurement:
  state: RoadEdgeDataState
  lateral_distance_m: float | None = None
  conservative_distance_m: float | None = None
  should_block: bool | None = None


@dataclass(frozen=True, slots=True)
class _SideState:
  blocked: bool = False
  block_timer_s: float = 0.0
  clear_timer_s: float = 0.0
  unavailable_timer_s: float = 0.0
  fallback_reported: bool = False


def evaluate_road_edge(edge: Any, std_m: Any, direction: int,
                       lane_width_m: float = LANE_CENTER_OFFSET_M) -> RoadEdgeMeasurement:
  if edge is None or std_m is None:
    return RoadEdgeMeasurement(RoadEdgeDataState.UNAVAILABLE)

  try:
    xs = edge.x
    ys = edge.y
    count = len(xs)
    y_count = len(ys)
  except (AttributeError, TypeError):
    return RoadEdgeMeasurement(RoadEdgeDataState.UNAVAILABLE)

  if count == 0 or y_count != count:
    return RoadEdgeMeasurement(RoadEdgeDataState.UNAVAILABLE)

  try:
    std = float(std_m)
  except (TypeError, ValueError):
    return RoadEdgeMeasurement(RoadEdgeDataState.INVALID)
  if not math.isfinite(std) or std < 0.0 or std > MAX_VALID_ROAD_EDGE_STD_M:
    return RoadEdgeMeasurement(RoadEdgeDataState.INVALID)

  lateral_distance_m: float | None = None
  for idx in range(count):
    try:
      x_m = float(xs[idx])
      y_m = float(ys[idx])
    except (IndexError, TypeError, ValueError):
      return RoadEdgeMeasurement(RoadEdgeDataState.UNAVAILABLE)
    if not math.isfinite(x_m) or not math.isfinite(y_m):
      return RoadEdgeMeasurement(RoadEdgeDataState.UNAVAILABLE)
    if not ROAD_EDGE_LOOKAHEAD_MIN_M <= x_m <= ROAD_EDGE_LOOKAHEAD_MAX_M:
      continue
    if ((direction == LaneChangeDirection.left and y_m >= 0.0) or
        (direction == LaneChangeDirection.right and y_m <= 0.0)):
      return RoadEdgeMeasurement(RoadEdgeDataState.INVALID)
    distance_m = abs(y_m)
    lateral_distance_m = distance_m if lateral_distance_m is None else min(lateral_distance_m, distance_m)

  if lateral_distance_m is None:
    return RoadEdgeMeasurement(RoadEdgeDataState.UNAVAILABLE)

  conservative_distance_m = lateral_distance_m - EDGE_CONFIDENCE_SIGMA * std
  required_distance_m = lane_width_m + VEHICLE_LATERAL_HALF_WIDTH_M + EDGE_CLEARANCE_MARGIN_M
  return RoadEdgeMeasurement(
    RoadEdgeDataState.VALID,
    lateral_distance_m,
    conservative_distance_m,
    conservative_distance_m < required_distance_m,
  )


def step_side_guard(state: _SideState, measurement: RoadEdgeMeasurement, speed_active: bool,
                    dt_s: float) -> tuple[_SideState, bool]:
  if not speed_active:
    return _SideState(), False

  if measurement.state == RoadEdgeDataState.UNAVAILABLE:
    unavailable_timer_s = state.unavailable_timer_s + dt_s
    if unavailable_timer_s < UNAVAILABLE_HOLD_S - TIMER_EPSILON_S:
      return _SideState(state.blocked, unavailable_timer_s=unavailable_timer_s,
                        fallback_reported=state.fallback_reported), False
    fallback_started = not state.fallback_reported
    return _SideState(unavailable_timer_s=unavailable_timer_s, fallback_reported=True), fallback_started

  should_block = bool(measurement.should_block) if measurement.state == RoadEdgeDataState.VALID else False
  if should_block == state.blocked:
    return _SideState(blocked=state.blocked), False

  if should_block:
    block_timer_s = state.block_timer_s + dt_s
    if block_timer_s >= BLOCK_DEBOUNCE_S - TIMER_EPSILON_S:
      return _SideState(blocked=True), False
    return _SideState(block_timer_s=block_timer_s), False

  clear_timer_s = state.clear_timer_s + dt_s
  if clear_timer_s >= CLEAR_DEBOUNCE_S - TIMER_EPSILON_S:
    return _SideState(), False
  return _SideState(blocked=True, clear_timer_s=clear_timer_s), False


class LateralEdgeGuard:
  def __init__(self, enabled: bool | None = None) -> None:
    self._params = Params() if enabled is None else None
    self._param_refresh_frame = 0
    self.enabled = self._read_enabled() if enabled is None else enabled
    self._active = False
    self._left = _SideState()
    self._right = _SideState()
    self.left_measurement = RoadEdgeMeasurement(RoadEdgeDataState.UNAVAILABLE)
    self.right_measurement = RoadEdgeMeasurement(RoadEdgeDataState.UNAVAILABLE)

  def _read_enabled(self) -> bool:
    try:
      return bool(self._params and self._params.get_bool("IQEdgeGuard"))
    except Exception:
      return False

  def _refresh_enabled(self) -> None:
    if self._params is not None and self._param_refresh_frame % PARAM_REFRESH_FRAMES == 0:
      self.enabled = self._read_enabled()
    self._param_refresh_frame += 1

  def _reset(self) -> None:
    self._left = _SideState()
    self._right = _SideState()
    self.left_measurement = RoadEdgeMeasurement(RoadEdgeDataState.UNAVAILABLE)
    self.right_measurement = RoadEdgeMeasurement(RoadEdgeDataState.UNAVAILABLE)

  @staticmethod
  def _model_side(modeldata: Any, side_index: int) -> tuple[Any | None, Any | None]:
    if modeldata is None:
      return None, None
    try:
      edges = modeldata.roadEdges
      stds = modeldata.roadEdgeStds
      if len(edges) <= side_index or len(stds) <= side_index:
        return None, None
      return edges[side_index], stds[side_index]
    except (AttributeError, TypeError):
      return None, None

  @staticmethod
  def _lane_line_prob(modeldata: Any, index: int) -> float | None:
    if modeldata is None:
      return None
    try:
      probs = modeldata.laneLineProbs
      if len(probs) <= index:
        return None
      value = float(probs[index])
    except (AttributeError, TypeError, IndexError, ValueError):
      return None
    return value if math.isfinite(value) else None

  @classmethod
  def _adjacent_lane_visible(cls, modeldata: Any, side_index: int) -> bool:
    prob = cls._lane_line_prob(modeldata, OUTER_LANE_LINE_INDEX[side_index])
    return prob is not None and prob > ADJACENT_LANE_LINE_PROB

  @classmethod
  def _measured_lane_width(cls, modeldata: Any) -> float:
    left_prob = cls._lane_line_prob(modeldata, EGO_LANE_LINE_INDEX[0])
    right_prob = cls._lane_line_prob(modeldata, EGO_LANE_LINE_INDEX[1])
    if left_prob is None or right_prob is None:
      return LANE_CENTER_OFFSET_M
    if left_prob <= EGO_LANE_LINE_PROB_MIN or right_prob <= EGO_LANE_LINE_PROB_MIN:
      return LANE_CENTER_OFFSET_M
    try:
      lines = modeldata.laneLines
      left_y = float(lines[EGO_LANE_LINE_INDEX[0]].y[0])
      right_y = float(lines[EGO_LANE_LINE_INDEX[1]].y[0])
    except (AttributeError, TypeError, IndexError, ValueError):
      return LANE_CENTER_OFFSET_M
    width = abs(right_y - left_y)
    if not math.isfinite(width):
      return LANE_CENTER_OFFSET_M
    return min(max(width, MIN_MEASURED_LANE_WIDTH_M), MAX_MEASURED_LANE_WIDTH_M)

  @staticmethod
  def _apply_lane_evidence(measurement: RoadEdgeMeasurement, lane_visible: bool) -> RoadEdgeMeasurement:
    if lane_visible and measurement.state == RoadEdgeDataState.VALID and measurement.should_block:
      return replace(measurement, should_block=False)
    return measurement

  def update(self, modeldata: Any, v_ego_mps: float, dt_s: float) -> None:
    self._refresh_enabled()
    if not self.enabled:
      if self._active:
        self._reset()
      self._active = False
      return
    if not self._active:
      self._reset()
      self._active = True

    dt = max(float(dt_s), 0.0)
    left_edge, left_std = self._model_side(modeldata, 0)
    right_edge, right_std = self._model_side(modeldata, 1)
    lane_width_m = self._measured_lane_width(modeldata)
    self.left_measurement = self._apply_lane_evidence(
      evaluate_road_edge(left_edge, left_std, LaneChangeDirection.left, lane_width_m),
      self._adjacent_lane_visible(modeldata, 0))
    self.right_measurement = self._apply_lane_evidence(
      evaluate_road_edge(right_edge, right_std, LaneChangeDirection.right, lane_width_m),
      self._adjacent_lane_visible(modeldata, 1))
    speed_active = math.isfinite(v_ego_mps) and v_ego_mps >= MIN_ACTIVE_SPEED_MPS
    self._left, left_fallback = step_side_guard(self._left, self.left_measurement, speed_active, dt)
    self._right, right_fallback = step_side_guard(self._right, self.right_measurement, speed_active, dt)
    if left_fallback:
      cloudlog.warning(f"lateral edge guard: left road edge unavailable for {UNAVAILABLE_HOLD_S:.2f} s; falling back to not blocking")
    if right_fallback:
      cloudlog.warning(f"lateral edge guard: right road edge unavailable for {UNAVAILABLE_HOLD_S:.2f} s; falling back to not blocking")

  def block_for_direction(self, direction: int) -> custom.IQLateralEdgeBlock:
    if not self.enabled:
      return LateralEdgeBlock.none
    if direction == LaneChangeDirection.left and self._left.blocked:
      return LateralEdgeBlock.left
    if direction == LaneChangeDirection.right and self._right.blocked:
      return LateralEdgeBlock.right
    return LateralEdgeBlock.none
