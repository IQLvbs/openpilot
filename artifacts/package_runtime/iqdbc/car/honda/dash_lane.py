"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos
"""
from dataclasses import dataclass, field

import numpy as np

POINT_COUNT = 40
POINTS_PER_FRAME = 4
SWEEP_INDICES = POINT_COUNT // POINTS_PER_FRAME

# the camera repeats each sweep index across four redundant banks: mux = index + bank*16,
# giving mux values 1-10, 17-26, 33-42 and 49-58 for logical indices 0-9
MUX_CYCLE = tuple(index + bank * 16 for bank in range(4) for index in range(1, SWEEP_INDICES + 1))

OFFSET_UNAVAILABLE = 2047
OFFSET_VALID_MAX = 2046

NEAR_M = 2.0
FAR_M = 100.0
LOOKAHEAD_M = np.linspace(NEAR_M, FAR_M, POINT_COUNT)

# full swing center -> max turn is slewed over this long so model jumps can't teleport the dash lane
SLEW_RATE_HZ = 50.0
SLEW_FULL_SCALE_S = 2.0
SLEW_MAX_STEP = OFFSET_VALID_MAX / (SLEW_FULL_SCALE_S * SLEW_RATE_HZ)


def _stock_gain(d):
  # raw offset units per meter of lateral, regressed from stock radar sweeps vs modelV2 lane centers
  return 29.3 + 0.243 * d - 0.00228 * d ** 2


def _legacy_gain(d):
  return 6.27 + 0.0106 * d + 0.000354 * d ** 2


GAIN = _stock_gain(LOOKAHEAD_M)


def gain_correction(d: float) -> float:
  # the HUD lead marker's lateral scale was tuned against lanes drawn with the legacy (flatter) gain
  # law, so the lead's lateral must ride this ratio to stay on the corrected lane rendering
  d = min(max(float(d), NEAR_M), FAR_M)
  return _stock_gain(d) / _legacy_gain(d)


LANE_LINE_ON = 3
LANE_LENGTH_MAX_VALUE = 33
LANE_WIDTH_DEFAULT = 32

LINE_PROB_ON = 0.25
LINE_PROB_OFF = 0.10
HALF_LANE_M = 1.65
FULL_REACH_SPEED = 27.0
FULL_REACH_LEAD_DIST = 70.0
MIN_REACH = 0.15


def encode_lane_path(x, y):
  x = np.asarray(x, dtype=float)
  y = np.asarray(y, dtype=float)
  if x.size < 2 or x.max() < FAR_M:
    return [OFFSET_UNAVAILABLE] * POINT_COUNT
  lat = np.interp(LOOKAHEAD_M, x, y)
  # stock encodes offsets with the opposite lateral sign to openpilot's +left convention
  raw = np.clip(np.round(-GAIN * lat), -OFFSET_VALID_MAX, OFFSET_VALID_MAX)
  return [int(v) for v in raw]


# The CAN FD dash has no LKAS_HUD_2 to carry the drawn length: it reads the path as a contiguous valid
# prefix ended by an in-band OFFSET_UNAVAILABLE terminator, idles at 6 valid zero offsets (never
# all-unavailable), and cross-checks the prefix length against RADAR_LEAD's LANE_PATH_LENGTH.
CANFD_MAX_VALID_PTS = 23
CANFD_MIN_VALID_PTS = 6
CANFD_IDLE_OFFSETS = [0] * CANFD_MIN_VALID_PTS + [OFFSET_UNAVAILABLE] * (POINT_COUNT - CANFD_MIN_VALID_PTS)

# stock valid-point count is a function of ego speed alone, fit from factory lanes-on RADAR_LEAD frames
CANFD_LEN_INTERCEPT = 6.74
CANFD_LEN_SLOPE = 0.862


@dataclass
class RenderedLane:
  offsets: list[int] = field(default_factory=lambda: [OFFSET_UNAVAILABLE] * POINT_COUNT)
  reach: float = 0.0
  left_line: bool = False
  right_line: bool = False
  lane_cross: int = 0
  v_ego: float = 0.0

  @property
  def blank(self) -> bool:
    return self.reach <= 0.0 or self.offsets[0] == OFFSET_UNAVAILABLE


def canfd_lane_length(lane: RenderedLane) -> int:
  if lane.blank:
    return CANFD_MIN_VALID_PTS
  n = round(CANFD_LEN_INTERCEPT + CANFD_LEN_SLOPE * lane.v_ego)
  return max(CANFD_MIN_VALID_PTS, min(CANFD_MAX_VALID_PTS, n))


def canfd_lane_offsets(lane: RenderedLane) -> list[int]:
  if lane.blank:
    return CANFD_IDLE_OFFSETS
  n_valid = canfd_lane_length(lane)
  return list(lane.offsets[:n_valid]) + [OFFSET_UNAVAILABLE] * (POINT_COUNT - n_valid)


def create_lane_path(packer, bus, offsets, mux):
  base = ((mux - 1) % 16) * POINTS_PER_FRAME
  values = {"MUX": mux}
  for i in range(POINTS_PER_FRAME):
    values[f"PATH_OFFSET_{i + 1}"] = offsets[base + i]
  return packer.make_can_msg("LANE_PATH", bus, values)


def create_lkas_hud_2(packer, bus, counter_2, reach=1.0, lane_cross=0, left_line=True, right_line=True):
  lane_length = max(0, min(LANE_LENGTH_MAX_VALUE, round(reach * LANE_LENGTH_MAX_VALUE)))
  shown = lane_length > 0
  values = {
    "COUNTER_2": counter_2,
    "SET_ME_X01": 1,
    "LANE_WIDTH": LANE_WIDTH_DEFAULT,
    "LEFT_LANE": LANE_LINE_ON if (shown and left_line) else 0,
    "RIGHT_LANE": LANE_LINE_ON if (shown and right_line) else 0,
    "LEFT_LANE_CROSSED": 1 if (shown and lane_cross < 0) else 0,
    "RIGHT_LANE_CROSSED": 1 if (shown and lane_cross > 0) else 0,
    "LANE_LENGTH": lane_length,
  }
  return packer.make_can_msg("LKAS_HUD_2", bus, values)


class LanePathRenderer:
  def __init__(self):
    self._left_on = False
    self._right_on = False
    self._shown = None

  def _lane_center(self, model):
    lls, probs = model.laneLines, model.laneLineProbs
    if len(lls) < 3 or len(probs) < 3 or len(lls[1].x) == 0:
      return None, None, False, False

    left = probs[1] >= (LINE_PROB_OFF if self._left_on else LINE_PROB_ON)
    right = probs[2] >= (LINE_PROB_OFF if self._right_on else LINE_PROB_ON)
    x = np.array(lls[1].x)
    yl, yr = np.array(lls[1].y), np.array(lls[2].y)
    if left and right:
      y = (yl + yr) / 2.0
    elif right:
      y = yr - HALF_LANE_M
    elif left:
      y = yl + HALF_LANE_M
    else:
      return None, None, False, False
    return x, y, left, right

  def _slew(self, offsets):
    # an all-sentinel fit draws nothing: pass through and reset so the next real fit shows unslewed
    if offsets[0] == OFFSET_UNAVAILABLE:
      self._shown = None
      return offsets
    target = np.asarray(offsets, dtype=float)
    if self._shown is None:
      self._shown = target
    else:
      self._shown = self._shown + np.clip(target - self._shown, -SLEW_MAX_STEP, SLEW_MAX_STEP)
    return [int(v) for v in np.round(self._shown)]

  def update(self, model, v_ego, lead_d) -> RenderedLane:
    x = y = None
    left_on = right_on = False
    if model is not None:
      x, y, left_on, right_on = self._lane_center(model)
    if x is None:
      self._shown = None
      return RenderedLane()
    self._left_on, self._right_on = left_on, right_on

    reach = float(np.clip(max(v_ego / FULL_REACH_SPEED, lead_d / FULL_REACH_LEAD_DIST, MIN_REACH), 0.0, 1.0))
    if round(reach * LANE_LENGTH_MAX_VALUE) <= 0:
      self._shown = None
      return RenderedLane()
    return RenderedLane(self._slew(encode_lane_path(x, y)), reach, left_on, right_on, v_ego=v_ego)
