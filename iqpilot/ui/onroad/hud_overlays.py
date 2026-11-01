"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos
"""
import math
import time

import numpy as np
import pyray as rl

from iqpilot.cereal import custom
from iqpilot.common.constants import CV
from iqpilot.common.filter_simple import FirstOrderFilter
from iqpilot.selfdrive.ui.ui_state import ui_state
from iqpilot.selfdrive.ui.mici.onroad.alert_renderer import IconSide, TURN_SIGNAL_BLINK_PERIOD
from iqpilot.system.ui.lib.application import gui_app, FontWeight
from iqpilot.system.ui.lib.multilang import tr
from iqpilot.system.ui.widgets import Widget
from iqpilot.selfdrive.ui.mici.onroad.torque_bar import TorqueBar
from iqpilot.system.ui.iqwidgets.lib import canvas
from iqdbc.car.volkswagen.values import VolkswagenFlags
from iqpilot.ui.onroad.theme import INACTIVE, tile, SECONDARY_TEXT, TEXT, header_tiles

def _feed():
  return ui_state.sm


def _speed_scale() -> float:
  return CV.MS_TO_KPH if ui_state.is_metric else CV.MS_TO_MPH


_BS_INSET = 20
_BS_DROP = 100
_BS_MIN = 0.01


class _BlindSide:
  def __init__(self, name: str):
    self.texture = gui_app.texture(f'icons_mici/onroad/blind_spot_{name}.png', 108, 128)
    self.glow = FirstOrderFilter(0, 0.15, 1 / gui_app.target_fps)
    self.on_left = name == "left"

  def feed(self, present: bool):
    self.glow.update(1.0 if present else 0.0)

  def lit(self) -> bool:
    return self.glow.x > _BS_MIN

  def place(self, rect):
    tex = self.texture
    x = rect.x + _BS_INSET if self.on_left else rect.x + rect.width - _BS_INSET - tex.width
    canvas.stamp(tex, x, rect.y + _BS_DROP, canvas.shade(255, 255, 255, int(255 * self.glow.x)))


class IQBlindSpotOverlay:
  def __init__(self):
    self._left = _BlindSide("left")
    self._right = _BlindSide("right")

  def update(self) -> None:
    cs = _feed()['carState']
    self._left.feed(cs.leftBlindspot)
    self._right.feed(cs.rightBlindspot)

  @property
  def detected(self) -> bool:
    return self._left.lit() or self._right.lit()

  def render(self, rect) -> None:
    if not ui_state.blindspot:
      return
    for side in (self._left, self._right):
      if side.lit():
        side.place(rect)

_MQB_CLUSTER_EXEMPT = (VolkswagenFlags.PQ | VolkswagenFlags.MLB | VolkswagenFlags.MEB | VolkswagenFlags.MEB_GEN2 | VolkswagenFlags.MQB_EVO)

class IQSpeedOverlay:
  def __init__(self):
    self.speed: float = 0.0
    self._cluster_ever_live: bool = False
    self._heavy = gui_app.font(FontWeight.BOLD)
    self._mid = gui_app.font(FontWeight.MEDIUM)

  def _source_speed(self, cs) -> float:
    cp = ui_state.CP
    if cp is not None and cp.brand == "volkswagen" and not (cp.flags & _MQB_CLUSTER_EXEMPT):
      return cs.vEgoCluster
    self._cluster_ever_live = self._cluster_ever_live or cs.vEgoCluster != 0.0
    return cs.vEgoCluster if self._cluster_ever_live else cs.vEgo

  def update(self) -> None:
    self.speed = max(0.0, self._source_speed(_feed()['carState']) * _speed_scale())

  def render(self, rect) -> None:
    _, speed_rect, _ = header_tiles(rect)

    speed = str(round(self.speed))
    font_size = 174 if len(speed) <= 2 else 152
    text_width = canvas.span(self._heavy, speed, font_size).x
    font_size = min(font_size, int(font_size * (speed_rect.width - 20) / max(1, text_width)))
    extent = canvas.span(self._heavy, speed, font_size)
    canvas.glyphs(self._heavy, speed,
                  canvas.Pt(speed_rect.x + (speed_rect.width - extent.x) / 2, speed_rect.y - 4),
                  font_size, TEXT)
    unit = tr("km/h") if ui_state.is_metric else tr("mph")
    unit_extent = canvas.span(self._mid, unit, 34)
    canvas.glyphs(self._mid, unit,
                  canvas.Pt(speed_rect.x + (speed_rect.width - unit_extent.x) / 2, speed_rect.y + 172),
                  34, SECONDARY_TEXT)

def clip_to_width(font, words: str, size: int, limit: float) -> str:
  if not words or limit <= 0:
    return ""
  if canvas.span(font, words, size).x <= limit:
    return words
  ellipsis = "..."
  if canvas.span(font, ellipsis, size).x > limit:
    return ""
  low, high = 0, len(words)
  while low < high:
    middle = (low + high + 1) // 2
    if canvas.span(font, words[:middle] + ellipsis, size).x <= limit:
      low = middle
    else:
      high = middle - 1
  return words[:low].rstrip() + ellipsis


class RoadNameBanner(Widget):
  TYPE_SIZE = 46
  MARGIN = 40
  GAP = 10
  TORQUE_SCALE = 3.0

  def __init__(self):
    super().__init__()
    self.road_name = ""
    self._face = gui_app.font(FontWeight.BOLD)

  def update(self):
    sm = _feed()
    if sm.recv_frame["carState"] < ui_state.started_frame:
      return
    if sm.updated["iqLiveData"]:
      self.road_name = " / ".join(dict.fromkeys(part.strip() for part in sm["iqLiveData"].roadName.split(";") if part.strip()))

  def _render(self, rect):
    if not self.road_name or not ui_state.road_name_toggle:
      return
    label = clip_to_width(self._face, self.road_name, self.TYPE_SIZE, rect.width - self.MARGIN)
    if not label:
      return
    extent = canvas.span(self._face, label, self.TYPE_SIZE)
    height = extent.y + 26
    top = min(TorqueBar.resting_bottom(rect, self.TORQUE_SCALE) + self.GAP,
              rect.y + rect.height - height)
    banner = rl.Rectangle(rect.x + (rect.width - extent.x) / 2 - 26, top, extent.x + 52, height)
    tile(banner, INACTIVE)
    canvas.glyphs(self._face, label, canvas.Pt(banner.x + 26, banner.y + 11), self.TYPE_SIZE, TEXT)


class RoadNameRenderer(RoadNameBanner):
  def _render(self, rect):
    if not self.road_name or not ui_state.road_name_toggle:
      return
    width = min(860, rect.width - 1000)
    if width <= 0:
      return
    label = clip_to_width(self._face, self.road_name, 32, width)
    if not label:
      return
    extent = canvas.span(self._face, label, 32)
    x = rect.x + (rect.width - extent.x) / 2
    y = rect.y + rect.height - 44
    canvas.glyphs(self._face, label, canvas.Pt(x + 1, y + 2), 32, rl.BLACK)
    canvas.glyphs(self._face, label, canvas.Pt(x, y), 32, TEXT)


ellipsize = clip_to_width

from dataclasses import dataclass, field

_ARROW = 'signal'
_WARN = 'blind_spot'


@dataclass(frozen=True)
class TurnSignalConfig:
  left_x: int = 240
  left_y: int = 110
  right_x: int = 240
  right_y: int = 110
  size: int = 150

class _IndicatorLamp(Widget):
  def __init__(self, direction: IconSide):
    super().__init__()
    self.mode: str | None = None
    self._epoch = 0.0
    self._art = {
      _ARROW: gui_app.texture(f'icons_mici/onroad/turn_signal_{direction}.png', 120, 109),
      _WARN: gui_app.texture(f'icons_mici/onroad/blind_spot_{direction}.png', 120, 109),
    }

  def set_mode(self, mode: str | None):
    if mode != self.mode:
      self._epoch = time.monotonic()
    self.mode = mode

  def _pulse(self) -> int:
    phase = (time.monotonic() - self._epoch) / TURN_SIGNAL_BLINK_PERIOD
    return round(255 * (0.5 + 0.5 * math.cos(math.tau * phase)))

  def _render(self, _):
    if self.mode is None:
      return
    alpha = self._pulse() if self.mode == _ARROW else 255
    tex = self._art[self.mode]
    canvas.stamp(tex, self._rect.x + (self._rect.width - tex.width) / 2,
                 self._rect.y + (self._rect.height - tex.height) / 2, canvas.shade(255, 255, 255, alpha))


def _lamp_modes(event_name: str, cs, remembered):
  if event_name == 'preLaneChangeLeft':
    return _ARROW, None, IconSide.left
  if event_name == 'preLaneChangeRight':
    return None, _ARROW, IconSide.right
  if event_name == 'laneChange':
    if remembered == IconSide.left:
      return _ARROW, None, remembered
    if remembered == IconSide.right:
      return None, _ARROW, remembered
    return None, None, remembered
  if event_name == 'laneChangeBlocked':
    side = IconSide.left if cs.leftBlinker else IconSide.right if cs.rightBlinker else remembered
    if side == IconSide.left:
      return _WARN, None, remembered
    if side == IconSide.right:
      return None, _WARN, remembered
    return None, None, remembered
  left = _WARN if cs.leftBlindspot else _ARROW if cs.leftBlinker else None
  right = _WARN if cs.rightBlindspot else _ARROW if cs.rightBlinker else None
  return left, right, None


class IQTurnSignalOverlay:
  def __init__(self, config: TurnSignalConfig | None = None):
    self._config = config or TurnSignalConfig()
    self._lamps = {IconSide.left: _IndicatorLamp(IconSide.left),
                   IconSide.right: _IndicatorLamp(IconSide.right)}
    self._remembered: IconSide | None = None

  def update(self):
    sm = _feed()
    alert = sm['selfdriveState'].alertType
    event_name = alert.split('/')[0] if alert else ''
    left, right, self._remembered = _lamp_modes(event_name, sm['carState'], self._remembered)
    self._lamps[IconSide.left].set_mode(left)
    self._lamps[IconSide.right].set_mode(right)

  def render(self, rect):
    if not ui_state.turn_signals:
      return
    c = self._config
    mid_x = rect.x + rect.width / 2
    spots = {
      IconSide.left: canvas.Box(mid_x - c.left_x - c.size, rect.y + c.left_y, c.size, c.size),
      IconSide.right: canvas.Box(mid_x + c.right_x, rect.y + c.right_y, c.size, c.size),
    }
    for side, lamp in self._lamps.items():
      if lamp.mode is not None:
        lamp.render(spots[side])

  @property
  def config(self) -> TurnSignalConfig:
    return self._config

  @config.setter
  def config(self, new_config: TurnSignalConfig):
    self._config = new_config

class ChevronOptions:
  OFF = 0
  DISTANCE_ONLY = 1
  SPEED_ONLY = 2
  TTC_ONLY = 3
  ALL = 4

_CH_FONT = 40
_CH_LINE = 50
_CH_MARGIN = 20
_CH_FADE_DOWN = 0.05
_CH_FADE_UP = 0.1
_CH_DEDUP = 3.0

def _gap_label(d_rel: float, _v_abs: float) -> str:
  val = max(0.0, d_rel)
  return f"{val:.0f} m" if ui_state.is_metric else f"{val * 3.28084:.0f} ft"

def _pace_label(_d_rel: float, v_abs: float) -> str:
  unit = "km/h" if ui_state.is_metric else "mph"
  return f"{max(0.0, v_abs * _speed_scale()):.0f} {unit}"

def _ttc_label(d_rel: float, _v_abs: float, v_ego: float) -> str:
  ttc = (d_rel / v_ego) if (d_rel > 0 and v_ego > 0) else 0.0
  return f"{ttc:.1f} s" if 0 < ttc < 200 else "---"

_CH_METRICS = (
  ((ChevronOptions.DISTANCE_ONLY, ChevronOptions.ALL), lambda d, va, ve: _gap_label(d, va)),
  ((ChevronOptions.SPEED_ONLY, ChevronOptions.ALL), lambda d, va, ve: _pace_label(d, va)),
  ((ChevronOptions.TTC_ONLY, ChevronOptions.ALL), _ttc_label),
)

class ChevronMetrics:
  def __init__(self):
    self._alpha: float = 0.0
    self._font = gui_app.font(FontWeight.SEMI_BOLD)

  def update_alpha(self, has_lead: bool):
    self._alpha = float(np.clip(self._alpha + (_CH_FADE_UP if has_lead else -_CH_FADE_DOWN), 0.0, 1.0))

  def should_render(self) -> bool:
    return ui_state.chevron_metrics != ChevronOptions.OFF and self._alpha > 0.0

  @staticmethod
  def _marker_size(d_rel: float) -> float:
    return float(np.clip((25 * 30) / (d_rel / 3 + 30), 15.0, 30.0)) * 2.35

  def _labels(self, d_rel: float, v_rel: float, v_ego: float) -> list[str]:
    mode = ui_state.chevron_metrics
    return [fn(d_rel, v_rel + v_ego, v_ego) for modes, fn in _CH_METRICS if mode in modes]

  def _top_y(self, anchor_y: float, size: float, n: int, rect) -> float:
    y = anchor_y + size + 15
    block = n * _CH_LINE
    floor = rect.y + rect.height - _CH_MARGIN
    if y + block > floor:
      y = max(rect.y + _CH_MARGIN, min(anchor_y, floor) - 15 - block)
    return y

  def _stack(self, lines, cx: float, top: float, rect):
    a = self._alpha
    fg = canvas.shade(255, 255, 255, int(255 * a))
    shadow = canvas.shade(0, 0, 0, int(200 * a))
    floor = rect.y + rect.height - _CH_MARGIN
    for i, line in enumerate(lines):
      y = int(top + i * _CH_LINE)
      if y + _CH_LINE > floor:
        break
      w = canvas.span(self._font, line, _CH_FONT).x
      x = int(np.clip(cx - w / 2, rect.x + _CH_MARGIN, rect.x + rect.width - w - _CH_MARGIN))
      canvas.glyphs(self._font, line, canvas.Pt(x + 2, y + 2), _CH_FONT, shadow)
      canvas.glyphs(self._font, line, canvas.Pt(x, y), _CH_FONT, fg)

  def _one_lead(self, lead, marker, v_ego: float, rect):
    if not self.should_render() or marker.center is None:
      return
    lines = self._labels(lead.dRel, lead.vRel, v_ego)
    if not lines:
      return
    if gui_app.big_ui():
      text = "   |   ".join(lines)
      width = canvas.span(self._font, text, 34).x
      x = max(rect.x + _CH_MARGIN, min(marker.center[0] - width / 2, rect.x + rect.width - width - _CH_MARGIN))
      floor = rect.y + rect.height - (270 if ui_state.torque_bar else _CH_MARGIN)
      y = marker.center[1] + 32
      if y + _CH_LINE > floor:
        y = max(rect.y + _CH_MARGIN, min(marker.center[1] - 32, floor) - _CH_LINE)
      shadow = canvas.shade(0, 0, 0, int(200 * self._alpha))
      color = canvas.shade(255, 255, 255, int(255 * self._alpha))
      canvas.glyphs(self._font, text, canvas.Pt(x + 2, y + 2), 34, shadow)
      canvas.glyphs(self._font, text, canvas.Pt(x, y), 34, color)
    else:
      self._stack(lines, marker.center[0], self._top_y(marker.center[1], self._marker_size(lead.dRel), len(lines), rect), rect)

  @staticmethod
  def _active_leads(radar_state, markers):
    tracked = []
    for lead, marker in zip((radar_state.leadOne, radar_state.leadTwo), markers, strict=False):
      if lead and lead.status and marker.center is not None:
        tracked.append((lead, marker))
    if len(tracked) == 2 and abs(tracked[0][0].dRel - tracked[1][0].dRel) <= _CH_DEDUP:
      tracked.pop()
    return tracked

  def draw_lead_status(self, sm, radar_state, rect, lead_vehicles):
    present = [radar_state.leadOne, radar_state.leadTwo]
    self.update_alpha(any(bool(x) and x.status for x in present))
    if not self.should_render():
      return
    v_ego = sm['carState'].vEgo
    for lead, marker in self._active_leads(radar_state, lead_vehicles):
      self._one_lead(lead, marker, v_ego, rect)

_TEAL = canvas.shade(0x0C, 0x94, 0x96, 0xFF)
_AMBER = canvas.shade(255, 188, 0, 255)
_GREEN = canvas.shade(0, 255, 0, 255)
_GREY = canvas.shade(145, 155, 149, 255)
_G = 9.81


@dataclass
class Readout:
  tag: str
  value: str
  unit: str = ""
  color: object = field(default_factory=lambda: canvas.WHITE)
  tag_text: str = ""
  value_text: str = ""
  unit_text: str = ""
  tag_w: float = 0.0
  value_w: float = 0.0
  unit_w: float = 0.0
  span: float = 0.0

  def size_up(self, font, px: int):
    self.tag_text = f"{self.tag} "
    self.value_text = self.value
    self.unit_text = f" {self.unit}" if self.unit else ""
    self.tag_w = canvas.span(font, self.tag_text, px, 0).x
    self.value_w = canvas.span(font, self.value_text, px, 0).x
    self.unit_w = canvas.span(font, self.unit_text, px, 0).x if self.unit else 0
    self.span = self.tag_w + self.value_w + self.unit_w

  @property
  def total_width(self):
    return self.span

  def measure(self, font, px):
    self.size_up(font, px)


UiElement = Readout

def _banded(magnitude, warn, crit, ok):
  if magnitude > crit:
    return canvas.RED
  return _AMBER if magnitude > warn else ok

def _steer_tint(sm):
  if not sm['carControl'].latActive:
    return canvas.WHITE
  return _GREY if sm['carState'].steeringPressed else _TEAL

def _angle_tint(sm, deg):
  floor = _steer_tint(sm) if sm['carControl'].latActive else canvas.WHITE
  return _banded(abs(deg), 90.0, 180.0, floor)

def _yaw_offset(sm):
  return sm['vehicleParameters'].angleOffsetAverageDeg if sm.valid['vehicleParameters'] else 0.0

def _bank(sm):
  return sm['vehicleParameters'].roll if sm.valid['vehicleParameters'] else 0.0

def steering_angle(sm, is_metric):
  deg = sm['carState'].steeringAngleDeg - _yaw_offset(sm)
  return Readout("R.S.", f"{deg:.1f}°", color=_angle_tint(sm, deg))

def desired_steering_angle(sm, is_metric):
  live = sm['carControl'].latActive
  off = _yaw_offset(sm)
  lat = sm['controlsState'].lateralControlState
  if lat.which() == 'angleState':
    want = lat.angleState.steeringAngleDesiredDeg - off
  else:
    want = sm['carControl'].actuators.steeringAngleDeg - off
  seen = sm['carState'].steeringAngleDeg - off
  tint = _banded(abs(seen), 90.0, 180.0, _TEAL) if live else canvas.WHITE
  return Readout("D.S.", f"{want:.1f}°" if live else "-", color=tint)

def desired_steering_pid(sm, is_metric):
  live = sm['carControl'].latActive
  off = _yaw_offset(sm)
  want = sm['controlsState'].lateralControlState.pidState.steeringAngleDesiredDeg - off
  seen = sm['carState'].steeringAngleDeg - off
  tint = _banded(abs(seen), 90.0, 180.0, _TEAL) if live else canvas.WHITE
  return Readout("D.S.", f"{want:.1f}°" if live else "-", color=tint)

def actual_lat_accel(sm, is_metric):
  a = sm['controlsState'].curvature * sm['carState'].vEgo ** 2 - _bank(sm) * _G
  return Readout("A.L.A.", f"{a:.2f}", "m/s^2", _steer_tint(sm))

def desired_lat_accel(sm, is_metric):
  live = sm['carControl'].latActive
  a = sm['controlsState'].desiredCurvature * sm['carState'].vEgo ** 2 - _bank(sm) * _G
  return Readout("D.L.A.", f"{a:.2f}" if live else "-", "m/s^2", _steer_tint(sm))

class IQDevMetricsOverlay(Widget):
  def __init__(self):
    super().__init__()
    self._face = gui_app.font(FontWeight.BOLD)
    self._label_face = gui_app.font(FontWeight.MEDIUM)

  def _render(self, rect) -> None:
    sm = ui_state.sm
    if not ui_state.developer_ui or sm.recv_frame["carState"] < ui_state.started_frame:
      return
    accel = f"L.A.  {sm['carState'].aEgo:.1f}"
    width = canvas.span(self._label_face, accel, 30, 0).x
    canvas.glyphs(self._label_face, accel, canvas.Pt(rect.x + (rect.width - width) / 2, rect.y + 12), 30, canvas.WHITE)
    desired = desired_steering_pid if sm['controlsState'].lateralControlState.which() == 'pidState' else desired_steering_angle
    for fraction, labels, probes in (
      (0.28, ("D.S.", "R.S."), (desired, steering_angle)),
      (0.72, ("D.L.A.", "A.L.A."), (desired_lat_accel, actual_lat_accel)),
    ):
      center_x = rect.x + rect.width * fraction
      for row, (label, probe) in enumerate(zip(labels, probes, strict=True)):
        cell = probe(sm, ui_state.is_metric)
        y = rect.y + 44 + row * 60
        label_width = canvas.span(self._label_face, label, 26, 0).x
        canvas.glyphs(self._label_face, label, canvas.Pt(center_x - 18 - label_width, y + 7), 26, canvas.WHITE)
        canvas.glyphs(self._face, cell.value, canvas.Pt(center_x + 18, y), 38, cell.color)

_SL_ASSIST = custom.IQPlan.SpeedLimit.AssistState
_SL_GREY = canvas.shade(145, 155, 149, 255)
_SL_DARK = canvas.shade(77, 77, 77, 255)

def _dim(color, alpha: float):
  return canvas.with_opacity(color, 255 * alpha)

class SpeedLimitState:
  def __init__(self):
    super().__init__()
    self.speed_limit = 0.0
    self.speed_limit_last = 0.0
    self.speed_limit_offset = 0.0
    self.speed_limit_valid = False
    self.speed_limit_last_valid = False
    self.speed_limit_final_last = 0.0
    self.assist_state = _SL_ASSIST.disabled
    self.speed = 0.0

  @property
  def _scale(self):
    return CV.MS_TO_KPH if ui_state.is_metric else CV.MS_TO_MPH

  def _take_plan(self, lp_iq):
    k = self._scale
    r = lp_iq.speedLimit.resolver
    self.speed_limit = r.speedLimit * k
    self.speed_limit_last = r.speedLimitLast * k
    self.speed_limit_offset = r.speedLimitOffset * k
    self.speed_limit_valid = r.speedLimitValid
    self.speed_limit_last_valid = r.speedLimitLastValid
    self.speed_limit_final_last = r.speedLimitFinalLast * k
    self.assist_state = lp_iq.speedLimit.assist.state

  def update(self):
    sm = _feed()
    if sm.recv_frame["carState"] < ui_state.started_frame:
      return
    if sm.updated["iqPlan"]:
      self._take_plan(sm["iqPlan"])
    cs = sm["carState"]
    v_ego = cs.vEgoCluster if cs.vEgoCluster != 0.0 else cs.vEgo
    self.speed = max(0.0, v_ego * self._scale)

  def _spec(self):
    has_limit = self.speed_limit_valid or self.speed_limit_last_valid
    value = str(round(self.speed_limit_last)) if has_limit else "---"
    badge = ""
    if self.speed_limit_offset != 0:
      badge = f"{'' if self.speed_limit_offset > 0 else '-'}{round(abs(self.speed_limit_offset))}"
    warn = ui_state.speed_limit_mode >= 2
    over = has_limit and round(self.speed_limit_final_last) < round(self.speed)
    tint = canvas.RED if (warn and over) else (_SL_GREY if not self.speed_limit_valid else canvas.BLACK)
    return value, badge, tint, has_limit
