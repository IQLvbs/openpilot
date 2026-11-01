"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""
import math
from enum import IntEnum

import pyray as rl

from iqpilot.common.params import Params
from iqpilot.system.ui.lib.text_measure import measure_text_cached

class SourceState(IntEnum):
  HIDDEN = 0
  LOADING = 1
  ACTIVE = 2
  FAILED = 3
  CROSSED = 4


class DockStatus(IntEnum):
  HIDDEN = 0
  READY = 1
  DEGRADED = 2
  SETUP = 3
  FAULT = 4


EGPU_FAULT_ALERTS = ("Offroad_EgpuPcieUnavailable", "Offroad_EgpuOverheated", "Offroad_EgpuFansObstructed",
                     "Offroad_EgpuUpdateFailed", "Offroad_EgpuNotDetected")
EGPU_DEGRADED_ALERTS = ("Offroad_EgpuUsbSlow", "Offroad_EgpuUncompiled")


_GREEN = rl.Color(46, 204, 113, 255)
_ORANGE = rl.Color(255, 115, 0, 255)
_WHITE = rl.Color(255, 255, 255, 255)


def _emac_state(params: Params, engaged: bool) -> SourceState:
  if not params.get_bool("MacModelReachable"):
    return SourceState.HIDDEN
  if params.get_bool("MacModelActive"):
    return SourceState.ACTIVE
  if params.get_bool("MacModelFailed"):
    return SourceState.CROSSED if engaged else SourceState.FAILED
  return SourceState.LOADING


def _egpu_state(params: Params, engaged: bool) -> SourceState:
  if not params.get_bool("UsbGpuPresent"):
    return SourceState.HIDDEN
  if params.get_bool("UsbGpuActive"):
    return SourceState.ACTIVE
  if params.get_bool("UsbGpuFailed"):
    return SourceState.CROSSED if engaged else SourceState.FAILED
  return SourceState.LOADING


def resolve_source(params: Params, engaged: bool) -> tuple[str, SourceState]:
  if params.get_bool("UsbGpuPresent") and not params.get_bool("IQEgpuDisabled"):
    return "GPU", _egpu_state(params, engaged)
  if params.get_bool("IQEmacEnabled"):
    return "MAC", _emac_state(params, engaged)
  if params.get_bool("IQEgpuEnabled"):
    return "GPU", _egpu_state(params, engaged)
  return "", SourceState.HIDDEN


def _alert_set(params: Params, keys: tuple[str, ...]) -> bool:
  # offroad alerts are JSON params: get_bool() on a raised alert reads False
  return any(params.get(k) is not None for k in keys)


def egpu_dock_status(params: Params, device_state) -> tuple[DockStatus, float]:
  if not getattr(device_state, "egpuDockPresent", False):
    return DockStatus.HIDDEN, 0.0
  if params.get_bool("UsbGpuFailed") or _alert_set(params, EGPU_FAULT_ALERTS):
    return DockStatus.FAULT, 0.0
  if params.get_bool("UsbGpuLoading") and not params.get_bool("UsbGpuCompiled"):
    try:
      progress = max(0.0, min(1.0, float(params.get("UsbGpuSetupProgress") or 0.0)))
    except (TypeError, ValueError):
      progress = 0.0
    return DockStatus.SETUP, progress
  if _alert_set(params, EGPU_DEGRADED_ALERTS):
    return DockStatus.DEGRADED, 0.0
  return DockStatus.READY, 0.0


def draw_source_label(font: rl.Font, label: str, state: SourceState,
                      pos: rl.Vector2, font_size: int) -> None:
  if state == SourceState.HIDDEN or not label:
    return
  if state == SourceState.ACTIVE:
    color, opacity, strike = _GREEN, 1.0, False
  elif state == SourceState.FAILED:
    color, opacity, strike = _ORANGE, 1.0, False
  elif state == SourceState.CROSSED:
    color, opacity, strike = _WHITE, 0.65, True
  else:
    color, opacity, strike = _WHITE, 0.35 + 0.65 * (0.5 - 0.5 * math.cos(rl.get_time() * 6.0)), False

  col = rl.Color(color.r, color.g, color.b, int(255 * opacity))
  rl.draw_text_ex(font, label, pos, font_size, 0, col)
  if strike:
    size = measure_text_cached(font, label, font_size)
    y = int(pos.y + size.y / 2)
    rl.draw_line_ex(rl.Vector2(pos.x - 2, y), rl.Vector2(pos.x + size.x + 2, y), 4, col)
