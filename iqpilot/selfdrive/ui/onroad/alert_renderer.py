import math
import time
from dataclasses import dataclass

import pyray as rl

from iqpilot.cereal import messaging, log
from iqpilot.selfdrive.ui.ui_state import ui_state
from iqpilot.system.hardware import TICI
from iqpilot.system.ui.lib.application import gui_app, FontWeight
from iqpilot.system.ui.lib.multilang import tr
from iqpilot.system.ui.lib.text_measure import measure_text_cached
from iqpilot.system.ui.widgets import Widget
from iqpilot.system.ui.widgets.label import Label
from iqpilot.ui.onroad.theme import CRITICAL, INACTIVE, PROMPT, ease_out, tone

AlertSize = log.SelfdriveState.AlertSize
AlertStatus = log.SelfdriveState.AlertStatus

ALERT_PADDING = 28
ALERT_BOTTOM_MARGIN = 36
ALERT_BORDER_RADIUS = 32
ALERT_ENTRY_DURATION = 0.36
ALERT_SIDE_MARGIN = 236
ALERT_MAP_GAP = 36
ALERT_MAP_RESERVED_WIDTH = 456


ALERT_HEIGHTS = {
  AlertSize.small: 176,
  AlertSize.mid: 240,
}

_CALIBRATION_EVENTS = {"calibrationIncomplete", "calibrationRecalibrating"}

SELFDRIVE_STATE_TIMEOUT = 5  # Seconds
SELFDRIVE_UNRESPONSIVE_TIMEOUT = 10  # Seconds

# Constants
ALERT_COLORS = {
  AlertStatus.normal: INACTIVE,
  AlertStatus.userPrompt: PROMPT,
  AlertStatus.critical: CRITICAL,
}


@dataclass
class Alert:
  text1: str = ""
  text2: str = ""
  size: int = 0
  status: int = 0
  event_name: str = ""


# Pre-defined alert instances
ALERT_STARTUP_PENDING = Alert(
  text1=tr("IQ.Pilot Unavailable"),
  text2=tr("Waiting to start"),
  size=AlertSize.mid,
  status=AlertStatus.normal,
)

ALERT_CRITICAL_TIMEOUT = Alert(
  text1=tr("TAKE CONTROL IMMEDIATELY"),
  text2=tr("System Unresponsive"),
  size=AlertSize.full,
  status=AlertStatus.critical,
)

ALERT_CRITICAL_REBOOT = Alert(
  text1=tr("System Unresponsive"),
  text2=tr("Reboot Device"),
  size=AlertSize.mid,
  status=AlertStatus.normal,
)


class AlertRenderer(Widget):
  def __init__(self):
    super().__init__()
    self.font_regular: rl.Font = gui_app.font(FontWeight.NORMAL)
    self.font_bold: rl.Font = gui_app.font(FontWeight.BOLD)
    self._alert_key: tuple | None = None
    self._appeared_at = 0.0
    self._visible_alert: Alert | None = None
    self._dismissed_at: float | None = None
    self._dismiss_progress = 1.0
    self._maps_visible = False

    # font size is set dynamically
    self._full_text1_label = Label("", font_size=0, font_weight=FontWeight.BOLD, text_alignment=rl.GuiTextAlignment.TEXT_ALIGN_CENTER,
                                   text_alignment_vertical=rl.GuiTextAlignmentVertical.TEXT_ALIGN_TOP)
    self._full_text2_label = Label("", font_size=56, text_alignment=rl.GuiTextAlignment.TEXT_ALIGN_CENTER,
                                   text_alignment_vertical=rl.GuiTextAlignmentVertical.TEXT_ALIGN_TOP)

  def get_alert(self, sm: messaging.SubMaster) -> Alert | None:
    """Generate the current alert based on selfdrive state."""
    ss = sm['selfdriveState']

    # Check if selfdriveState messages have stopped arriving
    recv_frame = sm.recv_frame['selfdriveState']
    if not sm.updated['selfdriveState']:
      time_since_onroad = time.monotonic() - ui_state.started_time

      # 1. Never received selfdriveState since going onroad
      waiting_for_startup = recv_frame < ui_state.started_frame
      if waiting_for_startup and time_since_onroad > 5:
        return ALERT_STARTUP_PENDING

      # 2. Lost communication with selfdriveState after receiving it
      if TICI and not waiting_for_startup:
        ss_missing = time.monotonic() - sm.recv_time['selfdriveState']
        if ss_missing > SELFDRIVE_STATE_TIMEOUT:
          if ss.enabled and (ss_missing - SELFDRIVE_STATE_TIMEOUT) < SELFDRIVE_UNRESPONSIVE_TIMEOUT:
            return ALERT_CRITICAL_TIMEOUT
          return ALERT_CRITICAL_REBOOT

    # No alert if size is none
    if ss.alertSize == 0:
      return None

    # Don't get old alert
    if recv_frame < ui_state.started_frame:
      return None

    event_name = ss.alertType.split('/')[0] if ss.alertType else ''
    if event_name in {'selfdrivedLagging', 'commIssue', 'commIssueAvgFreq'}:
      return None

    # Return current alert
    return Alert(text1=ss.alertText1, text2=ss.alertText2, size=ss.alertSize.raw,
                 status=ss.alertStatus.raw, event_name=event_name)

  def set_maps_visible(self, visible: bool) -> None:
    self._maps_visible = visible

  def has_tile(self, sm: messaging.SubMaster) -> bool:
    alert = self.get_alert(sm)
    return alert is not None and alert.size != AlertSize.full

  @staticmethod
  def _animation_key(alert: Alert) -> tuple:
    if alert.event_name in _CALIBRATION_EVENTS:
      return ("calibration", alert.size, alert.status)
    return (alert.event_name, alert.text1, alert.text2, alert.size, alert.status)

  def _render(self, rect: rl.Rectangle):
    alert = self.get_alert(ui_state.sm)

    if gui_app.iqpilot_ui():
      ui_state.onroad_brightness_handle_alerts(ui_state.started, alert)

    now = time.monotonic()
    if alert is not None:
      key = self._animation_key(alert)
      if key != self._alert_key or self._dismissed_at is not None:
        self._appeared_at = now
      self._alert_key = key
      self._visible_alert = alert
      self._dismissed_at = None
      progress = ease_out(now - self._appeared_at, ALERT_ENTRY_DURATION)
    else:
      alert = self._visible_alert
      if alert is None:
        return
      if alert.size == AlertSize.full:
        self._visible_alert = None
        self._alert_key = None
        return
      if self._dismissed_at is None:
        self._dismissed_at = now
        self._dismiss_progress = ease_out(now - self._appeared_at, ALERT_ENTRY_DURATION)
      progress = self._dismiss_progress * (1 - ease_out(now - self._dismissed_at, 0.24))
      if progress <= 0:
        self._visible_alert = None
        self._alert_key = None
        return

    elapsed = now - self._appeared_at
    alert_rect = self._get_alert_rect(rect, alert.size)
    if alert.size != AlertSize.full:
      alert_rect = self._entry_rect(alert_rect, progress)
    content_rect = alert_rect
    self._draw_background(alert_rect, alert, elapsed)

    text_rect = rl.Rectangle(
      content_rect.x + ALERT_PADDING,
      content_rect.y + ALERT_PADDING,
      content_rect.width - 2 * ALERT_PADDING,
      content_rect.height - 2 * ALERT_PADDING
    )
    rl.begin_scissor_mode(int(alert_rect.x), int(alert_rect.y), int(alert_rect.width), int(alert_rect.height))
    self._draw_text(text_rect, alert)
    rl.end_scissor_mode()

  def _get_alert_rect(self, rect: rl.Rectangle, size: int) -> rl.Rectangle:
    if size == AlertSize.full:
      return rect

    height = ALERT_HEIGHTS.get(size, 240)
    left = rect.x + ALERT_SIDE_MARGIN
    right = rect.x + rect.width - ALERT_SIDE_MARGIN
    maps_visible = getattr(self, "_maps_visible", getattr(self, "navigation_visible", False))
    if maps_visible:
      right = min(right, rect.x + rect.width - ALERT_MAP_RESERVED_WIDTH - ALERT_MAP_GAP)
    width = max(1, right - left)
    return rl.Rectangle(left, rect.y + rect.height - height - ALERT_BOTTOM_MARGIN, width, height)

  @staticmethod
  def _entry_rect(rect: rl.Rectangle, progress: float) -> rl.Rectangle:
    return rl.Rectangle(rect.x, rect.y + (1 - progress) * (rect.height + ALERT_BOTTOM_MARGIN), rect.width, rect.height)

  def _draw_background(self, rect: rl.Rectangle, alert: Alert, elapsed: float) -> None:
    color = ALERT_COLORS.get(alert.status, ALERT_COLORS[AlertStatus.normal])
    if alert.status == AlertStatus.critical:
      pulse_scale = 0.94 + 0.06 * (0.5 + 0.5 * math.sin(elapsed * math.tau / 1.2))
      color = tone(color, pulse_scale)

    if alert.size == AlertSize.full:
      rl.draw_rectangle_rec(rect, color)
      return

    roundness = min(0.5, ALERT_BORDER_RADIUS / (min(rect.width, rect.height) / 2))
    rl.draw_rectangle_rounded(rect, roundness, 16, rl.Color(16, 23, 29, 250))
    center = rl.Vector2(rect.x + 76, rect.y + rect.height / 2)
    accent = rl.Color(194, 207, 215, 255) if alert.status == AlertStatus.normal else color
    rl.draw_circle_v(center, 36, accent)
    ink = rl.Color(16, 23, 29, 255)
    is_information = alert.status == AlertStatus.normal
    stem_y = center.y - 6 if is_information else center.y - 18
    dot_y = center.y - 14 if is_information else center.y + 14
    rl.draw_rectangle_rounded(rl.Rectangle(center.x - 3, stem_y, 6, 24), 1.0, 8, ink)
    rl.draw_circle_v(rl.Vector2(center.x, dot_y), 4, ink)

  def _draw_text(self, rect: rl.Rectangle, alert: Alert) -> None:
    if alert.size != AlertSize.full:
      left = rect.x + 112
      width = rect.width - 128
      title_size = 58
      title_width = measure_text_cached(self.font_bold, alert.text1, title_size).x
      title_size = min(title_size, title_size * width / max(1, title_width))
      subtitle_size = 34
      subtitle_width = measure_text_cached(self.font_regular, alert.text2, subtitle_size).x
      subtitle_size = min(subtitle_size, subtitle_size * width / max(1, subtitle_width))
      height = title_size + (subtitle_size + 12 if alert.text2 else 0)
      top = rect.y + (rect.height - height) / 2
      rl.draw_text_ex(self.font_bold, alert.text1, rl.Vector2(left, top), title_size, 0, rl.WHITE)
      if alert.text2:
        rl.draw_text_ex(self.font_regular, alert.text2, rl.Vector2(left, top + title_size + 12),
                        subtitle_size, 0, rl.Color(201, 211, 218, 255))
    else:
      is_long = len(alert.text1) > 15
      font_size1 = 132 if is_long else 177

      top_offset = 200 if is_long or '\n' in alert.text1 else 270
      title_rect = rl.Rectangle(rect.x, rect.y + top_offset, rect.width, 600)
      self._full_text1_label.set_font_size(font_size1)
      self._full_text1_label.set_text(alert.text1)
      self._full_text1_label.render(title_rect)

      bottom_offset = 361 if is_long else 420
      subtitle_rect = rl.Rectangle(rect.x, rect.y + rect.height - bottom_offset, rect.width, 300)
      self._full_text2_label.set_text(alert.text2)
      self._full_text2_label.render(subtitle_rect)
