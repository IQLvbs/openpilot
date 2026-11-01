"""Offroad tile map with offline fallback, GPS tracking and touch controls."""
from __future__ import annotations

import math
import time

import pyray as rl

from iqpilot.common.params import Params
from iqpilot.selfdrive.ui.lib.nav_helpers import active_navigation_route, current_or_last_gps_position
from iqpilot.ui.onroad.nav_map_panel import MapboxTileProvider, OsmOfflineProvider
from iqpilot.ui.onroad.nav_map_utils import TILE_SIZE
from iqpilot.system.ui.lib.application import gui_app, FontWeight, MouseEvent, MousePos
from iqpilot.system.ui.lib.multilang import tr
from iqpilot.system.ui.lib.text_measure import measure_text_cached
from iqpilot.system.ui.lib.wrap_text import wrap_text
from iqpilot.system.ui.widgets import Widget

DEFAULT_ZOOM = 15.0
MIN_ZOOM = 4.0
MAX_ZOOM = 18.0
ZOOM_STEP = 1.0
TILE_UPDATE_S = 0.20          # fetch/prune cadence; drawing runs every frame
GPS_POLL_S = 0.5
DEST_POLL_S = 1.0
BTN = 92
PUCK_BLUE = rl.Color(23, 134, 246, 255)
MAP_BG = rl.Color(18, 18, 20, 255)
BTN_BG = rl.Color(28, 30, 36, 235)
BTN_BORDER = rl.Color(255, 255, 255, 45)


def _norm(lat: float, lon: float) -> tuple[float, float]:
  lat = max(-85.05, min(85.05, lat))
  x = (lon + 180.0) / 360.0
  s = math.sin(math.radians(lat))
  y = 0.5 - math.log((1.0 + s) / (1.0 - s)) / (4.0 * math.pi)
  return x, y


def _denorm(x: float, y: float) -> tuple[float, float]:
  lon = x * 360.0 - 180.0
  lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y))))
  return lat, lon


class InteractiveNavMap(Widget):
  def __init__(self, interactive: bool = True, on_destination=None):
    super().__init__()
    self._params = Params()
    self._tiles = MapboxTileProvider()
    self._offline = OsmOfflineProvider()
    self._interactive = interactive
    self._on_destination = on_destination
    self._online_enabled = True
    self._lat = 0.0
    self._lon = 0.0
    self._zoom = DEFAULT_ZOOM
    self._follow = True
    self._have_center = False

    self._drag_start: MousePos | None = None
    self._drag_center: tuple[float, float] | None = None
    self._dragging = False

    self._tiles_time = 0.0
    self._gps_cache: tuple[float, float, bool] = (0.0, 0.0, False)
    self._gps_time = 0.0
    self._dest_cache = None
    self._route_cache = None
    self._have_downloads = False
    self._dest_time = 0.0

    self._pin_icon = gui_app.texture("icons/iq/pin.png", 64, 64, keep_aspect_ratio=True)

  def _world_size(self) -> float:
    return TILE_SIZE * (2.0 ** self._zoom)

  def _poll_gps(self):
    now = time.monotonic()
    if now - self._gps_time > GPS_POLL_S:
      lat, lon, _, fix = current_or_last_gps_position(self._params)
      self._gps_cache = (lat, lon, fix)
      self._gps_time = now
    lat, lon, fix = self._gps_cache
    if fix and self._follow and not self._dragging:
      self._lat, self._lon = lat, lon
      self._have_center = True
    elif fix and not self._have_center:
      self._lat, self._lon = lat, lon
      self._have_center = True
    return lat, lon, fix

  def recenter(self):
    self._follow = True
    self._gps_time = 0.0

  # --- gestures ---------------------------------------------------------------
  def _handle_mouse_event(self, mouse_event: MouseEvent) -> None:
    super()._handle_mouse_event(mouse_event)
    if not self._interactive or mouse_event.slot != 0 or not self._have_center:
      return
    if mouse_event.left_pressed:
      if rl.check_collision_point_rec(mouse_event.pos, self._rect) and self._control_hit(mouse_event.pos) is None:
        self._drag_start = mouse_event.pos
        self._drag_center = _norm(self._lat, self._lon)
        self._dragging = False
      else:
        self._drag_start = None
    elif mouse_event.left_down and self._drag_start is not None:
      dx = mouse_event.pos.x - self._drag_start.x
      dy = mouse_event.pos.y - self._drag_start.y
      if not self._dragging and (abs(dx) > 10 or abs(dy) > 10):
        self._dragging = True
        self._follow = False
      if self._dragging:
        ws = self._world_size()
        nx = (self._drag_center[0] - dx / ws) % 1.0
        ny = max(0.001, min(0.999, self._drag_center[1] - dy / ws))
        self._lat, self._lon = _denorm(nx, ny)
    elif mouse_event.left_released:
      self._drag_start = None
      self._drag_center = None
      self._dragging = False

  def _controls(self) -> list[tuple[str, rl.Rectangle]]:
    if not self._interactive:
      return []
    r = self._rect
    x = r.x + r.width - BTN - 20
    ctrls = [("+", rl.Rectangle(x, r.y + 20, BTN, BTN)),
             ("-", rl.Rectangle(x, r.y + 20 + BTN + 14, BTN, BTN))]
    if not self._follow:
      ctrls.append(("recenter", rl.Rectangle(x, r.y + r.height - BTN - 20, BTN, BTN)))
    if not self._follow and self._on_destination is not None:
      ctrls.append(("destination", rl.Rectangle(r.x + 20, r.y + 20, 330, BTN)))
    return ctrls

  def _control_hit(self, pos: MousePos) -> str | None:
    for name, rect in self._controls():
      if rl.check_collision_point_rec(pos, rect):
        return name
    return None

  def _handle_mouse_release(self, mouse_pos: MousePos) -> None:
    hit = self._control_hit(mouse_pos)
    if hit == "+":
      self._zoom = min(MAX_ZOOM, self._zoom + ZOOM_STEP)
      self._tiles_time = 0.0
    elif hit == "-":
      self._zoom = max(MIN_ZOOM, self._zoom - ZOOM_STEP)
      self._tiles_time = 0.0
    elif hit == "recenter":
      self.recenter()
    elif hit == "destination" and self._on_destination is not None:
      self._on_destination(self._lat, self._lon)

  # --- rendering --------------------------------------------------------------
  def _render(self, rect: rl.Rectangle):
    gps_lat, gps_lon, fix = self._poll_gps()
    now = time.monotonic()

    if not self._have_center:
      if now - self._tiles_time > 2.0:
        self._have_downloads = self._offline.source_available()
        self._tiles_time = now
      rl.draw_rectangle_rounded(rect, 0.03, 20, MAP_BG)
      self._center_note(rect, tr("Waiting for GPS fix...") if self._have_downloads else self._download_note())
      return

    if now - self._tiles_time > TILE_UPDATE_S:
      self._online_enabled = self._params.get_bool("OnlineOSMaps")
      if self._online_enabled:
        self._tiles.update(self._lat, self._lon, self._zoom, rect.width, rect.height)
      # Offroad maps use downloaded tiles regardless of the onroad overlay switch.
      if not self._online_enabled or not self._tiles.viewport_complete():
        self._offline.update(self._lat, self._lon, self._zoom, rect.width, rect.height)
      self._tiles_time = now

    rl.draw_rectangle_rec(rect, MAP_BG)
    rl.begin_scissor_mode(int(rect.x), int(rect.y), int(rect.width), int(rect.height))
    drew_map = self._offline.draw(rect, self._lat, self._lon, self._zoom)
    if self._online_enabled:
      drew_map = self._tiles.draw(rect, self._lat, self._lon, self._zoom) or drew_map
    if not drew_map:
      note = self._download_note() if self._offline.viewport_missing() else tr("Loading map...")
      self._center_note(rect, note)
    if now - self._dest_time > DEST_POLL_S:
      self._dest_cache = self._params.get("NavigationDestination")
      self._route_cache = active_navigation_route(self._params)
      self._dest_time = now
    if drew_map:
      self._draw_route(rect)
    if fix and drew_map:
      self._draw_puck(rect, gps_lat, gps_lon)
    if drew_map:
      self._draw_destination(rect)
    rl.end_scissor_mode()
    rl.draw_rectangle_rounded_lines_ex(rect, 0.03, 20, 2, rl.Color(255, 255, 255, 38))
    self._draw_controls()
    if not self._follow and self._on_destination is not None:
      cx, cy = int(rect.x + rect.width / 2), int(rect.y + rect.height / 2)
      rl.draw_circle_lines(cx, cy, 12, rl.WHITE)
      rl.draw_line(cx - 20, cy, cx + 20, cy, rl.WHITE)
      rl.draw_line(cx, cy - 20, cx, cy + 20, rl.WHITE)
    if drew_map:
      font = gui_app.font(FontWeight.MEDIUM)
      rl.draw_text_ex(font, "© OpenStreetMap contributors", rl.Vector2(rect.x + 12, rect.y + rect.height - 30), 22, 0, rl.LIGHTGRAY)

  @staticmethod
  def _download_note() -> str:
    return tr("No maps for this area.\nOpen the konn3kt app to download maps.")

  def _center_note(self, rect: rl.Rectangle, text: str):
    font = gui_app.font(FontWeight.MEDIUM)
    size = 44
    lines = wrap_text(font, text, size, int(rect.width - (260 if self._interactive else 80)))
    line_height = measure_text_cached(font, "Ag", size).y + 12
    y = rect.y + (rect.height - len(lines) * line_height) / 2
    for line in lines:
      width = measure_text_cached(font, line, size).x
      rl.draw_text_ex(font, line, rl.Vector2(rect.x + (rect.width - width) / 2, y), size, 0, rl.Color(220, 222, 228, 255))
      y += line_height

  def _draw_route(self, rect: rl.Rectangle):
    if self._route_cache is None:
      return
    points = [rl.Vector2(*self._project(rect, lat, lon)) for lat, lon in self._route_cache.get("routePolylineSimplified", [])]
    for start, end in zip(points, points[1:]):
      rl.draw_line_ex(start, end, 12, PUCK_BLUE)

  def _project(self, rect: rl.Rectangle, lat: float, lon: float) -> tuple[float, float]:
    ws = self._world_size()
    cx, cy = _norm(self._lat, self._lon)
    px, py = _norm(lat, lon)
    return (rect.x + rect.width / 2 + (px - cx) * ws,
            rect.y + rect.height / 2 + (py - cy) * ws)

  def _draw_puck(self, rect: rl.Rectangle, lat: float, lon: float):
    x, y = self._project(rect, lat, lon)
    if rect.x <= x <= rect.x + rect.width and rect.y <= y <= rect.y + rect.height:
      rl.draw_circle(int(x), int(y), 24, rl.Color(255, 255, 255, 235))
      rl.draw_circle(int(x), int(y), 16, PUCK_BLUE)

  def _draw_destination(self, rect: rl.Rectangle):
    dest = self._dest_cache
    if not dest:
      return
    try:
      lat, lon = float(dest["latitude"]), float(dest["longitude"])
    except Exception:
      return
    x, y = self._project(rect, lat, lon)
    if rect.x <= x <= rect.x + rect.width and rect.y <= y <= rect.y + rect.height:
      rl.draw_texture(self._pin_icon, int(x - self._pin_icon.width / 2), int(y - self._pin_icon.height), rl.WHITE)

  def _draw_controls(self):
    font = gui_app.font(FontWeight.MEDIUM)
    for name, r in self._controls():
      rl.draw_rectangle_rounded(r, 0.35, 16, BTN_BG)
      rl.draw_rectangle_rounded_lines_ex(r, 0.35, 16, 2, BTN_BORDER)
      cx, cy = r.x + r.width / 2, r.y + r.height / 2
      if name == "recenter":
        rl.draw_circle_lines(int(cx), int(cy), 20, rl.WHITE)
        rl.draw_circle(int(cx), int(cy), 6, PUCK_BLUE)
        for ang in (0, 90, 180, 270):
          a = math.radians(ang)
          rl.draw_line_ex(rl.Vector2(cx + 20 * math.cos(a), cy + 20 * math.sin(a)),
                          rl.Vector2(cx + 30 * math.cos(a), cy + 30 * math.sin(a)), 3, rl.WHITE)
      elif name == "destination":
        label = tr("Navigate here")
        ts = measure_text_cached(font, label, 32)
        rl.draw_text_ex(font, label, rl.Vector2(cx - ts.x / 2, cy - ts.y / 2), 32, 0, rl.WHITE)
      else:
        ts = measure_text_cached(font, name, 56)
        rl.draw_text_ex(font, name, rl.Vector2(int(cx - ts.x / 2), int(cy - ts.y / 2)), 56, 0, rl.WHITE)

  def hide_event(self):
    super().hide_event()
    self.release()

  def release(self):
    self._tiles.release()
    self._offline.release()
    self._tiles_time = 0.0
    self._drag_start = None
    self._drag_center = None
    self._dragging = False
