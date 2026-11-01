import numpy as np
import pyray as rl
from iqpilot.cereal import car, custom
from dataclasses import dataclass, field
from iqpilot.common.filter_simple import FirstOrderFilter
from iqpilot.common.params import Params
from iqpilot.selfdrive.locationd.calibrationd import HEIGHT_INIT
from iqpilot.selfdrive.locationd.calibration_helpers import get_render_path_height
from iqpilot.selfdrive.ui.ui_state import ui_state, UIStatus, log_param_from_bytes
from iqpilot.system.ui.lib.application import gui_app
from iqpilot.system.ui.lib.shader_polygon import draw_polygon, Gradient
from iqpilot.system.ui.widgets import Widget
from iqpilot.ui.onroad.model_renderer_helpers import BACKUP_PATH_WIDTH, ModelRendererHelpers

from iqpilot.ui.onroad.hud_overlays import ChevronMetrics

CLIP_MARGIN = 500
MIN_DRAW_DISTANCE = 10.0
MAX_DRAW_DISTANCE = 100.0
# the head is trained against a 7% positive rate, so it reports ~0.08 at a real stop and never
# exceeds 0.024 on held-out highway frames; 0.5 gated out every stop the model actually found
STOP_POINT_MIN_PROBABILITY = 0.30
STOP_POINT_HALF_WIDTH = 1.0
STOP_POINT_MAX_OFFSET = 4.0
STOP_POINT_DEPTH = 0.45
STOP_POINT_COLOR = (255, 90, 60)

_VT_LABEL = custom.IQVehicleTracks.Track.Label
VEHICLE_TRACK_LABELS = (_VT_LABEL.car, _VT_LABEL.motorcycle, _VT_LABEL.bus, _VT_LABEL.truck)
SIGN_TRACK_COLORS = {
  _VT_LABEL.stopSign: rl.Color(255, 60, 45, 200),
  _VT_LABEL.trafficLight: rl.Color(255, 190, 0, 200),
}

THROTTLE_COLORS = [
  rl.Color(13, 248, 122, 102),   # HSLF(148/360, 0.94, 0.51, 0.4)
  rl.Color(114, 255, 92, 89),    # HSLF(112/360, 1.0, 0.68, 0.35)
  rl.Color(114, 255, 92, 0),     # HSLF(112/360, 1.0, 0.68, 0.0)
]

NO_THROTTLE_COLORS = [
  rl.Color(242, 242, 242, 102), # HSLF(148/360, 0.0, 0.95, 0.4)
  rl.Color(242, 242, 242, 89),  # HSLF(112/360, 0.0, 0.95, 0.35)
  rl.Color(242, 242, 242, 0),   # HSLF(112/360, 0.0, 0.95, 0.0)
]

LANE_ACTIVE_COLOR = rl.Color(0, 255, 245, 255)
LANE_LINE_COLORS = {
  UIStatus.DISENGAGED: rl.Color(0, 0, 0, 255),
  UIStatus.OVERRIDE: rl.Color(150, 150, 150, 255),
  UIStatus.LONG_ONLY: rl.Color(150, 150, 150, 255),
  UIStatus.LAT_ONLY: LANE_ACTIVE_COLOR,
  UIStatus.ENGAGED: LANE_ACTIVE_COLOR,
}

LEAD_LINE_WIDTH_AT_10M = 300.0
# Undo the radar-frame offset applied by radard when projecting into the camera frame.
LEAD_RADAR_TO_CAMERA = 1.52
LEAD_LINE_LAYERS = {
  status: tuple((height, rl.Color(base.r, base.g, base.b, alpha))
                for height, alpha in (((4, 255),) if status in (UIStatus.DISENGAGED, UIStatus.OVERRIDE) else
                                      ((48, 8), (36, 15), (26, 28), (18, 48), (10, 100), (4, 255))))
  for status, base in LANE_LINE_COLORS.items()
}


def lead_line_width(distance: float, screen_width: float) -> float:
  width = LEAD_LINE_WIDTH_AT_10M * (10.0 / max(distance, 1.0)) ** 0.5
  return min(width / 2160, 0.32) * screen_width


@dataclass
class ModelPoints:
  raw_points: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float32))
  projected_points: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.float32))
  glow_points: tuple = ()


@dataclass
class LeadVehicle:
  center: tuple[float, float] | None = None
  radius: float = 0.0
  sz: float = 0.0
  fill_alpha: int = 0
  width: float = 0.0


@dataclass
class VisionDot:
  x: float
  y: float
  tx: float
  ty: float
  radius: float
  tradius: float
  alpha: float = 0.0
  talpha: float = 1.0
  rgb: tuple[int, int, int] | None = None


_VD_EASE = 0.4


class ModelRenderer(ModelRendererHelpers, Widget):
  _lead_vehicle_type = LeadVehicle

  def __init__(self):
    Widget.__init__(self)
    self.chevron_metrics = ChevronMetrics()
    self._lead_orb = gui_app.texture("icons/lead_orb.png", 256, 256)
    self._longitudinal_control = False
    self._experimental_mode = False
    self._blend_filter = FirstOrderFilter(1.0, 0.25, 1 / gui_app.target_fps)
    self._prev_allow_throttle = True
    self._lane_line_probs = np.zeros(4, dtype=np.float32)
    self._road_edge_stds = np.zeros(2, dtype=np.float32)
    self._lead_vehicles = [LeadVehicle(), LeadVehicle()]
    self._track_dots: list[LeadVehicle] = []
    self._vision_dots: list[VisionDot] = []
    self._vt_frame = -1
    self._frame_transform: np.ndarray | None = None
    self._frame_transform_wide = False
    self._path_offset_z = HEIGHT_INIT[0]
    self._set_render_preferences(ui_state.render_preferences)
    # Initialize ModelPoints objects
    self._path = ModelPoints()
    self._lane_lines = [ModelPoints() for _ in range(4)]
    self._road_edges = [ModelPoints() for _ in range(2)]
    self._backup_path = ModelPoints()
    self._backup_alpha = FirstOrderFilter(0.0, 0.25, 1 / gui_app.target_fps)
    self._acceleration_x = np.empty((0,), dtype=np.float32)

    # Transform matrix (3x3 for car space to screen space)
    self._car_space_transform = np.zeros((3, 3), dtype=np.float32)
    self._transform_dirty = True
    self._clip_region = None

    self._exp_gradient = Gradient(
      start=(0.0, 1.0),  # Bottom of path
      end=(0.0, 0.0),  # Top of path
      colors=[],
      stops=[],
    )

    # Get longitudinal control setting from car parameters
    if (cp := log_param_from_bytes(Params(), "CarParams", car.CarParams)) is not None:
      self._longitudinal_control = cp.openpilotLongitudinalControl

  def set_transform(self, transform: np.ndarray):
    self._car_space_transform = transform.astype(np.float32)
    self._transform_dirty = True

  def set_frame_transform(self, transform: np.ndarray, is_wide: bool):
    self._frame_transform = transform.astype(np.float32)
    self._frame_transform_wide = is_wide

  def _render(self, rect: rl.Rectangle):
    sm = ui_state.sm

    # Check if data is up-to-date
    if (sm.recv_frame["extrinsicsCalibration"] < ui_state.started_frame or
        sm.recv_frame["modelV2"] < ui_state.started_frame):
      return

    # Set up clipping region
    self._clip_region = rl.Rectangle(
      rect.x - CLIP_MARGIN, rect.y - CLIP_MARGIN, rect.width + 2 * CLIP_MARGIN, rect.height + 2 * CLIP_MARGIN
    )

    # Update state
    self._experimental_mode = sm['selfdriveState'].experimentalMode

    live_calib = sm['extrinsicsCalibration']
    self._path_offset_z = get_render_path_height(live_calib)

    if sm.updated['carParams']:
      self._longitudinal_control = sm['carParams'].openpilotLongitudinalControl

    model = sm['modelV2']
    radar_state = sm['radarState'] if sm.valid['radarState'] else None
    lead_one = radar_state.leadOne if radar_state else None
    render_lead_indicator = radar_state is not None

    # Update model data when needed
    preferences_changed = self._set_render_preferences(ui_state.render_preferences)
    model_updated = sm.updated['modelV2'] or preferences_changed
    if model_updated or sm.updated['radarState'] or self._transform_dirty:
      if model_updated:
        self._update_raw_points(model)
        self._update_backup_raw_points(sm['iqDriveModelData'].backupPath)

      path_x_array = self._path.raw_points[:, 0]
      if path_x_array.size == 0:
        return

      self._update_model(lead_one, path_x_array)
      self._update_backup_path()
      if render_lead_indicator:
        self._update_leads(radar_state, path_x_array)
        if self._ambient_dots:
          self._update_track_dots(sm, radar_state, path_x_array)
        else:
          self._track_dots = []
      self._transform_dirty = False

    # Draw elements
    self._draw_lane_lines()
    self._draw_backup_path()
    self._draw_path(sm)
    self._draw_stop_point(model)

    if self._ambient_dots:
      self._update_vision_dots(sm)
      self._draw_vision_dots()

    if render_lead_indicator and radar_state:
      if self._ambient_dots:
        self._draw_track_dots()
      self._draw_lead_indicator()
      self.chevron_metrics.draw_lead_status(sm, radar_state, self._rect, self._lead_vehicles)

  def _update_track_dots(self, sm, radar_state, path_x_array):
    self._track_dots = []
    if not sm.valid['radarTracks']:
      return

    lead_track_ids = {lead.radarTrackId for lead in (radar_state.leadOne, radar_state.leadTwo)
                      if lead.status and lead.radarTrackId >= 0}

    for pt in sm['radarTracks'].points:
      if pt.trackId in lead_track_ids or pt.dRel < 1.0 or abs(pt.yRel) > 10.0:
        continue

      idx = self._get_path_length_idx(path_x_array, pt.dRel)
      z = self._path.raw_points[idx, 2] if idx < len(self._path.raw_points) else 0.0
      point = self._map_to_screen(pt.dRel, -pt.yRel, z + self._path_offset_z)
      if point is None:
        continue

      sz = np.clip((25 * 30) / (pt.dRel / 3 + 30), 15.0, 30.0) * 1.1
      radius = sz * 1.1
      x, y = point
      if not (self._rect.x <= x <= self._rect.x + self._rect.width and
              self._rect.y <= y <= self._rect.y + self._rect.height):
        continue

      self._track_dots.append(LeadVehicle(center=(float(x), float(y)), radius=float(radius), sz=float(sz)))

  def _update_vision_dots(self, sm):
    # iqVehicleTracks arrives at a few Hz; the dots are eased toward the latest
    # detection every render frame so they glide instead of teleporting.
    hidden = (self._frame_transform is None or
              not sm.alive['iqVehicleTracks'] or not sm.valid['iqVehicleTracks'])
    if not hidden:
      vt = sm['iqVehicleTracks']
      hidden = bool(vt.wide) != self._frame_transform_wide or vt.frameWidth == 0 or vt.frameHeight == 0

    if hidden:
      for d in self._vision_dots:
        d.talpha = 0.0
    elif vt.frameId != self._vt_frame:
      self._vt_frame = vt.frameId
      self._retarget_vision_dots(vt)

    for d in self._vision_dots:
      d.x += (d.tx - d.x) * _VD_EASE
      d.y += (d.ty - d.y) * _VD_EASE
      d.radius += (d.tradius - d.radius) * _VD_EASE
      d.alpha += (d.talpha - d.alpha) * _VD_EASE
    self._vision_dots = [d for d in self._vision_dots if d.alpha > 0.02 or d.talpha > 0.0]

  def _retarget_vision_dots(self, vt):
    fw, fh = vt.frameWidth, vt.frameHeight
    m = self._frame_transform
    occupied = [d.center for d in self._lead_vehicles + self._track_dots if d.center is not None]

    targets = []
    for t in vt.tracks:
      is_vehicle = t.label in VEHICLE_TRACK_LABELS
      sign_color = SIGN_TRACK_COLORS.get(t.label)
      if not is_vehicle and sign_color is None:
        continue
      cx_f = (t.x1 + t.x2) / 2.0 * fw
      cy_f = (t.y1 + t.y2) / 2.0 * fh
      x = m[0, 0] * cx_f + m[0, 2]
      y = m[1, 1] * cy_f + m[1, 2]
      if not (self._rect.x <= x <= self._rect.x + self._rect.width and
              self._rect.y <= y <= self._rect.y + self._rect.height):
        continue
      box_h = (t.y2 - t.y1) * fh * m[1, 1]
      radius = float(np.clip(box_h * 0.35, 14.0, 40.0))
      if sign_color is not None:
        targets.append((x, y, min(radius, 22.0), (sign_color.r, sign_color.g, sign_color.b)))
      elif not any((x - ox) ** 2 + (y - oy) ** 2 < (radius * 2.2) ** 2 for ox, oy in occupied):
        targets.append((x, y, radius, None))

    dots = self._vision_dots
    used = [False] * len(dots)
    for tx, ty, tr, rgb in targets:
      best, best_d2 = -1, 1e18
      for i, d in enumerate(dots):
        if used[i] or (d.rgb is None) != (rgb is None):
          continue
        d2 = (d.x - tx) ** 2 + (d.y - ty) ** 2
        if d2 < best_d2:
          best, best_d2 = i, d2
      if best >= 0 and best_d2 <= (max(tr, dots[best].radius) * 3.0) ** 2:
        d = dots[best]
        used[best] = True
        d.tx, d.ty, d.tradius, d.talpha, d.rgb = tx, ty, tr, 1.0, rgb
      else:
        dots.append(VisionDot(x=tx, y=ty, tx=tx, ty=ty, radius=tr, tradius=tr, alpha=0.0, talpha=1.0, rgb=rgb))
        used.append(True)

    for i, d in enumerate(dots):
      if not used[i]:
        d.talpha = 0.0

  def _update_model(self, lead, path_x_array):
    """Update model visualization data based on model message"""
    max_distance = np.clip(path_x_array[-1], MIN_DRAW_DISTANCE, MAX_DRAW_DISTANCE)
    max_idx = self._get_path_length_idx(self._lane_lines[0].raw_points[:, 0], max_distance)

    # Update lane lines using raw points
    for i, lane_line in enumerate(self._lane_lines):
      lane_line.projected_points = self._map_line_to_polygon(
        lane_line.raw_points, (0.16 if i in (1, 2) else 0.12) * self._lane_line_probs[i], 0.0, max_idx, max_distance
      )
      points = lane_line.projected_points
      half = len(points) // 2
      lane_line.glow_points = ()
      if half and len(points) == 2 * half:
        center = (points[:half] + points[half:][::-1]) * 0.5
        center = np.concatenate((center, center[::-1]))
        lane_line.glow_points = tuple(center + (points - center) * scale for scale in (2.2, 1.5))

    # Update road edges using raw points
    for road_edge in self._road_edges:
      road_edge.projected_points = self._map_line_to_polygon(road_edge.raw_points, 0.025, 0.0, max_idx, max_distance)

    # Update path using raw points
    if lead and lead.status:
      lead_d = lead.dRel * 2.0
      max_distance = np.clip(lead_d - min(lead_d * 0.35, 10.0), 0.0, max_distance)

    max_idx = self._get_path_length_idx(path_x_array, max_distance)
    self._path.projected_points = self._map_line_to_polygon(
      self._path.raw_points, 0.9, self._path_offset_z, max_idx, max_distance, allow_invert=False
    )

    self._update_experimental_gradient()

  def _update_backup_path(self):
    backup_x = self._backup_path.raw_points[:, 0]
    if backup_x.size == 0:
      return
    max_distance = np.clip(backup_x[-1], MIN_DRAW_DISTANCE, MAX_DRAW_DISTANCE)
    self._backup_path.projected_points = self._map_line_to_polygon(
      self._backup_path.raw_points, BACKUP_PATH_WIDTH, self._path_offset_z,
      self._get_path_length_idx(backup_x, max_distance), max_distance, allow_invert=False
    )

  def _update_experimental_gradient(self):
    """Pre-calculate experimental mode gradient colors"""
    if not self._experimental_mode:
      return

    max_len = min(len(self._path.projected_points) // 2, len(self._acceleration_x))

    segment_colors = []
    gradient_stops = []

    i = 0
    while i < max_len:
      # Some points (screen space) are out of frame (rect space)
      track_y = self._path.projected_points[i][1]
      if track_y < self._rect.y or track_y > (self._rect.y + self._rect.height):
        i += 1
        continue

      # Calculate color based on acceleration (0 is bottom, 1 is top)
      lin_grad_point = 1 - (track_y - self._rect.y) / self._rect.height

      # speed up: 120, slow down: 0
      path_hue = np.clip(60 + self._acceleration_x[i] * 35, 0, 120)

      saturation = min(abs(self._acceleration_x[i] * 1.5), 1)
      lightness = np.interp(saturation, [0.0, 1.0], [0.95, 0.62])
      alpha = np.interp(lin_grad_point, [0.75 / 2.0, 0.75], [0.4, 0.0])

      # Use HSL to RGB conversion
      color = self._hsla_to_color(path_hue / 360.0, saturation, lightness, alpha)

      gradient_stops.append(lin_grad_point)
      segment_colors.append(color)

      # Skip a point, unless next is last
      i += 1 + (1 if (i + 2) < max_len else 0)

    # Store the gradient in the path object
    self._exp_gradient = Gradient(
      start=(0.0, 1.0),  # Bottom of path
      end=(0.0, 0.0),  # Top of path
      colors=segment_colors,
      stops=gradient_stops,
    )

  def _update_leads(self, radar_state, path_x_array):
    self._lead_vehicles = [LeadVehicle(), LeadVehicle()]
    for i, lead in enumerate((radar_state.leadOne, radar_state.leadTwo)):
      if not lead or not lead.status:
        continue
      distance = lead.dRel + LEAD_RADAR_TO_CAMERA
      points = self._path.raw_points
      z = float(np.interp(distance, points[:, 0], points[:, 2])) if len(points) else 0.0
      point = self._map_to_screen(distance, -lead.yRel, z + self._path_offset_z)
      if point is not None:
        self._lead_vehicles[i] = self._update_lead_vehicle(lead.dRel, lead.vRel, point, self._rect)

  def _update_lead_vehicle(self, d_rel, v_rel, point, rect):
    width = lead_line_width(d_rel, rect.width)
    x = max(rect.x + width / 2, min(point[0], rect.x + rect.width - width / 2))
    y = max(rect.y + 24, min(point[1], rect.y + rect.height - 24))
    return LeadVehicle(center=(float(x), float(y)), width=width)

  def _get_lane_line_color(self, probability: float, ego_lane: bool = True) -> rl.Color:
    status = ui_state.status if ego_lane or ui_state.status == UIStatus.DISENGAGED else UIStatus.OVERRIDE
    base = LANE_LINE_COLORS.get(status, LANE_LINE_COLORS[UIStatus.DISENGAGED])
    return rl.Color(base.r, base.g, base.b, int(max(0.0, min(probability, 0.7)) * 255))

  def _draw_lane_lines(self):
    """Draw lane lines and road edges"""
    for i, lane_line in enumerate(self._lane_lines):
      if lane_line.projected_points.size == 0:
        continue

      ego_lane = i in (1, 2)
      color = self._get_lane_line_color(float(self._lane_line_probs[i]), ego_lane)
      if ego_lane and ui_state.status != UIStatus.DISENGAGED:
        for points, opacity in zip(lane_line.glow_points, (0.10, 0.18), strict=False):
          draw_polygon(self._rect, points, rl.Color(color.r, color.g, color.b, int(color.a * opacity)))
      draw_polygon(self._rect, lane_line.projected_points, color)

    for i, road_edge in enumerate(self._road_edges):
      if road_edge.projected_points.size == 0:
        continue

      draw_polygon(self._rect, road_edge.projected_points,
                   self._get_lane_line_color(float(1.0 - self._road_edge_stds[i]), ego_lane=False))

  def _draw_path(self, sm):
    """Draw path with dynamic coloring based on mode and throttle state."""
    if not self._path.projected_points.size:
      return

    allow_throttle = sm['longitudinalPlan'].allowThrottle or not self._longitudinal_control
    self._blend_filter.update(int(allow_throttle))


    if self._experimental_mode:
      # Draw with acceleration coloring
      if len(self._exp_gradient.colors) > 1:
        draw_polygon(self._rect, self._path.projected_points, gradient=self._exp_gradient)
      else:
        draw_polygon(self._rect, self._path.projected_points, rl.Color(255, 255, 255, 30))
    else:
      # Blend throttle/no throttle colors based on transition
      blend_factor = round(self._blend_filter.x * 100) / 100
      blended_colors = self._blend_colors(NO_THROTTLE_COLORS, THROTTLE_COLORS, blend_factor)
      gradient = Gradient(
        start=(0.0, 1.0),  # Bottom of path
        end=(0.0, 0.0),  # Top of path
        colors=blended_colors,
        stops=[0.0, 0.5, 1.0],
      )
      draw_polygon(self._rect, self._path.projected_points, gradient=gradient)

  # concentric layers (outer faint -> inner bright) build a soft center-out glow using only
  # draw_circle, which is signature-stable across raylib versions (draw_circle_gradient is not)
  _VD_GLOW = ((1.0, 26), (0.72, 38), (0.48, 54), (0.26, 78))

  def _draw_vision_dots(self):
    for dot in self._vision_dots:
      a = dot.alpha
      if a <= 0.02:
        continue
      x, y = int(dot.x), int(dot.y)
      cr, cg, cb = (40, 210, 200) if dot.rgb is None else dot.rgb
      for frac, base in self._VD_GLOW:
        rl.draw_circle(x, y, dot.radius * frac, rl.Color(cr, cg, cb, int(base * a)))

  def _draw_track_dots(self):
    src = rl.Rectangle(0, 0, self._lead_orb.width, self._lead_orb.height)
    for dot in self._track_dots:
      cx, cy = dot.center
      r = dot.radius
      dest = rl.Rectangle(cx, cy, r * 2.0, r * 2.0)
      rl.draw_texture_pro(self._lead_orb, src, dest, rl.Vector2(r, r), 0.0, rl.Color(255, 255, 255, 90))

  def _draw_stop_point(self, model):
    stop_point = getattr(model, 'stopPoint', None)
    if stop_point is None or not stop_point.valid or stop_point.probability < STOP_POINT_MIN_PROBABILITY:
      return

    points = self._path.raw_points
    if points.shape[0] < 2:
      return

    distance = float(stop_point.distance)
    # the plan collapses to a couple of metres once the car is stopped, so gating on its reach
    # hides the stop exactly when the model is surest of it; np.interp clamps past the last point
    if not 0.0 < distance <= MAX_DRAW_DISTANCE:
      return

    # np.interp needs an increasing x, and once the car halts the plan collapses and its lateral
    # values run to tens of metres; both would place the stop well off to the side of the road
    forward = points[:, 0]
    if np.all(np.diff(forward) > 0):
      centre_y = float(np.interp(distance, forward, points[:, 1]))
      centre_z = float(np.interp(distance, forward, points[:, 2]))
    else:
      centre_y, centre_z = 0.0, float(points[0, 2])
    centre_y = float(np.clip(centre_y, -STOP_POINT_MAX_OFFSET, STOP_POINT_MAX_OFFSET))
    centre_z += self._path_offset_z
    near, far = distance, distance + STOP_POINT_DEPTH
    corners = [self._map_to_screen(x, centre_y + offset, centre_z)
               for x, offset in ((near, -STOP_POINT_HALF_WIDTH), (near, STOP_POINT_HALF_WIDTH),
                                 (far, STOP_POINT_HALF_WIDTH), (far, -STOP_POINT_HALF_WIDTH))]
    if any(corner is None for corner in corners):
      return

    alpha = int(np.clip(stop_point.probability, 0.0, 1.0) * 200)
    draw_polygon(self._rect, np.array(corners, dtype=np.float32), rl.Color(*STOP_POINT_COLOR, alpha))

  def _draw_lead_indicator(self):
    layers = LEAD_LINE_LAYERS.get(ui_state.status, LEAD_LINE_LAYERS[UIStatus.DISENGAGED])
    for lead in self._lead_vehicles:
      if lead.center is None:
        continue
      cx, cy = lead.center
      for height, color in layers:
        rl.draw_rectangle_rounded(rl.Rectangle(cx - lead.width / 2, cy - height / 2, lead.width, height),
                                   1.0, 8, color)

  @staticmethod
  def _get_path_length_idx(pos_x_array: np.ndarray, path_distance: float) -> int:
    """Get the index corresponding to the given path distance"""
    if len(pos_x_array) == 0:
      return 0
    indices = np.where(pos_x_array <= path_distance)[0]
    return indices[-1] if indices.size > 0 else 0

  def _map_line_to_polygon(self, line: np.ndarray, y_off: float, z_off: float, max_idx: int, max_distance: float, allow_invert: bool = True) -> np.ndarray:
    """Convert 3D line to 2D polygon for rendering."""
    if line.shape[0] == 0:
      return np.empty((0, 2), dtype=np.float32)

    # Slice points and filter non-negative x-coordinates
    points = line[:max_idx + 1]

    # Interpolate around max_idx so path end is smooth. Only when max_distance genuinely
    if 0 < max_idx < line.shape[0] - 1:
      p0 = line[max_idx]
      p1 = line[max_idx + 1]
      x0, x1 = p0[0], p1[0]
      if x0 <= max_distance <= x1:
        interp_y = np.interp(max_distance, [x0, x1], [p0[1], p1[1]])
        interp_z = np.interp(max_distance, [x0, x1], [p0[2], p1[2]])
        interp_point = np.array([max_distance, interp_y, interp_z], dtype=points.dtype)
        points = np.concatenate((points, interp_point[None, :]), axis=0)

    points = points[points[:, 0] >= 0.5]
    if points.shape[0] == 0:
      return np.empty((0, 2), dtype=np.float32)

    N = points.shape[0]
    # Generate left and right 3D points in one array using broadcasting
    offsets = np.array([[0, -y_off, z_off], [0, y_off, z_off]], dtype=np.float32)
    points_3d = points[None, :, :] + offsets[:, None, :]  # Shape: 2xNx3
    points_3d = points_3d.reshape(2 * N, 3)  # Shape: (2*N)x3

    # Transform all points to projected space in one operation
    proj = self._project_car_points(points_3d)  # Shape: 3x(2*N)
    proj = proj.reshape(3, 2, N)
    left_proj = proj[:, 0, :]
    right_proj = proj[:, 1, :]

    # Filter points where z is sufficiently large
    valid_proj = (np.abs(left_proj[2]) >= 1e-6) & (np.abs(right_proj[2]) >= 1e-6)
    if not np.any(valid_proj):
      return np.empty((0, 2), dtype=np.float32)

    # Compute screen coordinates
    left_screen = left_proj[:2, valid_proj] / left_proj[2, valid_proj][None, :]
    right_screen = right_proj[:2, valid_proj] / right_proj[2, valid_proj][None, :]

    # Define clip region bounds
    clip = self._clip_region
    x_min, x_max = clip.x, clip.x + clip.width
    y_min, y_max = clip.y, clip.y + clip.height

    # Filter points within clip region
    left_in_clip = (
      (left_screen[0] >= x_min) & (left_screen[0] <= x_max) &
      (left_screen[1] >= y_min) & (left_screen[1] <= y_max)
    )
    right_in_clip = (
      (right_screen[0] >= x_min) & (right_screen[0] <= x_max) &
      (right_screen[1] >= y_min) & (right_screen[1] <= y_max)
    )
    both_in_clip = left_in_clip & right_in_clip

    if not np.any(both_in_clip):
      return np.empty((0, 2), dtype=np.float32)

    # Select valid and clipped points
    left_screen = left_screen[:, both_in_clip]
    right_screen = right_screen[:, both_in_clip]

    # Handle Y-coordinate inversion on hills
    if not allow_invert and left_screen.shape[1] > 1:
      y = left_screen[1, :]  # y-coordinates
      keep = y == np.minimum.accumulate(y)
      if not np.any(keep):
        return np.empty((0, 2), dtype=np.float32)
      left_screen = left_screen[:, keep]
      right_screen = right_screen[:, keep]

    return np.vstack((left_screen.T, right_screen[:, ::-1].T)).astype(np.float32)
