import numpy as np
import pyray as rl
from iqpilot.cereal import car
from dataclasses import dataclass, field
from iqpilot.common.params import Params
from iqpilot.common.filter_simple import FirstOrderFilter
from iqpilot.selfdrive.locationd.calibrationd import HEIGHT_INIT
from iqpilot.ui.onroad.hud_overlays import ChevronMetrics
from iqpilot.ui.onroad.lead_confidence import driving_confidence
from iqpilot.selfdrive.locationd.calibration_helpers import get_render_path_height
from iqpilot.selfdrive.ui.ui_state import ui_state, UIStatus, log_param_from_bytes
from iqpilot.selfdrive.ui.mici.onroad import blend_colors
from iqpilot.system.ui.lib.application import gui_app
from iqpilot.system.ui.lib.shader_polygon import draw_polygon, Gradient
from iqpilot.system.ui.widgets import Widget
from iqpilot.ui.onroad.model_renderer_helpers import BACKUP_PATH_WIDTH, ModelRendererHelpers
from iqpilot.ui.theme import NeonTheme

CLIP_MARGIN = 500
MIN_DRAW_DISTANCE = 10.0
MAX_DRAW_DISTANCE = 100.0

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

LANE_LINE_COLORS = {
  UIStatus.DISENGAGED: rl.Color(200, 200, 200, 255),
  UIStatus.OVERRIDE: rl.Color(255, 255, 255, 255),
  UIStatus.ENGAGED: rl.Color(0, 255, 64, 255),
}


@dataclass
class ModelPoints:
  raw_points: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float32))
  projected_points: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.float32))


@dataclass
class LeadVehicle:
  center: tuple[float, float] | None = None
  radius: float = 0.0
  sz: float = 0.0
  fill_alpha: int = 0


class ModelRenderer(ModelRendererHelpers, Widget):
  _lead_vehicle_type = LeadVehicle

  def __init__(self):
    super().__init__()
    self.chevron_metrics = ChevronMetrics()
    self._lead_orb = gui_app.texture("icons/lead_orb.png", 256, 256)
    self._longitudinal_control = False
    self._experimental_mode = False
    self._blend_filter = FirstOrderFilter(1.0, 0.25, 1 / gui_app.target_fps)
    self._prev_allow_throttle = True
    self._lane_line_probs = np.zeros(4, dtype=np.float32)
    self._road_edge_stds = np.zeros(2, dtype=np.float32)
    self._lead_vehicles = [LeadVehicle(), LeadVehicle()]
    self._path_offset_z = HEIGHT_INIT[0]

    # Initialize ModelPoints objects
    self._path = ModelPoints()
    self._lane_lines = [ModelPoints() for _ in range(4)]
    self._road_edges = [ModelPoints() for _ in range(2)]
    self._backup_path = ModelPoints()
    self._backup_alpha = FirstOrderFilter(0.0, 0.25, 1 / gui_app.target_fps)
    self._acceleration_x = np.empty((0,), dtype=np.float32)

    self._acceleration_x_filter = FirstOrderFilter(0.0, 0.1, 1 / gui_app.target_fps)
    self._acceleration_x_filter2 = FirstOrderFilter(0.0, 1, 1 / gui_app.target_fps)

    self._torque_filter = FirstOrderFilter(0, 0.1, 1 / gui_app.target_fps)
    self._ll_color_filter = FirstOrderFilter(0.0, 0.1, 1 / gui_app.target_fps)

    # Transform matrix (3x3 for car space to screen space)
    self._car_space_transform = np.zeros((3, 3), dtype=np.float32)
    self._transform_dirty = True
    self._clip_region = None

    self._set_render_preferences(ui_state.render_preferences)

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

  def _render(self, rect: rl.Rectangle):
    sm = ui_state.sm
    driving_confidence.update()

    self._torque_filter.update(-ui_state.sm['carOutput'].actuatorsOutput.torque)

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
    render_lead_indicator = self._longitudinal_control and radar_state is not None

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
      self._transform_dirty = False

    # Draw elements (hide when disengaged)
    if ui_state.status != UIStatus.DISENGAGED:
      self._draw_lane_lines()
      self._draw_backup_path((self._rect.x, self._rect.y))
      self._draw_path(sm)

    if render_lead_indicator and radar_state:
      self._draw_lead_indicator()
      self.chevron_metrics.draw_lead_status(sm, radar_state, self._rect, self._lead_vehicles)

  def _update_model(self, lead, path_x_array):
    """Update model visualization data based on model message"""
    max_distance = np.clip(path_x_array[-1], MIN_DRAW_DISTANCE, MAX_DRAW_DISTANCE)
    max_idx = self._get_path_length_idx(self._lane_lines[0].raw_points[:, 0], max_distance)

    # Update lane lines using raw points
    line_width_factor = 0.12
    for i, lane_line in enumerate(self._lane_lines):
      if i in (1, 2):
        line_width_factor = 0.16
      lane_line.projected_points = self._map_line_to_polygon(
        lane_line.raw_points, line_width_factor * self._lane_line_probs[i], 0.0, max_idx
      )

    # Update road edges using raw points
    for road_edge in self._road_edges:
      road_edge.projected_points = self._map_line_to_polygon(road_edge.raw_points, line_width_factor, 0.0, max_idx)

    # Update path using raw points
    if lead and lead.status:
      lead_d = lead.dRel * 2.0
      max_distance = np.clip(lead_d - min(lead_d * 0.35, 10.0), 0.0, max_distance)

    soon_acceleration = self._acceleration_x[len(self._acceleration_x) // 4] if len(self._acceleration_x) > 0 else 0
    self._acceleration_x_filter.update(soon_acceleration)
    self._acceleration_x_filter2.update(soon_acceleration)

    # make path width wider/thinner when initially braking/accelerating
    if self._experimental_mode and False:
      high_pass_acceleration = self._acceleration_x_filter.x - self._acceleration_x_filter2.x
      y_off = np.interp(high_pass_acceleration, [-1, 0, 1], [0.9 * 2, 0.9, 0.9 / 2])
    else:
      y_off = 0.9

    max_idx = self._get_path_length_idx(path_x_array, max_distance)
    self._path.projected_points = self._map_line_to_polygon(
      self._path.raw_points, y_off, self._path_offset_z, max_idx, allow_invert=False
    )

    self._update_experimental_gradient()

  def _update_backup_path(self):
    backup_x = self._backup_path.raw_points[:, 0]
    if backup_x.size == 0:
      return
    max_distance = np.clip(backup_x[-1], MIN_DRAW_DISTANCE, MAX_DRAW_DISTANCE)
    self._backup_path.projected_points = self._map_line_to_polygon(
      self._backup_path.raw_points, BACKUP_PATH_WIDTH, self._path_offset_z,
      self._get_path_length_idx(backup_x, max_distance), allow_invert=False
    )

  def _update_experimental_gradient(self):
    """Pre-calculate experimental mode gradient colors"""
    if not self._experimental_mode:
      return

    # reconstruct absolute (screen) points so the rect-space cull below stays correct
    path_pts = self._path.projected_points + np.array([self._rect.x, self._rect.y], dtype=np.float32)
    max_len = min(len(path_pts) // 2, len(self._acceleration_x))

    segment_colors = []
    gradient_stops = []

    i = 0
    while i < max_len:
      # Some points (screen space) are out of frame (rect space)
      track_y = path_pts[i][1]
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
    self._exp_gradient.colors = segment_colors
    self._exp_gradient.stops = gradient_stops

  def _update_lead_vehicle(self, d_rel, v_rel, point, rect):
    speed_buff, lead_buff = 10.0, 40.0

    # Calculate fill alpha
    fill_alpha = 0
    if d_rel < lead_buff:
      fill_alpha = 255 * (1.0 - (d_rel / lead_buff))
      if v_rel < 0:
        fill_alpha += 255 * (-1 * (v_rel / speed_buff))
      fill_alpha = min(fill_alpha, 255)

    # Calculate size and position. Distance-scaled orb radius (closer lead -> bigger orb).
    sz = np.clip((25 * 30) / (d_rel / 3 + 30), 15.0, 30.0) * 1
    radius = sz * 1.1
    # point is in absolute screen coords; clamp against the rect's absolute bounds so the orb stays
    # fully on-screen (rect-relative bounds mis-placed it when the camera pane is offset, e.g. split nav)
    x = np.clip(point[0], rect.x + radius, rect.x + rect.width - radius)
    y = np.clip(point[1], rect.y + radius, rect.y + rect.height - radius)

    return LeadVehicle(center=(float(x), float(y)), radius=float(radius), sz=float(sz), fill_alpha=int(fill_alpha))

  def _get_ll_color(self, prob: float, adjacent: bool, left: bool):
    alpha = np.clip(prob, 0.0, 0.7)
    if adjacent:
      if gui_app.iqpilot_ui() and ui_state.status == UIStatus.ENGAGED:
        _base_color = NeonTheme.glow()
      else:
        _base_color = LANE_LINE_COLORS.get(ui_state.status, LANE_LINE_COLORS[UIStatus.DISENGAGED])
      color = rl.Color(_base_color.r, _base_color.g, _base_color.b, int(alpha * 255))

      # turn adjacent lls orange if torque is high
      torque = self._torque_filter.x
      high_torque = abs(torque) > 0.6
      if high_torque and (left == (torque > 0)):
        color = blend_colors(
          color,
          rl.Color(255, 115, 0, int(alpha * 255)),  # orange
          np.interp(abs(torque), [0.6, 0.8], [0.0, 1.0])
        )
    else:
      color = rl.Color(255, 255, 255, int(alpha * 255))

    if ui_state.status == UIStatus.DISENGAGED:
      color = rl.Color(0, 0, 0, int(alpha * 255))

    return color

  def _draw_lane_lines(self):
    """Draw lane lines and road edges"""
    """Two closest lines should be green (lane line or road edges)"""
    # projected_points are origin-relative (rect.x/y kept out of the transform so it stays cached);
    # translate to the view's screen position here.
    offset = np.array([self._rect.x, self._rect.y], dtype=np.float32)
    for i, lane_line in enumerate(self._lane_lines):
      if lane_line.projected_points.size == 0:
        continue

      color = self._get_ll_color(float(self._lane_line_probs[i]), i in (1, 2), i in (0, 1))
      draw_polygon(self._rect, lane_line.projected_points + offset, color)

    for i, road_edge in enumerate(self._road_edges):
      if road_edge.projected_points.size == 0:
        continue

      # if closest lane lines are not confident, make road edges green
      color = self._get_ll_color(float(1.0 - self._road_edge_stds[i]), float(self._lane_line_probs[i + 1]) < 0.25, i == 0)
      draw_polygon(self._rect, road_edge.projected_points + offset, color)

  def _draw_path(self, sm):
    """Draw path with dynamic coloring based on mode and throttle state."""
    if not self._path.projected_points.size:
      return

    # projected_points are origin-relative; translate to the view's screen position
    path_pts = self._path.projected_points + np.array([self._rect.x, self._rect.y], dtype=np.float32)

    allow_throttle = sm['longitudinalPlan'].allowThrottle or not self._longitudinal_control
    self._blend_filter.update(int(allow_throttle))

    if self._experimental_mode:
      # Draw with acceleration coloring
      if ui_state.status == UIStatus.DISENGAGED:
        draw_polygon(self._rect, path_pts, rl.Color(0, 0, 0, 90))
      elif len(self._exp_gradient.colors) > 1:
        draw_polygon(self._rect, path_pts, gradient=self._exp_gradient)
      else:
        draw_polygon(self._rect, path_pts, rl.Color(255, 255, 255, 30))
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

      if ui_state.status == UIStatus.DISENGAGED:
        draw_polygon(self._rect, path_pts, rl.Color(0, 0, 0, 90))
      else:
        draw_polygon(self._rect, path_pts, gradient=gradient)

  def _draw_lead_indicator(self):
    tint, _ = driving_confidence.colors()
    src = rl.Rectangle(0, 0, self._lead_orb.width, self._lead_orb.height)
    for lead in self._lead_vehicles:
      if lead.center is None:
        continue
      cx, cy = lead.center
      r = lead.radius
      alpha = int(np.clip(140 + 115 * (lead.fill_alpha / 255.0), 0, 255))
      dest = rl.Rectangle(cx, cy, r * 2.0, r * 2.0)
      rl.draw_texture_pro(self._lead_orb, src, dest, rl.Vector2(r, r), 0.0, rl.Color(tint.r, tint.g, tint.b, alpha))

  @staticmethod
  def _get_path_length_idx(pos_x_array: np.ndarray, path_height: float) -> int:
    """Get the index corresponding to the given path height"""
    if len(pos_x_array) == 0:
      return 0
    indices = np.where(pos_x_array <= path_height)[0]
    return indices[-1] if indices.size > 0 else 0

  def _map_line_to_polygon(self, line: np.ndarray, y_off: float, z_off: float, max_idx: int, allow_invert: bool = True) -> np.ndarray:
    """Convert 3D line to 2D polygon for rendering."""
    if line.shape[0] == 0:
      return np.empty((0, 2), dtype=np.float32)

    # Slice points and filter non-negative x-coordinates
    points = line[:max_idx + 1]
    points = points[points[:, 0] >= 0]
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
