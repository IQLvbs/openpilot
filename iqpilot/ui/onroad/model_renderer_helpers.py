# Copyright (c) 2026 IQ.Lvbs. All rights reserved.
import colorsys
import numpy as np
import pyray as rl

from iqpilot.system.ui.lib.shader_polygon import Gradient, draw_polygon

BACKUP_PATH_COLORS = (
  rl.Color(170, 170, 170, 72),
  rl.Color(170, 170, 170, 56),
  rl.Color(170, 170, 170, 0),
)
BACKUP_PATH_WIDTH = 0.5


class ModelRendererHelpers:
  def _set_render_preferences(self, preferences):
    changed = getattr(self, '_render_preferences', None) != preferences
    self._render_preferences = preferences
    self._camera_offset = preferences.camera_offset
    self._ambient_dots = preferences.ambient_tracks
    return changed

  def _update_raw_points(self, model):
    groups = (([self._path], [model.position]), (self._lane_lines, model.laneLines), (self._road_edges, model.roadEdges))
    for targets, lines in groups:
      for target, line in zip(targets, lines, strict=False):
        target.raw_points = np.column_stack((line.x, line.y, line.z)).astype(np.float32)

    self._lane_line_probs = np.array(model.laneLineProbs, dtype=np.float32)
    self._road_edge_stds = np.array(model.roadEdgeStds, dtype=np.float32)
    self._acceleration_x = np.array(model.acceleration.x, dtype=np.float32)

  def _update_backup_raw_points(self, backup_path):
    self._backup_path.raw_points = np.column_stack((backup_path.x, backup_path.y, backup_path.z)).astype(np.float32).reshape(-1, 3)

  def _draw_backup_path(self, origin: tuple[float, float] = (0.0, 0.0)):
    alpha = self._backup_alpha.update(float(self._backup_path.raw_points.shape[0] > 0))
    if alpha < 0.01 or not self._backup_path.projected_points.size:
      return
    colors = [rl.Color(c.r, c.g, c.b, int(c.a * alpha)) for c in BACKUP_PATH_COLORS]
    gradient = Gradient(start=(0.0, 1.0), end=(0.0, 0.0), colors=colors, stops=[0.0, 0.5, 1.0])
    draw_polygon(self._rect, self._backup_path.projected_points + np.array(origin, dtype=np.float32), gradient=gradient)

  def _update_leads(self, radar_state, path_x_array):
    self._lead_vehicles = [self._lead_vehicle_type(), self._lead_vehicle_type()]
    leads = [radar_state.leadOne, radar_state.leadTwo]

    for i, lead_data in enumerate(leads):
      if lead_data and lead_data.status:
        d_rel, y_rel, v_rel = lead_data.dRel, lead_data.yRel, lead_data.vRel
        idx = self._get_path_length_idx(path_x_array, d_rel)

        z = self._path.raw_points[idx, 2] if idx < len(self._path.raw_points) else 0.0
        point = self._map_to_screen(d_rel, -y_rel, z + self._path_offset_z)
        if point:
          self._lead_vehicles[i] = self._update_lead_vehicle(d_rel, v_rel, point, self._rect)

  def _project_car_points(self, points: np.ndarray) -> np.ndarray:
    projected = self._car_space_transform @ points.T
    camera_translation = self._car_space_transform[:, 1:2] * self._camera_offset
    return projected + camera_translation

  def _map_to_screen(self, in_x, in_y, in_z):
    pt = self._project_car_points(np.array([[in_x, in_y, in_z]]))[:, 0]

    if abs(pt[2]) < 1e-6:
      return None

    x, y = pt[0] / pt[2], pt[1] / pt[2]

    clip = self._clip_region
    if not (clip.x <= x <= clip.x + clip.width and clip.y <= y <= clip.y + clip.height):
      return None

    return (x, y)

  @staticmethod
  def _hsla_to_color(h, s, l, a):
    rgb = colorsys.hls_to_rgb(h, l, s)
    return rl.Color(
      int(rgb[0] * 255),
      int(rgb[1] * 255),
      int(rgb[2] * 255),
      int(a * 255)
    )

  @staticmethod
  def _blend_colors(begin_colors, end_colors, t):
    if t >= 1.0:
      return end_colors
    if t <= 0.0:
      return begin_colors

    inv_t = 1.0 - t
    return [rl.Color(
      int(inv_t * start.r + t * end.r),
      int(inv_t * start.g + t * end.g),
      int(inv_t * start.b + t * end.b),
      int(inv_t * start.a + t * end.a)
    ) for start, end in zip(begin_colors, end_colors, strict=True)]
