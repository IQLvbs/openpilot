"""Big UI steering arc."""
import numpy as np
import pyray as rl

from iqpilot.selfdrive.ui.mici.onroad import blend_colors
from iqpilot.selfdrive.ui.mici.onroad.torque_bar import TorqueBar as BaseTorqueBar, arc_bar_pts, TORQUE_ANGLE_SPAN, TORQUE_REST_HEIGHT
from iqpilot.selfdrive.ui.ui_state import ui_state, UIStatus
from iqpilot.system.ui.lib.shader_polygon import draw_polygon, Gradient

TORQUE_REST_OFFSET = 16


class TorqueBar(BaseTorqueBar):
  @staticmethod
  def resting_bottom(rect: rl.Rectangle, scale: float = 1.0) -> float:
    return rect.y + rect.height - TORQUE_REST_OFFSET * scale

  def _render(self, rect: rl.Rectangle) -> None:
    torque_line_offset = np.interp(abs(self._torque_filter.x), [0.5, 1], [TORQUE_REST_OFFSET * self._scale, (TORQUE_REST_OFFSET + 4) * self._scale])
    torque_line_height = np.interp(abs(self._torque_filter.x), [0.5, 1], [TORQUE_REST_HEIGHT * self._scale, 56 * self._scale])

    # Events and disengagement use the same filter for both opacity and arc collapse.
    active = self._demo or ui_state.status not in (UIStatus.DISENGAGED, UIStatus.LONG_ONLY)
    self._torque_line_alpha_filter.update(self._visible and active)

    torque_line_bg_alpha = np.interp(abs(self._torque_filter.x), [0.5, 1.0], [0.25, 0.5])
    visible_alpha = self._torque_line_alpha_filter.x
    torque_line_bg_color = rl.Color(255, 255, 255, int(255 * torque_line_bg_alpha * visible_alpha))
    if ui_state.status not in (UIStatus.ENGAGED, UIStatus.LAT_ONLY) and not self._demo:
      torque_line_bg_color = rl.Color(255, 255, 255, int(255 * 0.15 * visible_alpha))

    torque_line_radius = 1200 * self._scale
    top_angle = -90
    torque_bg_angle_span = self._torque_line_alpha_filter.x * TORQUE_ANGLE_SPAN * 0.93
    torque_start_angle = top_angle - torque_bg_angle_span / 2
    torque_end_angle = top_angle + torque_bg_angle_span / 2
    mid_r = torque_line_radius + torque_line_height / 2

    cx = rect.x + rect.width / 2 + 8
    cy = rect.y + rect.height + torque_line_radius - torque_line_offset
    offset = np.array([cx, cy], dtype=np.float32)

    bg_pts = arc_bar_pts(
      mid_r, torque_line_height, torque_start_angle, torque_end_angle,
      cap_radius=7 * self._scale, max_points=384, cap_segs=16, px_per_seg=1.0,
    ) + offset
    draw_polygon(rect, bg_pts, color=torque_line_bg_color)

    a0s = top_angle
    a1s = a0s + torque_bg_angle_span / 2 * self._torque_filter.x
    sl_pts = arc_bar_pts(mid_r, torque_line_height, a0s, a1s, cap_radius=7 * self._scale) + offset

    start_grad_pt = cx / rect.width
    if self._torque_filter.x < 0:
      end_grad_pt = (cx * (1 - 0.65) + (min(bg_pts[:, 0]) * 0.65)) / rect.width
    else:
      end_grad_pt = (cx * (1 - 0.65) + (max(bg_pts[:, 0]) * 0.65)) / rect.width

    start_color = blend_colors(
      rl.Color(255, 255, 255, int(255 * 0.9 * visible_alpha)),
      rl.Color(255, 200, 0, int(255 * visible_alpha)),
      max(0, abs(self._torque_filter.x) - 0.75) * 4,
    )
    end_color = blend_colors(
      rl.Color(255, 255, 255, int(255 * 0.9 * visible_alpha)),
      rl.Color(255, 115, 0, int(255 * visible_alpha)),
      max(0, abs(self._torque_filter.x) - 0.75) * 4,
    )

    if ui_state.status not in (UIStatus.ENGAGED, UIStatus.LAT_ONLY) and not self._demo:
      start_color = end_color = rl.Color(255, 255, 255, int(255 * 0.35 * visible_alpha))

    gradient = Gradient(
      start=(start_grad_pt, 0),
      end=(end_grad_pt, 0),
      colors=[start_color, end_color],
      stops=[0.0, 1.0],
    )

    draw_polygon(rect, sl_pts, gradient=gradient)

    if abs(self._torque_filter.x) < 0.5:
      dot_y = self._rect.y + self._rect.height - torque_line_offset - torque_line_height / 2
      rl.draw_circle(int(cx), int(dot_y), 5 * self._scale,
                     rl.Color(182, 182, 182, int(255 * 0.9 * visible_alpha)))
