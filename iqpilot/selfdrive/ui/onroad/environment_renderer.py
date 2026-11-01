import numpy as np
import pyray as rl
from iqpilot.cereal import custom

from iqpilot.selfdrive.ui.ui_state import ui_state
from iqpilot.system.ui.lib.application import gui_app
from iqpilot.system.ui.widgets import Widget

MODE_OFF = 0
MODE_OVERLAY = 1
MODE_REPLACE = 2

_ENV_LABEL = custom.IQEnvironment.Object.Label
_OBJECT_COLORS = {
  _ENV_LABEL.car: (40, 210, 200),
  _ENV_LABEL.truck: (40, 210, 200),
  _ENV_LABEL.bus: (40, 210, 200),
  _ENV_LABEL.motorcycle: (90, 220, 255),
  _ENV_LABEL.bicycle: (90, 220, 255),
  _ENV_LABEL.person: (255, 210, 90),
  _ENV_LABEL.stopSign: (255, 60, 45),
  _ENV_LABEL.trafficLight: (255, 190, 0),
}

_BOX_EDGES = (
  (0, 1), (1, 3), (3, 2), (2, 0),
  (4, 5), (5, 7), (7, 6), (6, 4),
  (0, 4), (1, 5), (2, 6), (3, 7),
)

GRID_HALF_WIDTH = 12.0
GRID_MAX_DISTANCE = 90.0
GRID_STEP = 6.0


class EnvironmentRenderer(Widget):
  def __init__(self):
    Widget.__init__(self)
    self._car_space_transform = np.zeros((3, 3), dtype=np.float32)
    self._mode = MODE_OFF
    self._counter = 0

  def set_transform(self, transform: np.ndarray):
    self._car_space_transform = transform.astype(np.float32)

  def _project(self, pt: np.ndarray):
    p = self._car_space_transform @ pt
    if abs(p[2]) < 1e-6:
      return None
    return p[0] / p[2], p[1] / p[2]

  def _in_rect(self, x: float, y: float) -> bool:
    r = self._rect
    return r.x - 400 <= x <= r.x + r.width + 400 and r.y - 400 <= y <= r.y + r.height + 400

  def _render(self, rect: rl.Rectangle):
    sm = ui_state.sm
    if self._counter % 30 == 0:
      self._mode = int(ui_state.params.get("EnvironmentView", return_default=True) or 0) if ui_state.active_bundle else 0
    self._counter += 1

    if self._mode == MODE_OFF:
      return
    if sm.recv_frame["extrinsicsCalibration"] < ui_state.started_frame:
      return

    if self._mode == MODE_REPLACE:
      self._draw_backdrop(rect)
      self._draw_ground_grid()
      if sm.valid["modelV2"]:
        self._draw_model_scene(sm["modelV2"])

    if sm.alive["iqEnvironment"] and sm.valid["iqEnvironment"]:
      self._draw_objects(sm["iqEnvironment"])

  def _draw_backdrop(self, rect: rl.Rectangle):
    rl.draw_rectangle_gradient_v(int(rect.x), int(rect.y), int(rect.width), int(rect.height),
                                 rl.Color(14, 17, 22, 255), rl.Color(6, 8, 11, 255))

  def _draw_ground_grid(self):
    col = rl.Color(60, 70, 82, 90)
    dist = GRID_STEP
    while dist <= GRID_MAX_DISTANCE:
      a = self._project(np.array([dist, -GRID_HALF_WIDTH, 0.0]))
      b = self._project(np.array([dist, GRID_HALF_WIDTH, 0.0]))
      if a and b and self._in_rect(*a) and self._in_rect(*b):
        rl.draw_line_ex(rl.Vector2(*a), rl.Vector2(*b), 1.5, col)
      dist += GRID_STEP
    for off in np.arange(-GRID_HALF_WIDTH, GRID_HALF_WIDTH + 0.1, 3.0):
      a = self._project(np.array([GRID_STEP, float(off), 0.0]))
      b = self._project(np.array([GRID_MAX_DISTANCE, float(off), 0.0]))
      if a and b and self._in_rect(*a) and self._in_rect(*b):
        rl.draw_line_ex(rl.Vector2(*a), rl.Vector2(*b), 1.5, col)

  def _draw_polyline(self, xs, ys, zs, color, thick):
    pts = []
    for x, y, z in zip(xs, ys, zs, strict=False):
      if x < 0:
        continue
      s = self._project(np.array([x, y, z], dtype=np.float32))
      if s and self._in_rect(*s):
        pts.append(rl.Vector2(*s))
    for i in range(len(pts) - 1):
      rl.draw_line_ex(pts[i], pts[i + 1], thick, color)

  def _draw_model_scene(self, model):
    for i, lane in enumerate(model.laneLines):
      a = int(np.clip(model.laneLineProbs[i], 0.0, 0.9) * 255)
      self._draw_polyline(lane.x, lane.y, lane.z, rl.Color(235, 235, 235, a), 3.0)
    for edge in model.roadEdges:
      self._draw_polyline(edge.x, edge.y, edge.z, rl.Color(230, 70, 70, 180), 3.0)
    pos = model.position
    self._draw_polyline(pos.x, pos.y, pos.z, rl.Color(40, 210, 200, 220), 6.0)

  def _draw_objects(self, env):
    for obj in env.objects:
      self._draw_box(obj)

  def _draw_box(self, obj):
    hx, hy = obj.length / 2.0, obj.width / 2.0
    base = np.array([
      [obj.x - hx, obj.y - hy, obj.z], [obj.x - hx, obj.y + hy, obj.z],
      [obj.x + hx, obj.y - hy, obj.z], [obj.x + hx, obj.y + hy, obj.z],
      [obj.x - hx, obj.y - hy, obj.z + obj.height], [obj.x - hx, obj.y + hy, obj.z + obj.height],
      [obj.x + hx, obj.y - hy, obj.z + obj.height], [obj.x + hx, obj.y + hy, obj.z + obj.height],
    ], dtype=np.float32)

    screen = []
    for corner in base:
      s = self._project(corner)
      if s is None or not self._in_rect(*s):
        return
      screen.append(s)

    r, g, b = _OBJECT_COLORS.get(obj.label, (40, 210, 200))
    a = int(np.clip(obj.prob, 0.3, 1.0) * 210)
    floor = [rl.Vector2(*screen[i]) for i in (0, 1, 3, 2)]
    rl.draw_triangle(floor[0], floor[1], floor[2], rl.Color(r, g, b, a // 5))
    rl.draw_triangle(floor[0], floor[2], floor[3], rl.Color(r, g, b, a // 5))
    for i, j in _BOX_EDGES:
      rl.draw_line_ex(rl.Vector2(*screen[i]), rl.Vector2(*screen[j]), 2.0, rl.Color(r, g, b, a))
