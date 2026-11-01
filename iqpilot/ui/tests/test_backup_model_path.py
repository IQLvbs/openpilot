# Copyright (c) 2026 IQ.Lvbs. All rights reserved.
import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from iqpilot.common.filter_simple import FirstOrderFilter
from iqpilot.ui.render_preferences import RenderPreferences

LAYOUTS = ['onroad', 'mici.onroad']
X = np.linspace(1.0, 60.0, 12)
BACKUP = SimpleNamespace(x=list(X), y=list(0.4 + 0.01 * X), z=[-1.2] * len(X))


def renderer_for(layout, camera_offset=0.0):
  module = importlib.import_module(f'iqpilot.selfdrive.ui.{layout}.model_renderer')
  renderer = object.__new__(module.ModelRenderer)
  renderer._set_render_preferences(RenderPreferences(camera_offset))
  renderer._car_space_transform = np.array([[0.0, 100.0, 0.0], [0.0, 0.0, -100.0], [1.0, 0.0, 0.0]], dtype=np.float32)
  renderer._clip_region = module.rl.Rectangle(-1e4, -1e4, 2e4, 2e4)
  renderer._rect = module.rl.Rectangle(0, 0, 100, 100)
  renderer._path_offset_z = 0.0
  renderer._backup_path = module.ModelPoints()
  renderer._backup_alpha = FirstOrderFilter(0.0, 0.25, 1 / 20)
  return module, renderer


@pytest.mark.parametrize('layout', LAYOUTS)
@pytest.mark.parametrize('camera_offset', [0.0, 0.3])
def test_backup_path_centerline_is_the_backup_model_path_offset_once(layout, camera_offset):
  _, renderer = renderer_for(layout, camera_offset)
  renderer._update_backup_raw_points(BACKUP)
  renderer._update_backup_path()

  points = renderer._backup_path.projected_points
  half = len(points) // 2
  assert half == len(X)
  centerline = (points[:half] + points[half:][::-1]) * 0.5
  raw = np.column_stack((BACKUP.x, BACKUP.y, BACKUP.z)).astype(np.float32)
  projected = renderer._project_car_points(raw)
  np.testing.assert_allclose(centerline, (projected[:2] / projected[2]).T, rtol=1e-5, atol=1e-3)


@pytest.mark.parametrize('layout', LAYOUTS)
def test_missing_backup_path_keeps_nothing_to_project(layout):
  _, renderer = renderer_for(layout)
  renderer._update_backup_raw_points(SimpleNamespace(x=[], y=[], z=[]))
  renderer._update_backup_path()
  assert renderer._backup_path.raw_points.shape == (0, 3)
  assert renderer._backup_path.projected_points.size == 0


def test_backup_path_fades_in_with_the_iq_path_gradient_and_fades_out(monkeypatch):
  helpers = importlib.import_module('iqpilot.ui.onroad.model_renderer_helpers')
  _, renderer = renderer_for('onroad')
  drawn = []
  monkeypatch.setattr(helpers, 'draw_polygon', lambda rect, points, gradient: drawn.append((points, gradient)))

  renderer._update_backup_raw_points(BACKUP)
  renderer._update_backup_path()
  for _ in range(40):
    renderer._draw_backup_path((10.0, 20.0))
  alphas = [[c.a for c in gradient.colors] for _, gradient in drawn]
  assert alphas[0][0] < alphas[-1][0]
  assert alphas[-1][0] == pytest.approx(helpers.BACKUP_PATH_COLORS[0].a, abs=2)
  assert all(a[-1] == 0 for a in alphas)
  np.testing.assert_allclose(drawn[-1][0], renderer._backup_path.projected_points + [10.0, 20.0])

  renderer._update_backup_raw_points(SimpleNamespace(x=[], y=[], z=[]))
  count = len(drawn)
  for _ in range(40):
    renderer._draw_backup_path()
  assert count < len(drawn) < count + 40
  assert drawn[-1][1].colors[0].a < alphas[-1][0]
