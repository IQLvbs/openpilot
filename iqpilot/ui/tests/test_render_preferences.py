# Copyright (c) 2026 IQ.Lvbs. All rights reserved.
from types import SimpleNamespace

import numpy as np
import pytest

from iqpilot.ui.render_preferences import RenderPreferences


@pytest.mark.parametrize('raw,expected', [(0.2, 0.2), ('-0.15', -0.15), (None, 0.0), ('invalid', 0.0),
                                          (float('nan'), 0.0), (float('inf'), 0.0)])
def test_offset_is_finite(raw, expected):
  params = SimpleNamespace(get=lambda key, **_: raw if key == 'CameraOffset' else True)
  assert RenderPreferences.read(params, has_active_model=True) == RenderPreferences(expected, True)


def test_missing_active_model_ignores_stored_offset():
  calls = []
  params = SimpleNamespace(get=lambda *args, **kwargs: calls.append((args, kwargs)) or True)
  assert RenderPreferences.read(params, has_active_model=False) == RenderPreferences(0.0, True)
  assert calls == [(('AmbientTrackDots',), {'return_default': True})]


@pytest.mark.parametrize('layout', ['onroad', 'mici.onroad'])
def test_offset_change_reprojects_without_a_new_model_frame(layout, monkeypatch):
  import importlib
  module = importlib.import_module(f'iqpilot.selfdrive.ui.{layout}.model_renderer')
  line = SimpleNamespace(x=[0.0, 5.0], y=[1.0, 1.0], z=[0.0, 0.0])
  model = SimpleNamespace(position=line, laneLines=[line] * 4, roadEdges=[line] * 2,
                          laneLineProbs=[1.0] * 4, roadEdgeStds=[0.0] * 2,
                          acceleration=SimpleNamespace(x=[0.0, 0.0]))

  class Messages(dict):
    recv_frame = {'extrinsicsCalibration': 1, 'modelV2': 1}
    updated = {'carParams': False, 'modelV2': False, 'radarState': False}
    valid = {'radarState': False}

  sm = Messages(modelV2=model, selfdriveState=SimpleNamespace(experimentalMode=False),
                iqDriveModelData=SimpleNamespace(backupPath=SimpleNamespace(x=[], y=[], z=[])),
                extrinsicsCalibration=None, carOutput=SimpleNamespace(actuatorsOutput=SimpleNamespace(torque=0.0)))
  state = SimpleNamespace(sm=sm, started_frame=0, render_preferences=RenderPreferences(0.2),
                          status=module.UIStatus.DISENGAGED)
  monkeypatch.setattr(module, 'ui_state', state)
  monkeypatch.setattr(module, 'get_render_path_height', lambda _: 0.0)
  if layout.startswith('mici'):
    monkeypatch.setattr(module.driving_confidence, 'update', lambda: None)
  renderer = object.__new__(module.ModelRenderer)
  renderer._set_render_preferences(RenderPreferences())
  renderer._transform_dirty = False
  renderer._car_space_transform = np.eye(3, dtype=np.float32)
  renderer._longitudinal_control = False
  renderer._torque_filter = SimpleNamespace(update=lambda _: None)
  renderer._path = SimpleNamespace(raw_points=np.empty((0, 3)))
  renderer._lane_lines = [SimpleNamespace() for _ in range(4)]
  renderer._road_edges = [SimpleNamespace() for _ in range(2)]
  renderer._backup_path = module.ModelPoints()
  projections = []
  renderer._update_model = lambda *args: projections.append(args)
  renderer._draw_lane_lines = lambda: None
  renderer._draw_path = lambda _: None
  renderer._draw_backup_path = lambda *_: None
  rect = module.rl.Rectangle(0, 0, 100, 100)
  renderer._render(rect)
  np.testing.assert_allclose(renderer._project_car_points(renderer._path.raw_points)[1], [1.2, 1.2])
  np.testing.assert_allclose(renderer._project_car_points(renderer._road_edges[0].raw_points)[1], [1.2, 1.2])
  assert len(projections) == 1
  renderer._render(rect)
  assert len(projections) == 1
  state.render_preferences = RenderPreferences()
  renderer._render(rect)
  np.testing.assert_allclose(renderer._path.raw_points[:, 1], [1.0, 1.0])
  assert len(projections) == 2


@pytest.mark.parametrize('offset', [-0.35, 0.0, 0.35])
def test_projection_matches_translation_of_raw_points(offset):
  from iqpilot.ui.onroad.model_renderer_helpers import ModelRendererHelpers

  renderer = ModelRendererHelpers()
  renderer._set_render_preferences(RenderPreferences(offset))
  renderer._car_space_transform = np.array([[2.0, 0.2, 0.1], [0.1, 3.0, 0.3], [0.01, 0.02, 1.0]], dtype=np.float32)
  points = np.array([[1.0, -0.5, 0.1], [5.0, 0.2, 0.3], [40.0, 3.0, 0.4]], dtype=np.float32)
  original = points.copy()
  translated = points.copy()
  translated[:, 1] += offset
  expected = renderer._car_space_transform @ translated.T
  np.testing.assert_allclose(renderer._project_car_points(points), expected, rtol=1e-6, atol=1e-6)
  np.testing.assert_array_equal(points, original)
