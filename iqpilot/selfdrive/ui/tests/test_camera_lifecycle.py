# Copyright (c) 2026 IQ.Lvbs. All rights reserved.
from types import SimpleNamespace

import pytest

from iqpilot.cereal.visionipc import VisionStreamType
from iqpilot.selfdrive.ui.onroad import cameraview as base
from iqpilot.selfdrive.ui.mici.onroad import cameraview as mici


@pytest.fixture(params=[base.CameraView, mici.CameraView])
def camera(request, monkeypatch):
  callbacks = []
  monkeypatch.setattr(base.ui_state, '_offroad_transition_callbacks', callbacks)
  monkeypatch.setattr(base, 'EGL_DMA_BUF_SUPPORTED', False)
  monkeypatch.setattr(base.rl, 'load_shader_from_memory', lambda vertex, fragment: SimpleNamespace(id=1))
  monkeypatch.setattr(base.rl, 'get_shader_location', lambda shader, name: 1)
  monkeypatch.setattr(base.rl, 'unload_shader', lambda shader: None)
  monkeypatch.setattr(base, 'VisionIpcClient', lambda *args, **kwargs: SimpleNamespace(num_buffers=0))
  view = request.param('camerad', VisionStreamType.VISION_STREAM_ROAD)
  yield view
  view.close()
  assert callbacks == []


def test_stream_switch_waits_for_frame(camera):
  old = camera.client
  camera.switch_stream(VisionStreamType.VISION_STREAM_WIDE_ROAD)
  target = camera._target_client
  target.is_connected = lambda: True
  target.recv = lambda **kwargs: None
  camera._handle_switch()
  assert camera.client is old
  assert camera._switching
  frame = object()
  target.recv = lambda **kwargs: frame
  initialized = []
  camera._initialize_textures = lambda: initialized.append(True)
  camera._handle_switch()
  assert camera.client is target
  assert camera.frame is frame
  assert camera.stream_type == VisionStreamType.VISION_STREAM_WIDE_ROAD
  assert camera._texture_needs_update
  assert not camera._switching
  assert initialized == [True]


def test_offroad_transition_drops_frames_and_pending_switch(camera):
  camera.frame = object()
  camera.available_streams = [VisionStreamType.VISION_STREAM_ROAD]
  camera.switch_stream(VisionStreamType.VISION_STREAM_WIDE_ROAD)
  old = camera.client
  camera._offroad_transition()
  assert camera.frame is None
  assert camera.client is not old
  assert camera._target_client is None
  assert camera.available_streams == []
  assert not camera._switching


def test_texture_draw_updates_filter_inside_shader(camera, monkeypatch):
  calls = []
  camera.frame = object()
  camera.texture_y = SimpleNamespace(id=1)
  camera.texture_uv = SimpleNamespace(id=2)
  camera._texture_needs_update = False
  monkeypatch.setattr(base.rl, 'begin_shader_mode', lambda *args: calls.append('begin'))
  monkeypatch.setattr(camera, '_update_texture_color_filtering', lambda: calls.append('filter'))
  monkeypatch.setattr(base.rl, 'set_shader_value_texture', lambda *args: calls.append('uv'))
  monkeypatch.setattr(base.rl, 'draw_texture_pro', lambda *args: calls.append('draw'))
  monkeypatch.setattr(base.rl, 'end_shader_mode', lambda: calls.append('end'))
  monkeypatch.setattr(base.rl, 'unload_texture', lambda *args: None)
  camera._render_textures(base.rl.Rectangle(), base.rl.Rectangle())
  assert calls == ['begin', 'filter', 'uv', 'draw', 'end']
  camera.texture_y = camera.texture_uv = None


def test_egl_draw_reuses_images_and_binds_before_shader(camera, monkeypatch):
  calls = []
  camera.frame = SimpleNamespace(idx=3, width=1928, height=1208, stride=2048, fd=4, uv_offset=2473984)
  camera.egl_texture = SimpleNamespace(width=1, height=1)
  camera._external_texture_id = 7
  monkeypatch.setattr(base, 'create_egl_image', lambda *args: calls.append('create') or 'image')
  monkeypatch.setattr(base, 'bind_egl_image_to_texture', lambda *args: calls.append('bind'))
  monkeypatch.setattr(base.rl, 'begin_shader_mode', lambda *args: calls.append('begin'))
  monkeypatch.setattr(camera, '_update_texture_color_filtering', lambda: calls.append('filter'))
  monkeypatch.setattr(base.rl, 'draw_texture_pro', lambda *args: calls.append('draw'))
  monkeypatch.setattr(base.rl, 'end_shader_mode', lambda: calls.append('end'))
  for _ in range(2):
    camera._render_egl(base.rl.Rectangle(), base.rl.Rectangle())
  assert calls == ['create'] + ['bind', 'begin', 'filter', 'draw', 'end'] * 2
  assert (camera.egl_texture.width, camera.egl_texture.height) == (1928, 1208)
  camera.egl_texture = None
