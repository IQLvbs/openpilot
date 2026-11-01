# Copyright (c) 2026 IQ.Lvbs. All rights reserved.
import gc
import importlib
from types import SimpleNamespace
import weakref

import pytest

from iqpilot.system.ui.lib import application, emoji, text_measure, wrap_text
from iqpilot.selfdrive.ui.ui_state import device, ui_state


@pytest.fixture
def emoji_images(monkeypatch):
  loaded = []
  unloaded = []
  textures = []
  original_load = emoji.rl.load_image_from_memory
  original_unload = emoji.rl.unload_image
  monkeypatch.setattr(emoji, '_cache', type(emoji._cache)())

  def load(*args):
    image = original_load(*args)
    loaded.append(image)
    return image

  def unload(image):
    unloaded.append(image)
    original_unload(image)

  def upload(image):
    assert image.width == image.height == 128
    texture = SimpleNamespace(id=len(textures) + 1, width=128, height=128)
    textures.append(texture)
    return texture

  monkeypatch.setattr(emoji.rl, 'load_image_from_memory', load)
  monkeypatch.setattr(emoji.rl, 'unload_image', unload)
  monkeypatch.setattr(emoji.rl, 'load_texture_from_image', upload)
  monkeypatch.setattr(emoji.rl, 'unload_texture', lambda texture: None)
  yield loaded, unloaded
  for image in loaded:
    if not any(image.data == released.data for released in unloaded):
      original_unload(image)


def test_emoji_upload_releases_decoded_image(emoji_images):
  first = emoji.emoji_tex('😀')
  assert first is not None
  assert emoji.emoji_tex('😀') is first
  loaded, unloaded = emoji_images
  assert len(loaded) == len(unloaded) == 1


def test_failed_emoji_upload_releases_decoded_image(emoji_images, monkeypatch):
  def fail(image):
    raise RuntimeError('texture upload failed')

  monkeypatch.setattr(emoji.rl, 'load_texture_from_image', fail)
  with pytest.raises(RuntimeError, match='texture upload failed'):
    emoji.emoji_tex('😀')
  loaded, unloaded = emoji_images
  assert len(loaded) == len(unloaded) == 1
  assert not emoji._cache


def test_emoji_eviction_flushes_draws_before_releasing_texture(emoji_images, monkeypatch):
  events = []
  monkeypatch.setattr(emoji, 'MAX_CACHE_ENTRIES', 2)
  monkeypatch.setattr(emoji.rl, 'rl_draw_render_batch_active', lambda: events.append('flush'))
  monkeypatch.setattr(emoji.rl, 'unload_texture', lambda texture: events.append(texture.id))
  first = emoji.emoji_tex('😀')
  second = emoji.emoji_tex('😁')
  assert emoji.emoji_tex('😀') is first
  emoji.emoji_tex('😂')
  assert list(emoji._cache) == ['😀', '😂']
  assert events == ['flush', second.id]
  assert emoji.emoji_tex('😁') is not second
  assert len(emoji._cache) == 2


def test_font_cpu_release_preserves_metrics_and_texture(monkeypatch):
  images = [application.rl.gen_image_color(12, 24, application.rl.WHITE) for _ in range(3)]
  glyphs = application.rl.ffi.new('GlyphInfo[]', 3)
  font = application.rl.Font()
  font.texture.id = 5
  font.glyphCount = 3
  font.glyphs = glyphs
  for index, image in enumerate(images):
    glyphs[index].image = image
    glyphs[index].value = 65 + index
    glyphs[index].advanceX = 17
  monkeypatch.setattr(application.rl, 'get_font_default', lambda: SimpleNamespace(texture=SimpleNamespace(id=1)))
  application._release_font_cpu_images(font)
  assert font.texture.id == 5
  assert all(glyphs[index].image.data == application.rl.ffi.NULL for index in range(3))
  assert [glyphs[index].value for index in range(3)] == [65, 66, 67]
  assert [glyphs[index].advanceX for index in range(3)] == [17] * 3
  application._release_font_cpu_images(font)


def test_shared_default_font_images_are_preserved(monkeypatch):
  font = SimpleNamespace(texture=SimpleNamespace(id=1))
  monkeypatch.setattr(application.rl, 'get_font_default', lambda: font)
  application._release_font_cpu_images(font)


@pytest.mark.parametrize('atlas_available', [True, False])
def test_unused_fonts_allocate_only_when_requested(monkeypatch, tmp_path, atlas_available):
  loaded = []
  gui_fonts = []
  monkeypatch.setattr(application, 'FONT_DIR', tmp_path)
  if atlas_available:
    for weight in application.FontWeight:
      (tmp_path / weight).touch()

  def load(path, *args):
    font = application.rl.Font()
    font.texture.id = len(loaded) + 1
    loaded.append((path, args))
    return font

  monkeypatch.setattr(application.rl, 'load_font', load)
  monkeypatch.setattr(application.rl, 'load_font_ex', load)
  monkeypatch.setattr(application.rl, 'get_font_default', application.rl.Font)
  monkeypatch.setattr(application.rl, 'gen_texture_mipmaps', lambda texture: None)
  monkeypatch.setattr(application.rl, 'set_texture_filter', lambda *args: None)
  monkeypatch.setattr(application.rl, 'gui_set_font', gui_fonts.append)
  app = application.GuiApplication.__new__(application.GuiApplication)
  app._fonts = {}
  with pytest.raises(KeyError):
    app.font()
  assert loaded == []
  app._load_fonts()
  unused = (application.FontWeight.LIGHT, application.FontWeight.AUDIOWIDE)
  assert set(app._fonts) == set(application.FontWeight) - set(unused)
  assert len(loaded) == len(app._fonts)
  assert gui_fonts == [app.font()]
  for weight in unused:
    before = len(loaded)
    first = app.font(weight)
    assert first.texture.id > 0
    assert len(loaded) == before + 1
    assert app.font(weight) is first
    assert len(loaded) == before + 1
  assert set(app._fonts) == set(application.FontWeight)
  assert all(args == (() if atlas_available else (120, None, 0)) for _, args in loaded)


@pytest.fixture(params=['iqpilot.selfdrive.ui.onroad.cameraview', 'iqpilot.selfdrive.ui.mici.onroad.cameraview'])
def camera_module(request, monkeypatch):
  module = importlib.import_module(request.param)
  backend = importlib.import_module('iqpilot.selfdrive.ui.onroad.cameraview')
  shaders = []

  def load_shader(*args):
    shader = SimpleNamespace(id=42)
    shaders.append(shader)
    return shader

  monkeypatch.setattr(ui_state, '_offroad_transition_callbacks', [])
  monkeypatch.setattr(backend, 'EGL_DMA_BUF_SUPPORTED', False)
  monkeypatch.setattr(backend, 'VisionIpcClient', lambda *args, **kwargs: SimpleNamespace(num_buffers=0))
  monkeypatch.setattr(module.rl, 'load_shader_from_memory', load_shader)
  monkeypatch.setattr(module.rl, 'get_shader_location', lambda *args: -1)
  monkeypatch.setattr(module.rl, 'unload_shader', lambda shader: None)
  yield module
  for shader in shaders:
    shader.id = 0
  ui_state._offroad_transition_callbacks.clear()
  gc.collect()


def test_closed_camera_is_collectible(camera_module):
  view = camera_module.CameraView('memory-test', camera_module.VisionStreamType.VISION_STREAM_ROAD)
  ref = weakref.ref(view)
  view.close()
  del view
  gc.collect()
  assert ref() is None
  assert ui_state._offroad_transition_callbacks == []


def test_camera_close_releases_pending_stream(camera_module):
  view = camera_module.CameraView('memory-test', camera_module.VisionStreamType.VISION_STREAM_ROAD)
  view.switch_stream(camera_module.VisionStreamType.VISION_STREAM_WIDE_ROAD)
  assert view._target_client is not None
  view.close()
  assert view._target_client is None
  assert view._target_stream_type is None
  assert not view._switching


def test_camera_close_releases_shader_once(camera_module, monkeypatch):
  released = []
  monkeypatch.setattr(camera_module.rl, 'unload_shader', lambda shader: released.append(shader.id))
  view = camera_module.CameraView('memory-test', camera_module.VisionStreamType.VISION_STREAM_ROAD)
  view.close()
  view.close()
  assert released == [42]


def test_road_camera_releases_imported_frames_when_going_offroad(camera_module, monkeypatch):
  released = []
  monkeypatch.setattr(camera_module.rl, 'unload_texture', lambda texture: released.append(texture.id))
  monkeypatch.setattr(ui_state, 'is_onroad', lambda: False)
  view = camera_module.CameraView('memory-test', camera_module.VisionStreamType.VISION_STREAM_ROAD)
  old_client = view.client
  old_client.num_buffers = 20
  view.frame = object()
  view.texture_y = SimpleNamespace(id=100)
  view.texture_uv = SimpleNamespace(id=101)
  view.switch_stream(camera_module.VisionStreamType.VISION_STREAM_WIDE_ROAD)
  view._offroad_transition()
  assert view.frame is None
  assert view.client is not old_client
  assert view.client.num_buffers == 0
  assert view._target_client is None
  assert view.texture_y is view.texture_uv is None
  assert released == [100, 101]
  assert view.shader.id == 42
  view.close()


@pytest.mark.parametrize('camera_module', ['iqpilot.selfdrive.ui.onroad.cameraview'], indirect=True)
def test_hidden_road_view_releases_map_resources_on_offroad_transition(camera_module, monkeypatch, tmp_path):
  from iqpilot.selfdrive.ui.onroad.augmented_road_view import AugmentedRoadView
  from iqpilot.ui.onroad import nav_map_panel as nav

  released = []
  monkeypatch.setattr(nav, 'TILE_CACHE_ROOT', tmp_path)
  monkeypatch.setattr(nav.rl, 'unload_texture', lambda texture: released.append(texture.id))
  monkeypatch.setattr(nav.rl, 'unload_render_texture', lambda texture: released.append(texture.id))
  panel = object.__new__(nav.NavMapPanel)
  panel._mapbox = nav.MapboxTileProvider()
  panel._offline = nav.OsmOfflineProvider()
  panel._released = False
  panel._split_rt = SimpleNamespace(id=20)
  panel._mapbox._textures[(15, 0, 0)] = SimpleNamespace(id=10)
  panel._offline._textures[(15, 0, 0)] = SimpleNamespace(id=11)
  view = object.__new__(AugmentedRoadView)
  camera_module.CameraView.__init__(view, 'memory-test', camera_module.VisionStreamType.VISION_STREAM_ROAD)
  view._hud_renderer = SimpleNamespace(nav_map_panel=panel)
  try:
    monkeypatch.setattr(ui_state, 'is_onroad', lambda: True)
    for callback in tuple(ui_state._offroad_transition_callbacks):
      callback()
    assert not released
    monkeypatch.setattr(ui_state, 'is_onroad', lambda: False)
    for _ in range(2):
      for callback in tuple(ui_state._offroad_transition_callbacks):
        callback()
    assert sorted(released) == [10, 11, 20]
    assert not panel._mapbox._textures and not panel._offline._textures
    assert panel._split_rt is None
    assert panel._released
  finally:
    view.close()
  assert not ui_state._offroad_transition_callbacks


@pytest.fixture
def driver_dialog_environment(monkeypatch):
  values = {}
  monkeypatch.setattr(device, '_interactive_timeout_callbacks', [])
  monkeypatch.setattr(device, 'set_override_interactive_timeout', lambda value: None)
  monkeypatch.setattr(device, 'set_offroad_brightness', lambda value: None)
  monkeypatch.setattr(ui_state, 'params', SimpleNamespace(put_bool=values.__setitem__, remove=lambda key: values.pop(key, None)))
  monkeypatch.setattr(application.gui_app, 'texture', lambda *args: application.rl.Texture())
  monkeypatch.setattr(application.rl, 'get_time', lambda: 1.0)
  yield values
  device._interactive_timeout_callbacks.clear()


def test_driver_dialog_closes_without_accumulating_timeouts(camera_module, driver_dialog_environment, monkeypatch):
  module = importlib.import_module(camera_module.__name__.replace('cameraview', 'driver_camera_dialog'))
  if hasattr(module, 'messaging'):
    monkeypatch.setattr(module.messaging, 'PubMaster', lambda services: SimpleNamespace(send=lambda *args: None))
  for _ in range(20):
    dialog = module.DriverCameraDialog()
    dialog.show_event()
    dialog.hide_event()
    dialog.close()
  assert device._interactive_timeout_callbacks == []
  assert ui_state._offroad_transition_callbacks == []
  assert not driver_dialog_environment['IsDriverViewEnabled']


@pytest.mark.parametrize('camera_module', ['iqpilot.selfdrive.ui.mici.onroad.cameraview'], indirect=True)
def test_mici_driver_dialog_reopens_after_releasing_camera_and_publisher(camera_module, driver_dialog_environment, monkeypatch):
  module = importlib.import_module(camera_module.__name__.replace('cameraview', 'driver_camera_dialog'))
  monkeypatch.setattr(module.messaging, 'PubMaster', lambda services: SimpleNamespace(send=lambda *args: None))
  dialog = module.DriverCameraDialog(no_escape=True)
  dialog.show_event()
  first = dialog._camera_view
  dialog.hide_event()
  assert first.client is None
  assert first.shader.id == 0
  assert dialog._pm is None
  dialog.show_event()
  assert dialog._camera_view is not first
  assert dialog._camera_view.client is not None
  assert dialog._pm is not None
  dialog.hide_event()
  dialog.close()


@pytest.fixture
def interactive_timeout_environment(driver_dialog_environment, monkeypatch):
  monkeypatch.setattr(device, 'set_override_interactive_timeout', type(device).set_override_interactive_timeout.__get__(device))
  monkeypatch.setattr(device, '_override_interactive_timeout', None)
  monkeypatch.setattr(device, '_interaction_time', -1)
  monkeypatch.setattr(ui_state, 'custom_interactive_timeout', 0)
  monkeypatch.setattr(ui_state, 'ignition', False)
  monkeypatch.setattr(ui_state, 'night_mode', False)


@pytest.mark.parametrize('camera_module', ['iqpilot.selfdrive.ui.mici.onroad.cameraview'], indirect=True)
def test_training_timeout_survives_leaving_driver_camera(camera_module, interactive_timeout_environment, monkeypatch):
  from iqpilot.selfdrive.ui.mici.layouts.onboarding import TrainingGuide
  from iqpilot.system.ui.widgets import Widget
  module = importlib.import_module(camera_module.__name__.replace('cameraview', 'driver_camera_dialog'))
  monkeypatch.setattr(module.messaging, 'PubMaster', lambda services: SimpleNamespace(send=lambda *args: None))
  dialog = module.DriverCameraDialog(no_escape=True)
  guide = TrainingGuide.__new__(TrainingGuide)
  Widget.__init__(guide)
  guide._step = 0
  guide._completed_callback = None
  guide._steps = [dialog, SimpleNamespace(show_event=lambda: None, hide_event=lambda: None)]
  try:
    guide.show_event()
    assert device.interactive_timeout == 300
    guide._advance_step()
    assert device.interactive_timeout == 300
    assert dialog._camera_view is None and dialog._pm is None
  finally:
    guide.hide_event()
    dialog.close()
  assert device._override_interactive_timeout is None


@pytest.mark.parametrize('camera_module', ['iqpilot.selfdrive.ui.mici.onroad.cameraview'], indirect=True)
def test_training_timeout_survives_bad_face_page(camera_module, interactive_timeout_environment, monkeypatch):
  from iqpilot.selfdrive.ui.mici.layouts.onboarding import TrainingGuideDMTutorial
  from iqpilot.system.ui.widgets import Widget
  module = importlib.import_module(camera_module.__name__.replace('cameraview', 'driver_camera_dialog'))
  monkeypatch.setattr(module.messaging, 'PubMaster', lambda services: SimpleNamespace(send=lambda *args: None))
  tutorial = TrainingGuideDMTutorial.__new__(TrainingGuideDMTutorial)
  Widget.__init__(tutorial)
  tutorial._dialog = module.DriverCameraDialog(no_escape=True)
  tutorial._inactivity_callback = lambda: None
  tutorial._should_show_bad_face_page = False
  tutorial._bad_face_page = SimpleNamespace(show_event=lambda: None, hide_event=lambda: None)
  try:
    tutorial._dialog.show_event()
    assert device.interactive_timeout == 300
    tutorial._show_bad_face_page()
    assert device.interactive_timeout == 300
    assert tutorial._dialog._camera_view is None
  finally:
    tutorial.hide_event()
  assert device._override_interactive_timeout is None


def test_training_step_transition_releases_previous_step(driver_dialog_environment):
  from iqpilot.selfdrive.ui.mici.layouts.onboarding import TrainingGuide
  from iqpilot.system.ui.widgets import Widget
  events = []
  guide = TrainingGuide.__new__(TrainingGuide)
  Widget.__init__(guide)
  guide._step = 0
  guide._completed_callback = None
  guide._steps = [
    SimpleNamespace(show_event=lambda: events.append('show0'), hide_event=lambda: events.append('hide0')),
    SimpleNamespace(show_event=lambda: events.append('show1'), hide_event=lambda: events.append('hide1')),
  ]
  guide.show_event()
  guide._advance_step()
  guide.hide_event()
  assert events == ['show0', 'hide0', 'show1', 'hide1']


def test_onboarding_window_closes_active_training_resources(driver_dialog_environment):
  from iqpilot.selfdrive.ui.mici.layouts.onboarding import OnboardingState, OnboardingWindow
  from iqpilot.system.ui.widgets import Widget
  events = []
  window = OnboardingWindow.__new__(OnboardingWindow)
  Widget.__init__(window)
  window._state = OnboardingState.TERMS
  window._terms = SimpleNamespace(show_event=lambda: events.append('terms_show'), hide_event=lambda: events.append('terms_hide'))
  window._training_guide = SimpleNamespace(show_event=lambda: events.append('training_show'), hide_event=lambda: events.append('training_hide'))
  window._decline_page = SimpleNamespace()
  window.show_event()
  window._set_state(OnboardingState.ONBOARDING)
  window.hide_event()
  assert events == ['terms_show', 'terms_hide', 'training_show', 'training_hide']


def test_changing_onroad_text_does_not_accumulate_measurements(monkeypatch):
  monkeypatch.setattr(text_measure, '_cache', type(text_measure._cache)())
  monkeypatch.setattr(text_measure, 'font_fallback', lambda font: font)
  monkeypatch.setattr(text_measure.rl, 'measure_text_ex', lambda font, text, size, spacing: SimpleNamespace(x=len(text) * size, y=size))
  font = SimpleNamespace(texture=SimpleNamespace(id=4))
  for index in range(20_000):
    measured = text_measure.measure_text_cached(font, f'{index} m', 30)
    assert measured.x == pytest.approx(len(f'{index} m') * 30 * text_measure.FONT_SCALE)
  assert len(text_measure._cache) <= 4096


def test_changing_navigation_text_does_not_accumulate_wrapped_lines(monkeypatch):
  monkeypatch.setattr(wrap_text, '_cache', type(wrap_text._cache)())
  monkeypatch.setattr(wrap_text, 'font_fallback', lambda font: font)
  monkeypatch.setattr(wrap_text, 'measure_text_cached', lambda font, text, size, spacing=0: SimpleNamespace(x=len(text) * 10, y=size))
  font = SimpleNamespace(texture=SimpleNamespace(id=4))
  for index in range(8000):
    lines = wrap_text.wrap_text(font, f'Turn onto Street {index}', 30, 1000)
    assert lines == [f'Turn onto Street {index}']
  assert len(wrap_text._cache) <= 1024


def test_camera_switch_cancels_when_requested_stream_returns_to_active(camera_module):
  road = camera_module.VisionStreamType.VISION_STREAM_ROAD
  wide = camera_module.VisionStreamType.VISION_STREAM_WIDE_ROAD
  view = camera_module.CameraView('switch-test', road)
  view.switch_stream(wide)
  pending = view._target_client
  view.switch_stream(wide)
  assert view._target_client is pending
  view.switch_stream(road)
  assert view.stream_type == road
  assert view._target_client is None
  assert view._target_stream_type is None
  assert not view._switching
  view._handle_switch()
  assert view.stream_type == road
  view.close()


def test_camera_external_sampler_has_a_distinct_texture_target(camera_module, monkeypatch):
  backend = importlib.import_module('iqpilot.selfdrive.ui.onroad.cameraview')
  view = camera_module.CameraView('texture-test', camera_module.VisionStreamType.VISION_STREAM_ROAD)
  targets = {100: '2d', 101: 'external'}
  bound = []
  released = []
  monkeypatch.setattr(backend, 'EGL_DMA_BUF_SUPPORTED', True)
  monkeypatch.setattr(camera_module.rl, 'gen_image_color', lambda *args: object())
  monkeypatch.setattr(camera_module.rl, 'unload_image', lambda image: None)
  monkeypatch.setattr(camera_module.rl, 'load_texture_from_image', lambda image: SimpleNamespace(id=100, width=1, height=1))
  monkeypatch.setattr(backend, 'create_external_texture', lambda: 101)
  monkeypatch.setattr(backend, 'destroy_external_texture', lambda texture: released.append(texture))
  monkeypatch.setattr(camera_module.rl, 'unload_texture', lambda texture: released.append(texture.id))
  monkeypatch.setattr(backend, 'create_egl_image', lambda *args: object())
  monkeypatch.setattr(backend, 'destroy_egl_image', lambda image: None)
  monkeypatch.setattr(camera_module.rl, 'begin_shader_mode', lambda shader: None)
  monkeypatch.setattr(camera_module.rl, 'end_shader_mode', lambda: None)
  if hasattr(view, '_update_texture_color_filtering'):
    monkeypatch.setattr(view, '_update_texture_color_filtering', lambda: None)

  def bind(texture, image):
    assert targets[texture] == 'external'
    bound.append(image)

  def draw(texture, *args):
    assert targets[texture.id] == '2d'
    assert bound
    assert (texture.width, texture.height) == (1928, 1208)

  monkeypatch.setattr(backend, 'bind_egl_image_to_texture', bind)
  monkeypatch.setattr(camera_module.rl, 'draw_texture_pro', draw)
  view._initialize_textures()
  for index in (0, 1, 0):
    view.frame = SimpleNamespace(idx=index, width=1928, height=1208, stride=2048, fd=10 + index, uv_offset=2490368)
    view._render_egl(camera_module.rl.Rectangle(), camera_module.rl.Rectangle())
  assert bound[0] is bound[2]
  assert bound[0] is not bound[1]
  view.close()
  view.close()
  assert sorted(released) == [100, 101]
