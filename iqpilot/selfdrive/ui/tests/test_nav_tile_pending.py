# Copyright (c) 2026 IQ.Lvbs. All rights reserved.
import threading
from types import SimpleNamespace

import pytest

from iqpilot.ui.onroad import nav_map_panel as nav


class ImmediateThread:
  def __init__(self, target, args=(), **kwargs):
    self.target = target
    self.args = args

  def start(self):
    self.target(*self.args)


@pytest.fixture
def provider(tmp_path, monkeypatch):
  obj = nav.MapboxTileProvider.__new__(nav.MapboxTileProvider)
  obj._textures = {}
  obj._pending_tiles = {}
  obj._inflight = set()
  obj._lock = threading.Lock()
  obj._retry = nav.TileRetryGate()
  obj._generation = 0
  obj._cache_disabled = False
  obj._provider_disabled = False
  obj._cache_root = tmp_path
  obj._style = 'test'
  obj._keep_margin = 0
  obj._status = 'idle'
  obj._token = lambda: 'test'
  obj._prune_cache = lambda keep, visible: None
  monkeypatch.setattr(nav.threading, 'Thread', ImmediateThread)
  monkeypatch.setattr(nav, '_decode_tile_image', lambda payload: object())
  monkeypatch.setattr(nav.rl, 'unload_image', lambda image: None)
  return obj


def cache_tiles(provider, count):
  tiles = [(15, x, 10) for x in range(count)]
  for tile in tiles:
    path = provider._cache_path(tile)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'tile')
  return tiles


def test_fast_cache_workers_keep_decoded_images_bounded(provider):
  tiles = cache_tiles(provider, 64)
  for tile in tiles:
    assert provider._queue_cached_tile(tile)
  assert len(provider._pending_tiles) == nav.MAX_INFLIGHT_TILES
  assert not provider._inflight


def test_cache_capacity_includes_running_and_decoded_tiles(provider):
  tiles = cache_tiles(provider, 20)
  provider._inflight.update(tiles[:3])
  provider._pending_tiles.update({tile: object() for tile in tiles[3:8]})
  for tile in tiles[8:]:
    assert provider._queue_cached_tile(tile)
  assert len(provider._inflight) + len(provider._pending_tiles) == nav.MAX_INFLIGHT_TILES


def test_cached_pending_tiles_are_not_fetched_again(provider):
  tiles = cache_tiles(provider, 4)
  fetched = []
  provider._fetch_worker = lambda tile, token, generation: fetched.append(tile)
  provider._visible_tile_keys = lambda *args: (15, 1, 0, 0, tiles)
  provider.update(0, 0, 15, 100, 100)
  assert set(provider._pending_tiles) == set(tiles)
  assert fetched == []


@pytest.mark.parametrize('provider_class', [nav.MapboxTileProvider, nav.OsmOfflineProvider])
def test_worker_completion_during_viewport_calculation_stays_bounded(provider, provider_class):
  tiles = [(15, x, 10) for x in range(20)]
  obj = provider if provider_class is nav.MapboxTileProvider else provider_class.__new__(provider_class)
  if obj is not provider:
    obj.__dict__.update(provider.__dict__)
    obj._refresh_source = lambda *args: None
    obj._status = 'offline_ready'
    obj._source_tile_for_request = lambda tile: (tile, 0)
  launched = []
  obj._consume_pending = lambda: None
  obj._queue_cached_tile = lambda tile: False
  obj._fetch_worker = lambda tile, token, generation: launched.append(tile)
  obj._load_decode_worker = lambda tile, generation: launched.append(tile)

  def visible(*args):
    obj._pending_tiles.update({tile: object() for tile in tiles[:5]})
    obj._inflight.update(tiles[5:8])
    return 15, 1, 0, 0, tiles

  obj._visible_tile_keys = visible
  obj.update(0, 0, 15, 100, 100)
  assert launched == []


def test_online_uploads_one_tile_per_frame_without_discarding_pending(provider, monkeypatch):
  tiles = [(15, x, 10) for x in range(8)]
  provider._pending_tiles = dict.fromkeys(tiles, object())
  uploaded = []
  provider._pool = SimpleNamespace(acquire=lambda image: uploaded.append(image) or object())
  monkeypatch.setattr(nav, '_emit_nav_perf', lambda *args, **kwargs: None)
  for count in range(1, 9):
    provider._consume_pending()
    assert len(uploaded) == count
    assert len(provider._pending_tiles) == 8 - count
    assert set(provider._textures) == set(tiles[:count])


@pytest.mark.parametrize('source_zoom', [14, 15, 17])
def test_zoom_transition_draws_cached_coverage_until_replacement_ready(provider, monkeypatch, source_zoom):
  target = (16, 100, 100)
  source = (source_zoom, int(100 * 2 ** (source_zoom - 16)), int(100 * 2 ** (source_zoom - 16)))
  old = SimpleNamespace(width=448, height=448)
  new = SimpleNamespace(width=448, height=448)
  provider._textures[source] = old
  provider._visible_tile_keys = lambda *args: (16, 1, 100 * nav.TILE_SIZE, 100 * nav.TILE_SIZE, [target])
  drawn = []
  monkeypatch.setattr(nav.rl, 'draw_texture_pro', lambda texture, *args: drawn.append(texture))
  rect = nav.rl.Rectangle(0, 0, 256, 256)
  assert provider.draw(rect, 0, 0, 16)
  assert drawn == [old]
  provider._textures[target] = new
  drawn.clear()
  assert provider.draw(rect, 0, 0, 16)
  assert drawn == [new]
  assert not provider._fallback_tiles([target])


def test_partial_zoom_replacement_draws_new_tiles_over_parent(provider, monkeypatch):
  parent = (15, 50, 50)
  ready, missing = (16, 100, 100), (16, 101, 100)
  provider._textures = {parent: SimpleNamespace(width=448, height=448), ready: SimpleNamespace(width=448, height=448)}
  provider._visible_tile_keys = lambda *args: (16, 1, 100 * nav.TILE_SIZE, 100 * nav.TILE_SIZE, [ready, missing])
  drawn = []
  monkeypatch.setattr(nav.rl, 'draw_texture_pro', lambda texture, *args: drawn.append(texture))
  assert provider.draw(nav.rl.Rectangle(0, 0, 512, 256), 0, 0, 16)
  assert drawn == [provider._textures[parent], provider._textures[ready]]
  pruning = []
  provider._prune_cache = lambda keep, visible: pruning.append((keep, visible))
  provider._queue_cached_tile = lambda tile: False
  provider._fetch_worker = lambda *args: None
  provider.update(0, 0, 16, 512, 256)
  assert parent in pruning[0][0]
  assert pruning[0][1] == {ready, missing}


def test_same_zoom_missing_tile_keeps_resident_tiles_visible(provider, monkeypatch):
  ready, missing = (16, 100, 100), (16, 101, 100)
  texture = SimpleNamespace(width=448, height=448)
  provider._textures[ready] = texture
  provider._visible_tile_keys = lambda *args: (16, 1, 100 * nav.TILE_SIZE, 100 * nav.TILE_SIZE, [ready, missing])
  drawn = []
  monkeypatch.setattr(nav.rl, 'draw_texture_pro', lambda texture, *args: drawn.append(texture))
  assert provider.draw(nav.rl.Rectangle(0, 0, 512, 256), 0, 0, 16)
  assert drawn == [texture]


@pytest.mark.parametrize('cache_limit,visible_count', [(48, 25), (48, 64), (96, 25)])
def test_zoom_fallback_cannot_expand_original_texture_budget(provider, monkeypatch, cache_limit, visible_count):
  visible = [(15, x, 10) for x in range(visible_count)]
  children = {(16, x * 2 + dx, 20 + dy) for _, x, _ in visible for dx in (0, 1) for dy in (0, 1)}
  provider._cache_limit = cache_limit
  provider._pool = nav.TexturePool(cache_limit)
  provider._textures = {key: SimpleNamespace(width=448, height=448) for key in children}
  provider._visible_tile_keys = lambda *args: (15, 1, 0, 0, visible)
  provider._prune_cache = nav.MapboxTileProvider._prune_cache.__get__(provider)
  provider._queue_cached_tile = lambda tile: False
  provider._fetch_worker = lambda *args: None
  monkeypatch.setattr(nav.rl, 'unload_texture', lambda texture: None)
  monkeypatch.setattr(nav, '_emit_nav_perf', lambda *args, **kwargs: None)
  provider.update(0, 0, 15, 100, 100)
  budget = max(cache_limit, visible_count)
  assert len(provider._textures) + provider._pool._free_count <= budget
  assert provider._fallback_tiles(visible)
  for key in visible:
    provider._textures[key] = SimpleNamespace(width=448, height=448)
    provider.update(0, 0, 15, 100, 100)
    assert len(provider._textures) + provider._pool._free_count <= budget


@pytest.mark.parametrize('provider_class', [nav.MapboxTileProvider, nav.OsmOfflineProvider])
def test_pending_image_ownership_across_generation_change(provider_class, monkeypatch):
  obj = provider_class.__new__(provider_class)
  obj._lock = threading.Lock()
  obj._retry = nav.TileRetryGate()
  obj._generation = 4
  obj._pending_tiles = {}
  unloaded = []
  monkeypatch.setattr(nav.rl, 'unload_image', unloaded.append)
  key = (15, 1, 1)
  obj._stash_pending(key, 'first', 4)
  obj._stash_pending(key, 'replacement', 4)
  obj._stash_pending(key, 'stale', 3)
  assert obj._pending_tiles == {key: 'replacement'}
  assert unloaded == ['first', 'stale']


@pytest.mark.parametrize('provider_class', [nav.MapboxTileProvider, nav.OsmOfflineProvider])
def test_pruning_keeps_visible_tiles_and_reuses_pool(provider_class, monkeypatch):
  obj = provider_class.__new__(provider_class)
  obj._cache_limit = 2
  obj._textures = {key: key for key in range(5)}
  released = []
  trimmed = []
  obj._pool = SimpleNamespace(release=released.append, trim=trimmed.append)
  monkeypatch.setattr(nav, '_emit_nav_perf', lambda *args, **kwargs: None)
  obj._prune_cache({0, 1, 2}, {0, 1})
  assert list(obj._textures) == [0, 1]
  assert released == [3, 4, 2]
  assert trimmed == [-3, 0]


def test_offline_upload_batch_is_preserved(monkeypatch):
  obj = nav.OsmOfflineProvider.__new__(nav.OsmOfflineProvider)
  obj._lock = threading.Lock()
  obj._retry = nav.TileRetryGate()
  obj._pending_tiles = {key: key for key in range(8)}
  obj._textures = {}
  uploaded = []
  obj._pool = SimpleNamespace(acquire=lambda image: uploaded.append(image) or image)
  monkeypatch.setattr(nav.rl, 'unload_image', lambda image: None)
  monkeypatch.setattr(nav, '_emit_nav_perf', lambda *args, **kwargs: None)
  obj._consume_pending()
  assert uploaded == list(range(8))
  assert not obj._pending_tiles
  assert len(obj._textures) == 8
