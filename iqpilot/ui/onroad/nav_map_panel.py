import os
import math
import threading
import time
from typing import Any
from pathlib import Path
from functools import lru_cache

try:
  import sqlite3
except Exception:
  sqlite3 = None  # type: ignore[assignment]

import pyray as rl
import requests

from iqpilot.common.basedir import BASEDIR
from iqpilot.common.iq_perf import PerfSample, PerfTraceEmitter
from iqpilot.common.params import Params, UnknownKeyName
from iqpilot.selfdrive.ui.lib.nav_helpers import current_or_last_gps_position, resolve_mapbox_token
from iqpilot.ui.onroad.offline_tiles import (
  find_offline_mbtiles_path,
  find_offline_xyz_root,
  load_raster_tile_blob,
  load_raster_xyz_tile_blob,
  mbtiles_is_raster,
  mbtiles_zoom_bounds,
  open_mbtiles,
  xyz_zoom_bounds,
)
from iqpilot.ui.onroad.nav_map_utils import (
  MapMarkerMotion,
  build_mapbox_tile_url,
  choose_nav_camera,
  closest_polyline_segment,
  mercator_world_px_at_zoom,
  project_nav_point,
  project_nav_polyline,
)
from iqpilot.selfdrive.ui.ui_state import ui_state
from iqpilot.system.ui.lib.application import gui_app, FontWeight
from iqpilot.system.ui.widgets import Widget

PANEL_WIDTH = 560
MAP_HEIGHT = 392
ICON_ASSET_DIR = Path(BASEDIR) / "iqpilot" / "selfdrive" / "assets" / "navigation"
TILE_SIZE = 256
TILE_SCALE = int(os.getenv("IQPILOT_NAV_TILE_SCALE", "2"))
# Downscale each decoded tile before uploading to the GPU. Mapbox only serves integer-retina
# tiles (@2x = 512px), which cost ~1MB of dmabuf each; the resident cache of these is the bulk
# of the map's memory floor. Resizing the crisp @2x source down to TILE_SIZE*this (1.75 -> 448px,
# ~768KB) reclaims GPU memory while staying sharper than a native @1x (256px) fetch would be.
# Only ever downsamples — sources already at/under the target (e.g. native 256px offline tiles)
# are left untouched. Set to TILE_SCALE to keep full @2x resolution.
TILE_TEXTURE_SCALE = float(os.getenv("IQPILOT_NAV_TILE_TEXTURE_SCALE", "1.75"))
MAX_INFLIGHT_TILES = 8
TILE_RETRY_BASE_S = 2.0
TILE_RETRY_MAX_S = 60.0
TILE_RETRY_LIMIT = 512
TOKEN_REJECTED_RETRY_S = 600.0
CAMERA_SMOOTHING = 0.18
# Cache the map presentation independently of the 60 FPS onroad UI.
PANEL_RENDER_FPS = max(1, int(os.getenv("IQPILOT_NAV_PANEL_FPS", "12")))
PANEL_RENDER_INTERVAL = 1.0 / PANEL_RENDER_FPS
CACHE_MARGIN_TILES = int(os.getenv("IQPILOT_NAV_CACHE_MARGIN", "2"))
CACHE_LIMIT = int(os.getenv("IQPILOT_NAV_CACHE_LIMIT", "96"))
# Retention budget; larger viewports retain only their required visible coverage.
# Spare textures share this budget with resident tiles.
PANEL_CACHE_LIMIT = int(os.getenv("IQPILOT_NAV_PANEL_CACHE_LIMIT", "48"))
# How long mapbox must stay healthy before the offline fallback's tile cache is freed.
OFFLINE_RELEASE_AFTER_S = 60.0
PARAMS_REFRESH_S = 0.5
FALLBACK_POSITION_UPDATE_S = 0.5
NAV_FIX_TIMEOUT_S = 2.0
MAP_PROVIDER_UPDATE_S = 0.25
# Offline decode now runs off the render thread (worker pattern, same as mapbox), so when
# offline is the engaged provider it updates at the same cadence as the online one.
OFFLINE_PROVIDER_UPDATE_S = MAP_PROVIDER_UPDATE_S
ROUTE_PROJECTION_UPDATE_S = 0.15
TILE_CACHE_ROOT = Path(
  os.getenv(
    "IQPILOT_NAV_TILE_CACHE",
    "/data/iqpilot_nav_tiles" if Path("/data").exists() else "/tmp/iqpilot_nav_tiles",
  )
)
MAPBOX_PROVIDER_DISABLED = os.getenv("IQPILOT_DISABLE_MAPBOX_PROVIDER", "0") == "1"
MAPBOX_CACHE_DISABLED = os.getenv("IQPILOT_DISABLE_MAPBOX_CACHE", "0") == "1"
SQLITE_ERRORS = (sqlite3.Error,) if sqlite3 is not None else (Exception,)
NAV_TEXTURE_WARN_US = 20_000
NAV_TILE_OP_WARN_US = 10_000
NAV_PRUNE_WARN_US = 10_000
NAV_BURST_WARN_TILES = 4
NAV_PERF = PerfTraceEmitter("ui.nav")


def _downscale_tile_image(image) -> None:
  """Shrink a decoded tile image in place to the configured texture footprint.

  Downsample-only: leaves images already at/under the target size alone so native low-DPI
  (e.g. 256px offline) tiles are never upscaled. The resize is a CPU (stb) bicubic pass, so
  it must run OFF the render thread (in the fetch worker) — doing it inline in _consume_pending
  stalled the UI loop for several ms per tile and caused visible map lag on tile bursts."""
  target = int(round(TILE_SIZE * TILE_TEXTURE_SCALE))
  if target > 0 and image.width > target:
    rl.image_resize(image, target, target)


def _decode_tile_image(payload: bytes):
  """Decode a raw tile payload to an rl.Image and downscale it, on the calling thread.

  Pure CPU (stb) work — no GL context needed — so it is safe to run from a fetch worker
  thread. The returned Image is uploaded to a GPU texture later on the render thread.

  Raises ValueError on undecodable payloads: stb returns a 0x0 image instead of failing
  (e.g. webp, which stb can't read), and uploading that produces an invisible texture the
  cache then counts as content — the map draws blank while the badge claims tiles."""
  image = rl.load_image_from_memory(MapboxTileProvider._image_ext(payload), payload, len(payload))
  if image.width <= 0 or image.height <= 0:
    rl.unload_image(image)
    raise ValueError("undecodable tile payload")
  _downscale_tile_image(image)
  # pooled textures are refilled in place; contents must be RGBA8 regardless of source format
  rl.image_format(image, rl.PixelFormat.PIXELFORMAT_UNCOMPRESSED_R8G8B8A8)
  return image


def _emit_nav_perf(event_class: str, *, frame_id: int = 0, total_time_us: int = 0,
                   batch_size: int = 0, severity: str = "warning", detail: str = "",
                   sample: PerfSample | None = None, min_interval_s: float = 0.5) -> None:
  NAV_PERF.emit(
    event_class,
    severity=severity,
    frame_id=frame_id,
    total_time_us=total_time_us,
    batch_size=batch_size,
    samples=[sample] if sample is not None else None,
    detail=detail,
    min_interval_s=min_interval_s,
  )


class TexturePool:
  """Recycles GL texture objects between tiles of identical size/format.

  Creating/destroying thousands of texture objects per drive ratchets the GL
  driver's CPU-side pools (measured: hundreds of MB of driver-owned anon memory
  on both Adreno and Mac). Refilling an existing texture via rl.update_texture
  allocates nothing at steady state. Render thread only."""

  def __init__(self, max_free: int):
    self._max_free = max_free
    self._free: dict[tuple[int, int], list[Any]] = {}
    self._free_count = 0

  def acquire(self, image) -> Any:
    bucket = self._free.get((image.width, image.height))
    if bucket:
      texture = bucket.pop()
      self._free_count -= 1
      if not bucket:
        del self._free[(image.width, image.height)]
      rl.update_texture(texture, image.data)
      return texture
    texture = rl.load_texture_from_image(image)
    rl.set_texture_filter(texture, rl.TextureFilter.TEXTURE_FILTER_BILINEAR)
    rl.set_texture_wrap(texture, rl.TextureWrap.TEXTURE_WRAP_CLAMP)
    return texture

  def release(self, texture) -> None:
    if self._max_free <= 0:
      rl.unload_texture(texture)
      return
    if self._free_count >= self._max_free:
      oldest_size = next(iter(self._free))
      oldest_bucket = self._free[oldest_size]
      rl.unload_texture(oldest_bucket.pop())
      self._free_count -= 1
      if not oldest_bucket:
        del self._free[oldest_size]
    size = (texture.width, texture.height)
    bucket = self._free.pop(size, [])
    bucket.append(texture)
    self._free[size] = bucket
    self._free_count += 1

  def trim(self, limit: int) -> None:
    while self._free_count > max(0, limit):
      size = next(iter(self._free))
      bucket = self._free[size]
      rl.unload_texture(bucket.pop())
      self._free_count -= 1
      if not bucket:
        del self._free[size]

  def drain(self) -> None:
    for bucket in self._free.values():
      for texture in bucket:
        rl.unload_texture(texture)
    self._free.clear()
    self._free_count = 0


class TileRetryGate:
  def __init__(self, base_s: float = TILE_RETRY_BASE_S, max_s: float = TILE_RETRY_MAX_S, limit: int = TILE_RETRY_LIMIT):
    self._base_s = base_s
    self._max_s = max_s
    self._limit = limit
    self._lock = threading.Lock()
    self._entries: dict[tuple[int, int, int], tuple[int, float]] = {}

  def failed(self, tile_key: tuple[int, int, int]) -> None:
    with self._lock:
      failures = self._entries.pop(tile_key, (0, 0.0))[0] + 1
      delay = min(self._max_s, self._base_s * 2 ** (failures - 1))
      self._entries[tile_key] = (failures, time.monotonic() + delay)
      while len(self._entries) > self._limit:
        del self._entries[next(iter(self._entries))]

  def loaded(self, tile_key: tuple[int, int, int]) -> None:
    with self._lock:
      self._entries.pop(tile_key, None)

  def has_failed(self, tile_key: tuple[int, int, int]) -> bool:
    with self._lock:
      return tile_key in self._entries

  def blocked(self, tile_key: tuple[int, int, int]) -> bool:
    with self._lock:
      entry = self._entries.get(tile_key)
    return entry is not None and time.monotonic() < entry[1]

  def clear(self) -> None:
    with self._lock:
      self._entries.clear()


class RasterTileProvider:
  def _stash_pending(self, tile_key: tuple[int, int, int], image, generation: int) -> None:
    """Store a decoded tile Image for upload, unloading any Image it displaces.

    Safe to call from a fetch worker thread; only the render thread uploads/unloads textures."""
    with self._lock:
      if generation == self._generation:
        displaced = self._pending_tiles.get(tile_key)
        self._pending_tiles[tile_key] = image
      else:
        displaced = image
    if displaced is not None:
      rl.unload_image(displaced)

  def _visible_tile_keys(
    self, latitude: float, longitude: float, zoom: float, width: float, height: float
  ) -> tuple[int, float, float, float, list[tuple[int, int, int]]]:
    if self._rotated:
      # Cover the rotated viewport without clipping its corners.
      width = height = math.hypot(width, height)
    z = max(0, min(22, int(round(zoom))))
    scale = 2.0 ** (zoom - z)
    center_x, center_y = mercator_world_px_at_zoom(latitude, longitude, z, tile_size=TILE_SIZE)
    world_half_width = (width * 0.5) / max(scale, 1e-6)
    world_half_height = (height * 0.5) / max(scale, 1e-6)
    min_tile_x = int(math.floor((center_x - world_half_width) / TILE_SIZE)) - 1
    max_tile_x = int(math.floor((center_x + world_half_width) / TILE_SIZE)) + 1
    min_tile_y = int(math.floor((center_y - world_half_height) / TILE_SIZE)) - 1
    max_tile_y = int(math.floor((center_y + world_half_height) / TILE_SIZE)) + 1

    tile_count = 2 ** z
    visible = []
    for tile_y in range(max(0, min_tile_y), min(tile_count - 1, max_tile_y) + 1):
      for tile_x in range(min_tile_x, max_tile_x + 1):
        visible.append((z, tile_x % tile_count, tile_y))
    return z, scale, center_x, center_y, visible

  def set_rotated(self, rotated: bool) -> None:
    self._rotated = rotated
    self._keep_margin = 1 if rotated else CACHE_MARGIN_TILES

  def _prune_cache(self, keep_tiles: set[tuple[int, int, int]], visible_tiles: set[tuple[int, int, int]]) -> None:
    budget = max(self._cache_limit, len(visible_tiles))
    self._pool.trim(budget - len(self._textures))
    if len(self._textures) <= budget:
      return

    cache_before = len(self._textures)
    unload_count = 0
    started_ns = time.monotonic_ns()
    for tile_key in sorted(self._textures, key=lambda key: key in keep_tiles):
      if tile_key in visible_tiles:
        continue
      self._pool.release(self._textures.pop(tile_key))
      unload_count += 1
      if len(self._textures) <= budget:
        break
    self._pool.trim(budget - len(self._textures))
    prune_us = (time.monotonic_ns() - started_ns) // 1000
    if prune_us >= NAV_PRUNE_WARN_US or unload_count >= NAV_BURST_WARN_TILES:
      sample = PerfSample(
        texture_prune_us=int(prune_us),
        texture_cache_before=cache_before,
        texture_cache_after=len(self._textures),
        texture_unloaded=unload_count,
      )
      _emit_nav_perf(
        "nav_texture_prune",
        total_time_us=int(prune_us),
        batch_size=unload_count,
        detail=f"provider={self._perf_provider} prune_us={prune_us} cache_before={cache_before} cache_after={len(self._textures)} unload_count={unload_count}",
        sample=sample,
      )


class MapboxTileProvider(RasterTileProvider):
  _perf_provider = "mapbox"
  _rejected_token = ""
  _rejected_until = 0.0

  def __init__(self, cache_limit: int = CACHE_LIMIT):
    self._cache_limit = cache_limit
    self._pool = TexturePool(cache_limit)
    self._params = Params()
    self._session = requests.Session()
    # Decoded, downscaled rl.Images ready for GPU upload. Decoding + resizing happens off the
    # render thread (fetch worker) and is drained by _consume_pending, which only uploads.
    self._pending_tiles: dict[tuple[int, int, int], Any] = {}
    self._inflight: set[tuple[int, int, int]] = set()
    self._textures: dict[tuple[int, int, int], rl.Texture] = {}
    self._lock = threading.Lock()
    self._retry = TileRetryGate()
    self._generation = 0
    self._status = "idle"
    self._viewport_complete = False
    self._cache_root = TILE_CACHE_ROOT
    self._cache_root.mkdir(parents=True, exist_ok=True)
    self._provider_disabled = MAPBOX_PROVIDER_DISABLED
    self._cache_disabled = MAPBOX_CACHE_DISABLED
    self._day_mode = False
    self._style = "navigation-night-v1"
    self._rotated = False
    self._keep_margin = CACHE_MARGIN_TILES

  def set_day_mode(self, day: bool) -> None:
    """Switch between the day/night Mapbox styles. Frees the tile cache on a change —
    the resident textures are the wrong palette. Render-thread only (release touches GL)."""
    if day == self._day_mode:
      return
    self.release()
    self._day_mode = day
    self._style = "navigation-day-v1" if day else "navigation-night-v1"

  def _token(self) -> str:
    return resolve_mapbox_token(self._params)

  def _fetch_worker(self, tile_key: tuple[int, int, int], token: str, generation: int) -> None:
    z, x, y = tile_key
    try:
      with self._lock:
        if generation != self._generation:
          return
        style, day_mode = self._style, self._day_mode
      url = build_mapbox_tile_url(z, x, y, tile_size=TILE_SIZE, scale=TILE_SCALE, style=style)
      response = self._session.get(url, params={"access_token": token}, timeout=1.5)
      fallback_style = "light-v11" if day_mode else "dark-v11"
      if response.status_code == 401 and style != fallback_style:
        # some tokens can't access the navigation styles ("Direct access not allowed")
        style = fallback_style
        with self._lock:
          if generation != self._generation:
            return
          self._style = style
        url = build_mapbox_tile_url(z, x, y, tile_size=TILE_SIZE, scale=TILE_SCALE, style=style)
        response = self._session.get(url, params={"access_token": token}, timeout=3.0)
      if response.status_code in (401, 403):
        self._rejected_token = token
        self._rejected_until = time.monotonic() + TOKEN_REJECTED_RETRY_S
      response.raise_for_status()
      if not self._cache_disabled:
        cache_path = self._cache_path(tile_key, style)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(response.content)
      # Decode + downscale here on the worker thread (pure CPU/stb, no GL), so the render
      # thread only pays the cheap GPU upload in _consume_pending.
      self._stash_pending(tile_key, _decode_tile_image(response.content), generation)
      self._retry.loaded(tile_key)
      self._set_status("ready", generation)
    except (requests.RequestException, ValueError):
      if not self._queue_cached_tile(tile_key, inline=True, generation=generation):
        self._retry.failed(tile_key)
        self._set_status("error", generation)
    finally:
      self._inflight.discard(tile_key)

  def _set_status(self, status: str, generation: int) -> None:
    with self._lock:
      if generation == self._generation:
        self._status = status

  def _cache_path(self, tile_key: tuple[int, int, int], style: str | None = None) -> Path:
    z, x, y = tile_key
    # style-keyed: day and night tiles must never mix in the disk cache
    return self._cache_root / (self._style if style is None else style) / str(z) / str(x) / f"{y}@{TILE_SCALE}x.png"

  def _queue_cached_tile(self, tile_key: tuple[int, int, int], inline: bool = False, generation: int | None = None) -> bool:
    generation = self._generation if generation is None else generation
    if generation != self._generation:
      return False
    if self._cache_disabled:
      return False
    if tile_key in self._textures or tile_key in self._pending_tiles:
      return True
    # The render-thread path defers when the tile is already being fetched/decoded; the inline
    # (network-error fallback) path owns its own _inflight entry, so it must not short-circuit here.
    if not inline and (tile_key in self._inflight or self._retry.blocked(tile_key)):
      return True

    cache_path = self._cache_path(tile_key)
    if not cache_path.exists():
      return False

    if inline:
      # Already on a worker thread (network-error fallback) — decode here directly.
      try:
        payload = cache_path.read_bytes()
        self._stash_pending(tile_key, _decode_tile_image(payload), generation)
      except (OSError, ValueError):
        return False
      self._set_status("offline_cache", generation)
      return True

    # Called from the render thread: offload the disk read + decode + resize to a worker so the
    # UI loop never stalls, even when a whole screen of tiles is served from cache. Bounded by
    # MAX_INFLIGHT (shared with network fetches); over the cap we defer to a later frame.
    if len(self._inflight) + len(self._pending_tiles) >= MAX_INFLIGHT_TILES:
      return True
    self._inflight.add(tile_key)
    self._status = "offline_cache"
    threading.Thread(target=self._cache_decode_worker, args=(tile_key, cache_path, generation), daemon=True).start()
    return True

  def _cache_decode_worker(self, tile_key: tuple[int, int, int], cache_path: Path, generation: int) -> None:
    try:
      if generation != self._generation:
        return
      payload = cache_path.read_bytes()
      self._stash_pending(tile_key, _decode_tile_image(payload), generation)
    except (OSError, ValueError):
      self._retry.failed(tile_key)
    finally:
      self._inflight.discard(tile_key)

  def _consume_pending(self) -> None:
    with self._lock:
      tile_key = next(iter(self._pending_tiles), None)
      pending = [(tile_key, self._pending_tiles.pop(tile_key))] if tile_key is not None else []

    decode_us = 0
    upload_us = 0
    unload_us = 0
    unload_count = 0
    payload_bytes = 0
    cache_before = len(self._textures)
    for tile_key, image in pending:
      # Images arrive already decoded + downscaled from the fetch worker; the render thread
      # only uploads to a GPU texture here (decode cost is off-thread, so decode_us stays ~0).
      started_ns = time.monotonic_ns()
      texture = self._pool.acquire(image)
      upload_us += (time.monotonic_ns() - started_ns) // 1000
      rl.unload_image(image)
      old_texture = self._textures.get(tile_key)
      if old_texture is not None:
        started_ns = time.monotonic_ns()
        self._pool.release(old_texture)
        unload_us += (time.monotonic_ns() - started_ns) // 1000
        unload_count += 1
      self._textures[tile_key] = texture

    total_us = decode_us + upload_us + unload_us
    if pending and (
      total_us >= NAV_TEXTURE_WARN_US
      or decode_us >= NAV_TILE_OP_WARN_US
      or upload_us >= NAV_TILE_OP_WARN_US
      or len(pending) >= NAV_BURST_WARN_TILES
    ):
      sample = PerfSample(
        texture_decode_us=int(decode_us),
        texture_upload_us=int(upload_us),
        texture_unload_us=int(unload_us),
        texture_consume_us=int(total_us),
        texture_batch_size=len(pending),
        texture_bytes=payload_bytes,
        texture_cache_before=cache_before,
        texture_cache_after=len(self._textures),
        texture_unloaded=unload_count,
      )
      _emit_nav_perf(
        "nav_texture_burst",
        total_time_us=int(total_us),
        batch_size=len(pending),
        detail=(
          f"provider=mapbox decode_us={decode_us} upload_us={upload_us} unload_us={unload_us} "
          + f"tiles={len(pending)} bytes={payload_bytes} cache_before={cache_before} cache_after={len(self._textures)}"
        ),
        sample=sample,
      )


  @staticmethod
  def _image_ext(payload: bytes) -> str:
    if payload.startswith(b"\xff\xd8\xff"):
      return ".jpg"
    if payload.startswith(b"RIFF") and b"WEBP" in payload[:16]:
      return ".webp"
    return ".png"

  def update(self, latitude: float, longitude: float, zoom: float, width: float, height: float):
    if self._provider_disabled:
      self._status = "disabled"
      return

    self._consume_pending()

    z, scale, center_x, center_y, visible_tiles = self._visible_tile_keys(latitude, longitude, zoom, width, height)
    center_tile_x = center_x / TILE_SIZE
    center_tile_y = center_y / TILE_SIZE
    token = self._token()
    if token == self._rejected_token and time.monotonic() < self._rejected_until:
      token = ""

    for tile_key in visible_tiles:
      self._queue_cached_tile(tile_key)

    missing_tiles = [
      tile_key for tile_key in visible_tiles
      if tile_key not in self._textures and tile_key not in self._inflight and tile_key not in self._pending_tiles
      and not self._retry.blocked(tile_key)
    ]
    missing_tiles.sort(key=lambda tile_key: abs(tile_key[1] - center_tile_x) + abs(tile_key[2] - center_tile_y))
    if token:
      launch_count = max(0, MAX_INFLIGHT_TILES - len(self._inflight) - len(self._pending_tiles))
      for tile_key in missing_tiles[:launch_count]:
        self._inflight.add(tile_key)
        self._status = "loading"
        thread = threading.Thread(target=self._fetch_worker, args=(tile_key, token, self._generation), daemon=True)
        thread.start()
    elif not self._textures:
      self._status = "token_missing"

    visible_set = set(visible_tiles)
    self._viewport_complete = bool(visible_set) and all(tile_key in self._textures for tile_key in visible_set)
    if self._viewport_complete:
      self._status = "ready" if token else "offline_cache"
    elif self._inflight:
      self._status = "loading"
    elif self._textures:
      self._status = "offline_cache"

    tile_count = 2 ** z
    keep_tiles = set()
    for tile_key in visible_tiles:
      _, tile_x, tile_y = tile_key
      for dx in range(-self._keep_margin, self._keep_margin + 1):
        for dy in range(-self._keep_margin, self._keep_margin + 1):
          keep_y = tile_y + dy
          if 0 <= keep_y < tile_count:
            keep_tiles.add((z, (tile_x + dx) % tile_count, keep_y))
    self._prune_cache(keep_tiles | self._fallback_tiles(visible_tiles), set(visible_tiles))

  def _fallback_tiles(self, visible_tiles: list[tuple[int, int, int]]) -> set[tuple[int, int, int]]:
    fallback = set()
    for tile_key in visible_tiles:
      if tile_key in self._textures:
        continue
      z, x, y = tile_key
      for levels in (1, 2):
        parent = (z - levels, x >> levels, y >> levels)
        if parent in self._textures:
          fallback.add(parent)
          break
      else:
        fallback.update(
          child for dx in (0, 1) for dy in (0, 1)
          if (child := (z + 1, x * 2 + dx, y * 2 + dy)) in self._textures
        )
    return fallback

  def draw(self, rect: rl.Rectangle, latitude: float, longitude: float, zoom: float) -> bool:
    if self._provider_disabled:
      return False
    z, scale, center_x, center_y, visible_tiles = self._visible_tile_keys(latitude, longitude, zoom, rect.width, rect.height)
    drew_any = False
    half_width = rect.width * 0.5
    half_height = rect.height * 0.5

    fallback = sorted(self._fallback_tiles(visible_tiles))
    for tile_key in fallback + visible_tiles:
      texture = self._textures.get(tile_key)
      if texture is None:
        continue
      tile_z, tile_x, tile_y = tile_key
      tile_span = TILE_SIZE * 2.0 ** (z - tile_z)
      world_size = TILE_SIZE * 2 ** z
      offset_x = (tile_x * tile_span - center_x + world_size * 0.5) % world_size - world_size * 0.5
      tile_left = rect.x + half_width + offset_x * scale
      tile_top = rect.y + half_height + (tile_y * tile_span - center_y) * scale
      dst = rl.Rectangle(tile_left, tile_top, tile_span * scale, tile_span * scale)
      src = rl.Rectangle(0, 0, float(texture.width), float(texture.height))
      rl.draw_texture_pro(texture, src, dst, rl.Vector2(0, 0), 0.0, rl.WHITE)
      drew_any = True

    return drew_any

  def status(self) -> str:
    return self._status

  def has_content(self) -> bool:
    return bool(self._textures)

  def viewport_complete(self) -> bool:
    """True when every tile of the last update() viewport has a resident texture."""
    return self._viewport_complete

  def release(self) -> None:
    """Free all GPU tile textures and drop pending decodes.

    Called when the map panel goes inactive (offroad / maps disabled) so the tile
    cache — up to CACHE_LIMIT textures of dmabuf-backed GPU memory — isn't held
    resident while nothing is drawing it. Safe to call from the UI thread only
    (raylib GL context); update()/_consume_pending() already run here."""
    for texture in self._textures.values():
      rl.unload_texture(texture)
    self._textures.clear()
    self._pool.drain()
    with self._lock:
      self._generation += 1
      pending_images = list(self._pending_tiles.values())
      self._pending_tiles.clear()
    for image in pending_images:
      rl.unload_image(image)
    self._retry.clear()
    self._status = "idle"
    self._viewport_complete = False


class OsmOfflineProvider(RasterTileProvider):
  _perf_provider = "offline"

  def __init__(self, cache_limit: int = CACHE_LIMIT):
    self._cache_limit = cache_limit
    self._pool = TexturePool(cache_limit)
    self._conn: Any | None = None
    # Serializes sqlite access + source swaps: blob reads run on fetch worker threads against a
    # single shared read-only connection, and _refresh_source may close/reopen it under them.
    self._db_lock = threading.Lock()
    self._mbtiles_path: Path | None = None
    self._xyz_root: Path | None = None
    # Decoded, downscaled rl.Images ready for GPU upload — same off-render-thread pattern as
    # MapboxTileProvider: workers read the blob + decode + resize, _consume_pending only uploads.
    self._pending_tiles: dict[tuple[int, int, int], Any] = {}
    self._inflight: set[tuple[int, int, int]] = set()
    self._lock = threading.Lock()
    self._retry = TileRetryGate()
    self._generation = 0
    self._textures: dict[tuple[int, int, int], rl.Texture] = {}
    self._status = "offline_missing"
    self._visible_source: list[tuple[int, int, int]] = []
    self._min_zoom: int | None = None
    self._max_zoom: int | None = None
    self._day_mode = False
    self._rotated = False
    self._keep_margin = CACHE_MARGIN_TILES

  def set_day_mode(self, day: bool) -> None:
    self._day_mode = day

  def _close_conn_locked(self) -> None:
    if self._conn is not None:
      self._conn.close()
      self._conn = None

  def _drop_pending(self) -> None:
    with self._lock:
      self._generation += 1
      pending_images = list(self._pending_tiles.values())
      self._pending_tiles.clear()
    for image in pending_images:
      rl.unload_image(image)
    self._retry.clear()

  def source_available(self, latitude: float | None = None, longitude: float | None = None) -> bool:
    self._refresh_source(latitude, longitude)
    return self._status == "offline_ready"

  def viewport_missing(self) -> bool:
    return self._status != "offline_ready" or (
      bool(self._visible_source) and all(self._retry.has_failed(tile) for tile in self._visible_source)
    )

  def _refresh_source(self, latitude: float | None, longitude: float | None) -> None:
    mbtiles_path = find_offline_mbtiles_path(latitude, longitude, day=self._day_mode)
    xyz_root = find_offline_xyz_root(latitude, longitude)

    if self._mbtiles_path == mbtiles_path and self._xyz_root == xyz_root:
      return

    self._mbtiles_path = mbtiles_path
    self._xyz_root = xyz_root
    self._min_zoom = None
    self._max_zoom = None

    for texture in self._textures.values():
      rl.unload_texture(texture)
    self._textures.clear()
    self._drop_pending()

    with self._db_lock:
      self._close_conn_locked()
      if self._mbtiles_path is not None:
        try:
          self._conn = open_mbtiles(self._mbtiles_path)
          if mbtiles_is_raster(self._conn):
            self._min_zoom, self._max_zoom = mbtiles_zoom_bounds(self._conn)
            self._status = "offline_ready"
          else:
            self._status = "offline_invalid"
        except SQLITE_ERRORS:
          self._conn = None
          self._status = "offline_invalid"
      elif self._xyz_root is not None:
        self._min_zoom, self._max_zoom = xyz_zoom_bounds(self._xyz_root)
        self._status = "offline_ready"
      else:
        self._status = "offline_missing"

  def _source_tile_for_request(self, tile_key: tuple[int, int, int]) -> tuple[tuple[int, int, int], int]:
    z, x, y = tile_key
    source_z = z
    if self._max_zoom is not None and z > self._max_zoom:
      source_z = self._max_zoom
    elif self._min_zoom is not None and z < self._min_zoom:
      source_z = self._min_zoom

    delta = z - source_z
    if delta <= 0:
      return (source_z, x, y), 0

    return (source_z, x >> delta, y >> delta), delta

  def _load_decode_worker(self, tile_key: tuple[int, int, int], generation: int) -> None:
    """Read a tile blob from disk and decode + downscale it, entirely off the render thread.

    Mirrors MapboxTileProvider._fetch_worker: the render loop must never pay the sqlite/file
    read or the stb decode — with offline as the primary provider, an inline decode of a
    screenful of tiles was a visible render-loop hitch."""
    try:
      if generation != self._generation:
        return
      payload = self._load_blob(tile_key)
      if payload is None:
        self._retry.failed(tile_key)
      else:
        self._stash_pending(tile_key, _decode_tile_image(payload), generation)
        self._retry.loaded(tile_key)
    except Exception:
      self._retry.failed(tile_key)
    finally:
      self._inflight.discard(tile_key)

  def _consume_pending(self) -> None:
    with self._lock:
      pending = list(self._pending_tiles.items())
      self._pending_tiles.clear()

    upload_us = 0
    unload_us = 0
    unload_count = 0
    cache_before = len(self._textures)
    for tile_key, image in pending:
      # Images arrive already decoded + downscaled from the worker; only the GPU upload runs here.
      started_ns = time.monotonic_ns()
      texture = self._pool.acquire(image)
      upload_us += (time.monotonic_ns() - started_ns) // 1000
      rl.unload_image(image)
      old_texture = self._textures.get(tile_key)
      if old_texture is not None:
        started_ns = time.monotonic_ns()
        self._pool.release(old_texture)
        unload_us += (time.monotonic_ns() - started_ns) // 1000
        unload_count += 1
      self._textures[tile_key] = texture

    total_us = upload_us + unload_us
    if pending and (
      total_us >= NAV_TEXTURE_WARN_US
      or upload_us >= NAV_TILE_OP_WARN_US
      or len(pending) >= NAV_BURST_WARN_TILES
    ):
      sample = PerfSample(
        texture_upload_us=int(upload_us),
        texture_unload_us=int(unload_us),
        texture_consume_us=int(total_us),
        texture_batch_size=len(pending),
        texture_cache_before=cache_before,
        texture_cache_after=len(self._textures),
        texture_unloaded=unload_count,
      )
      _emit_nav_perf(
        "nav_texture_burst",
        total_time_us=int(total_us),
        batch_size=len(pending),
        detail=(
          f"provider=offline upload_us={upload_us} unload_us={unload_us} "
          + f"tiles={len(pending)} cache_before={cache_before} cache_after={len(self._textures)}"
        ),
        sample=sample,
      )

  def _load_blob(self, tile_key: tuple[int, int, int]) -> bytes | None:
    z, x, y = tile_key
    # Runs on worker threads: the shared read-only sqlite connection is not safe for
    # concurrent queries, and _refresh_source may swap it — serialize via _db_lock.
    with self._db_lock:
      if self._conn is not None and self._status == "offline_ready":
        try:
          return load_raster_tile_blob(self._conn, z, x, y)
        except SQLITE_ERRORS:
          self._status = "offline_invalid"
          return None
      xyz_root = self._xyz_root
    if xyz_root is not None:
      return load_raster_xyz_tile_blob(xyz_root, z, x, y)
    return None

  def update(self, latitude: float, longitude: float, zoom: float, width: float, height: float) -> None:
    self._refresh_source(latitude, longitude)
    if self._status != "offline_ready":
      return

    self._consume_pending()

    z, _, center_x, center_y, visible_tiles = self._visible_tile_keys(latitude, longitude, zoom, width, height)
    # Map requested tiles to their on-disk source tiles (overzoom clamps to the stored range).
    visible_source: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    for requested_tile in visible_tiles:
      source_tile, _ = self._source_tile_for_request(requested_tile)
      if source_tile not in seen:
        seen.add(source_tile)
        visible_source.append(source_tile)

    self._visible_source = visible_source

    missing_tiles = [
      tile_key for tile_key in visible_source
      if tile_key not in self._textures and tile_key not in self._inflight and tile_key not in self._pending_tiles
      and not self._retry.blocked(tile_key)
    ]

    def _center_distance(tile_key: tuple[int, int, int]) -> float:
      source_z, tile_x, tile_y = tile_key
      zoom_scale = 2.0 ** (z - source_z)
      return (
        abs(tile_x + 0.5 - (center_x / TILE_SIZE) / zoom_scale)
        + abs(tile_y + 0.5 - (center_y / TILE_SIZE) / zoom_scale)
      )

    missing_tiles.sort(key=_center_distance)
    launch_count = max(0, MAX_INFLIGHT_TILES - len(self._inflight) - len(self._pending_tiles))
    for tile_key in missing_tiles[:launch_count]:
      self._inflight.add(tile_key)
      threading.Thread(target=self._load_decode_worker, args=(tile_key, self._generation), daemon=True).start()

    # Keep a margin ring around the visible source tiles so panning doesn't churn re-decodes;
    # the cache limit bounds the resident GPU footprint exactly like the mapbox provider.
    keep_tiles: set[tuple[int, int, int]] = set()
    for source_z, tile_x, tile_y in visible_source:
      tile_count = 2 ** source_z
      for dx in range(-self._keep_margin, self._keep_margin + 1):
        for dy in range(-self._keep_margin, self._keep_margin + 1):
          keep_y = tile_y + dy
          if 0 <= keep_y < tile_count:
            keep_tiles.add((source_z, (tile_x + dx) % tile_count, keep_y))
    self._prune_cache(keep_tiles, set(visible_source))

  def draw(self, rect: rl.Rectangle, latitude: float, longitude: float, zoom: float) -> bool:
    if self._status != "offline_ready":
      return False

    z, scale, center_x, center_y, visible_tiles = self._visible_tile_keys(latitude, longitude, zoom, rect.width, rect.height)
    drew_any = False
    half_width = rect.width * 0.5
    half_height = rect.height * 0.5

    for requested_tile in visible_tiles:
      source_tile, delta = self._source_tile_for_request(requested_tile)
      texture = self._textures.get(source_tile)
      if texture is None:
        continue
      _, tile_x, tile_y = requested_tile
      tile_left = rect.x + half_width + ((tile_x * TILE_SIZE) - center_x) * scale
      tile_top = rect.y + half_height + ((tile_y * TILE_SIZE) - center_y) * scale
      dst = rl.Rectangle(tile_left, tile_top, TILE_SIZE * scale, TILE_SIZE * scale)
      if delta == 0:
        src = rl.Rectangle(0, 0, float(texture.width), float(texture.height))
      else:
        subdivisions = 2 ** delta
        src_width = float(texture.width) / subdivisions
        src_height = float(texture.height) / subdivisions
        src_x = float(tile_x % subdivisions) * src_width
        src_y = float(tile_y % subdivisions) * src_height
        src = rl.Rectangle(src_x, src_y, src_width, src_height)
      rl.draw_texture_pro(texture, src, dst, rl.Vector2(0, 0), 0.0, rl.WHITE)
      drew_any = True

    return drew_any

  def status(self) -> str:
    return self._status

  def has_content(self) -> bool:
    return bool(self._textures)

  def release(self) -> None:
    """Free all GPU tile textures and drop pending decodes. See MapboxTileProvider.release()."""
    for texture in self._textures.values():
      rl.unload_texture(texture)
    self._textures.clear()
    self._pool.drain()
    self._drop_pending()


class NavMapPanel(Widget):
  def __init__(self, force_visible: bool = False):
    super().__init__()
    self._params = Params()
    self._force_visible = force_visible
    self.active = False
    self._maps_enabled = False
    self.current_latitude = 0.0
    self.current_longitude = 0.0
    self.bearing_deg = 0.0
    self.zoom_hint = 16.0
    self.destination_latitude = 0.0
    self.destination_longitude = 0.0
    self.render_center_latitude = 0.0
    self.render_center_longitude = 0.0
    self.render_zoom = 16.0
    self.display_center_latitude = 0.0
    self.display_center_longitude = 0.0
    self.display_zoom = 16.0
    self.route_points = []
    self.next_distance = 0.0
    self.next_description = ""
    self.next_direction = 0
    self.next_type = 0
    self.next_valid = False
    self.nav_active = False
    self.road_name = ""
    self._has_render_fix = False
    self._route_ahead_index = 0
    self._projected_route_points: list[tuple[float, float]] = []
    self._projected_route_key: tuple | None = None
    self._route_camera: tuple[float, float, float, float, float] | None = None
    self._map_viewport_width: float = PANEL_WIDTH - 24
    self._map_viewport_height: float = MAP_HEIGHT
    self._mapbox = MapboxTileProvider(cache_limit=PANEL_CACHE_LIMIT)
    self._offline = OsmOfflineProvider(cache_limit=PANEL_CACHE_LIMIT)
    self._offline_idle_since = 0.0
    # OnlineOSMaps and OfflineOSMaps independently control their respective tile sources.
    self._online_maps_enabled = True
    self._offline_maps_enabled = False
    self._mapbox_mode_released = False
    self._heading_up = True
    self.display_bearing = 0.0
    self._title_font = gui_app.font(FontWeight.BOLD)
    self._icon_textures: dict[str, rl.Texture] = {}
    self._last_params_refresh = 0.0
    self._last_fallback_position_update = 0.0
    self._last_nav_fix = float("-inf")
    self._last_position_change = float("-inf")
    self._last_mapbox_update = 0.0
    self._last_offline_update = 0.0
    self._released = False
    self._last_projection_update = 0.0
    self._split_rt: Any = None
    self._last_split_render = 0.0
    self._map_camera: tuple[float, float, float, float] | None = None
    self._marker_motion = MapMarkerMotion()

  @staticmethod
  def _enum_value(value) -> int:
    return int(getattr(value, "raw", value))

  def _release_providers(self) -> None:
    """Free tile textures once when the panel transitions to inactive.

    Latched so it runs on the offroad/maps-off transition rather than every idle
    frame. force_visible (offroad NAV screen) keeps its own panel active, so this
    only fires for the onroad panel when it stops drawing."""
    gui_app.cancel_render_job(self)
    if self._released:
      return
    self._mapbox.release()
    self._offline.release()
    if self._split_rt is not None:
      rl.unload_render_texture(self._split_rt)
      self._split_rt = None
    self._last_split_render = 0.0
    self._map_camera = None
    self._marker_motion = MapMarkerMotion()
    self._last_position_change = float("-inf")
    self._released = True

  def _route_ahead_points(self):
    if len(self.route_points) < 2:
      self._route_ahead_index = 0
      return self.route_points

    best_idx = 0
    best_score = None
    longitude_scale = math.cos(math.radians(self.current_latitude))
    for idx in range(len(self.route_points) - 1):
      point, following = self.route_points[idx], self.route_points[idx + 1]
      x = (float(point.longitude) - self.current_longitude) * longitude_scale
      y = float(point.latitude) - self.current_latitude
      dx = (float(following.longitude) - float(point.longitude)) * longitude_scale
      dy = float(following.latitude) - float(point.latitude)
      length_squared = dx * dx + dy * dy
      fraction = min(1.0, max(0.0, -(x * dx + y * dy) / length_squared)) if length_squared else 0.0
      score = (x + fraction * dx) ** 2 + (y + fraction * dy) ** 2
      if best_score is None or score < best_score:
        best_score = score
        best_idx = idx

    self._route_ahead_index = best_idx
    return self.route_points[self._route_ahead_index:]

  def _refresh_route_projection(self, force: bool = False) -> None:
    route_points = self.route_points[self._route_ahead_index:]
    if len(route_points) < 2:
      self._projected_route_points = []
      self._projected_route_key = None
      return

    projection_key = (
      self._route_ahead_index,
      len(self.route_points),
      round(self.display_center_latitude, 5),
      round(self.display_center_longitude, 5),
      round(self.display_zoom, 2),
      round(self._map_viewport_width, 0),
      round(self._map_viewport_height, 0),
    )
    if not force and projection_key == self._projected_route_key:
      return

    started = time.monotonic()
    self._projected_route_points = project_nav_polyline(
      route_points,
      self.display_center_latitude,
      self.display_center_longitude,
      self.display_zoom,
      0.0,
      self._map_viewport_width,
      self._map_viewport_height,
    )
    self._route_camera = (self.display_center_latitude, self.display_center_longitude, self.display_zoom,
                          self._map_viewport_width, self._map_viewport_height)
    self._projected_route_key = projection_key

    project_us = int((time.monotonic() - started) * 1_000_000)
    if project_us > 3_000:
      sample = PerfSample(texture_consume_us=project_us, texture_batch_size=len(route_points))
      _emit_nav_perf(
        "nav_map_projection_slow",
        total_time_us=project_us,
        batch_size=len(route_points),
        detail=f"project_us={project_us} points={len(route_points)} active={int(self.active)}",
        sample=sample,
      )

  def _icon_asset_name(self) -> str | None:
    if not self._display_next_valid():
      return None
    dir_left = self._display_direction_is_left()
    if self.next_type == 1:
      return "direction_turn_left.png" if dir_left else "direction_turn_right.png"
    if self.next_type == 2:
      return "direction_off_ramp_left.png" if dir_left else "direction_off_ramp_right.png"
    if self.next_type == 3:
      return "direction_merge_left.png" if dir_left else "direction_merge_right.png"
    if self.next_type == 4:
      return "direction_fork_left.png" if dir_left else "direction_fork_right.png"
    if self.next_type == 6:
      return "direction_arrive.png"
    return "direction_turn_left.png" if dir_left else "direction_turn_right.png"

  def _display_next_valid(self) -> bool:
    return bool(
      self.next_valid and (
        self.next_distance > 1.0
        or bool(self.next_description)
        or self.next_type == 6
      )
    )

  def _display_direction_is_left(self) -> bool:
    description = f" {self.next_description.lower()} "
    if " left " in description and " right " not in description:
      return True
    if " right " in description and " left " not in description:
      return False
    return self.next_direction == 1

  def _icon_texture(self):
    name = self._icon_asset_name()
    if name is None:
      return None
    if name in self._icon_textures:
      return self._icon_textures[name]

    path = ICON_ASSET_DIR / name
    if not path.exists():
      return None

    texture = rl.load_texture(path.as_posix())
    rl.set_texture_filter(texture, rl.TextureFilter.TEXTURE_FILTER_BILINEAR)
    rl.set_texture_wrap(texture, rl.TextureWrap.TEXTURE_WRAP_CLAMP)
    self._icon_textures[name] = texture
    return texture

  def _destination_texture(self):
    name = "direction_flag.png"
    if name in self._icon_textures:
      return self._icon_textures[name]

    path = ICON_ASSET_DIR / name
    if not path.exists():
      return None

    texture = rl.load_texture(path.as_posix())
    rl.set_texture_filter(texture, rl.TextureFilter.TEXTURE_FILTER_BILINEAR)
    rl.set_texture_wrap(texture, rl.TextureWrap.TEXTURE_WRAP_CLAMP)
    self._icon_textures[name] = texture
    return texture

  @staticmethod
  def _lerp(start: float, end: float, alpha: float) -> float:
    return start + (end - start) * alpha

  @staticmethod
  def _lerp_angle(current: float, target: float, alpha: float) -> float:
    delta = ((target - current + 180.0) % 360.0) - 180.0
    return (current + delta * alpha) % 360.0

  def _refresh_position_from_fallback(self) -> bool:
    if time.monotonic() - self._last_nav_fix < NAV_FIX_TIMEOUT_S:
      return False
    lat, lon, bearing, have_fix = current_or_last_gps_position()
    if not have_fix:
      return False

    self.current_latitude = lat
    self.current_longitude = lon
    self.bearing_deg = bearing
    self.zoom_hint = self.zoom_hint if self.zoom_hint > 0.0 else 16.0
    self.render_center_latitude, self.render_center_longitude, self.render_zoom = choose_nav_camera(
      self.current_latitude,
      self.current_longitude,
      self.bearing_deg,
      self._route_ahead_points(),
      self._map_viewport_width,
      self._map_viewport_height,
      self.zoom_hint,
    )
    if self.display_center_latitude == 0.0 and self.display_center_longitude == 0.0:
      self.display_center_latitude = self.render_center_latitude
      self.display_center_longitude = self.render_center_longitude
      self.display_zoom = self.render_zoom
    self._has_render_fix = True
    self.active = True
    return True

  def _refresh_params(self, now: float) -> None:
    if now - self._last_params_refresh < PARAMS_REFRESH_S:
      return
    try:
      navigation_enabled = self._params.get_bool("NavigationEnabled")
      self._maps_enabled = self._force_visible or (navigation_enabled and self._params.get_bool("OnScreenNavigation"))
    except UnknownKeyName:
      navigation_enabled = False
      self._maps_enabled = self._force_visible
    try:
      self._online_maps_enabled = self._params.get_bool("OnlineOSMaps")
      self._offline_maps_enabled = self._params.get_bool("OfflineOSMaps")
    except UnknownKeyName:
      self._online_maps_enabled = True
      self._offline_maps_enabled = False

    has_downloaded_maps = find_offline_mbtiles_path() is not None or find_offline_xyz_root() is not None
    if navigation_enabled and gui_app.big_ui() and has_downloaded_maps:
      enable_panel = not self._maps_enabled
      enable_offline = not self._offline_maps_enabled
      self._maps_enabled = True
      self._offline_maps_enabled = True
      if not self._force_visible:
        if enable_panel:
          self._params.put_bool("OnScreenNavigation", True)
        if enable_offline:
          self._params.put_bool("OfflineOSMaps", True)
    try:
      self._heading_up = self._params.get_bool("OSMapsHeadingUp")
    except UnknownKeyName:
      self._heading_up = True
    self._last_params_refresh = now

  def update(self):
    now = time.monotonic()
    self._refresh_params(now)

    if not gui_app.big_ui() and not self._force_visible:
      self.active = False
      self._release_providers()
      return
    if not self._maps_enabled:
      self.active = False
      self._release_providers()
      return

    # We're onroad with maps enabled — tiles may load again, so re-arm the release latch.
    self._released = False

    sm = ui_state.sm
    if sm.updated["iqNavRenderState"]:
      rs = sm["iqNavRenderState"]
      rs_lat = float(rs.currentLatitude)
      rs_lon = float(rs.currentLongitude)
      has_fix = 0.001 < abs(rs_lat) <= 90.0 and 0.001 < abs(rs_lon) <= 180.0 and math.isfinite(float(rs.bearingDeg))
      if has_fix:
        self._last_nav_fix = now
        if (rs_lat, rs_lon) != (self.current_latitude, self.current_longitude):
          self._last_position_change = now
        self.current_latitude = rs_lat
        self.current_longitude = rs_lon
        self.bearing_deg = float(rs.bearingDeg)
        self.zoom_hint = float(rs.zoomHint) if float(rs.zoomHint) > 0.0 else 16.0
        self._has_render_fix = True
      route_points = list(rs.routePolylineSimplified) if len(rs.routePolylineSimplified) > 0 else list(rs.routePolyline)
      self.route_points = route_points
      self._projected_route_key = None
      if not route_points:
        self._projected_route_points = []
      self.next_distance = float(rs.nextManeuverDistance)
      self.next_direction = self._enum_value(rs.nextManeuverDirection)
      self.next_type = self._enum_value(rs.nextManeuverType)
      if self._has_render_fix:
        self.render_center_latitude, self.render_center_longitude, self.render_zoom = choose_nav_camera(
          self.current_latitude,
          self.current_longitude,
          self.bearing_deg,
          self._route_ahead_points(),
          self._map_viewport_width,
          self._map_viewport_height,
          self.zoom_hint,
        )
      self.destination_latitude = float(rs.destinationLatitude)
      self.destination_longitude = float(rs.destinationLongitude)
      if self.display_center_latitude == 0.0 and self.display_center_longitude == 0.0:
        self.display_center_latitude = self.render_center_latitude
        self.display_center_longitude = self.render_center_longitude
        self.display_zoom = self.render_zoom
      self.active = bool(rs.active) or self._has_render_fix or bool(self.route_points)

    if now - self._last_fallback_position_update >= FALLBACK_POSITION_UPDATE_S:
      self._refresh_position_from_fallback()
      self._last_fallback_position_update = now

    if sm.updated["iqNavState"]:
      nav = sm["iqNavState"]
      self.nav_active = bool(nav.active)
      if not self.nav_active:
        self.route_points = []
        self._projected_route_points = []
        self._projected_route_key = None
      self.next_valid = bool(nav.nextManeuverValid)
      self.next_description = nav.nextManeuverDescription if nav.nextManeuverValid else ""
      self.next_direction = self._enum_value(nav.nextManeuverDirection) if nav.nextManeuverValid else self.next_direction
      self.next_type = self._enum_value(nav.nextManeuverType) if nav.nextManeuverValid else self.next_type

    if sm.updated["iqLiveData"]:
      self.road_name = sm["iqLiveData"].roadName

    if self.active:
      self.display_center_latitude = self._lerp(self.display_center_latitude, self.render_center_latitude, CAMERA_SMOOTHING)
      self.display_center_longitude = self._lerp(self.display_center_longitude, self.render_center_longitude, CAMERA_SMOOTHING)
      self.display_zoom = self._lerp(self.display_zoom, self.render_zoom, CAMERA_SMOOTHING)
      if now - self._last_projection_update >= ROUTE_PROJECTION_UPDATE_S:
        self._refresh_route_projection()
        self._last_projection_update = now

      self._mapbox.set_day_mode(False)
      self._offline.set_day_mode(False)

      rotate = self._heading_up and self._has_render_fix
      self._mapbox.set_rotated(rotate)
      self._offline.set_rotated(rotate)
      target_bearing = self.bearing_deg if rotate else 0.0
      self.display_bearing = self._lerp_angle(self.display_bearing, target_bearing, 0.15)

      if self._online_maps_enabled:
        self._mapbox_mode_released = False
        if now - self._last_mapbox_update >= MAP_PROVIDER_UPDATE_S:
          self._mapbox.update(
            self.display_center_latitude,
            self.display_center_longitude,
            self.display_zoom,
            self._map_viewport_width,
            self._map_viewport_height,
          )
          self._last_mapbox_update = now
        else:
          self._mapbox._consume_pending()
      elif not self._mapbox_mode_released:
        # Online maps toggled off: free the mapbox tile cache once. No update() means no fetch
        # threads and no network traffic while offline-only mode is selected.
        self._mapbox.release()
        self._mapbox_mode_released = True

      mapbox_status = self._mapbox.status()
      if self._offline_maps_enabled:
        offline_engaged = not self._online_maps_enabled or (
          mapbox_status in {"disabled", "token_missing", "error"} or not self._mapbox.has_content()
        )
      else:
        offline_engaged = False

      if offline_engaged:
        if now - self._last_offline_update >= OFFLINE_PROVIDER_UPDATE_S:
          self._offline.update(
            self.display_center_latitude,
            self.display_center_longitude,
            self.display_zoom,
            self._map_viewport_width,
            self._map_viewport_height,
          )
          self._last_offline_update = now
        self._offline_idle_since = 0.0
      elif self._offline.has_content():
        if not self._offline_maps_enabled:
          # Offline maps toggled off mid-drive: free its tile cache immediately.
          self._offline.release()
          self._offline_idle_since = 0.0
        # Mapbox recovered: the offline fill-in cache (up to CACHE_LIMIT tiles of GPU
        # memory) would otherwise stay resident until the panel deactivates — one LTE
        # dropout per drive made it a permanent +tile-cache floor. It refills from the
        # on-disk cache in a few frames when needed, so free it after a healthy minute.
        elif self._offline_idle_since == 0.0:
          self._offline_idle_since = now
        elif now - self._offline_idle_since >= OFFLINE_RELEASE_AFTER_S:
          self._offline.release()
          self._offline_idle_since = 0.0

  def maps_enabled(self) -> bool:
    if self._force_visible:
      return True
    self._refresh_params(time.monotonic())
    return bool(self._maps_enabled)

  def set_maps_enabled(self, enabled: bool) -> None:
    self._params.put_bool("OnScreenNavigation", enabled)
    self._last_params_refresh = -math.inf
    self._refresh_params(time.monotonic())
    if not self._maps_enabled:
      self.active = False
      self._release_providers()

  def warm_up_tiles(self, timeout_s: float = 30.0) -> bool:
    """Block until the online provider covers the current viewport. For offline rendering
    (tools/clip), where frames aren't wall-clock paced and async tile fetches would lag the
    output. Pumps update() with the provider throttle bypassed; needs the panel active (a nav
    fix already fed through sm) and a resolvable Mapbox token. Render thread only."""
    if not (self._maps_enabled and self.active and self._online_maps_enabled):
      return False
    if not resolve_mapbox_token(self._params):
      return False
    deadline = time.monotonic() + timeout_s
    while True:
      self._last_mapbox_update = 0.0
      self.update()
      if self._mapbox.viewport_complete():
        return True
      if time.monotonic() >= deadline:
        return False
      time.sleep(0.05)

  @staticmethod
  def _draw_fallback_background(rect: rl.Rectangle):
    rl.draw_rectangle_rec(rect, rl.Color(10, 17, 23, 255))

  @staticmethod
  def _draw_ego_arrow(center_x: float, center_y: float, bearing_deg: float):
    rl.draw_circle_v(rl.Vector2(center_x, center_y + 8), 21.0, rl.Color(0, 0, 0, 120))
    heading = math.radians(bearing_deg)
    cos_heading, sin_heading = math.cos(heading), math.sin(heading)

    def rotate(px: float, py: float) -> rl.Vector2:
      rx = (px * cos_heading) - (py * sin_heading)
      ry = (px * sin_heading) + (py * cos_heading)
      return rl.Vector2(center_x + rx, center_y + ry)

    nose = rotate(0.0, -25.0)
    left = rotate(-16.0, 19.0)
    right = rotate(16.0, 19.0)
    rl.draw_triangle(nose, left, right, rl.Color(255, 255, 255, 245))

  def _draw_map_surface(self, map_rect: rl.Rectangle, draw_fade: bool = True) -> None:
    rl.begin_scissor_mode(int(map_rect.x), int(map_rect.y), int(map_rect.width), int(map_rect.height))
    self._draw_fallback_background(map_rect)
    rotation = self.display_bearing % 360.0
    rotated = min(rotation, 360.0 - rotation) > 0.2
    if rotated:
      rl.rl_push_matrix()
      rl.rl_translatef(map_rect.x + map_rect.width * 0.5, map_rect.y + map_rect.height * 0.5, 0.0)
      rl.rl_rotatef(-self.display_bearing, 0.0, 0.0, 1.0)
      rl.rl_translatef(-(map_rect.x + map_rect.width * 0.5), -(map_rect.y + map_rect.height * 0.5), 0.0)
    if self._online_maps_enabled:
      if self._offline_maps_enabled and self._offline.has_content():
        # Offline underlay fills any tiles mapbox is missing; it only holds content while
        # mapbox is degraded (released ~60s after recovery), so this is a no-op when healthy.
        self._offline.draw(map_rect, self.display_center_latitude, self.display_center_longitude, self.display_zoom)
      self._mapbox.draw(map_rect, self.display_center_latitude, self.display_center_longitude, self.display_zoom)
    elif self._offline_maps_enabled:
      self._offline.draw(map_rect, self.display_center_latitude, self.display_center_longitude, self.display_zoom)
    if rotated:
      rl.rl_pop_matrix()
    if draw_fade:
      fade = rl.Rectangle(map_rect.x, map_rect.y + map_rect.height - 128, map_rect.width, 128)
      rl.draw_rectangle_gradient_v(int(fade.x), int(fade.y), int(fade.width), int(fade.height), rl.Color(0, 0, 0, 0), rl.Color(4, 10, 16, 170))
    rl.end_scissor_mode()

  def _draw_route_overlay(self, map_rect: rl.Rectangle, pose, camera):
    center_latitude, center_longitude, display_zoom, display_bearing = camera
    if self.nav_active and len(self._projected_route_points) >= 2 and self._route_camera is not None:
      latitude, longitude, zoom, width, height = self._route_camera
      scale = 2.0 ** (display_zoom - zoom)
      center_x, center_y = project_nav_point(latitude, longitude, center_latitude,
                                             center_longitude, display_zoom, 0.0,
                                             map_rect.width, map_rect.height)
      offset_x = map_rect.x + center_x - width * scale * 0.5
      offset_y = map_rect.y + center_y - height * scale * 0.5
      marker_x, marker_y = project_nav_point(pose[0], pose[1], latitude, longitude, zoom, 0.0, width, height)
      first, fraction = closest_polyline_segment(self._projected_route_points, marker_x, marker_y)
      radius = math.hypot(map_rect.width, map_rect.height) * 0.5 + 16.0
      mid_x, mid_y = map_rect.x + map_rect.width * 0.5, map_rect.y + map_rect.height * 0.5
      for idx in range(first, len(self._projected_route_points) - 1):
        x1, y1 = self._projected_route_points[idx]
        x2, y2 = self._projected_route_points[idx + 1]
        if idx == first:
          x1 += (x2 - x1) * fraction
          y1 += (y2 - y1) * fraction
        if x1 == x2 and y1 == y2:
          continue
        x1, y1 = offset_x + x1 * scale, offset_y + y1 * scale
        x2, y2 = offset_x + x2 * scale, offset_y + y2 * scale
        if (min(x1, x2) > mid_x + radius or max(x1, x2) < mid_x - radius
            or min(y1, y2) > mid_y + radius or max(y1, y2) < mid_y - radius):
          continue
        p1 = rl.Vector2(x1, y1)
        p2 = rl.Vector2(x2, y2)
        rl.draw_line_ex(p1, p2, 16.0, rl.Color(8, 18, 30, 220))
        rl.draw_line_ex(p1, p2, 10.0, rl.Color(255, 255, 255, 210))
        rl.draw_line_ex(p1, p2, 6.0, rl.Color(24, 194, 181, 255))

    if self.nav_active and abs(self.destination_latitude) > 0.001 and abs(self.destination_longitude) > 0.001:
      dest_x, dest_y = project_nav_point(
        self.destination_latitude,
        self.destination_longitude,
        center_latitude,
        center_longitude,
        display_zoom,
        0.0,
        map_rect.width,
        map_rect.height,
      )
      center = rl.Vector2(map_rect.x + dest_x, map_rect.y + dest_y)
      rl.draw_circle_v(center, 12.0, rl.Color(7, 18, 26, 235))
      rl.draw_circle_lines(int(center.x), int(center.y), 12.0, rl.Color(255, 255, 255, 150))
      texture = self._destination_texture()
      if texture is not None:
        src = rl.Rectangle(0, 0, float(texture.width), float(texture.height))
        # counter-rotate about its own center so the flag stays upright in heading-up
        dst = rl.Rectangle(center.x, center.y, 18, 22)
        rl.draw_texture_pro(texture, src, dst, rl.Vector2(9, 11), display_bearing, rl.WHITE)

  def _update_pointer_pose(self, now: float) -> tuple[float, float, float]:
    sm = ui_state.sm
    car = sm['carState']
    motion_valid = sm.valid['carState'] and sm.alive['carState'] and 0 <= now - sm.recv_time['carState'] < 0.5
    motion_valid = motion_valid and str(car.gearShifter) != 'reverse'
    return self._marker_motion.update(now, self._last_position_change, self.current_latitude, self.current_longitude,
                                      self.bearing_deg, float(car.vEgo), float(car.yawRate), motion_valid)

  def _draw_live_pointer(self, rect: rl.Rectangle) -> None:
    if not self._has_render_fix or self._map_camera is None:
      self._marker_motion = MapMarkerMotion()
      return
    pose = self._update_pointer_pose(time.monotonic())
    if pose is None:
      return
    latitude, longitude, heading = pose
    center_latitude, center_longitude, zoom, bearing = self._map_camera
    # Project against the camera baked into the map texture, not the next camera update.
    x, y = project_nav_point(latitude, longitude, center_latitude, center_longitude,
                             zoom, bearing, rect.width, rect.height)
    top = rect.y + 24
    bottom = rect.y + rect.height - 24
    rl.begin_scissor_mode(int(rect.x + 24), int(top), int(rect.width - 48), max(0, int(bottom - top)))
    map_rect = rect
    rl.rl_push_matrix()
    rl.rl_translatef(map_rect.x + map_rect.width * 0.5, map_rect.y + map_rect.height * 0.5, 0.0)
    rl.rl_rotatef(-bearing, 0.0, 0.0, 1.0)
    rl.rl_translatef(-(map_rect.x + map_rect.width * 0.5), -(map_rect.y + map_rect.height * 0.5), 0.0)
    self._draw_route_overlay(map_rect, pose, self._map_camera)
    rl.rl_pop_matrix()
    self._draw_ego_arrow(rect.x + x, rect.y + y, heading - bearing)
    rl.end_scissor_mode()

  def _draw_maneuver_icon(self, bounds: rl.Rectangle):
    texture = self._icon_texture()
    if texture is not None and self._display_next_valid():
      src = rl.Rectangle(0, 0, float(texture.width), float(texture.height))
      rl.draw_texture_pro(texture, src, bounds, rl.Vector2(0, 0), 0.0, rl.WHITE)

  def render_details(self, rect: rl.Rectangle) -> None:
    if not self._display_next_valid():
      return
    fade = gui_app.texture('icons/onroad/nav_instruction_fade.png')
    rl.draw_texture_pro(fade, rl.Rectangle(0, 0, fade.width, fade.height),
                        rl.Rectangle(rect.x - 144, rect.y - 160, gui_app.width - rect.x + 144, gui_app.height - rect.y + 160),
                        rl.Vector2(0, 0), 0, rl.WHITE)
    self._draw_maneuver_icon(rl.Rectangle(rect.x + 12, rect.y + 7, 64, 64))
    distance = f"{self.next_distance / 1609.344:.1f} mi"
    if self.next_distance < 160.9344:
      distance = "< 0.1 mi"
    rl.draw_text_ex(self._title_font, distance, rl.Vector2(rect.x + 96, rect.y + 13), 44, 0, rl.WHITE)

  def _ensure_rt(self, attr: str, width: int, height: int):
    rt = getattr(self, attr)
    if rt is not None and (rt.texture.width != width or rt.texture.height != height):
      rl.unload_render_texture(rt)
      rt = None
      setattr(self, attr, None)
    if rt is None:
      try:
        rt = rl.load_render_texture(width, height)
        if not rl.is_render_texture_valid(rt):
          return None
        rl.set_texture_filter(rt.texture, rl.TextureFilter.TEXTURE_FILTER_BILINEAR)
        setattr(self, attr, rt)
      except Exception:
        setattr(self, attr, None)
        return None
    return rt

  @staticmethod
  @lru_cache(maxsize=4)
  def _rounded_perimeter(width: int, height: int):
    radius = min(24, width / 2, height / 2)
    corners = ((width - radius, radius, -90), (width - radius, height - radius, 0),
               (radius, height - radius, 90), (radius, radius, 180))
    perimeter = [(cx + radius * math.cos(math.radians(angle + step * 7.5)),
                  cy + radius * math.sin(math.radians(angle + step * 7.5)))
                 for cx, cy, angle in corners for step in range(13)]
    return perimeter

  @staticmethod
  def _blit_rt(rt, x: float, y: float) -> None:
    # RenderTexture color buffers are stored bottom-up, so flip vertically via negative src height.
    tex = rt.texture
    width, height = tex.width, tex.height
    perimeter = NavMapPanel._rounded_perimeter(width, height)
    rl.rl_set_texture(tex.id)
    rl.rl_begin(rl.RL_QUADS)
    rl.rl_color4ub(255, 255, 255, 255)
    for index, point in enumerate(perimeter):
      for px, py in ((width / 2, height / 2), perimeter[(index + 1) % len(perimeter)], point, point):
        rl.rl_tex_coord2f(px / width, 1 - py / height)
        rl.rl_vertex2f(x + px, y + py)
    rl.rl_end()
    rl.rl_set_texture(0)

  def _render(self, rect: rl.Rectangle):
    self.render_split(rect)

  def _draw_split_scene(self, rect: rl.Rectangle) -> None:
    local_rect = rl.Rectangle(rect.x, rect.y, rect.width, rect.height)
    map_rect = local_rect
    self._map_viewport_width = map_rect.width
    self._map_viewport_height = map_rect.height
    rotation = self.display_bearing % 360.0
    bearing = self.display_bearing if min(rotation, 360.0 - rotation) > 0.2 else 0.0
    self._map_camera = (self.display_center_latitude, self.display_center_longitude, self.display_zoom, bearing)
    self._draw_map_surface(map_rect, draw_fade=False)

  def _refresh_split_texture(self, width: int, height: int) -> None:
    target = self._ensure_rt("_split_rt", width, height)
    if target is None:
      return
    rl.begin_texture_mode(target)
    try:
      rl.clear_background(rl.BLANK)
      self._draw_split_scene(rl.Rectangle(0, 0, width, height))
    finally:
      rl.end_texture_mode()
    self._last_split_render = time.monotonic()

  def render_split(self, rect: rl.Rectangle) -> None:
    width, height = int(rect.width), int(rect.height)
    resized = self._split_rt is None or self._split_rt.texture.width != width or self._split_rt.texture.height != height
    interval = PANEL_RENDER_INTERVAL if self._has_render_fix else 1.0
    if resized or time.monotonic() - self._last_split_render >= interval:
      gui_app.queue_render_job(self, lambda: self._refresh_split_texture(width, height))
    if resized:
      self._draw_fallback_background(rect)
      return
    self._blit_rt(self._split_rt, rect.x, rect.y)
    self._draw_live_pointer(rect)
