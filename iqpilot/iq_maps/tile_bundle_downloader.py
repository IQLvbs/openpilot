#!/usr/bin/env python3
"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos
"""
import hashlib
import json
import platform
import threading
import time
from pathlib import Path

import requests

from iqpilot.common.params import Params
from iqpilot.common.swaglog import cloudlog
from iqpilot.ui.onroad.offline_tiles import offline_map_root

_BROKER_TOKEN_TTL_S = 50 * 60
_broker_lock = threading.Lock()
_broker_state: dict = {"token": "", "dongle": "", "until": 0.0}
_IQMAPS_RAW_PREFIXES = (
  "https://git.konn3kt.com/IQ.Lvbs/iqmaps/raw/branch/master/",
  "https://gitlvb.teallvbs.xyz/IQ.Lvbs/iqmaps/raw/branch/master/",
)
_HF_IQMAPS_PREFIX = "https://huggingface.co/datasets/T3vl/iqmaps/resolve/main/"


def _broker_dongle() -> str:
  try:
    value = Params().get("DongleId")
  except Exception:
    return ""
  value = value.decode() if isinstance(value, bytes) else (value or "")
  return "" if value in ("", "UnregisteredDevice") else value


def _broker_headers() -> dict:
  with _broker_lock:
    now = time.monotonic()
    if _broker_state["token"] and now < _broker_state["until"]:
      return {"Authorization": f"JWT {_broker_state['token']}"}
    dongle = _broker_dongle()
    if not dongle:
      return {}
    try:
      from iqpilot.common.api import Api
      token = Api(dongle).get_token(expiry_hours=1)
    except Exception:
      return {}
    _broker_state.update(token=token, dongle=dongle, until=now + _BROKER_TOKEN_TTL_S)
    return {"Authorization": f"JWT {token}"}


def _broker_asset_base() -> str:
  dongle = _broker_state["dongle"] or _broker_dongle()
  if not dongle:
    return ""
  try:
    from iqpilot.common.api.konn3kt import API_HOST
  except Exception:
    return ""
  return API_HOST.rstrip('/') + "/v1/devices/" + dongle + "/assets"


def _broker_iqmaps_url(url: str) -> str:
  base = _broker_asset_base()
  if not base:
    return ""
  for prefix in _IQMAPS_RAW_PREFIXES:
    if url.startswith(prefix):
      return base + "/iqmaps/" + url[len(prefix):]
  if url.startswith(_HF_IQMAPS_PREFIX):
    return base + "/iqmaps/" + url[len(_HF_IQMAPS_PREFIX):]
  return ""


def _broker_iqmaps_lfs_url(oid: str, size: int) -> str:
  base = _broker_asset_base()
  return (base + "/iqmaps-lfs/" + oid + "?size=" + str(int(size))) if base else ""

HF_TILE_BUNDLE_BASE_URL = "https://huggingface.co/datasets/T3vl/iqmaps/resolve/main"
DEFAULT_TILE_BUNDLE_BASE_URL = "https://git.konn3kt.com/IQ.Lvbs/iqmaps/raw/branch/master"
FALLBACK_TILE_BUNDLE_BASE_URL = "https://gitlvb.teallvbs.xyz/IQ.Lvbs/iqmaps/raw/branch/master"

LFS_POINTER_MAGIC = b"version https://git-lfs"
BASE_URL_PARAM = "OfflineTilesBaseUrl"
PROGRESS_PARAM = "OfflineTilesDownloadProgress"
REQUEST_PARAM = "OfflineTilesDownloadRequest"
CHUNK_BYTES = 1 << 20
# must match scripts/iqpilot/tile_factory/upload_bundles_lfs.py
PART_BYTES = 90 * 1024 * 1024
HTTP_TIMEOUT_S = 30.0
STREAM_RETRIES = 8


_BROKER_SENTINEL = "broker://"


def _broker_base_url() -> str:
  if _broker_headers is None:
    return ""
  headers = _broker_headers()
  if not headers:
    return ""
  return _BROKER_SENTINEL


def candidate_base_urls(params: Params) -> list[str]:
  override = params.get(BASE_URL_PARAM)
  if isinstance(override, bytes):
    override = override.decode("utf-8", errors="ignore")
  override = (override or "").strip()
  if override:
    return [override.rstrip("/")]
  urls: list[str] = []
  broker = _broker_base_url()
  if broker:
    urls.append(broker)
  urls.append(HF_TILE_BUNDLE_BASE_URL)
  urls.append(DEFAULT_TILE_BUNDLE_BASE_URL)
  urls.append(FALLBACK_TILE_BUNDLE_BASE_URL)
  seen: set[str] = set()
  return [u for u in urls if not (u in seen or seen.add(u))]


def _is_hf(url: str) -> bool:
  return "huggingface.co" in url.lower()


def _is_broker(url: str) -> bool:
  return url.startswith(_BROKER_SENTINEL)


def request_headers(url: str) -> dict:
  if _is_broker(url) and _broker_headers is not None:
    return _broker_headers()
  return {}


def _lfs_endpoint(base_url: str) -> str:
  return base_url.split("/raw/", 1)[0] + ".git/info/lfs"


def _resolve_oid_url_broker(oid: str, size: int) -> tuple[str, dict]:
  if _broker_iqmaps_lfs_url is None:
    raise requests.RequestException("broker LFS unavailable")
  url = _broker_iqmaps_lfs_url(oid, size)
  if not url:
    raise requests.RequestException("broker LFS URL empty")
  headers = _broker_headers() if _broker_headers is not None else {}
  if not headers:
    raise requests.RequestException("broker headers unavailable")
  return url, headers


def _resolve_oid_url(session: requests.Session, base_url: str, oid: str, size: int,
                     headers: dict) -> tuple[str, dict]:
  if _is_broker(base_url):
    return _resolve_oid_url_broker(oid, size)
  batch = session.post(f"{_lfs_endpoint(base_url)}/objects/batch",
                       data=json.dumps({"operation": "download", "transfers": ["basic"],
                                        "objects": [{"oid": oid, "size": size}]}),
                       headers={"Content-Type": "application/vnd.git-lfs+json",
                                "Accept": "application/vnd.git-lfs+json", **headers},
                       timeout=HTTP_TIMEOUT_S)
  batch.raise_for_status()
  entry = batch.json()["objects"][0]
  if "actions" not in entry:
    raise requests.RequestException(f"LFS object unavailable: {entry.get('error', oid)}")
  action = entry["actions"]["download"]
  return action["href"], action.get("header", {})


def _resolve_object_url_broker(url: str) -> tuple[str, dict]:
  headers = _broker_headers() if _broker_headers is not None else {}
  return url, headers


def _resolve_object_url(session: requests.Session, url: str, headers: dict) -> tuple[str, dict]:
  if _is_broker(url):
    return _resolve_object_url_broker(url)
  probe = session.get(url, headers={**headers, "Accept-Encoding": None}, stream=True,
                      timeout=HTTP_TIMEOUT_S)
  try:
    probe.raise_for_status()
    if int(probe.headers.get("content-length") or 0) >= 1024:
      return url, dict(headers)
    body = probe.content
  finally:
    probe.close()
  if not body.startswith(LFS_POINTER_MAGIC):
    return url, dict(headers)

  meta = dict(line.split(" ", 1) for line in body.decode().strip().splitlines() if " " in line)
  oid = meta["oid"].split(":", 1)[1]
  size = int(meta["size"])
  lfs_base = url.split("/raw/", 1)[0] + ".git/info/lfs"
  batch = session.post(f"{lfs_base}/objects/batch",
                       data=json.dumps({"operation": "download", "transfers": ["basic"],
                                        "objects": [{"oid": oid, "size": size}]}),
                       headers={"Content-Type": "application/vnd.git-lfs+json",
                                "Accept": "application/vnd.git-lfs+json", **headers},
                       timeout=HTTP_TIMEOUT_S)
  batch.raise_for_status()
  action = batch.json()["objects"][0]["actions"]["download"]
  return action["href"], action.get("header", {})


def _broker_index_url() -> str:
  if _broker_iqmaps_url is None:
    return ""
  return _broker_iqmaps_url(DEFAULT_TILE_BUNDLE_BASE_URL + "/index.json")


def fetch_index(base_url: str, session: requests.Session) -> dict:
  if _is_broker(base_url):
    index_url = _broker_index_url()
    if not index_url:
      raise ValueError("broker iqmaps URL unavailable")
    headers = request_headers(base_url)
  else:
    index_url = f"{base_url}/index.json"
    headers = request_headers(index_url)
  response = session.get(index_url, timeout=HTTP_TIMEOUT_S, headers=headers)
  response.raise_for_status()
  index = response.json()
  regions = index.get("regions")
  if not isinstance(regions, dict):
    raise ValueError("tile bundle index has no regions")
  return regions


def region_bundle_dir(selector: str) -> Path:
  return offline_map_root() / "regions" / selector


def region_bundle_path(selector: str) -> Path:
  return region_bundle_dir(selector) / "tiles" / "offline.mbtiles"


def region_valhalla_path(selector: str) -> Path:
  # valhalla mmaps this tar in place, so it stays uncompressed on disk
  return region_bundle_dir(selector) / "valhalla" / "tiles.tar"


def region_roads_path(selector: str) -> Path:
  return region_bundle_dir(selector) / "tiles" / "roads.mbtiles"


def region_valhalla_installed(selector: str) -> bool:
  return region_valhalla_path(selector).exists()


def installed_valhalla_selectors() -> list[str]:
  regions_root = offline_map_root() / "regions"
  if not regions_root.exists():
    return []
  return sorted(
    child.name for child in regions_root.iterdir()
    if child.is_dir() and (child / "valhalla" / "tiles.tar").exists()
  )


def region_bundle_installed(selector: str) -> bool:
  # The manifest is written only after a fully successful download, so a night-only partial (the
  # download died before the day tiles + manifest) must read as missing here or auto-restore never
  # finishes it. Require the manifest, plus the day file whenever the manifest declares one.
  if not region_bundle_path(selector).exists():
    return False
  try:
    manifest = json.loads((region_bundle_dir(selector) / "manifest.json").read_text())
  except (OSError, json.JSONDecodeError):
    return False
  if manifest.get("mbtiles_day") and not (region_bundle_dir(selector) / "tiles" / "offline_day.mbtiles").exists():
    return False
  return True


def _hash_existing(path: Path) -> tuple["hashlib._Hash", int]:
  digest = hashlib.sha256()
  size = 0
  with open(path, "rb") as f:
    while True:
      chunk = f.read(CHUNK_BYTES)
      if not chunk:
        break
      digest.update(chunk)
      size += len(chunk)
  return digest, size


def _write_manifest(selector: str, entry: dict) -> None:
  manifest = {
    "region": selector,
    "version": entry.get("version", ""),
    "mbtiles": {
      "bounds": entry.get("bounds", ""),
      "minzoom": entry.get("minzoom"),
      "maxzoom": entry.get("maxzoom"),
      "bytes": entry.get("bytes"),
      "sha256": entry.get("sha256", ""),
    },
  }
  if entry.get("day_path"):
    manifest["mbtiles_day"] = {
      "bytes": entry.get("day_bytes"),
      "sha256": entry.get("day_sha256", ""),
    }
  if entry.get("valhalla_path") and region_valhalla_path(selector).exists():
    manifest["valhalla"] = {
      "bytes": entry.get("valhalla_bytes"),
      "sha256": entry.get("valhalla_sha256", ""),
    }
  if entry.get("roads_path") and region_roads_path(selector).exists():
    manifest["roads"] = {
      "bytes": entry.get("roads_bytes"),
      "sha256": entry.get("roads_sha256", ""),
    }
  manifest_path = region_bundle_dir(selector) / "manifest.json"
  manifest_path.parent.mkdir(parents=True, exist_ok=True)
  manifest_path.write_text(json.dumps(manifest, indent=2))


class TileBundleDownloader:

  def __init__(self, params: Params | None = None, mem_params: Params | None = None,
               abort_check=None, include_display: bool = True):
    self.params = params if params is not None else Params()
    if mem_params is not None:
      self.mem_params = mem_params
    else:
      self.mem_params = Params("/dev/shm/params") if platform.system() != "Darwin" else self.params
    self.session = requests.Session()
    self._cancelled = threading.Event()
    self._abort_check = abort_check
    self.include_display = include_display

  def cancel(self) -> None:
    self._cancelled.set()

  def _should_abort(self) -> bool:
    if self._cancelled.is_set():
      return True
    if not self.mem_params.get(REQUEST_PARAM):
      self._cancelled.set()
      return True
    if self._abort_check is not None and self._abort_check():
      self._cancelled.set()
      return True
    return False

  def _publish_progress(self, region: str, downloaded: int, total: int, active: bool) -> None:
    self.mem_params.put(PROGRESS_PARAM, {
      "active": active,
      "region": region,
      "downloaded_bytes": int(downloaded),
      "total_bytes": int(total),
    })

  def _download_one(self, selector: str, entry: dict, base_url: str,
                    progress_offset: int, progress_total: int) -> bool:
    if not self.include_display:
      if not entry.get("valhalla_path"):
        cloudlog.warning(f"iq_maps: no routing extract published for {selector}")
        return False
      ok = self._download_file(
        selector, base_url, entry["valhalla_path"], int(entry.get("valhalla_bytes", 0)),
        str(entry.get("valhalla_sha256", "")).strip().lower(), region_valhalla_path(selector),
        progress_offset, progress_total, 1, entry.get("valhalla_objects"),
      )
      if ok:
        self._download_roads(selector, entry, base_url, progress_offset + int(entry.get("valhalla_bytes", 0)), progress_total)
        _write_manifest(selector, entry)
      return ok
    night_path = region_bundle_path(selector)
    ok = self._download_file(
      selector, base_url, entry["path"], int(entry.get("bytes", 0)),
      str(entry.get("sha256", "")).strip().lower(), night_path,
      progress_offset, progress_total, int(entry.get("parts", 1)), entry.get("objects"),
    )
    if not ok:
      return False
    if entry.get("day_path"):
      day_ok = self._download_file(
        selector, base_url, entry["day_path"], int(entry.get("day_bytes", 0)),
        str(entry.get("day_sha256", "")).strip().lower(),
        night_path.with_name("offline_day.mbtiles"),
        progress_offset + int(entry.get("bytes", 0)), progress_total,
        int(entry.get("day_parts", 1)), entry.get("day_objects"),
      )
      if not day_ok:
        cloudlog.warning(f"iq_maps: day-style bundle failed for {selector}; night set installed")
    if entry.get("valhalla_path"):
      # routing is additive: a region whose extract is missing or corrupt must still end up
      # with a usable map rather than failing the whole download
      try:
        nav_ok = self._download_file(
          selector, base_url, entry["valhalla_path"], int(entry.get("valhalla_bytes", 0)),
          str(entry.get("valhalla_sha256", "")).strip().lower(),
          region_valhalla_path(selector),
          progress_offset + int(entry.get("bytes", 0)) + int(entry.get("day_bytes", 0)),
          progress_total, 1, entry.get("valhalla_objects"),
        )
      except Exception as exc:
        nav_ok = False
        cloudlog.warning(f"iq_maps: routing extract errored for {selector}: {exc}")
      if not nav_ok:
        cloudlog.warning(f"iq_maps: routing extract failed for {selector}; map tiles installed")
    self._download_roads(selector, entry, base_url, progress_offset + int(entry.get("bytes", 0)) + int(entry.get("day_bytes", 0))
                         + int(entry.get("valhalla_bytes", 0)), progress_total)
    _write_manifest(selector, entry)
    cloudlog.info(f"iq_maps: installed tile bundle {selector}")
    return True

  def _download_roads(self, selector: str, entry: dict, base_url: str, progress_offset: int, progress_total: int) -> bool:
    if not entry.get("roads_path"):
      return False
    # roads are additive like routing: a missing or corrupt archive must never cost the region its map
    try:
      ok = self._download_file(
        selector, base_url, entry["roads_path"], int(entry.get("roads_bytes", 0)),
        str(entry.get("roads_sha256", "")).strip().lower(), region_roads_path(selector),
        progress_offset, progress_total, 1, entry.get("roads_objects"),
      )
    except Exception as exc:
      ok = False
      cloudlog.warning(f"iq_maps: roads archive errored for {selector}: {exc}")
    if not ok:
      cloudlog.warning(f"iq_maps: roads archive failed for {selector}")
    return ok

  def _download_file(self, selector: str, base_url: str, remote_path: str, expected_bytes: int,
                     expected_sha: str, final_path: Path,
                     progress_offset: int, progress_total: int, parts: int = 1,
                     objects: list | None = None) -> bool:
    # Bundles are published as <name>.pNN because Cloudflare caps proxied bodies at ~100MB.
    # They stream back-to-back into ONE .part file: concatenating afterwards would need
    # double the free space, which devices do not have.
    if _is_broker(base_url) and _broker_iqmaps_url is not None:
      broker_raw = _broker_iqmaps_url(DEFAULT_TILE_BUNDLE_BASE_URL + "/" + remote_path.lstrip('/'))
      base = broker_raw or f"{base_url}/{remote_path.lstrip('/')}"
    else:
      base = f"{base_url}/{remote_path.lstrip('/')}"
    count = len(objects) if objects else parts
    if objects and not _is_hf(base_url):
      urls = [None] * count
    else:
      urls = [base] if count <= 1 else [f"{base}.p{i:02d}" for i in range(count)]
    part_path = final_path.with_name(final_path.name + ".part")
    part_path.parent.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    digest = hashlib.sha256()
    last_error: Exception | None = None
    for attempt in range(STREAM_RETRIES):
      if self._should_abort():
        cloudlog.warning(f"iq_maps: tile bundle download cancelled for {selector}")
        return False
      if attempt:
        time.sleep(min(30.0, 2.0 * attempt))
      try:
        digest = hashlib.sha256()
        resume_from = 0
        if part_path.exists():
          digest, resume_from = _hash_existing(part_path)
          if expected_bytes and resume_from > expected_bytes:
            part_path.unlink()
            digest = hashlib.sha256()
            resume_from = 0

        # every part but the last is exactly PART_BYTES, so a byte offset maps to a part index
        first_part = resume_from // PART_BYTES if len(urls) > 1 else 0
        skip_in_part = resume_from - first_part * PART_BYTES if len(urls) > 1 else resume_from
        downloaded = resume_from
        mode = "ab" if resume_from else "wb"
        with open(part_path, mode) as f:
          for index in range(first_part, len(urls)):
            if objects and not _is_hf(base_url):
              url_headers = request_headers(base_url)
              object_url, object_headers = _resolve_oid_url(
                self.session, base_url, objects[index]["oid"], int(objects[index]["size"]),
                url_headers)
            else:
              url = urls[index]
              url_headers = request_headers(base_url) if _is_broker(base_url) else request_headers(url)
              object_url, object_headers = _resolve_object_url(self.session, url, url_headers)
            headers = dict(object_headers)
            offset = skip_in_part if index == first_part else 0
            if offset:
              headers["Range"] = f"bytes={offset}-"
            with self.session.get(object_url, headers=headers, stream=True,
                                  timeout=HTTP_TIMEOUT_S) as response:
              if offset and response.status_code != 206:
                # server ignored the range: restart this whole file cleanly
                f.close()
                part_path.unlink(missing_ok=True)
                raise requests.RequestException(f"range not honoured for part {index}")
              response.raise_for_status()
              for chunk in response.iter_content(chunk_size=CHUNK_BYTES):
                if self._should_abort():
                  cloudlog.warning(f"iq_maps: tile bundle download cancelled for {selector}")
                  return False
                f.write(chunk)
                digest.update(chunk)
                downloaded += len(chunk)
                self._publish_progress(selector, progress_offset + downloaded, progress_total, active=True)
        break
      except requests.RequestException as exc:
        last_error = exc
        cloudlog.warning(f"iq_maps: tile bundle stream interrupted for {selector} "
                         + f"(attempt {attempt + 1}/{STREAM_RETRIES}): {exc}")
    else:
      raise requests.RequestException(f"stream failed after {STREAM_RETRIES} attempts") from last_error

    if expected_bytes and downloaded != expected_bytes:
      cloudlog.error(f"iq_maps: tile bundle size mismatch for {selector}: {downloaded} != {expected_bytes}")
      part_path.unlink(missing_ok=True)
      return False
    if expected_sha and digest.hexdigest() != expected_sha:
      cloudlog.error(f"iq_maps: tile bundle sha256 mismatch for {selector}")
      part_path.unlink(missing_ok=True)
      return False

    part_path.replace(final_path)
    return True

  def download_regions(self, selectors: list[str]) -> bool:
    self._cancelled.clear()
    ok = True
    try:
      self.mem_params.put(REQUEST_PARAM, {"regions": list(selectors)})
      regions = None
      base_url = ""
      for candidate in candidate_base_urls(self.params):
        try:
          regions = fetch_index(candidate, self.session)
          base_url = candidate
          break
        except (requests.RequestException, ValueError, json.JSONDecodeError) as exc:
          cloudlog.warning(f"iq_maps: tile bundle index unavailable at {candidate}: {exc}")
      if regions is None:
        cloudlog.error("iq_maps: no tile bundle host reachable")
        return False

      wanted: list[tuple[str, dict]] = []
      roads_only: list[tuple[str, dict]] = []
      for selector in selectors:
        entry = regions.get(selector)
        if entry is None:
          cloudlog.warning(f"iq_maps: no tile bundle published for {selector}")
          ok = False
          continue
        current = ((self.include_display and region_bundle_installed(selector) and self._installed_matches(selector, entry))
                   or (not self.include_display and self._routing_matches(selector, entry)))
        if current:
          if not self._roads_matches(selector, entry):
            roads_only.append((selector, entry))
          continue
        wanted.append((selector, entry))

      progress_total = (sum(self._download_bytes(entry) for _, entry in wanted)
                        + sum(int(entry.get("roads_bytes", 0)) for _, entry in roads_only))
      progress_offset = 0
      for selector, entry in wanted:
        if self._should_abort():
          return False
        try:
          if not self._download_one(selector, entry, base_url, progress_offset, progress_total):
            ok = False
        except (requests.RequestException, OSError):
          cloudlog.exception(f"iq_maps: tile bundle download failed for {selector}")
          ok = False
        progress_offset += self._download_bytes(entry)
      for selector, entry in roads_only:
        if self._should_abort():
          return False
        if self._download_roads(selector, entry, base_url, progress_offset, progress_total):
          _write_manifest(selector, entry)
        progress_offset += int(entry.get("roads_bytes", 0))
      return ok
    finally:
      self._publish_progress("", 0, 0, active=False)
      try:
        self.mem_params.remove(REQUEST_PARAM)
      except Exception:
        pass

  @staticmethod
  def _routing_matches(selector: str, entry: dict) -> bool:
    if not entry.get("valhalla_path"):
      return False
    try:
      path = region_valhalla_path(selector)
      expected = str(entry.get("valhalla_sha256", "")).strip().lower()
      if not expected or path.stat().st_size != int(entry.get("valhalla_bytes", 0)):
        return False
      return _hash_existing(path)[0].hexdigest() == expected
    except (OSError, ValueError):
      return False

  @staticmethod
  def _roads_matches(selector: str, entry: dict) -> bool:
    if not entry.get("roads_path"):
      return True
    try:
      path = region_roads_path(selector)
      expected = str(entry.get("roads_sha256", "")).strip().lower()
      if not expected or path.stat().st_size != int(entry.get("roads_bytes", 0)):
        return False
      return _hash_existing(path)[0].hexdigest() == expected
    except (OSError, ValueError):
      return False

  def _download_bytes(self, entry: dict) -> int:
    display_bytes = int(entry.get("bytes", 0)) + int(entry.get("day_bytes", 0)) if self.include_display else 0
    return display_bytes + int(entry.get("valhalla_bytes", 0)) + int(entry.get("roads_bytes", 0))

  @staticmethod
  def _installed_matches(selector: str, entry: dict) -> bool:
    manifest_path = region_bundle_dir(selector) / "manifest.json"
    try:
      manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
      return False
    installed_sha = str(manifest.get("mbtiles", {}).get("sha256", "")).strip().lower()
    expected_sha = str(entry.get("sha256", "")).strip().lower()
    if not expected_sha or installed_sha != expected_sha:
      return False
    if entry.get("valhalla_path") and not TileBundleDownloader._routing_matches(selector, entry):
      return False
    if entry.get("day_path"):
      day_file = region_bundle_dir(selector) / "tiles" / "offline_day.mbtiles"
      installed_day = str(manifest.get("mbtiles_day", {}).get("sha256", "")).strip().lower()
      expected_day = str(entry.get("day_sha256", "")).strip().lower()
      if not day_file.exists() or installed_day != expected_day:
        return False
    return True
