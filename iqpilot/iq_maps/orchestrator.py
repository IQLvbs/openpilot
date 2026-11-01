#!/usr/bin/env python3
"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos
"""
import platform
import os
import glob
import shutil
import signal
import subprocess
import threading
import time
from datetime import datetime

import iqpilot.cereal.messaging as messaging
from iqpilot.cereal import custom
from iqpilot.common.params import Params
from iqpilot.common.realtime import Ratekeeper, config_realtime_process
from iqpilot.common.swaglog import cloudlog
from iqpilot.selfdrive.selfdrived.alertmanager import set_offroad_alert
from iqpilot.system.hardware.hw import Paths
from iqpilot.iq_maps import VENDOR_MAPD_BIN_DIR, VENDOR_MAPD_PATH
from iqpilot.iq_maps.tile_bundle_downloader import (TileBundleDownloader, region_bundle_installed, region_roads_path,
                                                    region_valhalla_installed)
from iqpilot.iq_maps.vendor_mapd_installer import VendorMapdInstaller

OfflineMapAction = custom.MapdInputType
_region_sync_worker: threading.Thread | None = None

_active_proc_lock = threading.Lock()
_active_proc: subprocess.Popen | None = None
_shutdown = threading.Event()
_tile_downloader: TileBundleDownloader | None = None


def _vendor_fetch_pidfile() -> str:
  return os.path.join(Paths.mapd_root(), ".vendor_fetch.pid")


def _pid_is_vendor_fetch(pid: int) -> bool:
  try:
    with open(f"/proc/{pid}/cmdline", "rb") as f:
      cmdline = f.read()
  except OSError:
    return False
  return VENDOR_MAPD_PATH.encode() in cmdline


def _reap_orphaned_vendor_fetch() -> None:
  pidfile = _vendor_fetch_pidfile()
  try:
    with open(pidfile) as f:
      pid = int(f.read().strip())
  except (OSError, ValueError):
    return
  try:
    if _pid_is_vendor_fetch(pid):
      cloudlog.warning(f"iq_maps: reaping orphaned vendor-fetch mapd pid={pid} from a prior run")
      os.kill(pid, signal.SIGTERM)
      for _ in range(20):
        time.sleep(0.1)
        if not _pid_is_vendor_fetch(pid):
          break
      else:
        os.kill(pid, signal.SIGKILL)
  except ProcessLookupError:
    pass
  finally:
    try:
      os.remove(pidfile)
    except OSError:
      pass


def _kill_active_proc() -> None:
  with _active_proc_lock:
    proc = _active_proc
  if proc is None or proc.poll() is not None:
    return
  proc.terminate()
  try:
    proc.wait(timeout=3)
  except Exception:
    proc.kill()
    try:
      proc.wait(timeout=2)
    except Exception:
      pass


def _handle_shutdown_signal(signum, _frame) -> None:
  cloudlog.warning(f"iq_maps: mapd_manager received signal {signum}, cleaning up vendor-fetch subprocess")
  _shutdown.set()
  _kill_active_proc()
  if _tile_downloader is not None:
    _tile_downloader.cancel()
  worker = _region_sync_worker
  if worker is not None and worker.is_alive():
    worker.join(timeout=3)
  raise SystemExit(0)


def _install_signal_handlers() -> None:
  signal.signal(signal.SIGINT, _handle_shutdown_signal)
  signal.signal(signal.SIGTERM, _handle_shutdown_signal)


def ensure_vendor_runtime() -> None:
  try:
    VendorMapdInstaller().verify()
  except Exception:
    cloudlog.exception("iq_maps: vendor runtime verification failed")

params = Params()
mem_params = Params("/dev/shm/params") if platform.system() != "Darwin" else params


def stale_region_artifacts() -> list[str]:
  patterns = [
    f"{Paths.mapd_root()}/db",
    f"{Paths.mapd_root()}/v*"
  ]
  stale_paths: list[str] = []
  for pattern in patterns:
    for match in glob.glob(pattern):
      stale_paths.append(match)
      if os.path.isdir(match):
        stale_paths.extend(glob.glob(match + '/**', recursive=True))
  if not os.path.isfile(VENDOR_MAPD_PATH):
    stale_paths.append(VENDOR_MAPD_PATH)
  return stale_paths


def purge_stale_region_artifacts(stale_paths: list[str]) -> None:
  for candidate in stale_paths:
    if candidate.endswith('/') and os.path.isfile(candidate[:-1]):
      candidate = candidate[:-1]
    if os.path.islink(candidate) or os.path.isfile(candidate):
      os.remove(candidate)
    elif os.path.isdir(candidate):
      shutil.rmtree(candidate, ignore_errors=False)


def _compose_region_selector(nations: list[str], states: list[str] | None = None) -> str:
  requested_paths: list[str] = []
  for state_code in (states or []):
    code = str(state_code).strip().upper()
    if code and code != "ALL":
      requested_paths.append(f"us_state.{code}")
  for nation_code in (nations or []):
    code = str(nation_code).strip().upper()
    if code:
      requested_paths.append(f"nation.{code}")
  return ",".join(requested_paths)


def _fetch_tile_bundles(region_selector: str, abort_check=None) -> None:
  global _tile_downloader
  include_display = params.get_bool("OfflineOSMaps") or params.get_bool("OsmLocal")
  selectors = [part for part in region_selector.split(",") if part]
  if not selectors:
    return
  try:
    _tile_downloader = TileBundleDownloader(params=params, mem_params=mem_params, abort_check=abort_check, include_display=include_display)
    _tile_downloader.download_regions(selectors)
  except Exception:
    cloudlog.exception("iq_maps: tile bundle download failed")
  finally:
    _tile_downloader = None


def _drive_vendor_fetch(region_selector: str, requested_regions: dict) -> None:
  global _active_proc
  proc = None
  cancelled = False
  try:
    mem_params.put("OSMDownloadLocations", requested_regions)
    proc = subprocess.Popen([VENDOR_MAPD_PATH], cwd=VENDOR_MAPD_BIN_DIR,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            start_new_session=True)
    with _active_proc_lock:
      _active_proc = proc
    with open(_vendor_fetch_pidfile(), "w") as f:
      f.write(str(proc.pid))

    pm = messaging.PubMaster(["mapdIn"])
    sm = messaging.SubMaster(["mapdExtendedOut"])
    time.sleep(4.0)

    for _ in range(10):
      msg = messaging.new_message("mapdIn")
      msg.mapdIn.type = OfflineMapAction.download
      msg.mapdIn.str = region_selector
      pm.send("mapdIn", msg)
      time.sleep(0.2)

    started = False
    deadline = time.monotonic() + 3600.0
    while time.monotonic() < deadline and not _shutdown.is_set():
      sm.update(500)
      dp = sm["mapdExtendedOut"].downloadProgress
      mem_params.put("OSMDownloadProgress", {
        "active": bool(dp.active),
        "total_files": int(dp.totalFiles),
        "downloaded_files": int(dp.downloadedFiles),
      })
      if dp.active:
        started = True
      elif started:
        break
      if not mem_params.get("OSMDownloadLocations"):
        cancelled = True
        cancel = messaging.new_message("mapdIn")
        cancel.mapdIn.type = OfflineMapAction.cancelDownload
        pm.send("mapdIn", cancel)
        break
    cloudlog.info(f"iq_maps: vendor map download finished for {region_selector}")
    if not cancelled and not _shutdown.is_set():
      _fetch_tile_bundles(region_selector, abort_check=lambda: _shutdown.is_set() or not mem_params.get("OSMDownloadLocations"))
  except Exception:
    cloudlog.exception("iq_maps: vendor map download failed")
  finally:
    try:
      mem_params.remove("OSMDownloadLocations")
    except Exception:
      pass
    if proc is not None:
      proc.terminate()
      try:
        proc.wait(timeout=5)
      except Exception:
        proc.kill()
    with _active_proc_lock:
      _active_proc = None
    try:
      os.remove(_vendor_fetch_pidfile())
    except OSError:
      pass


def queue_region_refresh(nations: list[str], states: list[str] | None = None) -> None:
  global _region_sync_worker
  params.put("OsmDownloadedDate", str(datetime.now().timestamp()))
  params.put_bool("OsmDbUpdatesCheck", False)

  region_selector = _compose_region_selector(nations, states)
  if not region_selector:
    cloudlog.warning("iq_maps: no region selected for offline map download")
    return
  if _region_sync_worker is not None and _region_sync_worker.is_alive():
    cloudlog.warning("iq_maps: vendor map download already in progress")
    return

  requested_regions = {"nations": nations, "states": states or [], "paths": region_selector}
  cloudlog.info(f"iq_maps: starting vendor map download for {region_selector}")
  _region_sync_worker = threading.Thread(
    target=_drive_vendor_fetch,
    args=(region_selector, requested_regions),
    daemon=True,
  )
  _region_sync_worker.start()


def normalize_region_selection(nations: list[str], states: list[str] | None = None) -> tuple[list[str], list[str]]:
  normalized_nations = list(nations)
  normalized_states = list(states or [])
  lowered_states = {entry.lower() for entry in normalized_states}

  if "US" in normalized_nations and normalized_states and "all" not in lowered_states:
    normalized_nations = [entry for entry in normalized_nations if entry != "US"]
  elif normalized_states:
    normalized_states = [entry for entry in normalized_states if entry.lower() != "all"]

  return normalized_nations, normalized_states


_AUTO_RESTORE_INTERVAL_S = 1800.0
_last_auto_restore_t = 0.0


def region_data_missing() -> bool:
  if not params.get_bool("OsmLocal"):
    return False
  if not params.get("OsmDownloadedDate"):
    return False
  if glob.glob(f"{Paths.mapd_root()}/db") or glob.glob(f"{Paths.mapd_root()}/v*"):
    return False
  if glob.glob(f"{Paths.mapd_root()}/offline/*/*"):
    return False
  country = params.get("OsmLocationName", return_default=True)
  return bool(country)


def configured_states() -> list[str]:
  try:
    states = params.get("OsmStateNames")
    if isinstance(states, bytes):
      import json as _json
      states = _json.loads(states.decode("utf-8"))
    if isinstance(states, str):
      import json as _json
      states = _json.loads(states)
    if isinstance(states, list) and states:
      return [str(s).strip().upper() for s in states if str(s).strip()]
  except Exception:
    pass
  state = params.get("OsmStateName", return_default=True)
  return [state] if state else []


def maybe_auto_restore_region() -> None:
  global _last_auto_restore_t
  if not region_data_missing():
    return
  if _region_sync_worker is not None and _region_sync_worker.is_alive():
    return
  now = time.monotonic()
  if now - _last_auto_restore_t < _AUTO_RESTORE_INTERVAL_S:
    return
  _last_auto_restore_t = now
  country = params.get("OsmLocationName", return_default=True)
  states = configured_states()
  nations, states_filtered = normalize_region_selection([country], states)
  cloudlog.warning(f"iq_maps: configured offline region {country}/{states} has no data on disk; auto-restoring")
  queue_region_refresh(nations, states_filtered)


_TILE_RESTORE_INTERVAL_S = 1800.0
_last_tile_restore_t = 0.0
_tile_only_worker: threading.Thread | None = None


def _configured_region_selector() -> str:
  country = params.get("OsmLocationName", return_default=True)
  states = configured_states()
  nations, states_filtered = normalize_region_selection([country] if country else [], states)
  return _compose_region_selector(nations, states_filtered)


def tile_bundles_missing() -> bool:
  selector = _configured_region_selector()
  if not selector:
    return False
  parts = [part for part in selector.split(",") if part]
  if (params.get_bool("OfflineOSMaps") or params.get_bool("OsmLocal")) and any(not region_bundle_installed(part) for part in parts):
    return True
  return any((region_bundle_installed(part) or region_valhalla_installed(part)) and not region_roads_path(part).exists() for part in parts)


def maybe_restore_tile_bundles() -> None:
  global _last_tile_restore_t, _tile_only_worker
  if not tile_bundles_missing():
    return
  if _region_sync_worker is not None and _region_sync_worker.is_alive():
    return
  if _tile_only_worker is not None and _tile_only_worker.is_alive():
    return
  now = time.monotonic()
  if now - _last_tile_restore_t < _TILE_RESTORE_INTERVAL_S:
    return
  _last_tile_restore_t = now
  selector = _configured_region_selector()
  cloudlog.warning(f"iq_maps: offline map tile bundles missing for {selector}; downloading")
  _tile_only_worker = threading.Thread(
    target=_fetch_tile_bundles,
    args=(selector,),
    kwargs={"abort_check": _shutdown.is_set},
    daemon=True,
  )
  _tile_only_worker.start()


def sync_osm_request_flags() -> None:
  maybe_auto_restore_region()
  maybe_restore_tile_bundles()
  if params.get_bool("OsmDbUpdatesCheck"):
    if _region_sync_worker is not None and _region_sync_worker.is_alive():
      return
    purge_stale_region_artifacts(stale_region_artifacts())
    country = params.get("OsmLocationName", return_default=True)
    states = configured_states()
    filtered_nations, filtered_states = normalize_region_selection([country], states)
    queue_region_refresh(filtered_nations, filtered_states)

  if not mem_params.get("OSMDownloadBounds"):
    mem_params.put("OSMDownloadBounds", "")

  if not mem_params.get("LastGPSPosition"):
    mem_params.put("LastGPSPosition", "{}")


def run_loop():
  ensure_vendor_runtime()
  config_realtime_process([0, 1, 2, 3], 5)

  rk = Ratekeeper(1, print_delay_threshold=None)

  try:
    os.mkdir(Paths.mapd_root())
  except FileExistsError:
    pass
  except PermissionError:
    cloudlog.exception(f"iq_maps: failed to make {Paths.mapd_root()}")
  _reap_orphaned_vendor_fetch()
  _install_signal_handlers()

  while not _shutdown.is_set():
    show_alert = stale_region_artifacts() and params.get_bool("OsmLocal")
    set_offroad_alert("Offroad_OSMUpdateRequired", show_alert, "This alert will be cleared when new maps are downloaded.")

    sync_osm_request_flags()
    rk.keep_time()


def main():
  run_loop()


if __name__ == "__main__":
  main()
