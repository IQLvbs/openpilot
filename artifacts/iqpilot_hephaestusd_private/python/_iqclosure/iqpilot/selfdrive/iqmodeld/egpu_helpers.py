"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import signal
import sys
import tempfile
import time

from iqpilot.common.swaglog import cloudlog
import urllib.request
from pathlib import Path

from iqpilot.system.hardware.usb import egpu_dock_ready

USB_SYSFS_ROOT = "/sys/bus/usb/devices"
FIRMWARE_MIRROR = os.getenv("IQ_EGPU_FIRMWARE_MIRROR", "/data/firmware/tinygrad")
TINYGRAD_CACHE = "/data/.cache"

COMMA_LFS_BATCH_URL = "https://gitlab.com/commaai/openpilot-lfs.git/info/lfs/objects/batch"

DOWNLOAD_CHUNK = 4 * 1024 * 1024
USB_SIGNAL_POLL_S = int(os.getenv("IQ_EGPU_SIGNAL_POLL_US", "500")) / 1e6
EGPU_POWER_LIMIT_W = 100
# tinygrad's USB+AMD path holds an exclusive flock on <tmp>/am_usb:<bus>.lock for the life of the opener
DOCK_LOCK_GLOB = "am_usb*.lock"
DOCK_LOCK_OURS = ("egpu_prefetch", "compile_egpu_model")
DOCK_LOCK_FREE_TIMEOUT_S = 15.0
PR_SET_PDEATHSIG = 1


def dock_lock_holders(proc_root: str = "/proc", tmp_dir: str | None = None) -> list[tuple[int, str]]:
  locks = {os.path.realpath(p) for p in glob.glob(os.path.join(tmp_dir or tempfile.gettempdir(), DOCK_LOCK_GLOB))}
  if not locks:
    return []
  me, holders = os.getpid(), []
  for pid in os.listdir(proc_root):
    if not pid.isdigit() or int(pid) == me:
      continue
    fd_dir = os.path.join(proc_root, pid, "fd")
    try:
      fds = os.listdir(fd_dir)
    except OSError:
      continue
    for fd in fds:
      try:
        target = os.path.realpath(os.readlink(os.path.join(fd_dir, fd)))
      except OSError:
        continue
      if target in locks:
        try:
          with open(os.path.join(proc_root, pid, "cmdline"), "rb") as f:
            cmd = f.read().replace(b"\0", b" ").decode(errors="replace").strip()
        except OSError:
          cmd = "?"
        holders.append((int(pid), cmd))
        break
  return holders


def free_dock_lock(timeout_s: float = DOCK_LOCK_FREE_TIMEOUT_S, proc_root: str = "/proc", tmp_dir: str | None = None,
                   kill=os.kill, sleep=time.sleep, clock=time.monotonic) -> list[tuple[int, str]]:
  deadline = clock() + timeout_s
  while True:
    holders = dock_lock_holders(proc_root, tmp_dir)
    if not holders:
      return []
    for pid, cmd in holders:
      if any(tag in cmd for tag in DOCK_LOCK_OURS):
        cloudlog.warning(f"egpu dock lock held by our own {cmd!r} (pid {pid}); killing it")
        try:
          kill(pid, signal.SIGKILL)
        except OSError:
          pass
    if clock() >= deadline:
      return holders
    sleep(0.25)


def apply_dock_power_limit() -> None:
  # tinygrad caches getenv, so this must run before the first AMD device open in the process
  os.environ.setdefault("AM_POWER_LIMIT", str(EGPU_POWER_LIMIT_W))


def die_with_parent() -> None:
  if sys.platform != "linux":
    return
  import ctypes
  ctypes.CDLL(None, use_errno=True).prctl(PR_SET_PDEATHSIG, signal.SIGKILL)


def usbgpu_present(sysfs_root: str = USB_SYSFS_ROOT) -> bool:
  return egpu_dock_ready(Path(sysfs_root))


def egpu_present_consented(params, sysfs_root: str = USB_SYSFS_ROOT) -> bool:
  try:
    if params is not None and params.get_bool("IQEgpuDisabled"):
      return False
  except Exception:
    pass
  return usbgpu_present(sysfs_root)


def egpu_selected(params, sysfs_root: str = USB_SYSFS_ROOT) -> bool:
  try:
    if params is not None and params.get_bool("IQEgpuDisabled"):
      return False
    if params is not None and params.get_bool("IQEgpuEnabled"):
      return True
  except Exception:
    pass
  return usbgpu_present(sysfs_root)


def resolve_backend(emac_enabled: bool, egpu_enabled: bool, egpu_present: bool = False) -> str | None:
  if egpu_present:
    return "egpu"
  if emac_enabled:
    return "emac"
  if egpu_enabled:
    return "egpu"
  return None


def egpu_pkl_path(meta: dict) -> str:
  from iqpilot.system.hardware.hw import Paths
  return os.path.join(Paths.model_root(), f"egpu_{meta['key']}_{meta['sha256'][:8]}_amd_tinygrad.pkl")


def egpu_policy_pkl_path(meta: dict) -> str:
  from iqpilot.system.hardware.hw import Paths
  return os.path.join(Paths.model_root(), f"egpu_{meta['key']}_{meta['sha256'][:8]}_amd_policy.pkl")


def egpu_oob_pkl_path(meta: dict) -> str:
  from iqpilot.system.hardware.hw import Paths
  return os.path.join(Paths.model_root(), f"egpu_{meta['key']}_{meta['sha256'][:8]}_amd_policy_oob.pkl")


def egpu_model_oob_pkl_path(meta: dict) -> str:
  from iqpilot.system.hardware.hw import Paths
  return os.path.join(Paths.model_root(), f"egpu_{meta['key']}_{meta['sha256'][:8]}_amd_model_oob2.pkl")


def drop_frozen_model_oob(meta: dict) -> None:
  from iqpilot.system.hardware.hw import Paths
  # warp-on-dock bundles saved under the old name were built before the prune fix and replay capture-time feeds
  stale = os.path.join(Paths.model_root(), f"egpu_{meta['key']}_{meta['sha256'][:8]}_amd_model_oob.pkl")
  try:
    os.remove(stale)
    cloudlog.warning(f"removed frozen-feed warp-on-dock artifact {stale}")
  except OSError:
    pass


def onnx_cache_path(meta: dict) -> str:
  from iqpilot.system.hardware.hw import Paths
  return os.path.join(Paths.model_root(), f"{meta['model_name']}_{meta['sha256'][:8]}.onnx")


def _sha256_file(path: str) -> str:
  digest = hashlib.sha256()
  with open(path, "rb") as f:
    while chunk := f.read(DOWNLOAD_CHUNK):
      digest.update(chunk)
  return digest.hexdigest()


def quarantine_artifact(path: str, why: str) -> None:
  try:
    if os.path.isfile(path):
      os.replace(path, path + ".unusable")
  except OSError:
    try:
      os.remove(path)
    except OSError:
      pass


def local_onnx(meta: dict) -> str | None:
  path = onnx_cache_path(meta)
  if not os.path.isfile(path):
    return None
  size = int(meta.get("download", {}).get("size", 0))
  if size and os.path.getsize(path) != size:
    quarantine_artifact(path, "onnx size mismatch")
    return None
  if _sha256_file(path) != meta["sha256"]:
    quarantine_artifact(path, "onnx sha256 mismatch")
    return None
  return path


def resolve_download_url(download_url: str, sha256: str, size: int, timeout: float = 30.0) -> str:
  if download_url.startswith("commalfs:"):
    oid = download_url.split(":", 1)[1]
    body = json.dumps({"operation": "download", "transfers": ["basic"],
                       "objects": [{"oid": oid, "size": size}]}).encode()
    req = urllib.request.Request(COMMA_LFS_BATCH_URL, data=body, headers={
      "Accept": "application/vnd.git-lfs+json", "Content-Type": "application/vnd.git-lfs+json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
      d = json.load(r)
    return d["objects"][0]["actions"]["download"]["href"]
  return download_url


def download_onnx(meta: dict, progress_cb=None) -> str:
  from iqpilot.selfdrive.iqmodeld.egpu_model import download_descriptor
  download_url, size = download_descriptor(meta)
  if not download_url:
    raise RuntimeError(f"model {meta['key']} has no download source; stage the onnx at {onnx_cache_path(meta)}")

  path = onnx_cache_path(meta)
  os.makedirs(os.path.dirname(path), exist_ok=True)
  try:
    from iqpilot.selfdrive.iqmodeld.model_bundle_downloader import download_hf_file
    return download_hf_file(f"onnx/{meta['sha256']}.onnx", path, meta["sha256"], int(size or 0), progress_cb=progress_cb)
  except Exception as e:
    cloudlog.warning(f"onnx {meta['key']} unavailable from HF ({e}); falling back to {download_url.split(':', 1)[0]}")
  url = resolve_download_url(download_url, meta["sha256"], size)
  tmp = path + ".part"
  digest = hashlib.sha256()
  got = 0
  with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
    while chunk := r.read(DOWNLOAD_CHUNK):
      f.write(chunk)
      digest.update(chunk)
      got += len(chunk)
      if progress_cb is not None and size:
        progress_cb(got / size)
  if size and got != size:
    os.remove(tmp)
    raise RuntimeError(f"onnx download truncated: {got}/{size} bytes")
  if digest.hexdigest() != meta["sha256"]:
    os.remove(tmp)
    raise RuntimeError(f"onnx sha256 mismatch for {meta['key']}")
  os.replace(tmp, path)
  return path


ARTIFACT_PATHS = {
  "egpu_model_oob_artifact": egpu_model_oob_pkl_path,
  "egpu_oob_artifact": egpu_oob_pkl_path,
  "egpu_policy_artifact": egpu_policy_pkl_path,
  "egpu_artifact": egpu_pkl_path,
}


def download_precompiled(meta: dict, progress_cb=None, policy: bool = False, oob: bool = False, field: str | None = None) -> str | None:
  field = field or ("egpu_oob_artifact" if oob else "egpu_policy_artifact" if policy else "egpu_artifact")
  art = meta.get(field)
  if not art or not (art.get("objects") or art.get("hf_path")):
    return None
  from iqpilot.selfdrive.iqmodeld.model_bundle_downloader import download_hf_file, download_lfs_bundle
  dest = ARTIFACT_PATHS[field](meta)
  if art.get("hf_path"):
    try:
      return download_hf_file(art["hf_path"], dest, art["sha256"], int(art.get("size", 0)), progress_cb=progress_cb)
    except Exception as e:
      cloudlog.warning(f"precompiled {meta['key']} unavailable from HF ({e}); trying LFS")
      if not art.get("objects"):
        raise
  return download_lfs_bundle(art["objects"], dest, art["sha256"], int(art.get("size", 0)), progress_cb=progress_cb)


def patch_tinygrad_fetch_fw() -> None:
  import pathlib

  import zstandard
  from tinygrad import helpers
  if getattr(helpers.fetch_fw, "_iq_patched", False):
    return
  _orig = helpers.fetch_fw

  def fetch_fw(path, name, sha256):
    mirror = pathlib.Path(FIRMWARE_MIRROR) / path / name
    if mirror.is_file():
      blob = mirror.read_bytes()
      if hashlib.sha256(blob).hexdigest() == sha256:
        return blob
    p = pathlib.Path(f"/lib/firmware/{path}/{name}.zst")
    if p.is_file():
      blob = zstandard.ZstdDecompressor().stream_reader(p.read_bytes()).read()
      if hashlib.sha256(blob).hexdigest() == sha256:
        return blob
    blob = _orig(path, name, sha256)
    # The dock's GPU firmware otherwise lives only in tinygrad's per-user download cache, which is
    # a network fetch the first time a new HOME sees it; onroad the car is usually offline.
    try:
      mirror.parent.mkdir(parents=True, exist_ok=True)
      tmp = mirror.with_suffix(mirror.suffix + ".part")
      tmp.write_bytes(blob)
      os.replace(tmp, mirror)
    except OSError:
      pass
    return blob

  fetch_fw._iq_patched = True
  helpers.fetch_fw = fetch_fw


def patch_tinygrad_prune_views() -> None:
  from tinygrad.engine import jit
  from tinygrad.uop.ops import Ops
  if getattr(jit.prune_linear, "_iq_patched", False):
    return

  # upstream prune only sees bare BUFFER args, so a kernel reading an input through a SLICE view ran once at capture and froze that input
  def prune_linear(linear, needed):
    kept, onetime = [], []
    for si in linear.src:
      si_bufs = {b for src in si.src[1:] for b in src.toposort() if b.op is Ops.BUFFER}
      if not si_bufs.isdisjoint(needed):
        kept.append(si)
        needed |= si_bufs
      else:
        onetime.append(si)
    return linear.replace(src=tuple(kept)), linear.replace(src=tuple(onetime))

  prune_linear._iq_patched = True
  jit.prune_linear = prune_linear


def patch_usb_signal_wait() -> None:
  import time

  from tinygrad.helpers import getenv
  from tinygrad.runtime.ops_amd import AMDSignal
  if getattr(AMDSignal.wait, "_iq_patched", False):
    return
  _orig = AMDSignal.wait

  def wait(self, value: int, timeout: int | None = None):
    owner = self.owner
    if owner is None or not owner.is_usb():
      return _orig(self, value, timeout)
    # a dock's signal lives in the USB bridge, so each poll is a USB read, and tinygrad only sleeps after 200 ms
    timeout_ms = timeout or getenv("HCQDEV_WAIT_TIMEOUT_MS", 30000)
    seen = self.value
    progress = time.perf_counter()
    while seen < value:
      time.sleep(USB_SIGNAL_POLL_S)
      current = self.value
      if current != seen:
        seen, progress = current, time.perf_counter()
      elif (time.perf_counter() - progress) * 1000 > timeout_ms:
        raise RuntimeError(f"Wait timeout: {timeout_ms} ms! (the signal is not set to {value}, but {current})")

  wait._iq_patched = True
  AMDSignal.wait = wait
