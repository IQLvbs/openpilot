"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""
from __future__ import annotations

import glob
import os
os.environ.setdefault("XDG_CACHE_HOME", "/data/.cache")
import subprocess
import sys
import time

from iqpilot.common.params import Params
from iqpilot.common.swaglog import cloudlog
from iqpilot.selfdrive.iqmodeld.egpu_helpers import (ARTIFACT_PATHS, apply_dock_power_limit, die_with_parent, download_precompiled,
                                                     patch_tinygrad_fetch_fw, usbgpu_present)
from iqpilot.selfdrive.iqmodeld.egpu_model import resolve_egpu_model

POLL_S = 30.0
RETRY_S = 120.0
WARM_FLAG = "--warm-firmware"
PREFETCH_ORDER = ("egpu_model_oob_artifact", "egpu_oob_artifact", "egpu_policy_artifact")
WARM_TIMEOUT_S = 180.0


def _selected_meta(params: Params) -> dict | None:
  key = params.get("IQEmacModel", encoding="utf8")
  try:
    return resolve_egpu_model(params, key)
  except Exception as e:
    cloudlog.warning(f"egpu_prefetch cannot resolve {key!r}: {e}")
    return None


def _drop_stale_partials(keep: str) -> None:
  root = os.path.dirname(keep)
  partial_globs = ("egpu_*_amd_policy.pkl.part", "egpu_*_amd_policy_oob.pkl.part", "egpu_*_amd_model_oob.pkl.part",
                   "egpu_*_amd_model_oob2.pkl.part",
                   "big_driving_supercombo_*.onnx.part")
  for path in [p for g in partial_globs for p in glob.glob(os.path.join(root, g))]:
    if not path.startswith(keep):
      try:
        os.remove(path)
      except OSError:
        pass


def prefetch_once(params: Params) -> bool:
  if params.get_bool("IQEgpuDisabled"):
    return False
  meta = _selected_meta(params)
  if meta is None:
    return False
  # same preference as iqegpumodeld, or the drive starts by downloading the warp-on-dock build it actually wants
  field = next((f for f in PREFETCH_ORDER if meta.get(f)), None)
  if field is None:
    return False
  dst = ARTIFACT_PATHS[field](meta)
  if os.path.isfile(dst):
    return True
  _drop_stale_partials(dst)
  params.put("UsbGpuSetupProgress", "0.0")
  last = [-1.0]

  def _prog(p: float) -> None:
    if p - last[0] >= 0.02 or p >= 1.0:
      last[0] = p
      params.put("UsbGpuSetupProgress", f"{p:.3f}")

  cloudlog.warning(f"egpu_prefetch downloading {meta['key']} {field} offroad")
  out = download_precompiled(meta, progress_cb=_prog, field=field)
  cloudlog.warning(f"egpu_prefetch ready -> {out}")
  return out is not None


_firmware_warm = False


def _open_dock_once() -> None:
  os.environ.setdefault("DEV", "USB+AMD:LLVM")
  os.environ.setdefault("GMMU", "0")
  apply_dock_power_limit()
  patch_tinygrad_fetch_fw()
  from tinygrad.device import Device
  Device["AMD"]


def warm_firmware() -> None:
  global _firmware_warm
  if _firmware_warm or not usbgpu_present():
    return
  # opening the dock takes tinygrad's exclusive flock until the opener exits; doing it here kept the lock for the whole
  # offroad session and iqegpumodeld then failed "Failed to acquire lock file am_usb" at every ignition
  proc = subprocess.run([sys.executable, "-m", "iqpilot.selfdrive.iqmodeld.egpu_prefetch", WARM_FLAG],
                        timeout=WARM_TIMEOUT_S, capture_output=True, text=True, preexec_fn=die_with_parent)
  if proc.returncode != 0:
    raise RuntimeError(f"firmware warm failed (rc={proc.returncode}): {(proc.stderr or proc.stdout).strip()[-300:]}")
  _firmware_warm = True
  cloudlog.warning("egpu_prefetch: firmware cached and mirrored; dock released")


def main() -> None:
  if WARM_FLAG in sys.argv:
    _open_dock_once()
    return
  params = Params()
  while True:
    try:
      warm_firmware()
    except Exception as e:
      cloudlog.warning(f"egpu_prefetch firmware warm failed: {e}")
    try:
      prefetch_once(params)
    except Exception as e:
      cloudlog.warning(f"egpu_prefetch failed: {e}")
      params.put("UsbGpuLastError", str(e)[:512])
      time.sleep(RETRY_S)
      continue
    time.sleep(POLL_S)


if __name__ == "__main__":
  main()
