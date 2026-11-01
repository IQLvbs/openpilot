#!/usr/bin/env python3
"""
IQ.OS compatibility check + in-place AGNOS update for the setup flow.

A chosen IQ.Pilot channel may target a newer IQ.OS than the device is running
(its cloned tree pins the required version in launch_env.sh AGNOS_VERSION). When
that differs from the running /VERSION, the setup flow flashes the target IQ.OS
via comma's own agnos.py BEFORE writing continue.sh, so the single reboot lands
on a compatible OS. The risky flashing is delegated entirely to agnos.py; this
module only reads versions, picks the right manifest, and streams coarse
progress.
"""
import json
import os
import re
import subprocess
import threading
from typing import Callable

VERSION_PATH = "/VERSION"


def current_os_version() -> str:
  try:
    with open(VERSION_PATH) as f:
      return f.read().strip()
  except Exception:
    return ""


def required_agnos_version(install_path: str) -> str:
  """Read the target OS version the cloned fork pins in launch_env.sh."""
  path = os.path.join(install_path, "launch_env.sh")
  try:
    with open(path) as f:
      for line in f:
        m = re.search(r'AGNOS_VERSION\s*=\s*"([^"]+)"', line)
        if m:
          return m.group(1).strip()
  except Exception:
    pass
  return ""


def _hardware_dir(install_path: str) -> str:
  nested = os.path.join(install_path, "iqpilot", "system", "hardware", "tici")
  if os.path.isdir(nested):
    return nested
  return os.path.join(install_path, "system", "hardware", "tici")


def agnos_manifest_path(install_path: str, device_type: str) -> str:
  # comma 3 (tici) uses a different AGNOS manifest than comma 3x (tizi) / comma 4 (mici).
  fname = "agnos_tici_15_1.json" if device_type == "tici" else "agnos.json"
  return os.path.join(_hardware_dir(install_path), fname)


def os_version_number(version: str) -> str:
  return version.strip().removeprefix("IQ.OS").strip()


def os_update_needed(install_path: str) -> tuple[bool, str, str]:
  """Returns (needed, current, required)."""
  current = current_os_version()
  required = required_agnos_version(install_path)
  needed = bool(required and current and os_version_number(required) != os_version_number(current))
  return needed, current, required


ProgressCb = Callable[[int, str], None]


def _supports_check_auth(agnos_py: str) -> bool:
  try:
    with open(agnos_py) as f:
      return "--check-auth" in f.read()
  except OSError:
    return False


def preflight_os_auth(install_path: str, device_type: str) -> tuple[bool, str]:
  """Confirm the device can authenticate to the IQ.OS image source before the user
  confirms a flash, so an unauthenticated device fails fast with a clear reason
  instead of stalling at 0%. A public manifest needs no auth and passes. Returns
  (ok, detail); a preflight that cannot even run does not block (the flash will
  still run and report)."""
  manifest = agnos_manifest_path(install_path, device_type)
  agnos_py = os.path.join(_hardware_dir(install_path), "agnos.py")
  if not os.path.isfile(manifest) or not _supports_check_auth(agnos_py):
    return True, ""
  try:
    proc = subprocess.run(
      [updater_interpreter(install_path), agnos_py, "--check-auth", manifest],
      cwd=install_path, env=updater_env(install_path),
      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=180)
  except Exception:
    return True, ""
  if proc.returncode == 0:
    return True, ""
  detail = ""
  for line in (proc.stdout or "").splitlines():
    if line.startswith("Error:"):
      detail = line[len("Error:"):].strip()
  return False, detail


def run_agnos_update(install_path: str, device_type: str, progress_cb: ProgressCb) -> bool:
  """Flash + swap to the target IQ.OS. Streams coarse partition-level progress
  via progress_cb(percent, note). Returns True on success. The device must be
  rebooted by the caller afterward for the new slot to take effect."""
  manifest = agnos_manifest_path(install_path, device_type)
  agnos_py = os.path.join(_hardware_dir(install_path), "agnos.py")
  if not os.path.isfile(manifest) or not os.path.isfile(agnos_py):
    progress_cb(0, "manifest_missing")
    return False

  try:
    with open(manifest) as f:
      total_partitions = max(1, len(json.load(f)))
  except Exception:
    total_partitions = 1

  progress_cb(1, "starting")
  try:
    proc = subprocess.Popen(
      [updater_interpreter(install_path), agnos_py, "--swap", manifest],
      cwd=install_path,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      text=True,
      env=updater_env(install_path),
    )
  except Exception:
    progress_cb(0, "launch_failed")
    return False

  completed = 0
  swapping = False
  last_error = ""
  assert proc.stdout is not None
  with proc:
    for line in proc.stdout:
      line = line.strip()
      if "Downloading and writing" in line or "Already flashed" in line:
        completed += 1
        pct = min(94, int((completed / total_partitions) * 90) + 2)
        progress_cb(pct, "flashing")
      elif "Swapping to slot" in line or "AGNOS ready" in line:
        swapping = True
        progress_cb(96, "swapping")
      elif "Error" in line or "Exception" in line:
        last_error = line[:160]
  if proc.returncode == 0:
    progress_cb(100, "done")
    return True
  reason = "failed" if not swapping else "swap_failed"
  progress_cb(0, f"{reason}: {last_error}" if last_error else reason)
  return False


def updater_interpreter(install_path: str) -> str:
  venv_python = os.path.join(install_path, ".venv", "bin", "python3")
  return venv_python if os.access(venv_python, os.X_OK) else "python3"


# hephaestusd runs on the system interpreter with the checkout's site-packages on PYTHONPATH; dropping it hides iqdbc from agnos.py.
def updater_env(install_path: str) -> dict[str, str]:
  env = dict(os.environ)
  inherited = env.get("PYTHONPATH", "")
  env["PYTHONPATH"] = install_path if not inherited else f"{install_path}{os.pathsep}{inherited}"
  return env


class OsUpdateCoordinator:
  """Bridges the setup UI's install thread and the BLE confirmation from the app.
  The install thread posts a required-update, waits for the phone's confirm, then
  runs the flash. On-screen setup can confirm locally too."""

  def __init__(self):
    self.decided = threading.Event()
    self.accepted = False
    self.needed = False
    self.current = ""
    self.required = ""

  def request(self, current: str, required: str) -> None:
    self.needed = True
    self.current = current
    self.required = required
    self.accepted = False
    self.decided.clear()

  def confirm(self) -> None:
    self.accepted = True
    self.decided.set()

  def decline(self) -> None:
    self.accepted = False
    self.decided.set()

  def wait_for_confirm(self, timeout: float | None = None) -> bool:
    return self.decided.wait(timeout=timeout) and self.accepted
