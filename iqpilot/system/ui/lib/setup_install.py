# Copyright (c) 2026 IQ.Lvbs. All rights reserved.
import os
import subprocess

IQ_SELABEL = "/usr/libexec/iqos/iq-selabel"


def relabel_code_tree(install_path: str) -> None:
  # A just-cloned tree is SELinux base_t until iqos-selinux.service relabels it at the
  # next boot; the custodian signer (iqkeeperd) only accepts a code_t peer, so the
  # in-place first IQ.OS update would stall before the reboot ever happens. Relabel now.
  if not install_path or not os.path.isdir(install_path):
    return
  try:
    if os.path.isfile(IQ_SELABEL):
      subprocess.run(["sudo", "-n", IQ_SELABEL], check=False, timeout=120)
    else:
      subprocess.run(["sudo", "-n", "chcon", "-R", "-t", "code_t", install_path], check=False, timeout=120)
  except Exception:
    pass


def write_setup_claim(ble):
  if ble is None or not getattr(ble, "phone_active", False):
    return
  try:
    import hashlib
    claim_id = hashlib.sha256(f"k3setup-claim:v1:{ble.code}:{ble.serial}".encode()).hexdigest()
    with open("/data/setup_claim_id", "w") as f:
      f.write(claim_id)
  except Exception:
    pass


def maybe_update_os(install_path: str, hardware, ble, download_failed, label: str, prompt_os_update) -> bool:
  # The freshly-installed fork pins the IQ.OS it needs in launch_env.sh. If it
  # differs from what we're running, flash it now (via comma's agnos.py) so the
  # upcoming single reboot lands on a compatible OS instead of dead-ending on
  # "update required". Returns False (and shows Download Failed) on abort.
  from iqpilot.system.proprietary_runtime.bundle_modes import normalize_tree
  from iqpilot.system.ui.lib.os_update import OsUpdateCoordinator, os_update_needed, preflight_os_auth, run_agnos_update
  relabel_code_tree(install_path)
  normalize_tree(install_path)
  try:
    needed, current, required = os_update_needed(install_path)
  except Exception:
    return True  # never block an install on a version-check failure
  if not needed:
    return True

  device_type = hardware.get_device_type() if hardware is not None else ""
  auth_ok, auth_detail = preflight_os_auth(install_path, device_type)
  if not auth_ok:
    if ble is not None:
      ble.set_install_progress("failed", error="os_update_auth_failed")
    download_failed(label, auth_detail or f"IQ.OS update to {required} failed: device could not authenticate to konn3kt.")
    return False

  coordinator = ble.os_update if ble is not None else OsUpdateCoordinator()
  coordinator.request(current, required)
  if ble is not None:
    ble.set_install_progress("os_update_required", 0, os_from=current, os_to=required)
  prompt_os_update(coordinator)
  if not coordinator.wait_for_confirm():
    if ble is not None:
      ble.set_install_progress("failed", error="os_update_not_confirmed")
    download_failed(label, f"IQ.OS update to {required} was declined. This software cannot install without it.")
    return False
  prompt_os_update(None)

  def _cb(pct: int, note: str):
    if ble is not None:
      ble.set_install_progress("os_updating", pct, error=note, os_from=current, os_to=required)

  if not run_agnos_update(install_path, hardware.get_device_type(), _cb):
    if ble is not None:
      ble.set_install_progress("failed", error="os_update_failed")
    download_failed(label, f"IQ.OS update to {required} failed. Please try again.")
    return False
  return True
