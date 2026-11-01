#!/usr/bin/env python3
"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""
import re
from pathlib import Path

from iqpilot.common.params import Params
from iqpilot.system.hardware.hw import Paths
from iqpilot.common.swaglog import cloudlog


UNREGISTERED_DONGLE_ID = "UnregisteredDevice"

_DONGLE_ID_RE = re.compile(r"^[a-fA-F0-9]{16}$")
IMEI_WAIT_TIMEOUT = 15.0


def _read_persist_dongle_id() -> str | None:
  p = Path(Paths.persist_root()) / "comma" / "dongle_id"
  try:
    if not p.is_file():
      return None
    s = p.read_text().strip()
    return s or None
  except Exception:
    cloudlog.exception("failed to read persist dongle_id")
    return None


def is_valid_dongle_id(dongle_id: str | None) -> bool:
  return isinstance(dongle_id, str) and _DONGLE_ID_RE.fullmatch(dongle_id) is not None


def get_cached_dongle_id(params: Params | None = None, prefer_readonly: bool = True) -> str | None:
  ro = _read_persist_dongle_id()
  ro = ro.lower() if is_valid_dongle_id(ro) else None
  if prefer_readonly and ro:
    return ro
  p = Params() if params is None else params
  v = p.get("DongleId")
  if is_valid_dongle_id(v):
    return v.lower()
  return ro or None


_impl = None


def _registration_impl():
  global _impl
  if _impl is None:
    from iqpilot._proprietary_loader import ProprietaryModuleMissing, load_private_module
    try:
      _impl = load_private_module(__name__, "iqpilot_private.konn3kt.registration.reg_core")
    except ProprietaryModuleMissing:
      from konn3kt_private.registration import reg_core as _impl
  return _impl


def ensure_dev_pairing_identity(params: Params | None = None, force_reset: bool = False) -> dict[str, str]:
  return _registration_impl().ensure_dev_pairing_identity(params, force_reset)


def get_registration_identifiers(wait_timeout: float = IMEI_WAIT_TIMEOUT, show_spinner: bool = False) -> tuple[str, str, str]:
  return _registration_impl().get_registration_identifiers(wait_timeout, show_spinner)


def register(show_spinner=False) -> str | None:
  return _registration_impl().register(show_spinner)


if __name__ == "__main__":
  print(register())
