# Copyright (c) 2026 IQ.Lvbs. All rights reserved.
import hashlib
import json
import os
import sys
from types import SimpleNamespace

import pytest

from iqpilot.system import proprietary_runtime
from iqpilot.system.proprietary_runtime import bundle_modes
from iqpilot.system.ui.lib import os_update, setup_install


@pytest.fixture(autouse=True)
def _no_relabel(monkeypatch):
  monkeypatch.setattr(setup_install, "relabel_code_tree", lambda path: None)


def _prompt(events, decision):
  def prompt(coordinator):
    events.append(("prompt", coordinator is not None))
    if coordinator is not None and decision is not None:
      (coordinator.confirm if decision else coordinator.decline)()
  return prompt


@pytest.mark.parametrize("device,manifest", [("tici", "agnos_tici_15_1.json"), ("tizi", "agnos.json"), ("mici", "agnos.json")])
@pytest.mark.parametrize("phone", [False, True])
@pytest.mark.parametrize("confirmed,success", [(False, True), (True, True), (True, False)])
def test_os_update_always_prompts(monkeypatch, mocker, device, manifest, phone, confirmed, success):
  events = []
  ble = None
  if phone:
    ble = SimpleNamespace(phone_active=True, os_update=os_update.OsUpdateCoordinator(),
                          set_install_progress=lambda state, *args, **kwargs: events.append(state))
  hardware = SimpleNamespace(get_device_type=lambda: device)
  monkeypatch.setattr(os_update, "os_update_needed", lambda path: (True, "old", "new"))

  def flash(path, device_type, callback):
    assert os_update.agnos_manifest_path(path, device_type).endswith("/" + manifest)
    events.append("flash")
    callback(50, "flashing")
    return success

  monkeypatch.setattr(os_update, "run_agnos_update", flash)
  failed = mocker.Mock()
  result = setup_install.maybe_update_os("/checkout", hardware, ble, failed, "release", _prompt(events, confirmed))
  assert result == (confirmed and success)
  assert events.count(("prompt", True)) == 1
  if not confirmed:
    assert "flash" not in events
    failed.assert_called_once_with("release", "IQ.OS update to new was declined. This software cannot install without it.")
  else:
    assert events.index(("prompt", True)) < events.index(("prompt", False)) < events.index("flash")
    assert failed.call_count == int(not success)


def test_phone_confirm_unblocks_on_screen_prompt(monkeypatch, mocker):
  coordinator = os_update.OsUpdateCoordinator()
  ble = SimpleNamespace(phone_active=True, os_update=coordinator, set_install_progress=lambda *args, **kwargs: None)
  monkeypatch.setattr(os_update, "os_update_needed", lambda path: (True, "old", "new"))
  flash = mocker.Mock(return_value=True)
  monkeypatch.setattr(os_update, "run_agnos_update", flash)
  assert setup_install.maybe_update_os("/checkout", SimpleNamespace(get_device_type=lambda: "mici"), ble, mocker.Mock(), "release",
                                       lambda c: c is not None and coordinator.confirm())
  flash.assert_called_once()


@pytest.mark.parametrize("result", [(False, "same", "same"), RuntimeError("unavailable")])
def test_version_check_does_not_block_install(monkeypatch, mocker, result):
  monkeypatch.setattr(os_update, "os_update_needed", mocker.Mock(side_effect=result if isinstance(result, Exception) else None, return_value=result))
  flash = mocker.Mock()
  prompt = mocker.Mock()
  monkeypatch.setattr(os_update, "run_agnos_update", flash)
  assert setup_install.maybe_update_os("/checkout", None, None, mocker.Mock(), "release", prompt)
  flash.assert_not_called()
  prompt.assert_not_called()


def test_install_normalizes_bundle_modes_before_os_update(monkeypatch, mocker, tmp_path):
  bundle = tmp_path / "artifacts" / "iqpilot_updater_private"
  lib = bundle / "python" / "iqpilot_private" / "updater" / "_hf_maps_blob.so"
  lib.parent.mkdir(parents=True)
  lib.write_bytes(b"\0")
  os.chmod(lib, 0o644)
  (bundle / "manifest.json").write_text(json.dumps({
    "bundle_version": "1",
    "python/iqpilot_private/updater/_hf_maps_blob.so": {"mode": 0o640, "sha256": "0" * 64},
  }))
  modes = []
  monkeypatch.setattr(os_update, "os_update_needed", lambda path: modes.append(os.stat(lib).st_mode & 0o777) or (False, "", ""))
  assert setup_install.maybe_update_os(str(tmp_path), None, None, mocker.Mock(), "release", mocker.Mock())
  assert modes == [0o640]


def test_verified_import_normalizes_first(monkeypatch, tmp_path):
  bundle = tmp_path / "iqpilot_updater_private"
  lib = bundle / "mod.so"
  lib.parent.mkdir(parents=True)
  lib.write_bytes(b"\0")
  os.chmod(lib, 0o644)
  (bundle / "manifest.json").write_text(json.dumps({"mod.so": {"mode": 0o640, "sha256": "0" * 64}}))
  monkeypatch.setenv("IQPILOT_PROPRIETARY_ROOT", str(tmp_path))
  monkeypatch.setattr(bundle_modes, "_normalized", set())
  seen = []
  fake = SimpleNamespace(import_verified_module=lambda b, m: seen.append(os.stat(lib).st_mode & 0o777) or m)
  monkeypatch.setitem(sys.modules, "iqpilot.system.proprietary_runtime._verified_import", fake)
  monkeypatch.delattr(proprietary_runtime, "_verified_import", raising=False)
  assert bundle_modes.import_verified_module("iqpilot_updater_private", "x.y") == "x.y"
  assert seen == [0o640]


@pytest.mark.parametrize("current,required,needed", [
  ("IQ.OS 4.9.9.6", "IQ.OS 4.9.9.6", False),
  ("4.9.9.7", "IQ.OS 4.9.9.7", False),
  ("IQ.OS 4.9.9.7", "4.9.9.7", False),
  ("IQ.OS 4.9.9.6", "4.9.9.7", True),
  ("16.1", "IQ.OS 4.9.9.7", True),
])
def test_os_update_needed_ignores_iqos_prefix(monkeypatch, current, required, needed):
  monkeypatch.setattr(os_update, "current_os_version", lambda: current)
  monkeypatch.setattr(os_update, "required_agnos_version", lambda path: required)
  assert os_update.os_update_needed("/checkout") == (needed, current, required)


def test_preflight_skips_flasher_without_check_auth(monkeypatch, tmp_path):
  hw = tmp_path / "system" / "hardware" / "tici"
  hw.mkdir(parents=True)
  (hw / "agnos.json").write_text("[]")
  (hw / "agnos.py").write_text('parser.add_argument("--swap")\n')
  run = []
  monkeypatch.setattr(os_update.subprocess, "run", lambda *args, **kwargs: run.append(args))
  assert os_update.preflight_os_auth(str(tmp_path), "mici") == (True, "")
  assert run == []


@pytest.mark.parametrize("active", [False, True])
def test_setup_claim_requires_active_phone(monkeypatch, mocker, active):
  opened = mocker.mock_open()
  monkeypatch.setattr("builtins.open", opened)
  setup_install.write_setup_claim(SimpleNamespace(phone_active=active, code="123456", serial="test-device"))
  if active:
    opened.assert_called_once_with("/data/setup_claim_id", "w")
    opened().write.assert_called_once_with(hashlib.sha256(b"k3setup-claim:v1:123456:test-device").hexdigest())
  else:
    opened.assert_not_called()
