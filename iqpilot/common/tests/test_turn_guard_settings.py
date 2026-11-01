# Copyright (c) 2026 IQ.Lvbs. All rights reserved.
import pytest

from iqpilot.common.params import ParamKeyFlag, Params
from iqpilot.common.turn_guard_settings import (
  get_turn_guards_enabled, reset_turn_guards, set_turn_guards_enabled, turn_guards_path,
)
from iqpilot.common.ui_defaults import apply_ui_defaults


@pytest.fixture
def params(tmp_path):
  return Params(str(tmp_path / "params"))


def test_defaults_off_without_writing(params):
  assert not get_turn_guards_enabled(params)
  assert not turn_guards_path(params).exists()


@pytest.mark.parametrize("enabled", [False, True])
def test_shared_value_survives_manager_and_drive_clears(params, tmp_path, enabled):
  set_turn_guards_enabled(params, enabled)
  for flag in (ParamKeyFlag.CLEAR_ON_MANAGER_START, ParamKeyFlag.CLEAR_ON_ONROAD_TRANSITION,
               ParamKeyFlag.CLEAR_ON_OFFROAD_TRANSITION, ParamKeyFlag.CLEAR_ON_IGNITION_ON):
    params.clear_all(flag)
    apply_ui_defaults(params)
    assert get_turn_guards_enabled(Params(str(tmp_path / "params"))) is enabled
  assert turn_guards_path(params).read_bytes() == (b"1" if enabled else b"0")


@pytest.mark.parametrize("contents", [b"", b"true", b"1\n", b"0", b"garbage"])
def test_malformed_or_disabled_values_are_off(params, contents):
  path = turn_guards_path(params)
  path.parent.mkdir(parents=True)
  path.write_bytes(contents)
  assert not get_turn_guards_enabled(params)


def test_params_roots_and_prefixes_are_isolated(tmp_path, monkeypatch):
  monkeypatch.setenv("OPENPILOT_PREFIX", "guard_a")
  first = Params(str(tmp_path / "first"))
  separate_root = Params(str(tmp_path / "second"))
  set_turn_guards_enabled(first, True)
  assert not get_turn_guards_enabled(separate_root)
  monkeypatch.setenv("OPENPILOT_PREFIX", "guard_b")
  separate_prefix = Params(str(tmp_path / "first"))
  assert not get_turn_guards_enabled(separate_prefix)
  assert turn_guards_path(first) != turn_guards_path(separate_prefix)
  assert get_turn_guards_enabled(first)


def test_failed_atomic_replace_preserves_previous_value_and_removes_temporary(params, monkeypatch):
  set_turn_guards_enabled(params, True)

  def fail_replace(*args):
    raise OSError("simulated write failure")

  monkeypatch.setattr("iqpilot.common.turn_guard_settings.os.replace", fail_replace)
  with pytest.raises(OSError, match="simulated write failure"):
    set_turn_guards_enabled(params, False)
  assert get_turn_guards_enabled(params)
  assert list(turn_guards_path(params).parent.iterdir()) == [turn_guards_path(params)]


def test_reset_is_idempotent(params):
  set_turn_guards_enabled(params, True)
  reset_turn_guards(params)
  reset_turn_guards(params)
  assert not get_turn_guards_enabled(params)


def test_read_error_defaults_off(params, monkeypatch):
  def fail_read(*args):
    raise OSError("simulated read failure")

  monkeypatch.setattr("iqpilot.common.turn_guard_settings.Path.read_bytes", fail_read)
  assert not get_turn_guards_enabled(params)
