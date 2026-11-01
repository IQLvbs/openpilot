# Copyright (c) 2026 IQ.Lvbs. All rights reserved.
import json
import os
import stat

import pytest

from iqpilot.system.ui.lib import os_update


def _install_tree(root, updater_script):
  hardware = root / "iqpilot" / "system" / "hardware" / "tici"
  hardware.mkdir(parents=True)
  (hardware / "agnos_tici_15_1.json").write_text(json.dumps([{"name": "boot"}, {"name": "system"}]))
  agnos = hardware / "agnos.py"
  agnos.write_text("")
  return agnos


def _fake_python(path, body):
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text("#!/bin/sh\n" + body)
  path.chmod(path.stat().st_mode | stat.S_IXUSR)
  return path


@pytest.fixture
def notes():
  return []


def test_updater_prefers_the_checkout_venv_and_keeps_inherited_pythonpath(tmp_path, monkeypatch, notes):
  _install_tree(tmp_path, None)
  record = tmp_path / "record.txt"
  _fake_python(tmp_path / ".venv" / "bin" / "python3",
               f'echo "$0|$PYTHONPATH|$*" > "{record}"\necho "Downloading and writing boot"\necho "Downloading and writing system"\necho "Swapping to slot 1"\n')
  monkeypatch.setenv("PYTHONPATH", "/opt/daemon/site-packages")

  assert os_update.run_agnos_update(str(tmp_path), "tici", lambda pct, note: notes.append((pct, note)))

  interpreter, pythonpath, args = record.read_text().strip().split("|")
  assert interpreter == str(tmp_path / ".venv" / "bin" / "python3")
  assert pythonpath.split(os.pathsep) == [str(tmp_path), "/opt/daemon/site-packages"]
  assert args.split()[1:] == ["--swap", str(tmp_path / "iqpilot" / "system" / "hardware" / "tici" / "agnos_tici_15_1.json")]
  assert [note for _, note in notes] == ["starting", "flashing", "flashing", "swapping", "done"]


def test_updater_falls_back_to_path_python_without_a_venv(tmp_path, monkeypatch, notes):
  _install_tree(tmp_path, None)
  record = tmp_path / "record.txt"
  bin_dir = tmp_path / "bin"
  _fake_python(bin_dir / "python3", f'echo "$0|$PYTHONPATH" > "{record}"\n')
  monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
  monkeypatch.delenv("PYTHONPATH", raising=False)

  assert os_update.run_agnos_update(str(tmp_path), "tici", lambda pct, note: notes.append((pct, note)))

  interpreter, pythonpath = record.read_text().strip().split("|")
  assert interpreter.endswith("python3")
  assert pythonpath == str(tmp_path)


def test_updater_failure_note_carries_the_last_error_line(tmp_path, notes):
  _install_tree(tmp_path, None)
  _fake_python(tmp_path / ".venv" / "bin" / "python3",
               'echo "Traceback (most recent call last):"\necho "ModuleNotFoundError: No module named \'iqdbc\'"\nexit 1\n')

  assert not os_update.run_agnos_update(str(tmp_path), "tici", lambda pct, note: notes.append((pct, note)))

  assert notes[-1] == (0, "failed: ModuleNotFoundError: No module named 'iqdbc'")
