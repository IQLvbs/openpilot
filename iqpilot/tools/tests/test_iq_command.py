# Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos

import os
from pathlib import Path
import subprocess


IQ_COMMAND = Path(__file__).parents[1] / "iq.sh"


def make_checkout(tmp_path: Path) -> Path:
  checkout = tmp_path / "checkout"
  (checkout / "iqpilot").mkdir(parents=True)
  (checkout / "launch_iqpilot.sh").touch()
  return checkout


def run_pkg(checkout: Path, path: Path) -> subprocess.CompletedProcess[str]:
  env = os.environ.copy()
  env["PATH"] = f"{path}:{env['PATH']}"
  return subprocess.run(
    ["bash", str(IQ_COMMAND), "--dir", str(checkout), "pkg"],
    check=False,
    capture_output=True,
    env=env,
    text=True,
  )


def run_iq(checkout: Path, command: str, *args: str) -> subprocess.CompletedProcess[str]:
  return subprocess.run(
    ["bash", str(IQ_COMMAND), "--dir", str(checkout), command, *args],
    check=False,
    capture_output=True,
    text=True,
  )


def test_public_checkout_skips_private_package_source_setup(tmp_path: Path):
  checkout = make_checkout(tmp_path)

  result = run_pkg(checkout, tmp_path)

  assert result.returncode == 0
  assert "private package sources are not present" in result.stdout
  assert "setup_private_packages.py" not in result.stderr


def test_internal_checkout_runs_private_package_source_setup(tmp_path: Path):
  checkout = make_checkout(tmp_path)
  setup_script = checkout / "iqpilot/tools/scripts/setup_private_packages.py"
  setup_script.parent.mkdir(parents=True)
  setup_script.touch()
  python_log = tmp_path / "python.log"
  python = tmp_path / "python3"
  python.write_text(f"#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" > {python_log!s}\n")
  python.chmod(0o755)

  result = run_pkg(checkout, tmp_path)

  assert result.returncode == 0
  assert python_log.read_text().strip() == "iqpilot/tools/scripts/setup_private_packages.py"


def test_public_checkout_runs_bundled_package_installer(tmp_path: Path):
  checkout = make_checkout(tmp_path)
  installer_log = tmp_path / "installer.log"
  installer = checkout / "artifacts/runtime/ensure_private_installed.sh"
  installer.parent.mkdir(parents=True)
  installer.write_text(f"printf installed > {installer_log!s}\n")

  result = run_pkg(checkout, tmp_path)

  assert result.returncode == 0
  assert installer_log.read_text() == "installed"


def test_cabana_command_runs_launcher_from_checkout(tmp_path: Path):
  checkout = make_checkout(tmp_path)
  log = tmp_path / "cabana.log"
  launcher = checkout / "iqpilot/tools/cabana/cabana"
  launcher.parent.mkdir(parents=True)
  launcher.write_text(f"#!/usr/bin/env bash\nprintf '%s\\n' \"$PWD|$*\" > {log!s}\n")
  launcher.chmod(0o755)

  result = run_iq(checkout, "cabana", "--msgq")

  assert result.returncode == 0
  assert log.read_text().strip() == f"{checkout}|--msgq"


def test_juggle_command_runs_launcher_from_checkout(tmp_path: Path):
  checkout = make_checkout(tmp_path)
  log = tmp_path / "juggle.log"
  launcher = checkout / "iqpilot/tools/jotpluggler/pluggle.py"
  launcher.parent.mkdir(parents=True)
  launcher.write_text(f"#!/usr/bin/env bash\nprintf '%s\\n' \"$PWD|$*\" > {log!s}\n")
  launcher.chmod(0o755)

  result = run_iq(checkout, "juggle", "route")

  assert result.returncode == 0
  assert log.read_text().strip() == f"{checkout}|route"


def test_desktop_tool_missing_from_host_checkout_fails(tmp_path: Path):
  checkout = make_checkout(tmp_path)

  result = run_iq(checkout, "cabana")

  assert result.returncode == 1
  assert "launcher is missing" in result.stderr
