# Copyright (c) 2026 IQ.Lvbs. All rights reserved.
import os
import site
import subprocess
import sys

AGNOS_PY = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "agnos.py"))


def _run_like_the_os(*args: str) -> subprocess.CompletedProcess:
  # -S skips .pth processing, so an editable install can't make iqpilot importable the way it never is on a device
  env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP")}
  env["PYTHONPATH"] = os.pathsep.join(site.getsitepackages())
  return subprocess.run([sys.executable, "-S", AGNOS_PY, *args], cwd="/", env=env,
                        capture_output=True, text=True, timeout=120)


class TestAgnosEntrypoint:

  def test_starts_without_pythonpath(self):
    result = _run_like_the_os("--help")
    output = result.stdout + result.stderr
    assert "ModuleNotFoundError" not in output, output
    assert result.returncode == 0, output
    assert "Flash and verify AGNOS update" in output
