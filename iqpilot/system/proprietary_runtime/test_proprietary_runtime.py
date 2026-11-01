"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos
"""
import os
import subprocess
import sys

from iqpilot.common.basedir import BASEDIR


def _runtime_paths(env):
  result = subprocess.run(
    [
      sys.executable,
      "-c",
      "import os; import iqpilot.system.proprietary_runtime; print(os.environ['OPENPILOT_BASEDIR']); "
      + "print(os.environ['IQPILOT_PROPRIETARY_ROOT'])",
    ],
    env=env,
    check=True,
    capture_output=True,
    text=True,
  )
  return result.stdout.splitlines()


def test_basedir_contract():
  env = os.environ.copy()
  env.pop("OPENPILOT_BASEDIR", None)
  env.pop("IQPILOT_PROPRIETARY_ROOT", None)
  assert _runtime_paths(env) == [BASEDIR, os.path.join(BASEDIR, "artifacts")]


def test_explicit_basedir_is_preserved(tmp_path):
  env = os.environ.copy()
  env["OPENPILOT_BASEDIR"] = str(tmp_path)
  env["IQPILOT_PROPRIETARY_ROOT"] = str(tmp_path / "bundles")
  assert _runtime_paths(env) == [str(tmp_path), str(tmp_path / "bundles")]
