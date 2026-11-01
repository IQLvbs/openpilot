#!/usr/bin/env python3
import os
import subprocess
import sys
from pathlib import Path

# NOTE: Do NOT import anything here that needs be built (e.g. params)
from iqpilot.common.basedir import BASEDIR
from iqpilot.common.spinner import Spinner
from iqpilot.common.text_window import TextWindow
from iqpilot.common.swaglog import cloudlog, add_file_handler
from iqpilot.system.hardware import HARDWARE, AGNOS
from iqpilot.system.version import get_build_metadata

MAX_CACHE_SIZE = 4e9 if "CI" in os.environ else 2e9
CACHE_DIR = Path("/data/scons_cache" if AGNOS else "/tmp/scons_cache")

TOTAL_SCONS_NODES = 5500
MAX_BUILD_PROGRESS = 100

def get_job_sequence() -> list[int]:
  env_override = os.environ.get("SCONS_MAX_JOBS")
  if env_override is not None:
    try:
      max_jobs = max(1, int(env_override))
    except ValueError:
      max_jobs = 1
  else:
    detected_jobs = os.cpu_count() or 2
    max_jobs = min(detected_jobs, 3 if AGNOS else detected_jobs)

  candidates = [max_jobs, max_jobs // 2, 1]
  jobs: list[int] = []
  for candidate in candidates:
    candidate = max(1, int(candidate))
    if candidate not in jobs:
      jobs.append(candidate)
  return jobs

class _SilentProgress:
  """Not a UI element: a do-nothing sink for build()'s progress calls, used when
  updated.py builds in the background at update-apply time so that NO spinner or
  window is shown. The normal boot path still uses the real Spinner."""
  def update(self, spinner_text: str) -> None:
    pass

  def update_progress(self, cur: float, total: float) -> None:
    pass

  def close(self) -> None:
    pass


REGISTRY_ARTIFACTS = [
  "iqpilot/cereal/services.h",
  "iqpilot/cereal/messaging/socketmaster.o",
  "iqpilot/cereal/libsocketmaster.a",
  "iqpilot/cereal/messaging/bridge",
  "iqpilot/selfdrive/iqlocd/iqlocd",
  "iqpilot/selfdrive/pandad/pandad",
  "iqpilot/system/camerad/camerad",
  "iqpilot/system/loggerd/loggerd",
  "iqpilot/system/loggerd/encoderd",
  "iqpilot/system/loggerd/bootlog",
]


def stale_registry_artifacts(basedir: str = BASEDIR) -> list[str]:
  from iqpilot.cereal.services import REGISTRY_TAG_PREFIX, registry_tag
  expected = registry_tag().encode()
  prefix = REGISTRY_TAG_PREFIX.encode()
  stale = []
  for rel in REGISTRY_ARTIFACTS:
    path = os.path.join(basedir, rel)
    if not os.path.isfile(path):
      continue
    with open(path, "rb") as f:
      data = f.read()
    if prefix in data and expected not in data:
      stale.append(rel)
  return stale


def purge_stale_registry_header(basedir: str = BASEDIR) -> bool:
  from iqpilot.cereal.services import registry_tag
  header = os.path.join(basedir, REGISTRY_ARTIFACTS[0])
  if not os.path.isfile(header):
    return False
  with open(header, "rb") as f:
    stamped = registry_tag().encode() in f.read()
  if stamped:
    return False
  purge_registry_artifacts([], basedir)
  return True


def purge_registry_artifacts(stale: list[str], basedir: str = BASEDIR) -> None:
  for rel in set(stale) | set(REGISTRY_ARTIFACTS[:3]):
    path = os.path.join(basedir, rel)
    if os.path.isfile(path):
      os.remove(path)


def build(spinner, dirty: bool = False, minimal: bool = False, show_error_window: bool = True, registry_retry: bool = False) -> None:
  env = os.environ.copy()
  env.pop('PWD', None)
  env['SCONS_PROGRESS'] = "1"

  extra_args = ["--minimal"] if minimal else []

  if AGNOS:
    HARDWARE.set_power_save(False)
    os.sched_setaffinity(0, range(8))  # ensure we can use the isolcpus cores

  # building with all cores can result in using too
  # much memory, so retry with less parallelism
  if purge_stale_registry_header():
    cloudlog.error("generated service registry header was stale, regenerating")

  compile_output: list[bytes] = []
  for n in get_job_sequence():
    compile_output.clear()
    command = [sys.executable, "-m", "SCons", f"-j{int(n)}", "--cache-populate", *extra_args]
    scons: subprocess.Popen = subprocess.Popen(command, cwd=BASEDIR, env=env, stderr=subprocess.PIPE)
    assert scons.stderr is not None

    # Read progress from stderr and update spinner
    while scons.poll() is None:
      try:
        line = scons.stderr.readline()
        if line is None:
          continue
        line = line.rstrip()

        prefix = b'progress: '
        if line.startswith(prefix):
          i = int(line[len(prefix):])
          spinner.update_progress(MAX_BUILD_PROGRESS * min(0.99, i / TOTAL_SCONS_NODES), 100.)
        elif len(line):
          compile_output.append(line)
          print(line.decode('utf8', 'replace'))
      except Exception:
        pass

    if scons.returncode == 0:
      spinner.update_progress(100, 100.)
      break

  if scons.returncode != 0:
    # Read remaining output
    if scons.stderr is not None:
      compile_output += scons.stderr.read().split(b'\n')

    # Build failed log errors
    error_s = b"\n".join(compile_output).decode('utf8', 'replace')
    add_file_handler(cloudlog)
    cloudlog.error("scons build failed\n" + error_s)

    # Show TextWindow
    spinner.close()
    if not os.getenv("CI") and show_error_window:
      with TextWindow("IQ.Pilot failed to build\n \n" + error_s) as t:
        t.wait_for_exit()
    exit(1)

  stale = stale_registry_artifacts()
  if stale and not registry_retry:
    cloudlog.error(f"compiled service registry is stale in {', '.join(stale)}, rebuilding messaging")
    purge_registry_artifacts(stale)
    build(spinner, dirty, minimal, show_error_window, registry_retry=True)
    return
  if stale:
    error_s = "compiled service registry is still stale after rebuild: " + ", ".join(stale)
    add_file_handler(cloudlog)
    cloudlog.error(error_s)
    spinner.close()
    if not os.getenv("CI") and show_error_window:
      with TextWindow("IQ.Pilot failed to build\n \n" + error_s) as t:
        t.wait_for_exit()
    exit(1)

  # enforce max cache size
  cache_files = [f for f in CACHE_DIR.rglob('*') if f.is_file()]
  cache_files.sort(key=lambda f: f.stat().st_mtime)
  cache_size = sum(f.stat().st_size for f in cache_files)
  for f in cache_files:
    if cache_size < MAX_CACHE_SIZE:
      break
    cache_size -= f.stat().st_size
    f.unlink()


if __name__ == "__main__":
  headless = "--headless" in sys.argv
  spinner = _SilentProgress() if headless else Spinner()
  spinner.update_progress(0, 100)
  build_metadata = get_build_metadata()
  build(spinner, build_metadata.openpilot.is_dirty, minimal = AGNOS, show_error_window = not headless)
