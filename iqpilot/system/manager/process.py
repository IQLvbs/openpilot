import importlib
import os
import signal
import time
from pathlib import Path
from collections.abc import Callable, ValuesView
from abc import ABC, abstractmethod
from multiprocessing import Process

from setproctitle import setproctitle

from iqpilot.cereal import car, log
import iqpilot.cereal.messaging as messaging
import iqpilot.system.sentry as sentry
from iqpilot.common.basedir import BASEDIR
from iqpilot.common.params import Params
from iqpilot.common.swaglog import cloudlog

MAX_CRASH_BACKOFF = 300.0
CRASH_RESET_TIME = 60.0
CRASH_LOOP_THRESHOLD = 6

try:
  from iqpilot.system.proprietary_runtime.runtime_paths import preferred_runner_path
except ModuleNotFoundError:
  _VERIFIED_RUNNER_PATH = Path("/usr/libexec/iqpilot/iqpilot_bundle_runner")
  _FALLBACK_RUNNER_PATH = Path("/data/openpilot/iqpilot/system/proprietary_runtime/iqpilot_bundle_runner")

  def preferred_runner_path() -> Path:
    if _VERIFIED_RUNNER_PATH.is_file() and os.access(_VERIFIED_RUNNER_PATH, os.X_OK):
      return _VERIFIED_RUNNER_PATH
    if os.getenv("IQPILOT_ALLOW_DEV_FALLBACKS") == "1" and _FALLBACK_RUNNER_PATH.is_file():
      return _FALLBACK_RUNNER_PATH
    return _VERIFIED_RUNNER_PATH


def launcher(proc: str, name: str, env: dict[str, str] | None = None) -> None:
  try:
    if env:
      os.environ.update(env)

    mod = importlib.import_module(proc)

    setproctitle(proc)

    messaging.reset_context()

    cloudlog.bind(daemon=name)
    sentry.set_tag("daemon", name)

    mod.main()
  except KeyboardInterrupt:
    cloudlog.warning(f"child {proc} got SIGINT")
  except Exception:
    sentry.capture_exception()
    raise


def nativelauncher(pargs: list[str], cwd: str, name: str, env: dict[str, str] | None = None) -> None:
  if env:
    os.environ.update(env)
  os.environ['MANAGER_DAEMON'] = name

  os.chdir(cwd)
  os.environ['PWD'] = cwd
  os.execvp(pargs[0], pargs)


def join_process(process: Process, timeout: float) -> None:
  # Process().join(timeout) will hang due to a python 3 bug: https://bugs.python.org/issue28382
  # We have to poll the exitcode instead
  t = time.monotonic()
  while time.monotonic() - t < timeout and process.exitcode is None:
    time.sleep(0.001)


class ManagerProcess(ABC):
  daemon = False
  sigkill = False
  should_run: Callable[[bool, Params, car.CarParams], bool]
  proc: Process | None = None
  enabled = True
  name = ""
  shutting_down = False
  restart_if_crash = False
  crash_count = 0
  last_restart_time = 0.0
  last_alive_time = 0.0
  crash_loop_logged = False

  @abstractmethod
  def prepare(self) -> None:
    pass

  @abstractmethod
  def start(self) -> None:
    pass

  def restart(self) -> None:
    self.stop(sig=signal.SIGKILL)
    self.start()

  def stop(self, retry: bool = True, block: bool = True, sig: signal.Signals | None = None, timeout: float = 5) -> int | None:
    if self.proc is None:
      return None

    if self.proc.exitcode is None:
      if not self.shutting_down:
        cloudlog.info(f"killing {self.name}")
        if sig is None:
          sig = signal.SIGKILL if self.sigkill else signal.SIGINT
        self.signal(sig)
        self.shutting_down = True

        if not block:
          return None

      join_process(self.proc, timeout)

      # If process failed to die send SIGKILL
      if self.proc.exitcode is None and retry:
        cloudlog.info(f"killing {self.name} with SIGKILL")
        self.signal(signal.SIGKILL)
        self.proc.join()

    ret = self.proc.exitcode
    cloudlog.info(f"{self.name} is dead with {ret}")

    if self.proc.exitcode is not None:
      self.shutting_down = False
      self.proc = None

    return ret

  def signal(self, sig: int) -> None:
    if self.proc is None:
      return

    # Don't signal if already exited
    if self.proc.exitcode is not None and self.proc.pid is not None:
      return

    # Can't signal if we don't have a pid
    if self.proc.pid is None:
      return

    cloudlog.info(f"sending signal {sig} to {self.name}")
    os.kill(self.proc.pid, sig)

  def get_process_state_msg(self):
    state = log.ManagerState.ProcessState.new_message()
    state.name = self.name
    if self.proc:
      state.running = self.proc.is_alive()
      state.shouldBeRunning = self.proc is not None and not self.shutting_down
      state.pid = self.proc.pid or 0
      state.exitCode = self.proc.exitcode or 0
    return state


class NativeProcess(ManagerProcess):
  def __init__(self, name, cwd, cmdline, should_run, enabled=True, sigkill=False, restart_if_crash=False, env=None):
    self.name = name
    self.cwd = cwd
    self.cmdline = cmdline
    self.should_run = should_run
    self.enabled = enabled
    self.sigkill = sigkill
    self.launcher = nativelauncher
    self.restart_if_crash = restart_if_crash
    self.env = env

  def prepare(self) -> None:
    pass

  def start(self) -> None:
    if self.shutting_down:
      self.stop()

    if self.proc is not None:
      return

    cwd = os.path.join(BASEDIR, self.cwd)
    cloudlog.info(f"starting process {self.name}")
    self.proc = Process(name=self.name, target=self.launcher, args=(self.cmdline, cwd, self.name, self.env))
    self.proc.start()
    self.shutting_down = False


def _normalize_bundle_modes(bundle: str) -> None:
  from iqpilot.system.proprietary_runtime.bundle_modes import normalize_bundle
  try:
    normalize_bundle(bundle, BASEDIR)
  except Exception:
    cloudlog.exception(f"failed to normalize bundle modes for {bundle}")


class BundleProcess(NativeProcess):
  def __init__(self, name, bundle, entry, should_run, enabled=True, sigkill=False, restart_if_crash=False, env=None):
    self.bundle = bundle
    self.entry = entry
    runner_path = preferred_runner_path()
    runner_cmd = str(runner_path) if runner_path.is_absolute() else "./iqpilot_bundle_runner"
    runner_cwd = ".iqpilot/runtime_root" if runner_path.is_absolute() else "system/proprietary_runtime"
    super().__init__(
      name=name,
      cwd=runner_cwd,
      cmdline=[
        runner_cmd,
        "--bundle", bundle,
        "--mode", "python-module",
        "--entry", entry,
        "--daemon-name", name,
      ],
      should_run=should_run,
      enabled=enabled,
      sigkill=sigkill,
      restart_if_crash=restart_if_crash,
      env=env,
    )

  def start(self) -> None:
    if self.proc is None:
      _normalize_bundle_modes(self.bundle)
    super().start()

  def stop(self, retry: bool = True, block: bool = True, sig: signal.Signals | None = None, timeout: float = 5) -> int | None:
    return super().stop(retry=retry, block=block, sig=signal.SIGTERM if sig is None else sig, timeout=timeout)


class PythonProcess(ManagerProcess):
  def __init__(self, name, module, should_run, enabled=True, sigkill=False, restart_if_crash=False, env=None):
    self.name = name
    self.module = module
    self.should_run = should_run
    self.enabled = enabled
    self.sigkill = sigkill
    self.launcher = launcher
    self.restart_if_crash = restart_if_crash
    self.env = env

  def prepare(self) -> None:
    if self.enabled:
      cloudlog.info(f"preimporting {self.module}")
      importlib.import_module(self.module)

  def start(self) -> None:
    if self.shutting_down:
      self.stop()

    if self.proc is not None:
      return

    cloudlog.info(f"starting python {self.module}")
    self.proc = Process(name=self.name, target=self.launcher, args=(self.module, self.name, self.env))
    self.proc.start()
    self.shutting_down = False


def ensure_running(procs: ValuesView[ManagerProcess], started: bool, params=None, CP: car.CarParams=None,
                   not_run: list[str] | None=None) -> list[ManagerProcess]:
  if not_run is None:
    not_run = []

  running = []
  now = time.monotonic()
  for p in procs:
    if p.enabled and p.name not in not_run and p.should_run(started, params, CP):
      if p.restart_if_crash and p.proc is not None and p.proc.is_alive():
        p.last_alive_time = now
      elif p.restart_if_crash and p.proc is not None:
        # uptime, not time-since-restart: the latter also counts the backoff wait,
        # which would reset the counter as soon as backoff exceeds CRASH_RESET_TIME
        if p.last_alive_time - p.last_restart_time > CRASH_RESET_TIME:
          p.crash_count = 0
          p.crash_loop_logged = False

        backoff = 0.0 if not p.crash_count else min(MAX_CRASH_BACKOFF, 2.0 ** (p.crash_count - 1))
        if now - p.last_restart_time >= backoff:
          p.crash_count += 1
          p.last_restart_time = now
          cloudlog.error(f'Restarting {p.name} (exitcode {p.proc.exitcode}) [crash {p.crash_count}]')
          if p.crash_count >= CRASH_LOOP_THRESHOLD and not p.crash_loop_logged:
            # never stop retrying: giving up on hardwared or ui is worse than restarting slowly
            cloudlog.error(f'{p.name} is in a crash loop, backing off to {MAX_CRASH_BACKOFF}s between restarts')
            p.crash_loop_logged = True
          p.restart()
      running.append(p)
    else:
      p.crash_count = 0
      p.crash_loop_logged = False
      p.last_alive_time = 0.0
      p.stop(block=False)

  for p in running:
    p.start()

  return running
