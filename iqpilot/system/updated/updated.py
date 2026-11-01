#!/usr/bin/env python3
import os
import re
import datetime
import subprocess
import tempfile
import psutil
import signal
import fcntl
import threading
import time
from collections import defaultdict
from pathlib import Path

from iqpilot.common.basedir import BASEDIR
from iqpilot.common.git_creds import fetch_remote, remove_legacy_source_remotes, retarget_missing_public_branch, run_remote
from iqpilot.common.params import Params
from iqpilot.common.time_helpers import system_time_valid
from iqpilot.common.markdown import parse_markdown
from iqpilot.common.swaglog import cloudlog
from iqpilot.selfdrive.selfdrived.alertmanager import set_offroad_alert
from iqpilot.system.hardware import AGNOS, HARDWARE
from iqpilot.system.version import get_build_metadata, IQ_BRANCH_MIGRATIONS


LOCK_FILE = os.getenv("UPDATER_LOCK_FILE", "/tmp/safe_staging_overlay.lock")
STAGING_ROOT = os.getenv("UPDATER_STAGING_ROOT", "/data/safe_staging")
OVERLAY_INIT = Path(os.path.join(BASEDIR, ".overlay_init"))

# do not allow to engage after this many hours onroad and this many routes
HOURS_NO_CONNECTIVITY_MAX = 27
ROUTES_NO_CONNECTIVITY_MAX = 84
# send an offroad prompt after this many hours onroad and this many routes
HOURS_NO_CONNECTIVITY_PROMPT = 23
ROUTES_NO_CONNECTIVITY_PROMPT = 80


class UserRequest:
  NONE = 0
  CHECK = 1
  FETCH = 2


class UpdateInstallMode:
  DOWNLOAD_ONLY = "download_only"
  DOWNLOAD_AND_INSTALL = "download_and_install"

class WaitTimeHelper:
  def __init__(self, params=None):
    self.ready_event = threading.Event()
    self.user_request = UserRequest.NONE
    signal.signal(signal.SIGHUP, self.update_now)
    signal.signal(signal.SIGUSR1, self.check_now)
    # The UI runs in a SELinux-confined domain under hidepid=2 and cannot
    # signal this process by PID. It writes UpdaterManualRequest instead,
    # which this thread polls and turns into the same request a signal would.
    self._params = params
    if params is not None:
      params.remove("UpdaterManualRequest")
      threading.Thread(target=self._poll_request_param, daemon=True).start()

  def _read_request_param(self) -> None:
    try:
      req = self._params.get("UpdaterManualRequest", encoding="utf-8")
      if req:
        self._params.remove("UpdaterManualRequest")
    except Exception:
      cloudlog.exception("updater: manual request unavailable")
      return
    if req == "fetch":
      self.update_now(0, None)
    elif req == "check":
      self.check_now(0, None)

  def _poll_request_param(self) -> None:
    while True:
      self._read_request_param()
      time.sleep(1)

  def update_now(self, signum: int, frame) -> None:
    cloudlog.info("update request received, attempting to download update")
    self.user_request = UserRequest.FETCH
    self.ready_event.set()

  def check_now(self, signum: int, frame) -> None:
    cloudlog.info("check request received, checking for updates")
    self.user_request = UserRequest.CHECK
    self.ready_event.set()

  def sleep(self, t: float) -> None:
    self.ready_event.wait(timeout=t)

def write_time_to_param(params, param) -> None:
  t = datetime.datetime.now(datetime.UTC).replace(tzinfo=None)
  params.put(param, t)

GIT_PROBE_TIMEOUT = 90.0
GIT_FETCH_TIMEOUT = 1200.0


def run(cmd: list[str], cwd: str | None = None, timeout: float | None = None) -> str:
  if timeout is None:
    return subprocess.check_output(cmd, cwd=cwd, stderr=subprocess.STDOUT, encoding='utf8')
  with subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        encoding='utf8', start_new_session=True) as proc:
    try:
      output, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
      try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
      except ProcessLookupError:
        pass
      proc.communicate()
      raise
  if proc.returncode != 0:
    raise subprocess.CalledProcessError(proc.returncode, cmd, output)
  return output


def has_git_repo(path: str) -> bool:
  return os.path.isdir(os.path.join(path, ".git"))


def display_commit_date(value: str) -> str:
  value = value.strip()
  match = re.fullmatch(r"'?\d+ (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) [+-]\d{4}'?", value)
  return match.group(1) if match else value.strip("'")


def cleanup_stale_overlay() -> None:
  try:
    merged = os.path.join(STAGING_ROOT, "merged")
    if os.path.ismount(merged):
      run(["sudo", "umount", "-l", merged])
    if os.path.isdir(STAGING_ROOT):
      run(["sudo", "rm", "-rf", STAGING_ROOT])
    OVERLAY_INIT.unlink(missing_ok=True)
  except Exception:
    cloudlog.exception("updater: failed to clean up stale overlay")


def parse_release_notes(basedir: str) -> bytes:
  try:
    with open(os.path.join(basedir, "iqpilot", "docs", "CHANGELOG.md"), "rb") as f:
      r = f.read().split(b'\n\n', 1)[0]  # Slice latest release notes
    try:
      return bytes(parse_markdown(r.decode("utf-8")), encoding="utf-8")
    except Exception:
      return r + b"\n"
  except FileNotFoundError:
    pass
  except Exception:
    cloudlog.exception("failed to parse release notes")
  return b""


def get_agnos_target_versions(launch_env_dir: str) -> tuple[str, list[str]]:
  out = run([
    "bash", "-c",
    r'unset AGNOS_VERSION AGNOS_COMPAT_VERSIONS && source launch_env.sh && printf "%s\n%s\n" "${AGNOS_VERSION}" "${AGNOS_COMPAT_VERSIONS}"',
  ], launch_env_dir)
  lines = out.splitlines()
  expected_version = lines[0].strip() if len(lines) > 0 else ""
  compat_versions = [v.strip() for v in (lines[1] if len(lines) > 1 else "").split(",") if v.strip()]
  return expected_version, compat_versions


def fetched_os_target_versions(cwd: str, commit: str) -> tuple[str, list[str]]:
  with tempfile.TemporaryDirectory() as directory:
    Path(directory, "launch_env.sh").write_bytes(subprocess.check_output(["git", "show", f"{commit}:launch_env.sh"], cwd=cwd))
    return get_agnos_target_versions(directory)


IQOS5_MANIFEST = "iqpilot/system/hardware/tici/iqos.json"


def flash_iqos5_for(commit: str) -> int | None:
  """Flash IQ.OS 5 to the other slot when the fetched commit needs it, before the checkout changes.

  The build being left does the work: the manifest is read from the fetched commit with git show and
  flashed with this build's flasher, so nothing from the new tree is imported. Returns the slot to switch
  to after the checkout is reset, or None when the commit doesn't ask for IQ.OS 5.
  """
  if not AGNOS:
    return None
  target, compat = fetched_os_target_versions(BASEDIR, commit)
  current = HARDWARE.get_os_version()
  if not is_iqos5(target) or is_iqos5(current) or agnos_version_allowed(current, target, compat):
    return None

  from iqpilot.system.hardware.tici.agnos import flash_iqos5_update, get_target_slot_number
  cloudlog.info(f"IQ.OS 5 update: current={current}, target={target}, compat={compat}")
  set_offroad_alert("Offroad_NeosUpdate", True)
  try:
    with tempfile.TemporaryDirectory() as directory:
      manifest = os.path.join(directory, "iqos.json")
      for name in (IQOS5_MANIFEST, f"{IQOS5_MANIFEST}.sig"):
        data = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=BASEDIR)
        Path(directory, os.path.basename(name)).write_bytes(data)
      target_slot_number = get_target_slot_number()
      flash_iqos5_update(manifest, target_slot_number, cloudlog)
  finally:
    set_offroad_alert("Offroad_NeosUpdate", False)
  return target_slot_number


def os_version_number(version: str) -> str:
  return version.strip().removeprefix("IQ.OS").strip()


def is_iqos5(version: str) -> bool:
  return version.strip().startswith("IQ.OS 5") or os_version_number(version).split(".")[0] == "5"


def agnos_version_allowed(current_version: str, expected_version: str, compat_versions: list[str]) -> bool:
  current_version = os_version_number(current_version)
  expected_version = os_version_number(expected_version)
  if current_version == expected_version or current_version.startswith(f"{expected_version}-"):
    return True

  return current_version in {os_version_number(v) for v in compat_versions}

# The public release repo at git.konn3kt.com/IQ.Lvbs/IQ.Pilot needs no auth.
# Earlier builds installed an insteadOf rule + JWT credential helper that
# rewrote origin through the authenticated k3 proxy; when the device can't sign
# a JWT the fetch fails. Clearing those leaves the public origin, which fetches
# directly. A device that installed from GitHub is already rewritten to it and
# is left untouched here.
PUBLIC_RELEASE_REMOTE = "https://git.konn3kt.com/IQ.Lvbs/IQ.Pilot.git"


def _clear_stale_proxy_config(cwd: str) -> None:
  subprocess.run(["git", "-C", cwd, "config", "--unset-all", "credential.helper"],
                 check=False, capture_output=True)
  try:
    raw = subprocess.check_output(["git", "-C", cwd, "config", "--get-regexp", r"^url\..*\.insteadOf$"],
                                  text=True, stderr=subprocess.DEVNULL)
    for line in raw.strip().splitlines():
      config_key = line.split()[0] if line.strip() else ""
      if config_key and "iqos-git" in config_key:
        subprocess.run(["git", "-C", cwd, "config", "--unset", config_key], check=False, capture_output=True)
  except subprocess.CalledProcessError:
    pass
  # If a prior build pointed origin at the k3 proxy or the GitHub mirror,
  # restore the public Gitea release remote so updates come from the same host
  # the prebuilt ships to and need no auth.
  origin_url = subprocess.check_output(["git", "-C", cwd, "config", "--get", "remote.origin.url"],
                                        text=True, stderr=subprocess.DEVNULL).strip().lower()
  from iqpilot.common.git_creds import ssh_to_https
  canonical_origin = ssh_to_https(origin_url).rstrip("/").removesuffix(".git")
  # Match complete public repo names: IQ.Pilot-Preview must retain its source.
  is_release_repo = "iqos-git" in origin_url or canonical_origin in (
    PUBLIC_RELEASE_REMOTE.lower().removesuffix(".git"),
    "https://github.com/iqinc/iqpilot", "https://github.com/iqinc/openpilot",
  )
  already_public = origin_url == PUBLIC_RELEASE_REMOTE.lower()
  if is_release_repo and not already_public:
    subprocess.run(["git", "-C", cwd, "remote", "set-url", "origin", PUBLIC_RELEASE_REMOTE],
                   check=False, capture_output=True)
    cloudlog.info("updater: restored public release remote git.konn3kt.com/IQ.Lvbs/IQ.Pilot")

def _strip_git_extraheaders(cwd: str) -> None:
  try:
    raw = subprocess.check_output(["git", "-C", cwd, "config", "--get-regexp", r"^http\..*\.extraheader$"],
                                  text=True, stderr=subprocess.DEVNULL)
  except subprocess.CalledProcessError:
    return
  for line in raw.strip().splitlines():
    config_key = line.split()[0] if line.strip() else ""
    if config_key:
      subprocess.run(["git", "-C", cwd, "config", "--unset-all", config_key], check=False, capture_output=True)


def configure_git_auth(cwd: str) -> None:
  if _configure_broker_git_proxy(cwd):
    return

  _clear_stale_proxy_config(cwd)

  try:
    from iqpilot.system.proprietary_runtime.bundle_modes import import_verified_module
    import_verified_module("iqpilot_updater_private", "iqpilot_private.updater.git_remote").configure(cwd)
  except Exception:
    cloudlog.info("updater: proprietary git auth unavailable; using ambient git credentials")

  try:
    from iqpilot.common.git_creds import configure as configure_user_creds
    configure_user_creds(cwd)
  except Exception:
    cloudlog.exception("updater: user git credentials unavailable")

  _strip_git_extraheaders(cwd)


def _configure_broker_git_proxy(cwd: str) -> bool:
  return False


def setup_git_options(cwd: str) -> None:
  # We sync FS object atimes (which NEOS doesn't use) and mtimes, but ctimes
  # are outside user control. Make sure Git is set up to ignore system ctimes,
  # because they change when we make hard links during finalize. Otherwise,
  # there is a lot of unnecessary churn. This appears to be a common need on
  # OSX as well: https://www.git-tower.com/blog/make-git-rebase-safe-on-osx/

  # We are using copytree to copy the directory, which also changes
  # inode numbers. Ignore those changes too.

  # Set protocol to the new version (default after git 2.26) to reduce data
  # usage on git fetch --dry-run from about 400KB to 18KB.
  git_cfg = [
    ("core.trustctime", "false"),
    ("core.checkStat", "minimal"),
    ("protocol.version", "2"),
    ("gc.auto", "0"),
    ("gc.autoDetach", "false"),
    ("http.lowSpeedLimit", "1000"),
    ("http.lowSpeedTime", "60"),
  ]
  for option, value in git_cfg:
    run(["git", "config", option, value], cwd)


def checkpoint(cwd: str) -> None:
  head = run(["git", "rev-parse", "HEAD"], cwd).strip()
  run(["git", "update-ref", "refs/iqos/checkpoint", head], cwd)
  stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%S.%fZ")
  # A failed stash is not a clean tree: abort before checkout destroys changes.
  dirty = run(["git", "stash", "create"], cwd).strip()
  if dirty:
    # Create only; even a timestamp collision must not replace recovery data.
    run(["git", "update-ref", f"refs/iqos/dirty/{stamp}", dirty, "0" * len(head)], cwd)
  cloudlog.info(f"updater: checkpoint {head[:7]}" + (f" + dirty tree {dirty[:7]}" if dirty else ""))


def cleanup_stale_prebuilt_marker(cwd: str, branch: str) -> None:
  prebuilt_path = os.path.join(cwd, "prebuilt")
  if not os.path.exists(prebuilt_path):
    return

  tracked = True
  try:
    run(["git", "ls-files", "--error-unmatch", "prebuilt"], cwd)
  except subprocess.CalledProcessError:
    tracked = False

  if not tracked and not branch.endswith("-prebuilt"):
    os.remove(prebuilt_path)
    cloudlog.info("removed stale untracked prebuilt marker on non-prebuilt branch %s", branch)


def handle_agnos_update() -> bool:
  from iqpilot.system.hardware.tici.agnos import flash_agnos_update, get_target_slot_number

  cur_version = HARDWARE.get_os_version()
  device_type = HARDWARE.get_device_type()
  is_tici_c3 = device_type in ("tici", "three")
  updated_version, compat_versions = get_agnos_target_versions(BASEDIR)

  cloudlog.info(f"AGNOS version check: current={cur_version}, target={updated_version}, compat={compat_versions}")
  if agnos_version_allowed(cur_version, updated_version, compat_versions):
    return False

  cloudlog.info(f"Beginning background installation for AGNOS {updated_version}")
  set_offroad_alert("Offroad_NeosUpdate", True)

  manifest_name = "agnos_tici_15_1.json" if is_tici_c3 else "agnos.json"
  manifest_path = os.path.join(BASEDIR, "iqpilot/system/hardware/tici", manifest_name)
  cloudlog.info(f"AGNOS manifest selected: device_type={device_type}, manifest={manifest_name}")
  target_slot_number = get_target_slot_number()
  flash_agnos_update(manifest_path, target_slot_number, cloudlog)
  set_offroad_alert("Offroad_NeosUpdate", False)
  return True


class Updater:
  def __init__(self):
    self.params = Params()
    self.branches = defaultdict(lambda: None)
    self._has_internet: bool = False
    self.os_flashed: bool = False

    # This snapshot survives updater-only restarts after an in-place fetch.
    # manager_init clears it before the next boot's processes start.
    runtime = self.params.get("UpdaterRuntimeState") or {}
    if isinstance(runtime, dict) and runtime.get("commit"):
      self.running_commit = runtime["commit"]
      self.running_branch = runtime["branch"]
      self.running_description = runtime["description"]
      return

    # The commit/branch the running processes booted from. We update BASEDIR in
    # place, so this is what we diff against to know a reboot is needed.
    try:
      self.running_commit = self.get_commit_hash(BASEDIR)
      self.running_branch = self.get_branch(BASEDIR)
      self.running_description = self.get_description(BASEDIR)
    except Exception:
      cloudlog.exception("updater: failed to capture running version")
      self.running_commit = ""
      self.running_branch = ""
      self.running_description = ""

  @property
  def git_mode(self) -> bool:
    return has_git_repo(BASEDIR)

  @property
  def has_internet(self) -> bool:
    return self._has_internet

  @property
  def install_mode(self) -> str:
    mode = self.params.get("UpdaterInstallMode")
    if mode not in (UpdateInstallMode.DOWNLOAD_ONLY, UpdateInstallMode.DOWNLOAD_AND_INSTALL):
      return UpdateInstallMode.DOWNLOAD_AND_INSTALL
    return mode

  @property
  def target_branch(self) -> str:
    b: str | None = self.params.get("UpdaterTargetBranch")
    if b is None:
      b = self.get_branch(BASEDIR)
    b = IQ_BRANCH_MIGRATIONS.get((HARDWARE.get_device_type(), b), b)
    return b

  @property
  def update_ready(self) -> bool:
    """True when the code on disk differs from what's running -> reboot to apply.
    After a reboot this is False again, since running_* is recaptured at start."""
    if not self.git_mode:
      return False
    on_disk_commit = self.get_commit_hash(BASEDIR)
    on_disk_branch = self.get_branch(BASEDIR)
    return (on_disk_commit != self.running_commit) or (on_disk_branch != self.running_branch)

  @property
  def update_available(self) -> bool:
    """True when the code on disk is behind the remote target -> can download."""
    if not self.git_mode:
      return False
    if len(self.branches) == 0:
      return False
    target = self.target_branch
    # A target the remote doesn't list can't be fetched; advertising it would fail every cycle.
    target_commit = self.branches.get(target)
    if target_commit is None:
      return False
    return self.get_commit_hash(BASEDIR) != target_commit or self.get_branch(BASEDIR) != target

  def get_branch(self, path: str) -> str:
    if not has_git_repo(path):
      try:
        return get_build_metadata(path).channel
      except Exception:
        return ""
    return run(["git", "rev-parse", "--abbrev-ref", "HEAD"], path).rstrip()

  def get_commit_hash(self, path: str = BASEDIR) -> str:
    if not has_git_repo(path):
      try:
        return get_build_metadata(path).openpilot.git_commit
      except Exception:
        return ""
    return run(["git", "rev-parse", "HEAD"], path).rstrip()

  def get_description(self, basedir: str) -> str:
    if not os.path.exists(basedir):
      return ""

    try:
      if has_git_repo(basedir):
        with open(os.path.join(basedir, "iqpilot", "common", "version.h")) as version_file:
          version = version_file.read().split('"')[1]
        branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"], basedir).rstrip()
        commit = run(["git", "rev-parse", "HEAD"], basedir).rstrip()[:7]
        commit_date = display_commit_date(run(["git", "show", "--no-patch", "--format='%ct %ci'", "HEAD"], basedir))
      else:
        metadata = get_build_metadata(basedir)
        branch = metadata.channel
        commit = metadata.openpilot.git_commit[:7]
        version = metadata.openpilot.version
        commit_date = display_commit_date(metadata.openpilot.git_commit_date)
      return f"{version} / {branch} / {commit} / {commit_date}"
    except Exception:
      cloudlog.exception("updater.get_description")
      return ""

  def set_params(self, update_success: bool, failed_count: int, exception: str | None) -> None:
    self.params.put("UpdateFailedCount", failed_count)
    self.params.put("UpdaterTargetBranch", self.target_branch)

    self.params.put_bool("UpdaterFetchAvailable", self.update_available)
    if len(self.branches):
      self.params.put("UpdaterAvailableBranches", ','.join(self.branches.keys()))

    last_uptime_onroad = self.params.get("UptimeOnroad", return_default=True)
    last_route_count = self.params.get("RouteCount", return_default=True)
    if update_success:
      self.params.put("LastUpdateTime", datetime.datetime.now(datetime.UTC).replace(tzinfo=None))
      self.params.put("LastUpdateUptimeOnroad", last_uptime_onroad)
      self.params.put("LastUpdateRouteCount", last_route_count)

    if exception is None:
      self.params.remove("LastUpdateException")
    else:
      self.params.put("LastUpdateException", exception)

    # Current = what's running (captured at boot, stable until reboot).
    # New = what's on disk now (changes once we fetch in place).
    self.params.put("UpdaterCurrentDescription", self.running_description)
    self.params.put("UpdaterCurrentReleaseNotes", parse_release_notes(BASEDIR))
    self.params.put("UpdaterNewDescription", self.get_description(BASEDIR))
    self.params.put("UpdaterNewReleaseNotes", parse_release_notes(BASEDIR))
    self.params.put_bool("UpdateAvailable", self.update_ready)

    # Handle user prompt
    for alert in ("Offroad_UpdateFailed", "Offroad_ConnectivityNeeded", "Offroad_ConnectivityNeededPrompt"):
      set_offroad_alert(alert, False)

    build_metadata = get_build_metadata()
    if failed_count > 15 and exception is not None and self.has_internet:
      if build_metadata.tested_channel:
        extra_text = "Ensure the software is correctly installed. Uninstall and re-install if this error persists."
      else:
        extra_text = exception
      set_offroad_alert("Offroad_UpdateFailed", True, extra_text=extra_text)

  def check_for_update(self) -> None:
    cloudlog.info("checking for updates")

    if not self.git_mode:
      cloudlog.info("updater: baked non-git deployment detected; skipping git update check")
      self._has_internet = False
      self.branches = defaultdict(lambda: None)
      return

    excluded_branches = ('release2', 'release2-staging')

    setup_git_options(BASEDIR)
    remove_legacy_source_remotes(BASEDIR)

    try:
      run_remote(["git", "ls-remote", "origin", "HEAD"], BASEDIR, self.target_branch, configure_git_auth, timeout=GIT_PROBE_TIMEOUT)
      self._has_internet = True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
      self._has_internet = False

    if not self._has_internet:
      cloudlog.info("updater: remote unreachable; skipping branch check")
      self.branches = defaultdict(lambda: None)
      return

    try:
      output = run_remote(["git", "ls-remote", "--heads"], BASEDIR, self.target_branch, configure_git_auth, timeout=GIT_PROBE_TIMEOUT)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
      cloudlog.exception("updater: ls-remote --heads failed")
      self._has_internet = False
      self.branches = defaultdict(lambda: None)
      return

    self.branches = defaultdict(lambda: None)
    for line in output.split('\n'):
      ls_remotes_re = r'(?P<commit_sha>\b[0-9a-f]{5,40}\b)(\s+)(refs\/heads\/)(?P<branch_name>.*$)'
      x = re.fullmatch(ls_remotes_re, line.strip())
      if x is not None and x.group('branch_name') not in excluded_branches:
        self.branches[x.group('branch_name')] = x.group('commit_sha')
    if self.branches:
      retarget_missing_public_branch(BASEDIR, self.target_branch, self.branches)

    cur_branch = self.get_branch(BASEDIR)
    cur_commit = self.get_commit_hash(BASEDIR)
    new_branch = self.target_branch
    new_commit = self.branches.get(new_branch)
    if (cur_branch, cur_commit) != (new_branch, new_commit):
      cloudlog.info(f"update available, {cur_branch} ({str(cur_commit)[:7]}) -> {new_branch} ({str(new_commit)[:7]})")
    else:
      cloudlog.info(f"up to date on {cur_branch} ({str(cur_commit)[:7]})")

  def fetch_update(self) -> None:
    if not self.git_mode:
      cloudlog.info("updater: baked non-git deployment detected; skipping git fetch")
      self.params.put("UpdaterState", "idle")
      return

    cloudlog.info("attempting git fetch and in-place reset")

    self.params.put("UpdaterState", "downloading...")
    self.params.put_bool("UpdateAvailable", False)

    setup_git_options(BASEDIR)

    run(["git", "config", "--replace-all", "remote.origin.fetch", "+refs/heads/*:refs/remotes/origin/*"], BASEDIR)

    branch = fetch_remote(BASEDIR, self.target_branch, configure_git_auth, timeout=GIT_FETCH_TIMEOUT)
    fetched_commit = run(["git", "rev-parse", "FETCH_HEAD"], BASEDIR).strip()
    cloudlog.info("git fetch success: %s (%s)", branch, fetched_commit)

    # raises before the checkout changes if the flash fails
    iqos5_slot = flash_iqos5_for(fetched_commit)
    previous_branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"], BASEDIR).strip()
    if iqos5_slot is not None:
      # bound before the reset, so the switch runs this build's code rather than the new tree's
      from iqpilot.system.hardware.tici.agnos import switch_slot

    checkpoint(BASEDIR)
    cloudlog.info("git reset in progress")
    cmds = [
      ["git", "checkout", "--force", "-B", branch, fetched_commit],
      ["git", "branch", "--set-upstream-to", f"origin/{branch}"],
      ["git", "reset", "--hard", fetched_commit],
    ]
    r = [run(cmd, BASEDIR) for cmd in cmds]
    cloudlog.info("git reset success: %s", '\n'.join(r))
    cleanup_stale_prebuilt_marker(BASEDIR, branch)

    if iqos5_slot is not None:
      try:
        switch_slot(iqos5_slot, cloudlog)
      except Exception:
        cloudlog.exception("IQ.OS 5 slot switch failed; restoring the previous checkout")
        if previous_branch == "HEAD":
          run(["git", "checkout", "--force", "--detach", "refs/iqos/checkpoint"], BASEDIR)
        else:
          run(["git", "checkout", "--force", "-B", previous_branch, "refs/iqos/checkpoint"], BASEDIR)
        run(["git", "reset", "--hard", "refs/iqos/checkpoint"], BASEDIR)
        raise
      self.os_flashed = True
    # TODO: show agnos download progress
    elif AGNOS and handle_agnos_update():
      self.os_flashed = True

    cloudlog.info("update applied to disk; reboot to finish")


# cap the reboot quiet-window deferral so an idle SSH session can't hold updates forever
MAX_SSH_DEFER_S = 2 * 60 * 60


def has_active_ssh_session() -> bool:
  """True when someone is interactively logged in over SSH. tmux's own panes
  register in utmp with a tmux(...) host and don't count."""
  try:
    out = subprocess.check_output(["who"], encoding="utf8", timeout=5)
  except Exception:
    return False
  for line in out.splitlines():
    fields = line.split()
    if len(fields) >= 2 and fields[1].startswith("pts/") and "tmux(" not in line:
      return True
  return False


def prepare_environment() -> bool:
  """Run the boot-time env sync + build right after an update lands on disk,
  while the device is offroad and the stack is idle -- so the reboot that
  follows starts on a warm cache with nothing heavy left to do at boot.
  launch_chffrplus.sh keeps the identical steps as its fallback path."""
  # same env the launch script builds under: repo on PYTHONPATH, venv tools
  # (uv, cython, ...) on PATH
  env = os.environ.copy()
  env["PYTHONPATH"] = BASEDIR + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
  env["PATH"] = os.pathsep.join([os.path.join(BASEDIR, ".venv", "bin"), "/usr/local/venv/bin", env.get("PATH", "")])

  def _run(cmd: list[str], cwd: str) -> None:
    subprocess.check_output(cmd, cwd=cwd, stderr=subprocess.STDOUT, encoding="utf8", env=env)

  try:
    env_sync = os.path.join(BASEDIR, "artifacts", "runtime", "env_sync.sh")
    if os.path.exists(env_sync):
      cloudlog.info("update prep: syncing python env")
      _run(["bash", env_sync], BASEDIR)
    venv_python = os.path.join(BASEDIR, ".venv", "bin", "python3")
    build_py = os.path.join(BASEDIR, "iqpilot", "system", "manager", "build.py")
    if not os.path.exists(os.path.join(BASEDIR, "prebuilt")) and os.path.exists(venv_python) and os.path.exists(build_py):
      cloudlog.info("update prep: building")
      _run([venv_python, build_py, "--headless"], os.path.dirname(build_py))
    cloudlog.info("update prep: done, next boot starts on a warm cache")
    return True
  except subprocess.CalledProcessError as e:
    output = e.output[-4096:] if e.output else ""
    cloudlog.event("update prep failed", cmd=e.cmd, output=output, returncode=e.returncode, error=True)
    return False


def publish_runtime(params: Params, updater: Updater, disabled: bool) -> None:
  params.put("UpdaterRuntimeState", {
    "disabled": disabled, "commit": updater.running_commit,
    "branch": updater.running_branch, "description": updater.running_description,
  })


def publish_running_version(params: Params) -> None:
  try:
    updater = Updater()
    publish_runtime(params, updater, True)
    release_notes = parse_release_notes(BASEDIR)
    params.put("UpdaterState", "idle")
    params.put("UpdaterTargetBranch", updater.target_branch)
    params.put("UpdaterCurrentDescription", updater.running_description)
    params.put("UpdaterCurrentReleaseNotes", release_notes)
    params.put("UpdaterNewDescription", updater.running_description)
    params.put("UpdaterNewReleaseNotes", release_notes)
    params.put_bool("UpdaterFetchAvailable", False)
    params.put_bool("UpdateAvailable", False)
  except Exception:
    cloudlog.exception("updater: failed to publish running version")


def main() -> None:
  params = Params()

  if params.get_bool("DisableUpdates"):
    cloudlog.warning("updates are disabled by the DisableUpdates param")
    publish_running_version(params)
    return

  with open(LOCK_FILE, 'w') as ov_lock_fd:
    try:
      fcntl.flock(ov_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as e:
      raise RuntimeError("couldn't get overlay lock; is another instance running?") from e

    # Set low io priority
    proc = psutil.Process()
    if psutil.LINUX:
      proc.ionice(psutil.IOPRIO_CLASS_BE, value=7)

    cleanup_stale_overlay()

    if not params.get("InstallDate"):
      t = datetime.datetime.now(datetime.UTC).replace(tzinfo=None)
      params.put("InstallDate", t)

    updater = Updater()
    update_failed_count = 0 # TODO: Load from param?
    wait_helper = WaitTimeHelper(params)

    params.put("UpdaterState", "idle")
    params.put_bool("UpdateAvailable", False)

    publish_runtime(params, updater, False)

    # Run the update loop
    first_run = True
    # commit hash the apply-time prep (env sync + build) last succeeded for
    prepared_commit = updater.running_commit
    # when the pending reboot first became eligible, for the SSH quiet-window cap
    install_eligible_since = None
    while True:
      wait_helper.ready_event.clear()

      # Attempt an update
      exception = None
      install_pending = False
      try:
        # ensure we have some params written soon after startup
        updater.set_params(False, update_failed_count, exception)

        if not system_time_valid() or first_run:
          first_run = False
          wait_helper.sleep(60)
          continue

        update_failed_count += 1

        # check for update
        params.put("UpdaterState", "checking...")
        updater.check_for_update()

        # download update
        last_fetch = params.get("UpdaterLastFetchTime")
        timed_out = last_fetch is None or (datetime.datetime.now(datetime.UTC).replace(tzinfo=None) - last_fetch > datetime.timedelta(days=3))
        user_requested_fetch = wait_helper.user_request == UserRequest.FETCH
        if params.get("IsOnroad") is not False:
          # the update download + AGNOS flash hold hundreds of MB; onroad that trips lowMemory,
          # so defer the heavy work until parked
          cloudlog.info("skipping fetch, device is onroad")
        elif params.get_bool("NetworkMetered") and not timed_out and not user_requested_fetch:
          cloudlog.info("skipping fetch, connection metered")
        elif wait_helper.user_request == UserRequest.CHECK:
          cloudlog.info("skipping fetch, only checking")
        elif updater.update_available or user_requested_fetch:
          updater.fetch_update()
          write_time_to_param(params, "UpdaterLastFetchTime")
        else:
          cloudlog.info("already up to date, skipping fetch")

        # Finish a fetched update. update_ready stays true across cycles (on-disk
        # vs running commit), so failed prep and deferred reboots retry here on
        # the short sleep instead of being one-shot at fetch time.
        if updater.update_ready and params.get("IsOnroad") is False:
          on_disk_commit = updater.get_commit_hash(BASEDIR)
          if updater.os_flashed and prepared_commit != on_disk_commit:
            # the new tree targets the OS just flashed to the other slot; its env prep on this OS is
            # wasted or fails outright, and the launch script swaps slots and rebuilds on next boot
            cloudlog.info("update prep: skipped, the OS update swaps slots on reboot")
            prepared_commit = on_disk_commit
          if prepared_commit != on_disk_commit:
            params.put("UpdaterState", "preparing...")
            if prepare_environment():
              prepared_commit = on_disk_commit
          if prepared_commit != on_disk_commit:
            # prep failed: hold the reboot and retry next cycle. The boot-time
            # fallback path still exists, but rebooting into it is exactly the
            # heavy all-at-once boot that takes devices down.
            install_pending = True
          elif params.get("IsOnroad") is not False:
            # Preparation may take long enough for the car to start driving.
            install_eligible_since = None
          elif updater.install_mode == UpdateInstallMode.DOWNLOAD_AND_INSTALL:
            if install_eligible_since is None:
              install_eligible_since = time.monotonic()
            if has_active_ssh_session() and time.monotonic() - install_eligible_since < MAX_SSH_DEFER_S:
              cloudlog.info("update ready; deferring reboot, active SSH session")
              install_pending = True
            else:
              cloudlog.info("update ready; auto-install mode enabled, triggering reboot")
              params.put_bool("DoReboot", True)
        else:
          install_eligible_since = None
        update_failed_count = 0
      except subprocess.CalledProcessError as e:
        cloudlog.event(
          "update process failed",
          cmd=e.cmd,
          output=e.output,
          returncode=e.returncode
        )
        exception = f"command failed: {e.cmd}\n{e.output}"
      except Exception as e:
        cloudlog.exception("uncaught updated exception, shouldn't happen")
        exception = str(e)

      try:
        params.put("UpdaterState", "idle")
        update_successful = (update_failed_count == 0)
        updater.set_params(update_successful, update_failed_count, exception)
      except Exception:
        cloudlog.exception("uncaught updated exception while setting params, shouldn't happen")

      # infrequent attempts if we successfully updated recently; short cadence
      # while an install is pending (failed prep retry or deferred reboot)
      wait_helper.user_request = UserRequest.NONE
      wait_helper.sleep(5*60 if (update_failed_count > 0 or install_pending) else 1.5*60*60)


if __name__ == "__main__":
  main()
