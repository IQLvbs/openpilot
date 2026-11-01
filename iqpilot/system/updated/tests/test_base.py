import os
import pathlib
import shutil
import signal
import stat
import subprocess
import tempfile
import time
import pytest

from iqpilot.common.params import Params
from iqpilot.system.manager.process import ManagerProcess
from iqpilot.selfdrive.test.helpers import processes_context


def run(args, **kwargs):
  return subprocess.check_output(args, **kwargs)


def update_release(directory, name, version, agnos_version, release_notes):
  (directory / "iqpilot" / "docs").mkdir(parents=True, exist_ok=True)
  with open(directory / "iqpilot" / "docs" / "CHANGELOG.md", "w") as f:
    f.write(release_notes)

  (directory / "iqpilot" / "common").mkdir(parents=True, exist_ok=True)

  with open(directory / "iqpilot" / "common" / "version.h", "w") as f:
    f.write(f'#define COMMA_VERSION "{version}"')

  launch_env = directory / "launch_env.sh"
  with open(launch_env, "w") as f:
    f.write(f'export AGNOS_VERSION="{agnos_version}"')

  st = os.stat(launch_env)
  os.chmod(launch_env, st.st_mode | stat.S_IEXEC)

  test_symlink = directory / "test_symlink"
  if not os.path.exists(str(test_symlink)):
    os.symlink("iqpilot/common/version.h", test_symlink)


def get_version(path: str) -> str:
  with open(os.path.join(path, "iqpilot", "common", "version.h")) as f:
    return f.read().split('"')[1]


@pytest.mark.linux
@pytest.mark.slow
class BaseUpdateTest:
  def setup_method(self):
    self.tmpdir = tempfile.mkdtemp()
    self.mock_update_path = pathlib.Path(self.tmpdir)

    self.params = Params()
    # Downloads require the parked state normally published by the manager.
    self.params.put_bool("IsOnroad", False)

    self.basedir = self.mock_update_path / "openpilot"
    self.basedir.mkdir()

    self.staging_root = self.mock_update_path / "safe_staging"

    self.remote_dir = self.mock_update_path / "remote"
    self.remote_dir.mkdir()

    os.environ["UPDATER_STAGING_ROOT"] = str(self.staging_root)
    os.environ["UPDATER_LOCK_FILE"] = str(self.mock_update_path / "safe_staging_overlay.lock")

    self.MOCK_RELEASES = {
      "release3": ("0.1.2", "1.2", "0.1.2 release notes"),
      "master": ("0.1.3", "1.2", "0.1.3 release notes"),
    }

  @pytest.fixture(autouse=True)
  def mock_basedir(self, mocker):
    mocker.patch("iqpilot.common.basedir.BASEDIR", self.basedir)
    mocker.patch("iqpilot.system.updated.updated.BASEDIR", str(self.basedir))
    mocker.patch("iqpilot.system.updated.updated.STAGING_ROOT", str(self.staging_root))
    mocker.patch("iqpilot.system.updated.updated.LOCK_FILE", str(self.mock_update_path / "safe_staging_overlay.lock"))
    mocker.patch("iqpilot.system.updated.updated.OVERLAY_INIT", self.basedir / ".overlay_init")

  def set_target_branch(self, branch):
    self.params.put("UpdaterTargetBranch", branch)

  def setup_basedir_release(self, release):
    self.params = Params()
    self.set_target_branch(release)

  def update_remote_release(self, release):
    raise NotImplementedError("")

  def setup_remote_release(self, release):
    raise NotImplementedError("")

  def additional_context(self):
    raise NotImplementedError("")

  def teardown_method(self):
    shutil.rmtree(self.tmpdir)

  def wait_for_condition(self, condition, timeout=12):
    start = time.monotonic()
    while True:
      waited = time.monotonic() - start
      if condition():
        print(f"waited {waited}s for condition ")
        return waited

      if waited > timeout:
        raise TimeoutError("timed out waiting for condition")

      time.sleep(1)

  def _test_finalized_update(self, branch, version, agnos_version, release_notes):
    assert get_version(str(self.basedir)) == version
    assert os.access(str(self.basedir / "launch_env.sh"), os.X_OK)

    with open(self.basedir / "test_symlink") as f:
      assert version in f.read()

class ParamsBaseUpdateTest(BaseUpdateTest):
  def _test_finalized_update(self, branch, version, agnos_version, release_notes):
    assert self.params.get("UpdaterNewDescription").startswith(f"{version} / {branch}")
    assert self.params.get("UpdaterNewReleaseNotes") == f"{release_notes}\n".encode()
    super()._test_finalized_update(branch, version, agnos_version, release_notes)

  def send_check_for_updates_signal(self, updated: ManagerProcess):
    updated.signal(signal.SIGUSR1.value)

  def send_download_signal(self, updated: ManagerProcess):
    updated.signal(signal.SIGHUP.value)

  def _test_params(self, branch, fetch_available, update_available):
    assert self.params.get("UpdaterTargetBranch") == branch
    assert self.params.get_bool("UpdaterFetchAvailable") == fetch_available
    assert self.params.get_bool("UpdateAvailable") == update_available

  def wait_for_idle(self):
    self.wait_for_condition(lambda: self.params.get("UpdaterState") == "idle")

  def wait_for_failed(self):
    self.wait_for_condition(lambda: self.params.get("UpdateFailedCount") is not None and \
                                              self.params.get("UpdateFailedCount") > 0)

  def wait_for_fetch_available(self):
    self.wait_for_condition(lambda: self.params.get_bool("UpdaterFetchAvailable"))

  def wait_for_update_available(self):
    self.wait_for_condition(lambda: self.params.get_bool("UpdateAvailable"))

  def wait_for_reboot_requested(self):
    self.wait_for_condition(lambda: self.params.get_bool("DoReboot"))

  def test_no_update(self):
    # Start on release3, ensure we don't fetch any updates
    self.setup_remote_release("release3")
    self.setup_basedir_release("release3")

    with self.additional_context(), processes_context(["updated"]) as [updated]:
      self._test_params("release3", False, False)
      self.wait_for_idle()
      self._test_params("release3", False, False)

      self.send_check_for_updates_signal(updated)

      self.wait_for_idle()

      self._test_params("release3", False, False)

  def test_new_release(self):
    # Start on release3, simulate a release3 commit, ensure we fetch that update properly
    self.setup_remote_release("release3")
    self.setup_basedir_release("release3")

    with self.additional_context(), processes_context(["updated"]) as [updated]:
      self._test_params("release3", False, False)
      self.wait_for_idle()
      self._test_params("release3", False, False)

      self.MOCK_RELEASES["release3"] = ("0.1.3", "1.2", "0.1.3 release notes")
      self.update_remote_release("release3")

      self.send_check_for_updates_signal(updated)

      self.wait_for_fetch_available()

      self._test_params("release3", True, False)

      self.send_download_signal(updated)

      self.wait_for_update_available()

      self._test_params("release3", False, True)
      self._test_finalized_update("release3", *self.MOCK_RELEASES["release3"])

  def test_switch_branches(self):
    # Start on release3, request to switch to master manually, ensure we switched
    self.setup_remote_release("release3")
    self.setup_remote_release("master")
    self.setup_basedir_release("release3")

    with self.additional_context(), processes_context(["updated"]) as [updated]:
      self._test_params("release3", False, False)
      self.wait_for_idle()
      self._test_params("release3", False, False)

      self.set_target_branch("master")
      self.send_check_for_updates_signal(updated)

      self.wait_for_fetch_available()

      self._test_params("master", True, False)

      self.send_download_signal(updated)

      self.wait_for_update_available()

      self._test_params("master", False, True)
      self._test_finalized_update("master", *self.MOCK_RELEASES["master"])

  def test_download_only_does_not_auto_install(self):
    self.setup_remote_release("release3")
    self.setup_basedir_release("release3")
    self.params.put("UpdaterInstallMode", "download_only")

    with self.additional_context(), processes_context(["updated"]) as [updated]:
      self.wait_for_idle()

      self.MOCK_RELEASES["release3"] = ("0.1.3", "1.2", "0.1.3 release notes")
      self.update_remote_release("release3")

      self.send_check_for_updates_signal(updated)
      self.wait_for_fetch_available()

      self.send_download_signal(updated)
      self.wait_for_update_available()

      assert not self.params.get_bool("DoReboot")

  def test_download_and_install_auto_installs(self):
    self.setup_remote_release("release3")
    self.setup_basedir_release("release3")
    self.params.put("UpdaterInstallMode", "download_and_install")

    with self.additional_context(), processes_context(["updated"]) as [updated]:
      self.wait_for_idle()

      self.MOCK_RELEASES["release3"] = ("0.1.3", "1.2", "0.1.3 release notes")
      self.update_remote_release("release3")

      self.send_check_for_updates_signal(updated)
      self.wait_for_fetch_available()

      self.send_download_signal(updated)
      self.wait_for_reboot_requested()

  def test_agnos_update(self, mocker):
    # Start on release3, push an update with an agnos change
    self.setup_remote_release("release3")
    self.setup_basedir_release("release3")

    with self.additional_context(), processes_context(["updated"]) as [updated]:
      mocker.patch("iqpilot.system.hardware.AGNOS", "True")
      mocker.patch("iqpilot.system.hardware.tici.hardware.Tici.get_os_version", "1.2")
      mocker.patch("iqpilot.system.hardware.tici.agnos.get_target_slot_number")
      mocker.patch("iqpilot.system.hardware.tici.agnos.flash_agnos_update")

      self._test_params("release3", False, False)
      self.wait_for_idle()
      self._test_params("release3", False, False)

      self.MOCK_RELEASES["release3"] = ("0.1.3", "1.3", "0.1.3 release notes")
      self.update_remote_release("release3")

      self.send_check_for_updates_signal(updated)

      self.wait_for_fetch_available()

      self._test_params("release3", True, False)

      self.send_download_signal(updated)

      self.wait_for_update_available()

      self._test_params("release3", False, True)
      self._test_finalized_update("release3", *self.MOCK_RELEASES["release3"])
