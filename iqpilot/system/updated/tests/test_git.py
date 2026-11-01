import contextlib
import os
import subprocess
from iqpilot.system.updated.tests.test_base import ParamsBaseUpdateTest, run, update_release
from iqpilot.system.updated.updated import _strip_git_extraheaders, cleanup_stale_prebuilt_marker


class TestUpdateDGitStrategy(ParamsBaseUpdateTest):
  def update_remote_release(self, release):
    update_release(self.remote_dir, release, *self.MOCK_RELEASES[release])
    run(["git", "add", "."], cwd=self.remote_dir)
    run(["git", "commit", "-m", f"openpilot release {release}"], cwd=self.remote_dir)

  def setup_remote_release(self, release):
    run(["git", "init"], cwd=self.remote_dir)
    run(["git", "config", "user.name", "IQPilot Test"], cwd=self.remote_dir)
    run(["git", "config", "user.email", "test@iqpilot.local"], cwd=self.remote_dir)
    run(["git", "checkout", "-b", release], cwd=self.remote_dir)
    self.update_remote_release(release)

  def setup_basedir_release(self, release):
    super().setup_basedir_release(release)
    run(["git", "clone", "-b", release, self.remote_dir, self.basedir])

  @contextlib.contextmanager
  def additional_context(self):
    yield

  def test_cleanup_stale_prebuilt_marker(self):
    self.setup_remote_release("release3")
    self.setup_basedir_release("release3")

    prebuilt_path = os.path.join(self.basedir, "prebuilt")
    with open(prebuilt_path, "w") as f:
      f.write("")

    cleanup_stale_prebuilt_marker(str(self.basedir), "release3")
    assert not os.path.exists(prebuilt_path)

  def test_keep_prebuilt_marker_for_prebuilt_branch(self):
    self.setup_remote_release("release3")
    self.setup_basedir_release("release3")

    prebuilt_path = os.path.join(self.basedir, "prebuilt")
    with open(prebuilt_path, "w") as f:
      f.write("")

    cleanup_stale_prebuilt_marker(str(self.basedir), "release3-prebuilt")
    assert os.path.exists(prebuilt_path)

  def test_strip_git_extraheaders(self):
    self.setup_remote_release("release3")
    self.setup_basedir_release("release3")
    key = "http.https://git.konn3kt.com/x.git/.extraheader"
    run(["git", "config", key, "Authorization: Basic stale"], cwd=self.basedir)
    _strip_git_extraheaders(str(self.basedir))
    got = subprocess.run(["git", "config", "--get-regexp", r"^http\..*\.extraheader$"],
                         cwd=self.basedir, capture_output=True, text=True)
    assert got.returncode != 0 and not got.stdout.strip()


class TestUpdatedRunTimeout:
  def test_run_returns_output(self):
    from iqpilot.system.updated import updated
    assert updated.run(["echo", "hi"], timeout=10).strip() == "hi"

  def test_run_nonzero_raises(self):
    import subprocess
    import pytest
    from iqpilot.system.updated import updated
    with pytest.raises(subprocess.CalledProcessError):
      updated.run(["sh", "-c", "exit 4"], timeout=10)

  def test_run_timeout_kills_child_tree(self, tmp_path):
    import os
    import subprocess
    import time
    import pytest
    from iqpilot.system.updated import updated
    pidfile = tmp_path / "child.pid"
    start = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
      updated.run(["sh", "-c", f"sleep 45 & echo $! > {pidfile}; wait"], timeout=2)
    assert time.monotonic() - start < 20
    child_pid = int(pidfile.read_text().strip())
    time.sleep(0.5)
    with pytest.raises(ProcessLookupError):
      os.kill(child_pid, 0)
