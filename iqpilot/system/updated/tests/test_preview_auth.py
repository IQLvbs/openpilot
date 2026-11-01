import subprocess
from types import SimpleNamespace

import pytest

from iqpilot.common import git_creds
from iqpilot.system.updated.updated import configure_git_auth
from iqpilot.system.updated import updated


@pytest.mark.parametrize('origin', [
  'https://git.konn3kt.com/IQ.Lvbs/IQ.Pilot-Preview.git',
  'git@ssh.gitlvb.teallvbs.xyz:IQ.Lvbs/IQ.Pilot-Preview.git',
])
def test_legacy_updater_preserves_preview_source(tmp_path, monkeypatch, origin):
  monkeypatch.setattr(git_creds, 'get_credentials', lambda: None)
  subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
  subprocess.run(['git', '-C', str(tmp_path), 'remote', 'add', 'origin', origin], check=True)
  configure_git_auth(str(tmp_path))
  actual = subprocess.check_output(['git', '-C', str(tmp_path), 'remote', 'get-url', 'origin'], text=True).strip()
  assert actual == origin


def test_legacy_updater_still_retires_the_old_public_proxy(tmp_path, monkeypatch):
  monkeypatch.setattr(git_creds, 'get_credentials', lambda: None)
  subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
  subprocess.run(['git', '-C', str(tmp_path), 'remote', 'add', 'origin',
                  'https://example.invalid/iqos-git/IQ.Lvbs/IQ.Pilot.git'], check=True)
  configure_git_auth(str(tmp_path))
  actual = subprocess.check_output(['git', '-C', str(tmp_path), 'remote', 'get-url', 'origin'], text=True).strip()
  assert actual == 'https://git.konn3kt.com/IQ.Lvbs/IQ.Pilot.git'


def test_fetch_checks_out_the_recovered_public_target(monkeypatch):
  params = {"UpdaterTargetBranch": "feature/preview-only"}
  updater = updated.Updater.__new__(updated.Updater)
  monkeypatch.setattr(updated.Updater, "git_mode", property(lambda _self: True))
  updater.params = SimpleNamespace(get=lambda key, **_kwargs: params.get(key), put=params.__setitem__, put_bool=params.__setitem__)
  commands = []

  def recover_public_target(repo, branch, configure_auth, timeout=None):
    assert branch == "feature/preview-only"
    params["UpdaterTargetBranch"] = "alpha"
    return "alpha"

  monkeypatch.setattr(updated, "fetch_remote", recover_public_target)
  monkeypatch.setattr(updated, "run", lambda cmd, _repo: commands.append(cmd) or "fetched-commit")
  monkeypatch.setattr(updated, "setup_git_options", lambda _repo: None)
  monkeypatch.setattr(updated, "cleanup_stale_prebuilt_marker", lambda _repo, _branch: None)
  monkeypatch.setattr(updated, "AGNOS", False)
  updater.fetch_update()

  assert ["git", "checkout", "--force", "-B", "alpha", "fetched-commit"] in commands
  assert ["git", "branch", "--set-upstream-to", "origin/alpha"] in commands
