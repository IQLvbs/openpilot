"""Leave Preview while offline, then reconnect, through the real updater with native Params."""
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from iqpilot.common import git_creds
from iqpilot.common.params import Params
from iqpilot.system.updated import updated

PUBLIC = 'https://git.konn3kt.com/IQ.Lvbs/IQ.Pilot.git'
MIRROR = 'https://github.com/iqinc/openpilot.git'
PREVIEW = 'https://git.konn3kt.com/IQ.Lvbs/IQ.Pilot-Preview.git'
BRANCH = 'feature/preview-only'


def git(repo, *args):
  return subprocess.check_output(['git', '-c', 'commit.gpgsign=false', '-C', str(repo), *args],
                                 text=True, stderr=subprocess.STDOUT).strip()


@pytest.fixture
def device(tmp_path, monkeypatch):
  """A device on a Preview-only branch whose tester token was removed while offline."""
  assert Params.__module__ == 'iqpilot.common.params_pyx', 'requires native Params, which returns text for branches'
  monkeypatch.setenv('GIT_CONFIG_GLOBAL', str(tmp_path / 'global.gitconfig'))
  monkeypatch.setenv('GIT_CONFIG_NOSYSTEM', '1')
  for key in list(os.environ):
    if key.startswith(('GIT_CONFIG_KEY_', 'GIT_CONFIG_VALUE_')) or key in (
        'GIT_CONFIG_COUNT', 'GIT_CONFIG_PARAMETERS', 'GIT_DIR', 'GIT_WORK_TREE', 'GIT_COMMON_DIR'):
      monkeypatch.delenv(key)
  repo = tmp_path / 'device'
  git(tmp_path, 'init', '-q', '-b', BRANCH, str(repo))
  git(repo, 'config', 'user.name', 'Updater Test')
  git(repo, 'config', 'user.email', 'updater@example.invalid')
  version = repo / 'iqpilot/common/version.h'
  version.parent.mkdir(parents=True)
  version.write_text('#define COMMA_VERSION "test"\n')
  git(repo, 'add', '.')
  git(repo, 'commit', '-qm', 'installed preview')
  git(repo, 'remote', 'add', 'origin', PREVIEW)
  public = tmp_path / 'public'
  git(tmp_path, 'init', '-q', '--bare', str(public))

  params = Params(str(tmp_path / 'params'))
  params.put('UpdaterTargetBranch', BRANCH)
  params.put_bool('IsOnroad', False)
  params.put_bool('IsOffroad', True)
  monkeypatch.setattr(git_creds, 'Params', lambda: params)
  monkeypatch.setattr(git_creds, 'PARAMS_DIR', params.get_param_path())
  monkeypatch.setattr(git_creds, 'DEFAULT_REPO_DIR', str(repo))
  monkeypatch.setattr(updated, 'Params', lambda: params)
  monkeypatch.setattr(updated, 'BASEDIR', str(repo))
  monkeypatch.setattr(updated, 'setup_git_options', lambda _: None)
  monkeypatch.setattr(updated, 'set_offroad_alert', lambda *args, **kwargs: None)
  monkeypatch.setattr(updated, 'get_build_metadata', lambda: SimpleNamespace(tested_channel=False))
  monkeypatch.setattr(updated, 'AGNOS', False)
  state = SimpleNamespace(repo=repo, public=public, params=params, online=False, fetches=[])
  run = subprocess.run

  def local_command(args):
    # Serve both public sources from a local repository, or nothing while offline.
    if args[0] == 'git' and ('ls-remote' in args or 'fetch' in args):
      if 'fetch' in args:
        state.fetches.append(list(args))
      backing = public if state.online else tmp_path / 'offline'
      args = ['git', '-c', f'url.{backing.as_uri()}.insteadOf={PUBLIC}', '-c', f'url.{backing.as_uri()}.insteadOf={MIRROR}', *args[1:]]
    return args

  popen = subprocess.Popen
  monkeypatch.setattr(subprocess, 'run', lambda args, **kwargs: run(local_command(args), **kwargs))
  monkeypatch.setattr(subprocess, 'Popen', lambda args, **kwargs: popen(local_command(args), **kwargs))
  git_creds.clear_credentials()
  assert git(repo, 'config', '--get', 'remote.origin.url') == PUBLIC
  assert params.get('UpdaterTargetBranch') == BRANCH
  state.online = True
  return state


def publish(device, branch):
  git(device.repo, 'commit', '--allow-empty', '-qm', f'public {branch}')
  commit = git(device.repo, 'rev-parse', 'HEAD')
  git(device.repo, 'push', '-q', str(device.public), f'HEAD:refs/heads/{branch}')
  git(device.repo, 'reset', '-q', '--hard', 'HEAD~1')
  return commit


def test_reconnecting_moves_to_public_alpha(device):
  alpha = publish(device, 'alpha')
  updater = updated.Updater()
  updater.check_for_update()
  assert device.params.get('UpdaterTargetBranch') == 'alpha'
  assert updater.update_available
  updater.fetch_update()
  assert git(device.repo, 'rev-parse', 'HEAD') == alpha
  assert git(device.repo, 'branch', '--show-current') == 'alpha'


@pytest.mark.parametrize('installed', [BRANCH, 'feature/other-preview'])
@pytest.mark.parametrize('listing', ['empty', 'no alpha'])
def test_without_a_public_fallback_nothing_is_fetched(device, listing, installed):
  if listing == 'no alpha':
    publish(device, 'beta')
  if installed != BRANCH:
    # Running one Preview branch with another queued as the target.
    git(device.repo, 'checkout', '-q', '-b', installed)
  updater = updated.Updater()
  for cycle in range(2):
    updater.check_for_update()
    assert not updater.update_available
    updater.set_params(True, cycle, None)
    assert not device.params.get_bool('UpdaterFetchAvailable')
  assert device.fetches == []
  assert device.params.get('UpdaterTargetBranch') == BRANCH
  assert BRANCH not in updater.branches


def test_a_branch_saved_while_recovery_waits_for_the_lock_is_kept(device):
  import fcntl
  import threading
  lock = os.open(os.path.join(os.path.dirname(os.path.normpath(device.params.get_param_path())), '.lock'), os.O_CREAT | os.O_RDONLY)
  fcntl.flock(lock, fcntl.LOCK_EX)
  recovery = threading.Thread(target=git_creds._replace_target, args=(BRANCH, 'alpha'))
  try:
    recovery.start()
    recovery.join(0.3)
    assert recovery.is_alive(), 'recovery must wait for the Params lock before reading the target'
    # A user's save that finished first, as native Params.put leaves it.
    Path(device.params.get_param_path('UpdaterTargetBranch')).write_text('beta')
  finally:
    os.close(lock)
  recovery.join(5)
  assert device.params.get('UpdaterTargetBranch') == 'beta'


def test_recovery_replaces_an_unchanged_target(device):
  git_creds._replace_target(BRANCH, 'alpha')
  assert device.params.get('UpdaterTargetBranch') == 'alpha'
  assert [name for name in os.listdir(os.path.dirname(os.path.normpath(device.params.get_param_path())))
          if name.startswith('.tmp_value_')] == []
