"""Exercise recovery and fetched-branch identity through alpha's in-place updater."""
import datetime
import subprocess
from types import SimpleNamespace

import pytest

from iqpilot.common.params import Params
from iqpilot.system.updated import updated


def git(repo, *args):
  return subprocess.check_output(['git', '-c', 'commit.gpgsign=false', '-C', str(repo), *args],
                                 text=True, stderr=subprocess.STDOUT).strip()


@pytest.fixture
def local_update(tmp_path, monkeypatch):
  remote = tmp_path / 'remote'
  repo = tmp_path / 'checkout'
  remote.mkdir()
  git(remote, 'init', '-q', '-b', 'alpha')
  git(remote, 'config', 'user.name', 'Updater Test')
  git(remote, 'config', 'user.email', 'updater@example.invalid')
  version = remote / 'iqpilot/common/version.h'
  version.parent.mkdir(parents=True)
  version.write_text('#define COMMA_VERSION \"test\"\n')
  (remote / 'tracked').write_text('old release\n')
  git(remote, 'add', '.')
  git(remote, 'commit', '-qm', 'old release')
  git(tmp_path, 'clone', '-q', str(remote), str(repo))
  git(repo, 'config', 'user.name', 'Updater Test')
  git(repo, 'config', 'user.email', 'updater@example.invalid')
  params = Params(str(tmp_path / 'params'))
  monkeypatch.setattr(updated, 'BASEDIR', str(repo))
  monkeypatch.setattr(updated, 'Params', lambda: params)
  monkeypatch.setattr(updated, 'AGNOS', False)
  # Keep fetch/configure on the production path but use a local remote with no
  # fleet credentials. No network, OS flash, build, or reboot is invoked.
  monkeypatch.setattr(updated, 'configure_git_auth', lambda _cwd: None)
  updater = updated.Updater()
  (remote / 'tracked').write_text('new release\n')
  git(remote, 'commit', '-qam', 'new release')
  return remote, repo, updater


@pytest.mark.parametrize('dirty', [False, True])
def test_fetch_preserves_recoverable_tracked_state(local_update, tmp_path, dirty):
  remote, repo, updater = local_update
  old_head = git(repo, 'rev-parse', 'HEAD')
  if dirty:
    (repo / 'tracked').write_text('staged edit\n')
    git(repo, 'add', 'tracked')
    (repo / 'tracked').write_text('unstaged edit\n')
    (repo / 'added').write_text('staged addition\n')
    git(repo, 'add', 'added')
  (repo / 'untracked').write_text('kept local file\n')
  updater.fetch_update()
  assert git(repo, 'rev-parse', 'HEAD') == git(remote, 'rev-parse', 'HEAD')
  assert (repo / 'tracked').read_text() == 'new release\n'
  assert (repo / 'untracked').read_text() == 'kept local file\n'
  assert git(repo, 'rev-parse', 'refs/iqos/checkpoint') == old_head
  refs = git(repo, 'for-each-ref', '--format=%(refname)', 'refs/iqos/dirty').splitlines()
  assert len(refs) == int(dirty)
  recovered = tmp_path / 'recovered'
  git(tmp_path, 'clone', '-q', str(repo), str(recovered))
  git(recovered, 'checkout', '--detach', old_head)
  if dirty:
    stash = git(repo, 'rev-parse', refs[0])
    assert git(repo, 'rev-parse', stash + '^1') == old_head
    git(recovered, 'fetch', str(repo), refs[0])
    git(recovered, 'stash', 'apply', '--index', 'FETCH_HEAD')
    assert (recovered / 'tracked').read_text() == 'unstaged edit\n'
    assert git(recovered, 'show', ':tracked') == 'staged edit'
    assert (recovered / 'added').read_text() == 'staged addition\n'
    assert git(recovered, 'diff', '--name-only', '--cached') == 'added\ntracked'
  else:
    assert (recovered / 'tracked').read_text() == 'old release\n'
    assert git(recovered, 'status', '--porcelain') == ''


def test_checkpoint_does_not_modify_the_index_or_worktree(local_update):
  _remote, repo, _updater = local_update
  (repo / 'tracked').write_text('staged\n')
  git(repo, 'add', 'tracked')
  (repo / 'tracked').write_text('unstaged\n')
  before = git(repo, 'status', '--porcelain'), (repo / 'tracked').read_bytes(), git(repo, 'write-tree')
  updated.checkpoint(str(repo))
  assert (git(repo, 'status', '--porcelain'), (repo / 'tracked').read_bytes(), git(repo, 'write-tree')) == before


def test_checkpoint_retains_two_dirty_trees_in_one_second(local_update, monkeypatch):
  _remote, repo, _updater = local_update
  real_datetime = datetime.datetime
  stamps = iter([real_datetime(2026, 10, 21, 12, 0, 0, us, tzinfo=datetime.UTC) for us in (1, 2)])
  monkeypatch.setattr(updated, 'datetime', SimpleNamespace(UTC=datetime.UTC,
                                                        datetime=SimpleNamespace(now=lambda _tz: next(stamps))))
  for content in ('first edit\n', 'second edit\n'):
    (repo / 'tracked').write_text(content)
    updated.checkpoint(str(repo))
  refs = git(repo, 'for-each-ref', '--format=%(refname)', 'refs/iqos/dirty').splitlines()
  assert len(refs) == 2
  assert {git(repo, 'show', ref + ':tracked') for ref in refs} == {'first edit', 'second edit'}


def test_fetch_aborts_if_dirty_tree_cannot_be_saved(local_update):
  _remote, repo, updater = local_update
  old_head = git(repo, 'rev-parse', 'HEAD')
  # An unresolved index makes git stash create fail. Preserve it for manual
  # recovery rather than resetting through the conflict without a dirty ref.
  blob = git(repo, 'rev-parse', 'HEAD:tracked')
  stages = f'100644 {blob} 1\ttracked\n100644 {blob} 2\ttracked\n100644 {blob} 3\ttracked\n'
  git(repo, 'update-index', '--force-remove', 'tracked')
  subprocess.run(['git', '-C', str(repo), 'update-index', '--index-info'], input=stages, text=True, check=True)
  (repo / 'tracked').write_text('unresolved local conflict\n')
  index = git(repo, 'ls-files', '--stage')
  with pytest.raises(subprocess.CalledProcessError):
    updater.fetch_update()
  assert git(repo, 'rev-parse', 'HEAD') == old_head
  assert (repo / 'tracked').read_text() == 'unresolved local conflict\n'
  assert git(repo, 'ls-files', '--stage') == index
  assert git(repo, 'for-each-ref', '--format=%(refname)', 'refs/iqos/dirty') == ''


def test_checkpoint_never_overwrites_an_existing_dirty_ref(local_update, monkeypatch):
  _remote, repo, _updater = local_update
  stamp = datetime.datetime(2026, 10, 21, 12, 0, 0, 1, tzinfo=datetime.UTC)
  monkeypatch.setattr(updated, 'datetime', SimpleNamespace(UTC=datetime.UTC,
                                                        datetime=SimpleNamespace(now=lambda _tz: stamp)))
  (repo / 'tracked').write_text('first edit\n')
  updated.checkpoint(str(repo))
  ref = git(repo, 'for-each-ref', '--format=%(refname)', 'refs/iqos/dirty')
  saved = git(repo, 'rev-parse', ref)
  (repo / 'tracked').write_text('second edit\n')
  with pytest.raises(subprocess.CalledProcessError):
    updated.checkpoint(str(repo))
  assert git(repo, 'rev-parse', ref) == saved
  assert (repo / 'tracked').read_text() == 'second edit\n'


def test_fetch_does_not_relabel_a_commit_if_target_changes_during_fetch(local_update, monkeypatch):
  from iqpilot.common import git_creds
  remote, repo, updater = local_update
  # Fetches run through the bounded remote helper, not subprocess.check_output.
  remote_output = git_creds._remote_output

  def change_target_after_fetch(cmd, *args, **kwargs):
    output = remote_output(cmd, *args, **kwargs)
    if cmd == ['git', 'fetch', 'origin', 'alpha']:
      updater.params.put('UpdaterTargetBranch', 'next-request')
    return output

  monkeypatch.setattr(git_creds, '_remote_output', change_target_after_fetch)
  updater.fetch_update()
  assert git(repo, 'rev-parse', 'HEAD') == git(remote, 'rev-parse', 'HEAD')
  assert git(repo, 'symbolic-ref', '--short', 'HEAD') == 'alpha'
  assert updater.params.get('UpdaterTargetBranch') == 'next-request'


def test_failed_checkpoint_pin_aborts_before_checkout(local_update):
  _remote, repo, updater = local_update
  head = git(repo, 'rev-parse', 'HEAD')
  (repo / 'tracked').write_text('local edit\n')
  refs = repo / '.git/refs/iqos'
  refs.mkdir(parents=True)
  (refs / 'checkpoint.lock').touch()
  with pytest.raises(subprocess.CalledProcessError):
    updater.fetch_update()
  assert git(repo, 'rev-parse', 'HEAD') == head
  assert (repo / 'tracked').read_text() == 'local edit\n'


def test_checkpoint_is_pinned_before_each_destructive_command(local_update, monkeypatch):
  _remote, repo, updater = local_update
  head = git(repo, 'rev-parse', 'HEAD')
  (repo / 'tracked').write_text('local edit\n')
  run = updated.run
  replacements = []

  def observe(cmd, cwd=None):
    if cmd[:2] in (['git', 'checkout'], ['git', 'reset']):
      assert cwd == str(repo)
      assert git(repo, 'rev-parse', 'refs/iqos/checkpoint') == head
      ref = git(repo, 'for-each-ref', '--format=%(refname)', 'refs/iqos/dirty')
      assert git(repo, 'show', ref + ':tracked') == 'local edit'
      replacements.append(cmd[1])
    return run(cmd, cwd)

  monkeypatch.setattr(updated, 'run', observe)
  updater.fetch_update()
  assert replacements == ['checkout', 'reset']


def test_fetch_keeps_recovered_public_branch_when_another_request_arrives(local_update, monkeypatch):
  from iqpilot.common import git_creds
  remote, repo, updater = local_update
  updater.params.put('UpdaterTargetBranch', 'feature/preview-only')
  git(repo, 'remote', 'set-url', 'origin', str(repo / 'missing-preview'))
  monkeypatch.setattr(git_creds, '_params_get', updater.params.get)

  def fallback(cwd, branch, _configure):
    assert branch == 'feature/preview-only'
    git(repo, 'remote', 'set-url', 'origin', str(remote))
    updater.params.put('UpdaterTargetBranch', 'alpha')
    return 'alpha'

  monkeypatch.setattr(git_creds, '_fallback_to_public', fallback)
  remote_output = git_creds._remote_output

  def next_request(cmd, *args, **kwargs):
    output = remote_output(cmd, *args, **kwargs)
    if cmd == ['git', 'fetch', 'origin', 'alpha']:
      updater.params.put('UpdaterTargetBranch', 'next-request')
    return output

  monkeypatch.setattr(git_creds, '_remote_output', next_request)
  updater.fetch_update()
  assert git(repo, 'symbolic-ref', '--short', 'HEAD') == 'alpha'
  assert git(repo, 'rev-parse', 'HEAD') == git(remote, 'rev-parse', 'HEAD')
  assert updater.params.get('UpdaterTargetBranch') == 'next-request'
