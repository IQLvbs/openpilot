"""IQ.OS 4 -> 5 through alpha's updater: the build being left flashes and switches, before and after the reset."""
import subprocess

import pytest

from iqpilot.common.params import Params
from iqpilot.system.hardware.tici import agnos
from iqpilot.system.updated import updated

IQOS5_ENV = 'export AGNOS_VERSION="IQ.OS 5.0.12"\nexport AGNOS_COMPAT_VERSIONS="IQ.OS 5.0.11"\n'
IQOS4_ENV = 'export AGNOS_VERSION="IQ.OS 4.9.9.4"\nexport AGNOS_COMPAT_VERSIONS=""\n'
MANIFEST = b'[{"name": "abl"}]\n'
SIGNATURE = b'c2lnbmF0dXJl\n'


def git(repo, *args):
  return subprocess.check_output(['git', '-c', 'commit.gpgsign=false', '-C', str(repo), *args],
                                 text=True, stderr=subprocess.STDOUT).strip()


def commit_release(remote, launch_env, message):
  (remote / 'launch_env.sh').write_text(launch_env)
  manifest = remote / 'iqpilot/system/hardware/tici/iqos.json'
  manifest.parent.mkdir(parents=True, exist_ok=True)
  manifest.write_bytes(MANIFEST)
  manifest.with_name('iqos.json.sig').write_bytes(SIGNATURE)
  git(remote, 'add', '.')
  git(remote, 'commit', '-qm', message)


@pytest.fixture(params=["IQ.OS 4.9.9.4", "IQ.OS 4.9.9.5"])
def agnos_device(tmp_path, monkeypatch, request):
  remote = tmp_path / 'remote'
  repo = tmp_path / 'checkout'
  remote.mkdir()
  git(remote, 'init', '-q', '-b', 'alpha')
  git(remote, 'config', 'user.name', 'Updater Test')
  git(remote, 'config', 'user.email', 'updater@example.invalid')
  commit_release(remote, IQOS4_ENV, 'IQ.OS 4 release')
  git(tmp_path, 'clone', '-q', str(remote), str(repo))
  params = Params(str(tmp_path / 'params'))
  monkeypatch.setattr(updated, 'BASEDIR', str(repo))
  monkeypatch.setattr(updated, 'Params', lambda: params)
  monkeypatch.setattr(updated, 'AGNOS', True)
  monkeypatch.setattr(updated, 'configure_git_auth', lambda _cwd: None)
  monkeypatch.setattr(updated, 'set_offroad_alert', lambda *_args: None)
  monkeypatch.setattr(updated.HARDWARE, 'get_os_version', lambda: request.param)
  monkeypatch.setattr(agnos, 'get_target_slot_number', lambda: 0)
  monkeypatch.setattr(updated, 'handle_agnos_update', lambda: pytest.fail('the new tree must not flash'))

  events = []

  def flash(manifest, slot, _log):
    with open(manifest, 'rb') as f:
      manifest_bytes = f.read()
    with open(f'{manifest}.sig', 'rb') as f:
      signature_bytes = f.read()
    events.append(('flash', slot, manifest_bytes, signature_bytes, git(repo, 'rev-parse', 'HEAD')))

  monkeypatch.setattr(agnos, 'flash_iqos5_update', flash)
  monkeypatch.setattr(agnos, 'switch_slot', lambda slot, _log: events.append(('switch', slot, git(repo, 'rev-parse', 'HEAD'))))
  return remote, repo, updated.Updater(), events


def test_flashes_before_the_checkout_changes_and_switches_after(agnos_device):
  remote, repo, updater, events = agnos_device
  old_head = git(repo, 'rev-parse', 'HEAD')
  commit_release(remote, IQOS5_ENV, 'IQ.OS 5 release')
  new_head = git(remote, 'rev-parse', 'HEAD')

  updater.fetch_update()

  assert events == [('flash', 0, MANIFEST, SIGNATURE, old_head), ('switch', 0, new_head)]
  assert git(repo, 'rev-parse', 'HEAD') == new_head
  assert updater.os_flashed


def test_failed_switch_restores_the_previous_checkout(agnos_device, monkeypatch):
  remote, repo, updater, events = agnos_device
  old_head = git(repo, 'rev-parse', 'HEAD')
  commit_release(remote, IQOS5_ENV, 'IQ.OS 5 release')

  def refuse(_slot, _log):
    raise Exception('switching to slot 0 failed')

  monkeypatch.setattr(agnos, 'switch_slot', refuse)
  with pytest.raises(Exception, match='switching to slot 0 failed'):
    updater.fetch_update()

  assert git(repo, 'rev-parse', 'HEAD') == old_head
  assert git(repo, 'symbolic-ref', '--short', 'HEAD') == 'alpha'
  assert not updater.os_flashed


def test_failed_flash_leaves_the_checkout_alone(agnos_device, monkeypatch):
  remote, repo, updater, events = agnos_device
  old_head = git(repo, 'rev-parse', 'HEAD')
  commit_release(remote, IQOS5_ENV, 'IQ.OS 5 release')

  def fail(*_args):
    raise Exception('download failed')

  monkeypatch.setattr(agnos, 'flash_iqos5_update', fail)
  with pytest.raises(Exception, match='download failed'):
    updater.fetch_update()

  assert git(repo, 'rev-parse', 'HEAD') == old_head
  assert git(repo, 'for-each-ref', 'refs/iqos/checkpoint') == ''


@pytest.mark.parametrize('current', ['IQ.OS 5.0.11', 'IQ.OS 5.0.12'])
def test_iqos5_device_is_left_to_the_existing_path(agnos_device, monkeypatch, current):
  remote, repo, updater, events = agnos_device
  commit_release(remote, IQOS5_ENV, 'IQ.OS 5 release')
  monkeypatch.setattr(updated.HARDWARE, 'get_os_version', lambda: current)
  monkeypatch.setattr(updated, 'handle_agnos_update', lambda: False)

  updater.fetch_update()

  assert events == []
  assert git(repo, 'rev-parse', 'HEAD') == git(remote, 'rev-parse', 'HEAD')


def test_iqos4_release_keeps_the_agnos_path(agnos_device, monkeypatch):
  remote, repo, updater, events = agnos_device
  commit_release(remote, IQOS4_ENV.replace('4.9.9.4', '4.9.9.5'), 'IQ.OS 4 update')
  calls = []
  monkeypatch.setattr(updated, 'handle_agnos_update', lambda: calls.append(git(repo, 'rev-parse', 'HEAD')) or True)

  updater.fetch_update()

  assert events == []
  assert calls == [git(remote, 'rev-parse', 'HEAD')]
  assert updater.os_flashed


@pytest.mark.parametrize('current,expected,compat,allowed', [
  ('IQ.OS 4.9.9.6', 'IQ.OS 4.9.9.6', [], True),
  ('4.9.9.7', 'IQ.OS 4.9.9.7', [], True),
  ('IQ.OS 4.9.9.7', '4.9.9.7', [], True),
  ('4.9.9.7-dev', '4.9.9.7', [], True),
  ('5.0.11', 'IQ.OS 5.0.12', ['IQ.OS 5.0.11'], True),
  ('IQ.OS 4.9.9.6', '4.9.9.7', [], False),
  ('16.1', 'IQ.OS 4.9.9.7', [], False),
])
def test_version_allowed_ignores_iqos_prefix(current, expected, compat, allowed):
  assert updated.agnos_version_allowed(current, expected, compat) == allowed


@pytest.mark.parametrize('version,iqos5', [('IQ.OS 5.0.12', True), ('5.0.12', True), ('IQ.OS 4.9.9.7', False), ('4.9.9.7', False), ('15.1', False)])
def test_iqos5_detection_accepts_bare_versions(version, iqos5):
  assert updated.is_iqos5(version) == iqos5
