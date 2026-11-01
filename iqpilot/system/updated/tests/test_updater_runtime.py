"""Exercise the PR23 updater lifecycle with native Params and actual Git."""
import json
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from iqpilot.common.params import Params, ParamKeyFlag
from iqpilot.system.updated import updated


def git(repo, *args):
  try:
    return subprocess.check_output(['git', '-c', 'commit.gpgsign=false', '-C', str(repo), *args],
                                   text=True, stderr=subprocess.STDOUT).strip()
  except subprocess.CalledProcessError as e:
    raise AssertionError(e.output) from e


@pytest.fixture
def runtime_checkout(tmp_path, monkeypatch):
  repo = tmp_path / 'checkout'
  repo.mkdir()
  git(repo, 'init', '-q', '-b', 'alpha')
  git(repo, 'config', 'user.name', 'Updater Test')
  git(repo, 'config', 'user.email', 'updater@example.invalid')
  version = repo / 'iqpilot/common/version.h'
  version.parent.mkdir(parents=True)
  version.write_text('#define COMMA_VERSION "test"\n')
  git(repo, 'add', '.')
  git(repo, 'commit', '-qm', 'running release')
  params = Params(str(tmp_path / 'params'))
  assert Params.__module__ == 'iqpilot.common.params_pyx', 'requires the native Params registry and JSON conversion'
  monkeypatch.setattr(updated, 'BASEDIR', str(repo))
  monkeypatch.setattr(updated, 'Params', lambda: params)
  monkeypatch.setattr(updated, 'LOCK_FILE', str(tmp_path / 'updater.lock'))
  monkeypatch.setattr(updated, 'STAGING_ROOT', str(tmp_path / 'staging'))
  monkeypatch.setattr(updated, 'OVERLAY_INIT', repo / '.overlay_init')
  monkeypatch.setattr(updated.psutil, 'Process', lambda: SimpleNamespace(ionice=lambda *_a, **_kw: None))
  return repo, params


class FirstIdle(BaseException):
  pass


def start_enabled(monkeypatch):
  # Run main through its real publication and first set_params, stopping at the
  # initial sleep before any network, environment prep, or reboot can occur.
  monkeypatch.setattr(updated.WaitTimeHelper, '_poll_request_param', lambda _self: None)
  monkeypatch.setattr(updated.WaitTimeHelper, 'sleep', lambda *_a: (_ for _ in ()).throw(FirstIdle()))
  with pytest.raises(FirstIdle):
    updated.main()


def read_runtime(params):
  raw = Path(params.get_param_path('UpdaterRuntimeState')).read_text()
  runtime = json.loads(raw)
  assert isinstance(runtime, dict)  # Catch JSON double encoding with native Params.
  assert params.get('UpdaterRuntimeState') == runtime
  assert set(runtime) == {'disabled', 'commit', 'branch', 'description'}
  return runtime


def test_enabled_main_publishes_runtime(runtime_checkout, monkeypatch):
  repo, params = runtime_checkout
  start_enabled(monkeypatch)
  runtime = read_runtime(params)
  assert runtime['disabled'] is False
  assert runtime['commit'] == git(repo, 'rev-parse', 'HEAD')
  assert runtime['branch'] == 'alpha'
  assert runtime['description'] == updated.Updater().running_description
  assert params.get('UpdaterState') == 'idle'
  assert params.get('UpdaterTargetBranch') == 'alpha'


def test_disabled_main_publishes_runtime(runtime_checkout, monkeypatch):
  repo, params = runtime_checkout
  params.put_bool('DisableUpdates', True)
  monkeypatch.setattr(updated, 'cleanup_stale_overlay', lambda: pytest.fail('disabled updater started work'))
  updated.main()
  runtime = read_runtime(params)
  assert runtime['disabled'] is True
  assert runtime['commit'] == git(repo, 'rev-parse', 'HEAD')
  assert runtime['branch'] == 'alpha'
  assert params.get('UpdaterState') == 'idle'
  assert params.get('UpdaterFetchAvailable') is False


def test_runtime_survives_updater_restart_until_manager_clear(runtime_checkout, monkeypatch):
  repo, params = runtime_checkout
  git(repo, 'commit', '--allow-empty', '-qm', 'running build')
  start_enabled(monkeypatch)
  running = read_runtime(params)
  assert running['commit'] == git(repo, 'rev-parse', 'HEAD')

  git(repo, 'checkout', '-qb', 'next-release')
  git(repo, 'commit', '--allow-empty', '-qm', 'next release')
  start_enabled(monkeypatch)
  assert read_runtime(params) == running
  assert updated.Updater().update_ready

  params.put_bool('DisableUpdates', True)
  updated.main()
  assert read_runtime(params) == dict(running, disabled=True)

  # manager_init performs this native lifecycle clear before starting processes.
  params.put('DongleId', 'stable-device-identity')
  params.clear_all(ParamKeyFlag.CLEAR_ON_MANAGER_START)
  assert params.get('UpdaterRuntimeState') is None
  assert params.get('DongleId') == 'stable-device-identity'
  params.put_bool('DisableUpdates', False)
  start_enabled(monkeypatch)
  rebooted = read_runtime(params)
  assert rebooted['commit'] == git(repo, 'rev-parse', 'HEAD')
  assert rebooted['commit'] != running['commit']
  assert rebooted['branch'] == 'next-release'
  assert not updated.Updater().update_ready


@pytest.fixture
def run_cycles(runtime_checkout, monkeypatch):
  repo, params = runtime_checkout
  booted = updated.Updater()
  updated.publish_runtime(params, booted, False)
  git(repo, 'commit', '--allow-empty', '-qm', 'downloaded release')
  updater = updated.Updater()
  updater.check_for_update = Mock()
  updater.fetch_update = Mock()
  monkeypatch.setattr(updated, 'Updater', lambda: updater)
  monkeypatch.setattr(updated, 'system_time_valid', lambda: True)
  monkeypatch.setattr(updated, 'has_active_ssh_session', lambda: False)

  def run(*, onroads=(False,), mode='download_and_install', prepare_results=(True,), ssh=False, clock=None, os_flashed=False):
    params.put('UpdaterInstallMode', mode)
    updater.os_flashed = os_flashed
    helper = SimpleNamespace(ready_event=threading.Event(), user_request=updated.UserRequest.FETCH)
    sleeps, reboots = [], []
    results = iter(prepare_results)
    preparations = []

    def prepare():
      preparations.append(params.get('UpdaterState'))
      return next(results)

    def sleep(seconds):
      sleeps.append(seconds)
      reboots.append(params.get_bool('DoReboot'))
      if len(sleeps) > len(onroads):
        raise FirstIdle()
      onroad = onroads[len(sleeps) - 1]
      if onroad is None:
        params.remove('IsOnroad')
      else:
        params.put_bool('IsOnroad', onroad)
      if len(sleeps) == 1:
        helper.user_request = updated.UserRequest.FETCH

    helper.sleep = sleep
    monkeypatch.setattr(updated, 'WaitTimeHelper', lambda _params: helper)
    monkeypatch.setattr(updated, 'prepare_environment', prepare)
    monkeypatch.setattr(updated, 'has_active_ssh_session', lambda: ssh)
    if clock is not None:
      monkeypatch.setattr(updated.time, 'monotonic', clock)
    with pytest.raises(FirstIdle):
      updated.main()
    return params, updater, sleeps, reboots, preparations
  return run


@pytest.mark.parametrize('onroad', [True, None])
def test_manual_fetch_requires_confirmed_offroad(run_cycles, onroad):
  params, updater, sleeps, reboots, preparations = run_cycles(onroads=(onroad,))
  updater.fetch_update.assert_not_called()
  assert not params.get_bool('DoReboot')
  assert preparations == []
  assert reboots == [False, False]
  assert sleeps == [60, 5400]


@pytest.mark.parametrize('mode, reboot', [('download_only', False), ('download_and_install', True)])
def test_auto_install_prepares_while_busy_and_respects_install_mode(run_cycles, mode, reboot):
  params, updater, sleeps, reboots, preparations = run_cycles(mode=mode)
  updater.fetch_update.assert_called_once()
  assert preparations == ['preparing...']
  assert params.get_bool('DoReboot') is reboot
  assert params.get('UpdaterState') == 'idle'
  assert sleeps == [60, 5400]
  assert reboots == [False, reboot]


def test_failed_preparation_holds_reboot_and_retries_on_short_cadence(run_cycles):
  params, updater, sleeps, reboots, preparations = run_cycles(onroads=(False, False), prepare_results=(False, True))
  updater.fetch_update.assert_called_once()
  assert preparations == ['preparing...', 'preparing...']
  assert params.get_bool('DoReboot')
  assert sleeps == [60, 300, 5400]
  assert reboots == [False, False, True]


@pytest.mark.parametrize('onroad', [True, None])
@pytest.mark.parametrize('resume', [False, True])
def test_auto_install_waits_for_offroad_after_preparation(runtime_checkout, run_cycles, onroad, resume):
  _repo, params = runtime_checkout

  def preparation():
    if onroad is None:
      params.remove('IsOnroad')
    else:
      params.put_bool('IsOnroad', onroad)
    yield True

  params, updater, _sleeps, reboots, preparations = run_cycles(onroads=(False, False) if resume else (False,), prepare_results=preparation())
  updater.fetch_update.assert_called_once()
  assert preparations == ['preparing...']
  assert reboots == ([False, False, True] if resume else [False, False])
  assert params.get_bool('DoReboot') is resume
  assert params.get('UpdaterState') == 'idle'


def test_ssh_quiet_window_still_defers_until_cap(run_cycles):
  ticks = iter([0, 0, 7199, 7200])
  params, updater, sleeps, reboots, preparations = run_cycles(onroads=(False, False, False), ssh=True, clock=lambda: next(ticks))
  updater.fetch_update.assert_called_once()
  assert preparations == ['preparing...']
  assert params.get_bool('DoReboot')
  assert sleeps == [60, 300, 300, 5400]
  assert reboots == [False, False, False, True]


@pytest.mark.parametrize('command, expected', [('check', 1), ('fetch', 2), ('invalid', 0)])
def test_manual_request_wakes_existing_path(runtime_checkout, command, expected):
  _repo, params = runtime_checkout
  helper = updated.WaitTimeHelper.__new__(updated.WaitTimeHelper)
  helper._params = params
  helper.ready_event = threading.Event()
  helper.user_request = updated.UserRequest.NONE
  params.put('UpdaterManualRequest', command)
  helper._read_request_param()
  assert params.get('UpdaterManualRequest') is None
  assert helper.user_request == expected
  assert helper.ready_event.is_set() == bool(expected)


def test_manual_request_retries_after_transient_remove_error(runtime_checkout):
  _repo, params = runtime_checkout
  helper = updated.WaitTimeHelper.__new__(updated.WaitTimeHelper)
  helper.ready_event = threading.Event()
  helper.user_request = updated.UserRequest.NONE
  params.put('UpdaterManualRequest', 'fetch')
  helper._params = SimpleNamespace(get=params.get, remove=Mock(side_effect=OSError('transient failure')))
  helper._read_request_param()
  assert not helper.ready_event.is_set()
  assert params.get('UpdaterManualRequest') == 'fetch'
  helper._params = params
  helper._read_request_param()
  assert helper.user_request == updated.UserRequest.FETCH
  assert params.get('UpdaterManualRequest') is None


@pytest.mark.parametrize('onroad', [True, None])
def test_pending_update_resumes_when_offroad_is_confirmed(run_cycles, onroad):
  params, updater, sleeps, reboots, preparations = run_cycles(onroads=(onroad, False))
  updater.fetch_update.assert_not_called()
  assert preparations == ['preparing...']
  assert reboots == [False, False, True]
  assert sleeps == [60, 5400, 5400]
  assert params.get_bool('DoReboot')


@pytest.mark.parametrize('onroad', [True, None, False])
def test_flashed_os_still_requires_confirmed_offroad_before_reboot(run_cycles, onroad):
  params, _updater, _sleeps, reboots, preparations = run_cycles(onroads=(onroad,), os_flashed=True)
  assert preparations == []
  assert params.get_bool('DoReboot') is (onroad is False)
  assert reboots == [False, onroad is False]
