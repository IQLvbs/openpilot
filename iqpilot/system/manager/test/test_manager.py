import os
import pytest
import signal
import time

from iqpilot.cereal import car
from iqpilot.common.params import Params
import iqpilot.system.manager.manager as manager
from iqpilot.system.manager.process import BundleProcess, NativeProcess, ensure_running
from iqpilot.system.manager.process_config import managed_processes, procs
from iqpilot.system.hardware import HARDWARE

os.environ['FAKEUPLOAD'] = "1"

MAX_STARTUP_TIME = 3
BLACKLIST_PROCS = ['manage_hephaestusd', 'pandad', 'pigeond']


class TestManager:
  def setup_method(self):
    HARDWARE.set_power_save(False)

    # ensure clean CarParams
    params = Params()
    params.clear_all()

  def teardown_method(self):
    manager.manager_cleanup()

  @pytest.mark.linux
  def test_manager_prepare(self):
    os.environ['PREPAREONLY'] = '1'
    manager.main()

  def test_duplicate_procs(self):
    assert len(procs) == len(managed_processes), "Duplicate process names"

  def test_models_manager_uses_private_bundle(self):
    proc = managed_processes["models_manager"]

    assert isinstance(proc, BundleProcess)
    assert proc.bundle == "iqpilot_model_selector_private"
    assert proc.entry == "iqpilot_private.models.manager"

  def test_modeld_watchdog_restarts_stalled_process(self, mocker):
    proc = mocker.Mock()
    proc.proc.is_alive.return_value = True

    deadline = manager.update_modeld_watchdog(None, True, False, proc, 10.0)
    assert deadline == 10.0 + manager.MODELD_FIRST_PUBLISH_TIMEOUT
    assert manager.update_modeld_watchdog(deadline, True, False, proc, deadline - 0.1) == deadline
    proc.restart.assert_not_called()

    assert manager.update_modeld_watchdog(deadline, True, False, proc, deadline) is None
    proc.restart.assert_called_once_with()
    assert manager.update_modeld_watchdog(None, True, False, proc, deadline + 1) == deadline + 1 + manager.MODELD_FIRST_PUBLISH_TIMEOUT

  def test_modeld_watchdog_slow_first_load_is_not_a_stall(self, mocker):
    proc = mocker.Mock()
    proc.proc.is_alive.return_value = True

    deadline = manager.update_modeld_watchdog(None, True, False, proc, 0.0)
    assert manager.update_modeld_watchdog(deadline, True, False, proc, manager.MODELD_WATCHDOG_TIMEOUT + 1) == deadline
    proc.restart.assert_not_called()
    deadline = manager.update_modeld_watchdog(deadline, True, True, proc, 90.0)
    assert deadline == 90.0 + manager.MODELD_WATCHDOG_TIMEOUT
    manager.update_modeld_watchdog(deadline, True, False, proc, deadline)
    proc.restart.assert_called_once_with()

  def test_modeld_watchdog_tracks_output_and_resets(self, mocker):
    proc = mocker.Mock()
    proc.proc.is_alive.return_value = True

    deadline = manager.update_modeld_watchdog(20.0, True, True, proc, 15.0)
    assert deadline == 15.0 + manager.MODELD_WATCHDOG_TIMEOUT
    assert manager.update_modeld_watchdog(deadline, False, False, proc, 16.0) is None

    proc.proc.is_alive.return_value = False
    assert manager.update_modeld_watchdog(deadline, True, False, proc, 16.0) is None
    proc.restart.assert_not_called()

  def test_bundle_process_keeps_restart_if_crash(self):
    proc = BundleProcess("test", "bundle", "entry", lambda *_: True, restart_if_crash=True)
    assert proc.restart_if_crash is True
    assert BundleProcess("test", "bundle", "entry", lambda *_: True).restart_if_crash is False

  def test_bundle_process_stops_with_sigterm(self, mocker):
    proc = BundleProcess("test", "bundle", "entry", lambda *_: True)
    proc.proc = mocker.Mock(exitcode=None, pid=123)
    proc.proc.is_alive.return_value = True
    signal_mock = mocker.patch.object(proc, "signal")

    proc.stop(block=False)

    signal_mock.assert_called_once_with(signal.SIGTERM)

  def test_native_process_stop_timeout(self, mocker):
    proc = NativeProcess("test", ".", ["true"], lambda *_: True)
    native_process = mocker.Mock(exitcode=None, pid=123)
    proc.proc = native_process
    signal_mock = mocker.patch.object(proc, "signal")
    join_mock = mocker.patch("iqpilot.system.manager.process.join_process", side_effect=lambda process, _: setattr(process, "exitcode", 0))

    assert proc.stop(timeout=30) == 0
    signal_mock.assert_called_once_with(signal.SIGINT)
    join_mock.assert_called_once_with(native_process, 30)

  def test_blacklisted_procs(self):
    # TODO: ensure there are blacklisted procs until we have a dedicated test
    assert len(BLACKLIST_PROCS), "No blacklisted procs to test not_run"

  @pytest.mark.linux
  def test_set_params_with_default_value(self):
    params = Params()
    params.clear_all()

    os.environ['PREPAREONLY'] = '1'
    manager.main()
    for k in params.all_keys():
      default_value = params.get_default_value(k)
      if default_value is not None:
        assert params.get(k) == default_value
    assert params.get("OpenpilotEnabledToggle")
    assert params.get("RouteCount") == 0

  @pytest.mark.tici
  def test_clean_exit(self, subtests):
    """
      Ensure all processes exit cleanly when stopped.
    """
    HARDWARE.set_power_save(False)
    manager.manager_init()

    CP = car.CarParams.new_message()
    procs = ensure_running(managed_processes.values(), True, Params(), CP, not_run=BLACKLIST_PROCS)

    time.sleep(10)

    for p in procs:
      with subtests.test(proc=p.name):
        state = p.get_process_state_msg()
        assert state.running, f"{p.name} not running"
        exit_code = p.stop(retry=False)

        assert p.name not in BLACKLIST_PROCS, f"{p.name} was started"

        assert exit_code is not None, f"{p.name} failed to exit"

        # TODO: interrupted blocking read exits with 1 in cereal. use a more unique return code
        exit_codes = [0, 1]
        if p.sigkill:
          exit_codes = [-signal.SIGKILL]
        assert exit_code in exit_codes, f"{p.name} died with {exit_code}"
