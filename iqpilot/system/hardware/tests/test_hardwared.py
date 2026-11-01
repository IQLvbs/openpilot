import re
import queue
import threading
from pathlib import Path

import pytest

from iqpilot.common.params import Params
from iqpilot.common.realtime import DT_HW
from iqpilot.system.hardware import hardwared
from iqpilot.system.hardware.hardwared import (
  CAN_STARTUP_RECOVERY_COOLDOWN,
  CAN_STARTUP_RECOVERY_DELAY,
  CAN_STARTUP_RECOVERY_MAX_ATTEMPTS,
  CanStartupRecovery,
)


@pytest.fixture
def run_network_samples(tmp_path, monkeypatch, mocker):
  params = mocker.Mock(wraps=Params(str(tmp_path)))
  monkeypatch.setattr(hardwared, "Params", lambda: params)
  monkeypatch.setattr(hardwared, "AGNOS", False)
  hardware = mocker.Mock()
  hardware.get_modem_temperatures.return_value = []
  hardware.get_modem_data_usage.return_value = (0, 0)
  hardware.get_modem_version.return_value = None
  monkeypatch.setattr(hardwared, "HARDWARE", hardware)
  for name in ("get_usb_state", "get_link_error_count", "usb3_lane"):
    monkeypatch.setattr(hardwared, name, mocker.Mock())
  monkeypatch.setattr(hardwared.cloudlog, "exception", mocker.Mock())

  def run(samples):
    end_event = threading.Event()
    hw_queue = queue.Queue()
    hardware.get_network_metered.side_effect = samples
    ticks = 0

    def sleep(dt):
      nonlocal ticks
      assert dt == DT_HW
      ticks += 1
      if ticks >= len(samples) * int(10. / DT_HW):
        end_event.set()

    monkeypatch.setattr(hardwared.time, "sleep", sleep)
    hardwared.hw_state_thread(end_event, hw_queue)
    assert [hw_queue.get_nowait().network_metered for _ in samples] == samples
    assert hw_queue.empty()

  return params, run


@pytest.mark.parametrize("initial,samples,writes", [
  (None, [False] * 100, [False]),
  (None, [True] * 100, [True]),
  (False, [False, False], []),
  (True, [True, True], []),
  (True, [False, False], [False]),
  (False, [True, True], [True]),
  (False, [False, True, True, False, False], [True, False]),
])
def test_network_metered_persists_only_missing_or_changed_values(run_network_samples, monkeypatch, mocker, initial, samples, writes):
  params, run = run_network_samples
  if initial is not None:
    params.put_bool("NetworkMetered", initial)
  put = mocker.Mock(wraps=params.put_bool)
  monkeypatch.setattr(params, "put_bool", put)
  run(samples)
  assert put.call_args_list == [(("NetworkMetered", value),) for value in writes]
  assert params.get("NetworkMetered") == samples[-1]


@pytest.mark.parametrize("raises", [False, True])
def test_network_metered_retries_failed_persistence(run_network_samples, monkeypatch, mocker, raises):
  params, run = run_network_samples
  params.put_bool("NetworkMetered", False)
  original_put = params.put_bool

  def fail_first(key, value):
    if put.call_count == 1:
      if raises:
        raise OSError("write failed")
      return
    return original_put(key, value)

  put = mocker.Mock(side_effect=fail_first)
  monkeypatch.setattr(params, "put_bool", put)
  run([True, True, True])
  assert put.call_count == 2
  assert params.get("NetworkMetered") is True


def recovery_update(recovery: CanStartupRecovery, now: float, **kwargs) -> bool:
  defaults = {
    "ignition": True,
    "started": True,
    "engaged": False,
    "car_state_alive": True,
    "can_timeout": True,
    "v_ego": 0.,
  }
  return recovery.update(now, **(defaults | kwargs))


def test_can_startup_recovery_requires_persistent_timeout():
  recovery = CanStartupRecovery()
  assert not recovery_update(recovery, 10.)
  assert not recovery_update(recovery, 10. + CAN_STARTUP_RECOVERY_DELAY - 0.1)
  assert recovery_update(recovery, 10. + CAN_STARTUP_RECOVERY_DELAY)


def test_can_startup_recovery_only_when_safe():
  for unsafe_state in (
    {"started": False},
    {"engaged": True},
    {"car_state_alive": False},
    {"can_timeout": False},
    {"v_ego": 0.2},
  ):
    recovery = CanStartupRecovery()
    assert not recovery_update(recovery, 10., **unsafe_state)
    assert not recovery_update(recovery, 10. + CAN_STARTUP_RECOVERY_DELAY, **unsafe_state)


def test_can_startup_recovery_is_bounded_and_resets_next_ignition():
  recovery = CanStartupRecovery()
  now = 10.
  for _ in range(CAN_STARTUP_RECOVERY_MAX_ATTEMPTS):
    assert not recovery_update(recovery, now)
    now += CAN_STARTUP_RECOVERY_DELAY
    assert recovery_update(recovery, now)
    now += CAN_STARTUP_RECOVERY_COOLDOWN

  assert not recovery_update(recovery, now)
  assert not recovery_update(recovery, now + CAN_STARTUP_RECOVERY_DELAY)

  assert not recovery_update(recovery, now + 10., ignition=False)
  assert not recovery_update(recovery, now + 11.)
  assert recovery_update(recovery, now + 11. + CAN_STARTUP_RECOVERY_DELAY)


def test_hardwared_only_touches_declared_params():
  hardware_dir = Path(__file__).resolve().parents[1]
  declared = set(re.findall(r'^\s*\{"([A-Za-z0-9_]+)",', (hardware_dir.parents[1] / "common" / "params_keys.h").read_text(), re.M))
  used = set(re.findall(r'params\.[a-z_]+\("([A-Za-z0-9_]+)"', (hardware_dir / "hardwared.py").read_text()))
  assert used
  assert used <= declared, sorted(used - declared)
  assert "ForceOnroadUntil" not in declared | used
