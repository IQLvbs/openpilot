"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""
import re
import time
from pathlib import Path

import pytest

from iqpilot.selfdrive.iqmodeld import egpu_helpers as eh
from iqpilot.selfdrive.iqmodeld import egpu_sched as es

IQMODELD = Path(__file__).resolve().parents[1]


class SchedParam:
  def __init__(self, priority):
    self.sched_priority = priority


def _linux_sched(monkeypatch, affinity, scheduler):
  monkeypatch.setattr(es, "_can_schedule", lambda: True)
  monkeypatch.setattr(es.os, "sched_param", SchedParam, raising=False)
  monkeypatch.setattr(es.os, "SCHED_FIFO", 1, raising=False)
  monkeypatch.setattr(es.os, "SCHED_OTHER", 0, raising=False)
  monkeypatch.setattr(es.os, "sched_setaffinity", affinity, raising=False)
  monkeypatch.setattr(es.os, "sched_setscheduler", scheduler, raising=False)


def _fifo_priority(path: Path) -> int:
  m = re.search(r"config_realtime_process\((\d+|\[[^\]]*\]),\s*(\d+)\)", path.read_text())
  assert m, f"{path.name} no longer sets a realtime priority"
  return int(m.group(2))


def test_setup_never_runs_on_the_isolated_cores():
  assert not es.SETUP_CORES & {es.CAMERA_CORE, es.MODEL_CORE}


def test_the_frame_loop_never_shares_camerads_core():
  # iqegpumodeld pinned {4, 5, 6} at nice -10, stuck on isolated core 6, and took camerad from ~9 ms to 20-66 ms
  assert es.MODEL_CORE != es.CAMERA_CORE
  src = (IQMODELD / "iqegpumodeld.py").read_text()
  assert "sched_setaffinity" not in src, "core placement goes through egpu_sched"
  assert not re.search(r"os\.nice\(\s*-", src), "priority is raised only through egpu_sched"


def test_the_plan_outranks_the_fallback_on_the_shared_core():
  small = _fifo_priority(IQMODELD / "daemon.py")
  dmon = _fifo_priority(IQMODELD.parent / "dmonitoringmodeld" / "dmonitoringmodeld.py")
  assert es.EGPU_RT_PRIORITY > small > dmon


def test_dmonitoring_is_lifted_above_the_worker_from_the_priority_it_really_runs_at():
  # route ed2717e2c1912f87/0000000a: under the worker dmonitoring's frame time went 46 -> 101 ms and it ran at 8 Hz
  assert es.DMONITORING_PRIORITY == _fifo_priority(IQMODELD.parent / "dmonitoringmodeld" / "dmonitoringmodeld.py")
  assert es.DMONITORING_LIFTED > es.EGPU_RT_PRIORITY


def _fake_proc(tmp_path, procs):
  for pid, (cmdline, tids) in procs.items():
    (tmp_path / str(pid) / "task").mkdir(parents=True)
    (tmp_path / str(pid) / "cmdline").write_bytes(cmdline)
    for tid in tids:
      (tmp_path / str(pid) / "task" / str(tid)).mkdir()
  (tmp_path / "self").mkdir()
  return str(tmp_path)


@pytest.fixture
def dm_sched(monkeypatch, tmp_path):
  proc = _fake_proc(tmp_path, {
    100: (b"iqpilot.selfdrive.dmonitoringmodeld.dmonitoringmodeld\x00", [100, 101, 102]),
    200: (b"iqpilot.selfdrive.monitoring.dmonitoringd\x00", [200]),
    300: (b"iqpilot.selfdrive.iqmodeld.daemon\x00", [300]),
  })
  real_tids = es._dmonitoring_tids
  monkeypatch.setattr(es, "_dmonitoring_tids", lambda: real_tids(proc))
  policy = {100: (1, 5), 101: (1, 5), 102: (0, 0), 200: (1, 5), 300: (1, 54)}
  monkeypatch.setattr(es, "_can_schedule", lambda: True)
  monkeypatch.setattr(es.os, "SCHED_FIFO", 1, raising=False)
  monkeypatch.setattr(es.os, "sched_param", SchedParam, raising=False)
  monkeypatch.setattr(es.os, "sched_getscheduler", lambda tid: policy[tid][0], raising=False)
  monkeypatch.setattr(es.os, "sched_getparam", lambda tid: SchedParam(policy[tid][1]), raising=False)

  def setsched(tid, pol, param):
    policy[tid] = (pol, param.sched_priority)
  monkeypatch.setattr(es.os, "sched_setscheduler", setsched, raising=False)
  return policy


def test_only_dmonitorings_realtime_threads_are_lifted_and_restored(dm_sched):
  assert es.lift_dmonitoring() == 2
  assert dm_sched[100] == dm_sched[101] == (1, es.DMONITORING_LIFTED)
  assert dm_sched[102] == (0, 0) and dm_sched[200] == (1, 5) and dm_sched[300] == (1, 54)
  assert es.lift_dmonitoring() == 0
  assert es.restore_dmonitoring() == 2
  assert dm_sched[100] == dm_sched[101] == (1, 5)


def test_a_restarted_dmonitoring_is_lifted_again_on_the_next_healthy_window():
  wall, cpu, lifts = Clock(), Clock(), []
  guard = es.CpuBudgetGuard(budget=0.5, window_s=10.0, clock=wall, cpu_clock=cpu, demote=lambda: None,
                            lift=lambda: lifts.append(1))
  wall.t, cpu.t = 10.0, 4.0
  guard.tick()
  wall.t, cpu.t = 20.0, 8.0
  guard.tick()
  assert lifts == [1, 1]
  wall.t, cpu.t = 30.0, 17.0
  guard.tick()
  assert lifts == [1, 1] and not guard.realtime


def test_dropping_realtime_hands_dmonitoring_its_own_priority_back(dm_sched, monkeypatch):
  monkeypatch.setattr(es.os, "SCHED_OTHER", 0, raising=False)
  es.lift_dmonitoring()
  es.leave_realtime()
  assert dm_sched[100] == dm_sched[101] == (1, 5)


def test_enter_realtime_pins_only_the_model_core(monkeypatch):
  calls = []
  _linux_sched(monkeypatch, lambda pid, cores: calls.append(("affinity", pid, set(cores))),
               lambda pid, pol, param: calls.append(("sched", pid, param.sched_priority)))
  monkeypatch.setattr(es, "_dmonitoring_tids", list)
  assert es.enter_realtime() is True
  assert calls == [("affinity", 0, {es.MODEL_CORE}), ("sched", 0, es.EGPU_RT_PRIORITY)]


def test_a_failed_realtime_request_is_reported_not_fatal(monkeypatch):
  def deny(*a):
    raise PermissionError("no CAP_SYS_NICE")
  _linux_sched(monkeypatch, lambda *a: None, deny)
  assert es.enter_realtime() is False


class Clock:
  def __init__(self):
    self.t = 0.0

  def __call__(self):
    return self.t


def test_guard_drops_realtime_once_the_worker_outgrows_its_share():
  wall, cpu, demoted = Clock(), Clock(), []
  guard = es.CpuBudgetGuard(budget=0.5, window_s=10.0, clock=wall, cpu_clock=cpu, demote=lambda: demoted.append(1))
  wall.t, cpu.t = 10.0, 4.0
  guard.tick()
  assert guard.realtime and not demoted
  wall.t, cpu.t = 20.0, 11.5
  guard.tick()
  assert not guard.realtime and demoted == [1]
  wall.t, cpu.t = 30.0, 21.5
  guard.tick()
  assert demoted == [1]


def test_guard_waits_for_a_full_window():
  wall, cpu, demoted = Clock(), Clock(), []
  guard = es.CpuBudgetGuard(budget=0.5, window_s=10.0, clock=wall, cpu_clock=cpu, demote=lambda: demoted.append(1))
  wall.t, cpu.t = 5.0, 5.0
  guard.tick()
  assert guard.realtime and not demoted


class FakeOwner:
  def __init__(self, usb):
    self.usb = usb

  def is_usb(self):
    return self.usb


class FakeSignal:
  def __init__(self, done_after_s, owner, target):
    self.owner, self.target, self.reads = owner, target, 0
    self._done_at = time.perf_counter() + done_after_s

  @property
  def value(self):
    self.reads += 1
    return self.target if time.perf_counter() >= self._done_at else self.target - 1


@pytest.fixture
def patched(monkeypatch):
  from tinygrad.runtime.ops_amd import AMDSignal
  monkeypatch.setattr(AMDSignal, "wait", AMDSignal.wait)
  eh.patch_usb_signal_wait()
  return AMDSignal


def test_a_usb_wait_sleeps_between_polls_instead_of_spinning(patched, monkeypatch):
  # every poll of a dock signal is a USB read; the stock loop only sleeps after 200 ms, so it spun a whole core
  monkeypatch.setattr(eh, "USB_SIGNAL_POLL_S", 0.0005)
  sig = FakeSignal(0.03, FakeOwner(usb=True), target=7)
  st = time.perf_counter()
  patched.wait(sig, 7)
  took = time.perf_counter() - st
  assert 0.03 <= took < 0.045
  assert sig.reads < 0.03 / 0.0005 * 1.5


def test_a_finished_signal_returns_without_sleeping(patched):
  sig = FakeSignal(0.0, FakeOwner(usb=True), target=3)
  st = time.perf_counter()
  patched.wait(sig, 3)
  assert time.perf_counter() - st < 0.001 and sig.reads == 1


def test_a_stuck_usb_signal_still_times_out(patched):
  sig = FakeSignal(60.0, FakeOwner(usb=True), target=9)
  with pytest.raises(RuntimeError, match="Wait timeout"):
    patched.wait(sig, 9, timeout=20)


def test_non_usb_devices_keep_the_stock_wait(monkeypatch):
  from tinygrad.runtime.ops_amd import AMDSignal
  seen = []
  monkeypatch.setattr(AMDSignal, "wait", lambda self, value, timeout=None: seen.append((value, timeout)))
  eh.patch_usb_signal_wait()
  AMDSignal.wait(FakeSignal(0.0, FakeOwner(usb=False), target=1), 1, 50)
  assert seen == [(1, 50)]


def test_patching_twice_does_not_wrap_twice(patched):
  first = patched.wait
  eh.patch_usb_signal_wait()
  assert patched.wait is first


def test_the_default_budget_allows_real_inference_work_and_still_catches_a_spin():
  # route ed2717e2c1912f87/5: cinque used 55% of core 7, the old 50% budget demoted it, the loop fell to 13 Hz
  assert 0.6 < es.CPU_BUDGET < 0.95
