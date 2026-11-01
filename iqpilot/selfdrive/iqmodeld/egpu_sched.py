"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""
from __future__ import annotations

import atexit
import os
import sys
import time

from iqpilot.common.swaglog import cloudlog
from iqpilot.system.hardware import PC

# isolcpus=6,7: nothing is ever balanced off them; a model thread on camerad's core 6 starves the camera ISP
CAMERA_CORE = 6
MODEL_CORE = 7
SETUP_CORES = frozenset({0, 1, 2, 3, 4, 5})
# above the fallback small model (FIFO 54 on the same core) so it never delays the plan
EGPU_RT_PRIORITY = 55
DMONITORING_PRIORITY = 5
# dmonitoring needs ~6 ms of core 7 per frame; below this worker it fell from 20 to 8 Hz (route ed2717e2c1912f87/0000000a)
DMONITORING_LIFTED = EGPU_RT_PRIORITY + 1
DMONITORING_CMDLINE = b"dmonitoringmodeld"
# the small model (~33%) and dmonitoring (~13%) share core 7; past this the worker would starve them
# a runaway USB signal spin pins the core near 100%; cinque legitimately uses ~55-60% of it, and a 50% budget demoted it to 13 Hz
CPU_BUDGET = 0.85
BUDGET_WINDOW_S = 10.0


def _can_schedule() -> bool:
  return sys.platform == "linux" and not PC


def enter_setup() -> None:
  if not _can_schedule():
    return
  try:
    os.sched_setaffinity(0, SETUP_CORES)
  except OSError as e:
    cloudlog.warning(f"iqegpumodeld setup affinity failed ({e})")


def _dmonitoring_tids(proc: str = "/proc") -> list[int]:
  tids = []
  try:
    pids = os.listdir(proc)
  except OSError:
    return tids
  for pid in pids:
    if not pid.isdigit():
      continue
    try:
      with open(os.path.join(proc, pid, "cmdline"), "rb") as f:
        if DMONITORING_CMDLINE not in f.read():
          continue
      tids += [int(t) for t in os.listdir(os.path.join(proc, pid, "task")) if t.isdigit()]
    except OSError:
      continue
  return tids


def _move_dmonitoring(frm: int, to: int) -> int:
  if not _can_schedule():
    return 0
  moved = 0
  for tid in _dmonitoring_tids():
    try:
      if os.sched_getscheduler(tid) == os.SCHED_FIFO and os.sched_getparam(tid).sched_priority == frm:
        os.sched_setscheduler(tid, os.SCHED_FIFO, os.sched_param(to))
        moved += 1
    except OSError:
      continue
  return moved


def lift_dmonitoring() -> int:
  return _move_dmonitoring(DMONITORING_PRIORITY, DMONITORING_LIFTED)


def restore_dmonitoring() -> int:
  return _move_dmonitoring(DMONITORING_LIFTED, DMONITORING_PRIORITY)


def enter_realtime() -> bool:
  if not _can_schedule():
    return False
  try:
    os.sched_setaffinity(0, {MODEL_CORE})
    os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(EGPU_RT_PRIORITY))
  except OSError as e:
    cloudlog.warning(f"iqegpumodeld realtime setup failed ({e}); running at normal priority on core {MODEL_CORE}")
    return False
  atexit.register(restore_dmonitoring)
  cloudlog.warning(f"iqegpumodeld lifted {lift_dmonitoring()} dmonitoring thread(s) above the eGPU worker")
  return True


def leave_realtime() -> None:
  if not _can_schedule():
    return
  try:
    os.sched_setscheduler(0, os.SCHED_OTHER, os.sched_param(0))
  except OSError as e:
    cloudlog.warning(f"iqegpumodeld could not drop realtime ({e})")
  restore_dmonitoring()


class CpuBudgetGuard:
  def __init__(self, budget: float = CPU_BUDGET, window_s: float = BUDGET_WINDOW_S,
               clock=time.monotonic, cpu_clock=time.thread_time, demote=leave_realtime, lift=lift_dmonitoring) -> None:
    self.budget, self.window_s = budget, window_s
    self._clock, self._cpu_clock, self._demote, self._lift = clock, cpu_clock, demote, lift
    self.realtime = True
    self.last_share = 0.0
    self._t0, self._c0 = clock(), cpu_clock()

  def tick(self) -> None:
    now = self._clock()
    if not self.realtime or now - self._t0 < self.window_s:
      return
    self.last_share = (self._cpu_clock() - self._c0) / (now - self._t0)
    self._t0, self._c0 = now, self._cpu_clock()
    if self.last_share <= self.budget:
      self._lift()
    else:
      self.realtime = False
      self._demote()
      cloudlog.error(f"iqegpumodeld used {self.last_share:.0%} of core {MODEL_CORE} (budget {self.budget:.0%}); " +
                     "dropped realtime so dmonitoring and the fallback model are not starved")
