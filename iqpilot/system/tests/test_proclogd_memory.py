# Copyright (c) 2026 IQ.Lvbs. All rights reserved.
import io
from types import SimpleNamespace

import pytest

from iqpilot.cereal import messaging
from iqpilot.system import proclogd


@pytest.fixture
def processes(monkeypatch):
  monkeypatch.setattr(proclogd, '_proc_cache', {})
  monkeypatch.setattr(proclogd, '_cpu_times', list)
  monkeypatch.setattr(proclogd, '_mem_info', lambda: dict.fromkeys(
    ('MemTotal:', 'MemFree:', 'MemAvailable:', 'Buffers:', 'Cached:', 'Active:', 'Inactive:', 'Shmem:'), 0))
  monkeypatch.setattr(proclogd, 'os', SimpleNamespace(readlink=lambda path: '/usr/bin/python'))
  monkeypatch.setattr(proclogd, 'open', lambda *args, **kwargs: io.BytesIO(b'python\0worker\0'), raising=False)
  rows = []
  monkeypatch.setattr(proclogd, '_procs', lambda: rows)
  return rows


def proc(pid, start):
  values = dict.fromkeys(('ppid', 'utime', 'stime', 'cutime', 'cstime', 'priority', 'nice', 'num_threads', 'vms', 'rss', 'processor'), 0)
  return dict(values, pid=pid, starttime=start, name='worker', state='S')


def test_exited_processes_do_not_accumulate_metadata(processes):
  for index in range(500):
    processes[:] = [proc(10_000 + index, index)]
    message = messaging.new_message('procLog')
    proclogd.build_proc_log_message(message)
    assert len(message.procLog.procs) == 1
    assert message.procLog.procs[0].pid == 10_000 + index
  assert len(proclogd._proc_cache) == 1


def test_reused_pid_refreshes_process_metadata(processes, monkeypatch):
  processes[:] = [proc(10_000, 10)]
  proclogd.build_proc_log_message(messaging.new_message('procLog'))
  processes[:] = [proc(10_000, 20)]
  monkeypatch.setattr(proclogd, 'open', lambda *args, **kwargs: io.BytesIO(b'python\0replacement\0'))
  message = messaging.new_message('procLog')
  proclogd.build_proc_log_message(message)
  assert list(message.procLog.procs[0].cmdline) == ['python', 'replacement']
