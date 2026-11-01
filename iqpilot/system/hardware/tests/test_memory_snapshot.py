# Copyright (c) 2026 IQ.Lvbs. All rights reserved.
import queue
import threading
from types import SimpleNamespace

from iqpilot.system.hardware import hardwared


def test_memory_snapshot_worker_is_bounded_and_recovers(monkeypatch):
  entered = threading.Event()
  release = threading.Event()
  done = threading.Event()
  end = threading.Event()
  enqueued = threading.Event()
  snapshots = []
  errors = []
  calls = 0

  def collect():
    nonlocal calls
    calls += 1
    if calls == 1:
      entered.set()
      assert release.wait(5)
      raise OSError('process disappeared')
    return [{'pid': 123, 'rss_mb': 32}]

  def record(*args, **kwargs):
    snapshots.append(kwargs)
    done.set()

  monkeypatch.setattr(hardwared, 'get_top_memory_processes', collect)
  monkeypatch.setattr(hardwared, 'cloudlog', SimpleNamespace(event=record, exception=errors.append))
  requests = queue.Queue(maxsize=1)
  worker = threading.Thread(target=hardwared.memory_snapshot_thread, args=(end, requests))
  worker.start()
  try:
    requests.put_nowait(96)
    assert entered.wait(5)
    def enqueue():
      requests.put_nowait(97)
      for _ in range(1000):
        try:
          requests.put_nowait(98)
        except queue.Full:
          pass
      enqueued.set()

    producer = threading.Thread(target=enqueue)
    producer.start()
    assert enqueued.wait(1)
    producer.join(1)
    assert requests.qsize() == 1
    release.set()
    assert done.wait(5)
    assert errors == ['Error collecting low memory snapshot']
    assert len(snapshots) == 1
    assert snapshots[0]['memory_usage_percent'] == 97
    assert snapshots[0]['top_processes'] == [{'pid': 123, 'rss_mb': 32}]
  finally:
    end.set()
    release.set()
    worker.join(2)
  assert not worker.is_alive()
