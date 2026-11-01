# Copyright (c) 2026 IQ.Lvbs. All rights reserved.
import gc
import weakref

from iqpilot.common import realtime


class Node:
  pass


def test_cycle_collector_frees_cycles_while_the_collector_is_disabled(monkeypatch):
  now = [0.0]
  monkeypatch.setattr(realtime.time, "monotonic", lambda: now[0])
  was_enabled = gc.isenabled()
  gc.disable()
  try:
    collector = realtime.CycleCollector(interval_s=30.0)
    node = Node()
    node.me = node
    ref = weakref.ref(node)
    del node
    now[0] = 29.0
    collector.tick()
    assert ref() is not None
    now[0] = 30.0
    collector.tick()
    assert ref() is None
  finally:
    gc.unfreeze()
    if was_enabled:
      gc.enable()
