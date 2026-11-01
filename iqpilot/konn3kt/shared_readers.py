# Copyright (c) 2026 IQ.Lvbs. All rights reserved.
from collections import OrderedDict
import threading
import weakref

from iqpilot.cereal import messaging


class _SharedSubMaster(messaging.SubMaster):
  def __init__(self, services: tuple[str, ...]):
    super().__init__(list(services))
    self.lock = threading.Lock()


class SharedMessageReaders:
  def __init__(self, recent_limit: int = 8):
    self._recent_limit = recent_limit
    self._fixed: dict[tuple[str, ...], _SharedSubMaster] = {}
    self._recent: OrderedDict[tuple[str, ...], _SharedSubMaster] = OrderedDict()
    self._active: weakref.WeakValueDictionary[tuple[str, ...], _SharedSubMaster] = weakref.WeakValueDictionary()
    self._lock = threading.Lock()

  def get(self, services: tuple[str, ...], transient: bool = False):
    key = tuple(sorted(set(services)))
    with self._lock:
      reader = self._active.get(key)
      if reader is None:
        if transient and len(self._recent) >= self._recent_limit and self._recent:
          self._recent.popitem(last=False)
        reader = _SharedSubMaster(key)
        self._active[key] = reader
      if not transient:
        self._fixed[key] = reader
        self._recent.pop(key, None)
      elif key not in self._fixed:
        self._recent[key] = reader
        self._recent.move_to_end(key)
        while len(self._recent) > self._recent_limit:
          self._recent.popitem(last=False)
      return reader, reader.lock
