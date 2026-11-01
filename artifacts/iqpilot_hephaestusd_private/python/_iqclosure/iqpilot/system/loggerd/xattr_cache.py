import errno
import os
from collections import OrderedDict
import threading

if hasattr(os, "getxattr"):
  from os import getxattr as _getxattr, setxattr as _setxattr
else:
  from xattr import getxattr as _getxattr, setxattr as _setxattr

MAX_CACHE_ENTRIES = 4096
_cached_attributes: OrderedDict[tuple[str, str], tuple[tuple[int, int, int], bytes | None]] = OrderedDict()
_cache_lock = threading.Lock()


def _remember_attribute(key: tuple[str, str], identity: tuple[int, int, int], value: bytes | None):
  with _cache_lock:
    _cached_attributes[key] = (identity, value)
    _cached_attributes.move_to_end(key)
    if len(_cached_attributes) > MAX_CACHE_ENTRIES:
      _cached_attributes.popitem(last=False)

def getxattr(path: str, attr_name: str) -> bytes | None:
  key = (path, attr_name)
  st = os.stat(path)
  identity = (st.st_dev, st.st_ino, st.st_ctime_ns)
  with _cache_lock:
    cached = _cached_attributes.get(key)
    if cached is not None and cached[0] == identity:
      _cached_attributes.move_to_end(key)
      return cached[1]
  try:
    response = _getxattr(path, attr_name)
  except OSError as e:
    # ENODATA (Linux) or ENOATTR (macOS) means attribute hasn't been set
    if e.errno == errno.ENODATA or (hasattr(errno, 'ENOATTR') and e.errno == errno.ENOATTR):
      response = None
    else:
      raise
  _remember_attribute(key, identity, response)
  return response

def setxattr(path: str, attr_name: str, attr_value: bytes) -> None:
  _setxattr(path, attr_name, attr_value)
  st = os.stat(path)
  _remember_attribute((path, attr_name), (st.st_dev, st.st_ino, st.st_ctime_ns), attr_value)
