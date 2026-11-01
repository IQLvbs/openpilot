# Copyright (c) 2026 IQ.Lvbs. All rights reserved.
import errno
from types import SimpleNamespace

import pytest

from iqpilot.system.loggerd import xattr_cache


@pytest.fixture
def attributes(monkeypatch):
  monkeypatch.setattr(xattr_cache, '_cached_attributes', type(xattr_cache._cached_attributes)())
  monkeypatch.setattr(xattr_cache, 'os', SimpleNamespace(stat=lambda path: SimpleNamespace(st_dev=1, st_ino=hash(path), st_ctime_ns=1)))
  values = {}

  def get(path, name):
    if (path, name) not in values:
      raise OSError(errno.ENODATA, 'attribute absent')
    return values[path, name]

  monkeypatch.setattr(xattr_cache, '_getxattr', get)
  monkeypatch.setattr(xattr_cache, '_setxattr', lambda path, name, value: values.__setitem__((path, name), value))
  return values


def test_route_churn_keeps_attribute_cache_bounded(attributes):
  for index in range(20_000):
    assert xattr_cache.getxattr(f'/routes/route--{index}', 'user.preserve') is None
  assert len(xattr_cache._cached_attributes) <= 4096


def test_attribute_eviction_keeps_preserve_flags_correct(attributes):
  path = '/routes/preserved--0'
  xattr_cache.setxattr(path, 'user.preserve', b'1')
  for index in range(5000):
    xattr_cache.getxattr(f'/routes/route--{index}', 'user.preserve')
  assert xattr_cache.getxattr(path, 'user.preserve') == b'1'
  xattr_cache.setxattr(path, 'user.preserve', b'0')
  assert xattr_cache.getxattr(path, 'user.preserve') == b'0'


def test_changed_file_identity_refreshes_attribute(attributes, monkeypatch):
  path = '/routes/replaced--0'
  attributes[path, 'user.preserve'] = b'1'
  assert xattr_cache.getxattr(path, 'user.preserve') == b'1'
  attributes[path, 'user.preserve'] = b'0'
  monkeypatch.setattr(xattr_cache.os, 'stat', lambda value: SimpleNamespace(st_dev=1, st_ino=2, st_ctime_ns=2))
  assert xattr_cache.getxattr(path, 'user.preserve') == b'0'


def test_read_failure_is_not_cached(attributes, monkeypatch):
  path = '/routes/preserved--0'
  attributes[path, 'user.preserve'] = b'1'
  get = xattr_cache._getxattr

  def fail_read(*args):
    raise OSError(errno.EIO, 'read failed')

  monkeypatch.setattr(xattr_cache, '_getxattr', fail_read)
  with pytest.raises(OSError) as error:
    xattr_cache.getxattr(path, 'user.preserve')
  assert error.value.errno == errno.EIO
  monkeypatch.setattr(xattr_cache, '_getxattr', get)
  assert xattr_cache.getxattr(path, 'user.preserve') == b'1'


def test_filesystem_attribute_round_trip(tmp_path):
  path = str(tmp_path / 'segment')
  (tmp_path / 'segment').mkdir()
  assert xattr_cache.getxattr(path, 'user.preserve') is None
  xattr_cache.setxattr(path, 'user.preserve', b'1')
  assert xattr_cache.getxattr(path, 'user.preserve') == b'1'
  xattr_cache.setxattr(path, 'user.preserve', b'0')
  assert xattr_cache.getxattr(path, 'user.preserve') == b'0'


def test_native_backend_needs_no_xattr_package(monkeypatch, tmp_path):
  import os
  import runpy
  import sys

  monkeypatch.setattr(os, 'getxattr', xattr_cache._getxattr, raising=False)
  monkeypatch.setattr(os, 'setxattr', xattr_cache._setxattr, raising=False)
  monkeypatch.setitem(sys.modules, 'xattr', None)
  backend = runpy.run_path(xattr_cache.__file__)
  assert backend['_getxattr'] is os.getxattr
  assert backend['_setxattr'] is os.setxattr
  path = str(tmp_path)
  backend['setxattr'](path, 'user.preserve', b'1')
  backend['_cached_attributes'].clear()
  assert backend['getxattr'](path, 'user.preserve') == b'1'
