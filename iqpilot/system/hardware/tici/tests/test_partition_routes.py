"""Partition I/O on IQ.OS 4.9.9.4, where the service can't open otaflash partitions itself (see agnos.OTAFLASH_GROUP)."""
import builtins
import hashlib
import os
import stat

import pytest

from iqpilot.system.hardware.tici import agnos


GNU_DD = """#!/usr/bin/env python3
import os
import sys

args = dict(arg.split("=", 1) for arg in sys.argv[1:])


def size(text):
  units = {"K": 1024, "M": 1024 * 1024}
  return int(text[:-1]) * units[text[-1]] if text[-1] in units else int(text)


block = size(args.get("bs", "512"))
flags = set(args.get("iflag", "").split(",")) | set(args.get("oflag", "").split(","))
skip = int(args.get("skip", "0")) * (1 if "skip_bytes" in flags else block)
seek = int(args.get("seek", "0")) * (1 if "seek_bytes" in flags else block)
count = args.get("count")
limit = None if count is None else int(count) * (1 if "count_bytes" in flags else block)

source = open(args["if"], "rb") if "if" in args else sys.stdin.buffer
if source.seekable():
  source.seek(skip)
else:
  source.read(skip)
data = source.read() if limit is None else source.read(limit)

if "of" in args:
  mode = "r+b" if os.path.exists(args["of"]) and "notrunc" in args.get("conv", "") else "wb"
  with open(args["of"], mode) as out:
    out.seek(seek)
    out.write(data)
else:
  sys.stdout.buffer.write(data)
"""


@pytest.fixture
def otaflash_only(tmp_path, monkeypatch):
  """A fake partition that direct opens can't reach and a fake `sg` that only grants otaflash."""
  partition = tmp_path / 'dev' / 'abl_a'
  partition.parent.mkdir()
  partition.write_bytes(b'\x00' * 8192)
  bin_dir = tmp_path / 'bin'
  bin_dir.mkdir()
  sg = bin_dir / 'sg'
  sg.write_text('#!/bin/sh\n[ "$1" = otaflash ] && [ "$2" = -c ] || exit 1\nexec sh -c "$3"\n')
  sg.chmod(sg.stat().st_mode | stat.S_IXUSR)
  (bin_dir / 'sudo').write_text('#!/bin/sh\necho "sudo: a password is required" >&2\nexit 1\n')
  (bin_dir / 'sudo').chmod(0o755)
  (bin_dir / 'dd').write_text(GNU_DD)
  (bin_dir / 'dd').chmod(0o755)
  monkeypatch.setenv('PATH', f"{bin_dir}{os.pathsep}{os.environ['PATH']}")

  real_open = builtins.open

  def guarded_open(path, *args, **kwargs):
    if str(path).startswith(str(partition.parent)):
      raise PermissionError(13, 'Permission denied', str(path))
    return real_open(path, *args, **kwargs)

  monkeypatch.setattr(agnos, 'open', guarded_open, raising=False)
  monkeypatch.setattr(agnos, '_partition_routes', {})
  return partition


class Log:
  def __init__(self):
    self.lines = []

  def info(self, line):
    self.lines.append(line)


def test_preflight_falls_back_to_sg_and_logs_it(otaflash_only):
  log = Log()
  agnos.preflight_partition_io([str(otaflash_only)], log)
  assert agnos._partition_routes == {str(otaflash_only): 'sg'}
  assert log.lines == [f'partition I/O preflight: {otaflash_only} via sg']
  assert otaflash_only.read_bytes() == b'\x00' * 8192


def test_preflight_refuses_before_writing_when_nothing_works(otaflash_only, tmp_path):
  (tmp_path / 'bin' / 'sg').write_text('#!/bin/sh\nexit 1\n')
  log = Log()
  with pytest.raises(PermissionError, match='no route'):
    agnos.preflight_partition_io([str(otaflash_only)], log)
  assert log.lines == [f'partition I/O preflight: {otaflash_only} via nothing']


def test_sg_route_reads_writes_and_hashes(otaflash_only, monkeypatch):
  path = str(otaflash_only)
  agnos.preflight_partition_io([path], Log())

  writer = agnos.open_partition_writer(path)
  image = os.urandom(5000)
  writer.write(image[:3000])
  writer.write(image[3000:])
  writer.close()
  agnos.write_partition_at(path, 5000, b'f' * 64)

  data = otaflash_only.read_bytes()
  assert data[:5000] == image
  assert data[5000:5064] == b'f' * 64
  assert len(data) == 8192
  assert b''.join(agnos.read_partition(path, 5000, 64)) == b'f' * 64
  assert agnos.get_raw_hash(path, 5000) == hashlib.sha256(image).hexdigest()
  partition = {'name': 'abl', 'size': 5000, 'hash_raw': 'f' * 64, 'full_check': False}
  monkeypatch.setattr(agnos, 'get_partition_path', lambda _slot, _partition: path)
  assert agnos.verify_partition(0, partition)


def test_direct_route_is_preferred(tmp_path, monkeypatch):
  partition = tmp_path / 'abl_a'
  partition.write_bytes(b'\x00' * 4096)
  monkeypatch.setattr(agnos, '_partition_routes', {})
  agnos.preflight_partition_io([str(partition)], Log())
  assert agnos._partition_routes == {str(partition): 'direct'}


def test_board_filter_keeps_only_this_boards_xbl():
  update = [
    {'name': 'abl', 'hash': 'ours'},
    {'name': 'xbl', 'hash': '7af49ed5', 'board': 'tici'},
    {'name': 'xbl', 'hash': 'dd45c0fe', 'board': 'tizi'},
    {'name': 'xbl', 'hash': 'dd45c0fe', 'board': 'mici'},
  ]
  assert agnos.select_for_board(update, 'tizi') == [update[0], update[2]]
  with pytest.raises(RuntimeError, match='no xbl for unknown'):
    agnos.select_for_board(update, 'unknown')
  with pytest.raises(RuntimeError, match='model could not be read'):
    agnos.select_for_board(update, '')
