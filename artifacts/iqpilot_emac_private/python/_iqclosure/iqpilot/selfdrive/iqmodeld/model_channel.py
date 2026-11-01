"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""
from __future__ import annotations

import mmap
import os
import pickle
import struct

from iqpilot.common.swaglog import cloudlog

SMALL_CHANNEL = "/dev/shm/iqpilot_smallmodel"
BIG_CHANNEL = "/dev/shm/iqpilot_bigmodel"
SHM_SIZE = 8 * 1024 * 1024
HEADER = struct.Struct("<QqQ")


class ModelChannel:
  def __init__(self, path: str, create: bool):
    if create:
      fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
      os.ftruncate(fd, SHM_SIZE)
    else:
      fd = os.open(path, os.O_RDWR)
    self.mm = mmap.mmap(fd, SHM_SIZE)
    os.close(fd)
    if create:
      self.mm[:HEADER.size] = HEADER.pack(0, -1, 0)

  def write(self, frame_id: int, payload: dict) -> None:
    data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    if HEADER.size + len(data) > SHM_SIZE:
      cloudlog.error(f"model payload {len(data)} bytes exceeds shm {SHM_SIZE}, dropping frame {frame_id}")
      return
    seq = HEADER.unpack(self.mm[:HEADER.size])[0]
    HEADER.pack_into(self.mm, 0, seq + 1, frame_id, len(data))
    self.mm[HEADER.size:HEADER.size + len(data)] = data
    HEADER.pack_into(self.mm, 0, seq + 2, frame_id, len(data))

  def peek_frame_id(self) -> int | None:
    seq, frame_id, length = HEADER.unpack(self.mm[:HEADER.size])
    if seq == 0 or seq % 2 != 0 or length == 0:
      return None
    return frame_id

  def peek(self) -> tuple[int | None, bool]:
    seq, frame_id, length = HEADER.unpack(self.mm[:HEADER.size])
    if seq % 2 != 0:
      return None, True
    if seq == 0 or length == 0:
      return None, False
    return frame_id, False

  def read(self) -> tuple[int, dict] | None:
    seq1, frame_id, length = HEADER.unpack(self.mm[:HEADER.size])
    if seq1 == 0 or seq1 % 2 != 0 or length == 0:
      return None
    data = bytes(self.mm[HEADER.size:HEADER.size + length])
    seq2 = HEADER.unpack(self.mm[:HEADER.size])[0]
    if seq1 != seq2:
      return None
    try:
      return frame_id, pickle.loads(data)
    except Exception:
      return None
