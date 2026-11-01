"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""
from __future__ import annotations

import argparse
import os
import pickletools
import struct

from iqpilot.selfdrive.iqmodeld.egpu_policy import OOB_MAGIC

MIN_OOB_BYTES = 1 << 16
NEXT_BUFFER = b"\x97"
READONLY_BUFFER = b"\x98"


def rewrite_oob(src: str, dst: str, min_bytes: int = MIN_OOB_BYTES) -> tuple[int, int]:
  # tinygrad pickles device buffers as PickleBuffers, which land in-band as BYTEARRAY8/BINBYTES8
  # without a buffer_callback; moving those opcodes out-of-band is byte-for-byte what a protocol-5
  # dump with a buffer_callback produces, so nothing has to be unpickled (no dock needed).
  with open(src, "rb") as f:
    data = f.read()
  ops = list(pickletools.genops(data))
  proto = next((arg for op, arg, _ in ops if op.name == "PROTO"), 0)
  if proto < 5:
    raise ValueError(f"{src} is pickle protocol {proto}; out-of-band buffers need protocol 5")
  moved = 0
  tmp = dst + ".part"
  with open(tmp, "wb") as out, open(tmp + ".buf", "wb") as bufs:
    ops_stream = bytearray()
    for i, (op, arg, pos) in enumerate(ops):
      end = ops[i + 1][2] if i + 1 < len(ops) else len(data)
      if op.name in ("BYTEARRAY8", "BINBYTES8", "BINBYTES") and len(arg) >= min_bytes:
        ops_stream += NEXT_BUFFER
        if op.name != "BYTEARRAY8":
          ops_stream += READONLY_BUFFER
        bufs.write(struct.pack("<q", len(arg)))
        bufs.write(arg)
        moved += 1
      else:
        ops_stream += data[pos:end]
    out.write(OOB_MAGIC)
    out.write(struct.pack("<q", len(ops_stream)))
    out.write(ops_stream)
  with open(tmp, "ab") as out, open(tmp + ".buf", "rb") as bufs:
    while chunk := bufs.read(1 << 24):
      out.write(chunk)
  os.remove(tmp + ".buf")
  os.replace(tmp, dst)
  return moved, len(ops)


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("src")
  p.add_argument("dst")
  args = p.parse_args()
  moved, total = rewrite_oob(args.src, args.dst)
  print(f"{args.dst}: moved {moved} buffers out-of-band ({total} opcodes, {os.path.getsize(args.dst) / 1e6:.1f} MB)")


if __name__ == "__main__":
  main()
