"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""
from __future__ import annotations

import argparse
import gc
import os

os.environ.setdefault("DEV", "USB+AMD:LLVM")
os.environ.setdefault("GMMU", "0")

from iqpilot.selfdrive.iqmodeld.egpu_helpers import patch_tinygrad_fetch_fw
from iqpilot.selfdrive.iqmodeld.egpu_policy import dump_oob, is_oob, load_bundle


def convert(src: str, dst: str) -> str:
  patch_tinygrad_fetch_fw()
  if is_oob(src):
    if src != dst:
      os.replace(src, dst)
    return dst
  bundle = load_bundle(src)
  tmp = dst + ".part"
  with open(tmp, "wb") as f:
    dump_oob(bundle, f)
  del bundle
  gc.collect()
  load_bundle(tmp)
  os.replace(tmp, dst)
  return dst


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("src")
  p.add_argument("--out", default=None)
  args = p.parse_args()
  out = convert(args.src, args.out or args.src)
  print(f"converted -> {out} ({os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
  main()
