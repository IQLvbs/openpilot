"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""
import hashlib
import sys

from iqpilot.system.hardware.egpu_dock.flash import (
  Flash, RomFallback, bundled_version, find_dock, in_rom_bootloader, link_up, stable_read,
)


def main() -> int:
  path, vid_pid, product = find_dock()
  if path is None:
    print("no eGPU dock enumerated")
    return 1
  print(f"dock at {path}")
  print(f"  vid:pid   {vid_pid[0]}:{vid_pid[1]}")
  print(f"  product   {product!r}")
  print(f"  bundled   {bundled_version()!r}")
  print(f"  match     {product == bundled_version()}")
  with open(path + "/speed") as fs:
    speed = int(fs.read())
  print(f"  usb speed {speed} Mbps ({'USB3' if speed >= 5000 else 'USB2 - register reads capped at 64B'})")
  if in_rom_bootloader(vid_pid, product):
    print("  state     ROM bootloader (config page lost or firmware invalid)")
    return 2
  print(f"  pcie link {'L0 (trained)' if link_up() else 'not trained'}")

  flash = Flash()
  try:
    flash.connect()
    config = stable_read(flash, 0, 0x100, 3)
    print(f"  config    sha256={hashlib.sha256(config).hexdigest()[:16]} "
          f"(stable over 3 reads, {sum(1 for b in config if b != 0xFF)} non-blank bytes)")
  except (RomFallback, OSError, RuntimeError, TimeoutError) as e:
    print(f"  config    read failed: {type(e).__name__}: {e}")
    return 3
  finally:
    flash.close()
  print("all read-only checks passed")
  return 0


if __name__ == "__main__":
  sys.exit(main())
