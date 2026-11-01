#!/usr/bin/env python3
"""Verify the runtime packages are completely installed: every file each
package's wheel RECORD names must exist on disk. A top-level `import tinygrad`
passes on a partially extracted install (backends load lazily), and
importlib.metadata's `files` silently drops missing entries, so RECORD is
parsed directly here. Runs under the interpreter whose site-packages should be
checked; prints missing files and exits nonzero on any damage.
"""
import csv
import importlib.util
import sys
from importlib import metadata
from pathlib import Path

PACKAGES = ("iqdbc", "msgq", "panda", "tinygrad", "pycapnp")
IMPORT_NAME = {"pycapnp": "capnp"}
REQUIRED_DATA = {"iqdbc": ("car/car.capnp", "car/rlog.capnp", "car/include/c++.capnp")}


def _verify_without_metadata(name):
  imp = IMPORT_NAME.get(name, name)
  spec = importlib.util.find_spec(imp)
  if spec is None or spec.origin is None:
    return [f"{name}: not installed"]
  base = Path(spec.origin).parent
  return [f"{name}: {rel}" for rel in REQUIRED_DATA.get(imp, ()) if not (base / rel).is_file()]


missing = []
for name in PACKAGES:
  try:
    dist = metadata.distribution(name)
  except metadata.PackageNotFoundError:
    missing.extend(_verify_without_metadata(name))
    continue
  record = dist.read_text("RECORD")
  if not record:
    missing.append(f"{name}: no RECORD")
    continue
  base = Path(str(dist._path)).parent
  for row in csv.reader(record.splitlines()):
    if not row or row[0].endswith((".pyc", "/")):
      continue
    if not (base / row[0]).exists():
      missing.append(f"{name}: {row[0]}")

if missing:
  print("\n".join(missing[:20]), file=sys.stderr)
  print(f"runtime packages damaged: {len(missing)} missing files", file=sys.stderr)
  sys.exit(1)
