#!/usr/bin/env python3
import csv
import importlib.util
import sys
from importlib import metadata
from pathlib import Path

from iqpilot.system.runtime_wheel_requirements import PACKAGE_IMPORTS, PACKAGE_NAMES

_IMPORT_NAME = dict(zip(PACKAGE_NAMES, PACKAGE_IMPORTS, strict=True))


def _verify_without_metadata(name):
  spec = importlib.util.find_spec(_IMPORT_NAME.get(name, name))
  return [] if spec is not None and spec.origin is not None else [f"{name}: not installed"]


missing = []
for name in PACKAGE_NAMES:
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
  print(f"runtime wheels damaged: {len(missing)} missing files", file=sys.stderr)
  sys.exit(1)
