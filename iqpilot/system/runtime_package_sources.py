#!/usr/bin/env python3
import re
import sys
from pathlib import Path


PACKAGE_NAMES = ("msgq", "iqdbc", "panda", "tinygrad", "pycapnp")
root = Path(sys.argv[1]).resolve()
text = (root / "pyproject.toml").read_text()
for name in PACKAGE_NAMES:
  vendor_source = root / "artifacts" / "package_sources" / name
  if vendor_source.is_dir():
    source = str(vendor_source)
  else:
    match = re.search(rf'"({name} @ git\+https://[^@]+@[0-9a-f]{{40}})"', text)
    if match is None:
      raise RuntimeError(f"missing {name} package source")
    source = match.group(1)
  print(f"{name}\t{source}")
