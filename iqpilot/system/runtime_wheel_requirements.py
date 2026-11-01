#!/usr/bin/env python3
import hashlib
import sys
import tomllib
from pathlib import Path


PACKAGE_NAMES = ("libdatachannel-py",)
PACKAGE_IMPORTS = ("libdatachannel",)
PACKAGE_WHEELS = {
  "libdatachannel-py": "libdatachannel_py-2026.1.0.dev2-cp312-cp312-manylinux_2_35_aarch64.whl",
}


def locked_packages(root: Path) -> dict[str, dict]:
  lock = tomllib.loads((root / "uv.lock").read_text())
  packages = {package["name"]: package for package in lock["package"] if package["name"] in PACKAGE_NAMES}
  if packages.keys() != set(PACKAGE_NAMES):
    raise RuntimeError("missing locked runtime wheel")
  return packages


def requirements(root: Path) -> list[str]:
  packages = locked_packages(root)
  return [f"{name}=={packages[name]['version']}" for name in PACKAGE_NAMES]


def install_sources(root: Path) -> list[str]:
  packages = locked_packages(root)
  sources = []
  for name in PACKAGE_NAMES:
    filename = PACKAGE_WHEELS[name]
    wheel = root / "artifacts" / "runtime_wheels" / filename
    locked_wheel = next((item for item in packages[name]["wheels"] if item["url"].endswith(f"/{filename}")), None)
    if locked_wheel is None or not wheel.is_file():
      raise RuntimeError(f"missing vendored runtime wheel for {name}")
    expected = locked_wheel["hash"].removeprefix("sha256:")
    if hashlib.sha256(wheel.read_bytes()).hexdigest() != expected:
      raise RuntimeError(f"invalid vendored runtime wheel for {name}")
    sources.append(str(wheel))
  return sources


if __name__ == "__main__":
  root = Path(sys.argv[1]).resolve()
  values = install_sources(root) if "--sources" in sys.argv[2:] else requirements(root)
  print("\n".join(values))
