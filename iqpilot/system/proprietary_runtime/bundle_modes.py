# Copyright (c) 2026 IQ.Lvbs. All rights reserved.
import glob
import json
import os

from iqpilot.common.basedir import BASEDIR

_normalized: set[str] = set()


def normalize_root(root: str) -> bool:
  manifest_path = os.path.join(root, "manifest.json")
  if not os.path.isfile(manifest_path):
    return False
  with open(manifest_path) as f:
    manifest = json.load(f)
  for rel, meta in manifest.items():
    if not (isinstance(meta, dict) and "mode" in meta and "sha256" in meta):
      continue
    path = os.path.join(root, rel)
    try:
      if os.path.isfile(path) and (os.stat(path).st_mode & 0o777) != meta["mode"]:
        os.chmod(path, meta["mode"])
    except OSError:
      pass
  return True


def bundle_root(bundle: str, basedir: str = BASEDIR) -> str | None:
  candidates = []
  if env_root := os.environ.get("IQPILOT_PROPRIETARY_ROOT"):
    candidates += [os.path.join(env_root, bundle), env_root]
  candidates += [
    os.path.join(basedir, ".iqpilot", "bundles", bundle),
    os.path.join(os.path.dirname(basedir), ".iqpilot", "bundles", bundle),
    os.path.join(basedir, "artifacts", bundle),
  ]
  return next((c for c in candidates if os.path.isfile(os.path.join(c, "manifest.json"))), None)


def normalize_bundle(bundle: str, basedir: str = BASEDIR) -> None:
  root = bundle_root(bundle, basedir)
  if root is not None:
    normalize_root(root)


def normalize_tree(basedir: str) -> None:
  for manifest_path in sorted(glob.glob(os.path.join(basedir, "artifacts", "*", "manifest.json"))):
    try:
      normalize_root(os.path.dirname(manifest_path))
    except (OSError, ValueError):
      pass


# git checkouts write every file 0644 while signed manifests pin some to 0640, and the native verifier aborts the whole process on a mode mismatch.
def import_verified_module(bundle: str, module: str):
  if bundle not in _normalized:
    try:
      normalize_bundle(bundle)
    except (OSError, ValueError):
      pass
    _normalized.add(bundle)
  from iqpilot.system.proprietary_runtime import _verified_import
  return _verified_import.import_verified_module(bundle, module)
