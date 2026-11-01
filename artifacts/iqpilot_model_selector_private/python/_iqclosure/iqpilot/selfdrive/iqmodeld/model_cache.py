import os
import re
from pathlib import Path


_RUNTIME_ARTIFACT_PATTERNS = (
  re.compile(r"egpu_.+_amd_(?:tinygrad|policy|policy_oob|model_oob2)\.pkl$"),
  re.compile(r"emac_warp_\d+x\d+_tinygrad\.pkl$"),
  re.compile(r".+_usbemac\.pkl$"),
  re.compile(r".+\.onnx$"),
)


def is_runtime_artifact(filename: str) -> bool:
  return any(pattern.fullmatch(filename) for pattern in _RUNTIME_ARTIFACT_PATTERNS)


def model_cache_size(root: str | os.PathLike[str]) -> int:
  path = Path(root)
  if not path.is_dir():
    return 0

  total = 0
  for entry in path.iterdir():
    try:
      if entry.is_file() and not is_runtime_artifact(entry.name):
        total += entry.stat().st_size
    except OSError:
      continue
  return total


def clear_model_cache(root: str | os.PathLike[str], keep: set[str]) -> None:
  path = Path(root)
  if not path.is_dir():
    return

  for entry in path.iterdir():
    try:
      if entry.name not in keep and not is_runtime_artifact(entry.name) and entry.is_file():
        entry.unlink()
    except OSError:
      continue


def bundle_index(bundle) -> int | None:
  try:
    return int(getattr(bundle, "index", -1))
  except (TypeError, ValueError):
    return None


def bundle_matches(left, right) -> bool:
  if left is None or right is None:
    return False

  left_index = bundle_index(left)
  right_index = bundle_index(right)
  if left_index is not None and right_index is not None and left_index == right_index:
    return True

  for attr in ("ref", "internalName", "displayName"):
    left_value = getattr(left, attr, None)
    if left_value and left_value == getattr(right, attr, None):
      return True
  return False


def safe_model_path(root: str, filename: str) -> str | None:
  if not filename or os.path.basename(filename) != filename:
    return None

  root = os.path.realpath(root)
  path = os.path.realpath(os.path.join(root, filename))
  try:
    if os.path.commonpath([root, path]) != root:
      return None
  except ValueError:
    return None
  return path


def remove_bundle_files(root: str, bundle) -> None:
  for model in getattr(bundle, "models", []) or []:
    for artifact in (getattr(model, "metadata", None), getattr(model, "artifact", None)):
      filename = getattr(artifact, "fileName", "") if artifact is not None else ""
      path = safe_model_path(root, filename)
      if path is None:
        continue
      for candidate in (path, f"{path}.download"):
        try:
          if os.path.isfile(candidate):
            os.remove(candidate)
        except OSError:
          pass
