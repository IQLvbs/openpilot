#!/usr/bin/env python3
"""
Copyright (c) IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos
"""
import json
import hashlib
import os
import shutil
from pathlib import Path

from iqpilot.cereal import custom
from iqpilot.common.params import Params
from iqpilot.common.swaglog import cloudlog
from iqpilot._proprietary_loader import ProprietaryModuleMissing, load_private_module
from iqpilot.system.hardware.hw import Paths

try:
  load_private_module(__name__, "iqpilot_private.models.helpers")
except ProprietaryModuleMissing:
  try:
    from iqpilot.models_private_src.helpers import *  # noqa: F403
  except ImportError:
    pass


ModelBundle = custom.IQModelManager.ModelBundle
Runner = custom.IQModelManager.Runner
_MODEL_ROOT = Path(Paths.model_root())
_ACTIVE_BUNDLE_KEY = "ModelManager_ActiveBundle"
_MODELS_CACHE_KEY = "ModelManager_ModelsCache"
_RUNNER_CACHE_KEY = "ModelRunnerTypeCache"
_DOWNLOAD_INDEX_KEY = "ModelManager_DownloadIndex"
_PENDING_INDEX_KEY = "ModelManager_PendingIndex"
_PENDING_MODEL_RESTORE_FILE = "/data/k3_pending_model_restore"
_STOCK_RUNNER = int(Runner.stock)
_TINYGRAD_RUNNER = int(Runner.tinygrad)
_SNPE_RUNNER = int(Runner.snpe)

_DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[1] / "default_model"
_DEFAULT_BUNDLE_JSON = _DEFAULT_MODEL_DIR / "bundle.json"
_DEFAULT_BUNDLE_REF = "default"


def get_default_model_bundle(_bundles):
  return None


def _coerce_runner_value(value) -> int | None:
  raw = getattr(value, "raw", value)
  try:
    return int(raw)
  except (TypeError, ValueError):
    return None


def _bundle_models(bundle) -> list:
  models = getattr(bundle, "models", None)
  return list(models) if models is not None else []


def _bundle_needs_runtime_upgrade(bundle) -> bool:
  if bundle is None:
    return False

  if _coerce_runner_value(getattr(bundle, "runner", None)) == _SNPE_RUNNER:
    return True

  for model in _bundle_models(bundle):
    file_name = getattr(getattr(model, "artifact", None), "fileName", "") or ""
    if file_name.endswith(".thneed"):
      return True

  return False


def _load_cached_manifest_bundles(params: Params):
  cached = params.get(_MODELS_CACHE_KEY) or {}
  bundles = []
  for raw_bundle in cached.get("bundles", []):
    try:
      min_selector_version = int(raw_bundle.get("minimumSelectorVersion", raw_bundle.get("minimum_selector_version", 0)))
      compatibility_view = dict(raw_bundle)
      compatibility_view["minimumSelectorVersion"] = min_selector_version
      is_compatible = globals().get("is_bundle_version_compatible")
      if is_compatible is not None and not is_compatible(compatibility_view):
        continue

      if "short_name" in raw_bundle:
        from iqpilot.selfdrive.iqmodeld.models.fetcher import ManifestDecoder
        bundles.append(ManifestDecoder._decode_bundle(raw_bundle))
        continue

      if "internalName" in raw_bundle:
        bundles.append(ModelBundle(**raw_bundle))
        continue

      bundle = ModelBundle()
      bundle.index = int(raw_bundle["index"])
      bundle.internalName = raw_bundle.get("short_name")
      bundle.displayName = raw_bundle.get("display_name")
      bundle.status = 0
      bundle.generation = int(raw_bundle["generation"])
      bundle.environment = raw_bundle["environment"]
      bundle.runner = raw_bundle.get("runner", Runner.tinygrad)
      bundle.is20hz = raw_bundle.get("is_20hz", False)
      bundle.minimumSelectorVersion = int(min_selector_version)
      bundle.ref = raw_bundle.get("ref")
      bundle.overrides = []
      for key, value in raw_bundle.get("overrides", {}).items():
        override = custom.IQModelManager.Override()
        override.key = key
        override.value = value
        bundle.overrides.append(override)

      bundle.models = []
      for raw_model in raw_bundle.get("models", []):
        model = custom.IQModelManager.Model()
        model.type = raw_model.get("type")
        for attr_name in ("artifact", "metadata"):
          raw_artifact = raw_model.get(attr_name)
          if not raw_artifact:
            continue
          artifact = custom.IQModelManager.Artifact()
          artifact.fileName = raw_artifact.get("file_name")
          download_uri = custom.IQModelManager.DownloadUri()
          download_uri.uri = raw_artifact.get("download_uri", {}).get("url")
          download_uri.sha256 = raw_artifact.get("download_uri", {}).get("sha256")
          artifact.downloadUri = download_uri
          setattr(model, attr_name, artifact)
        bundle.models.append(model)

      bundles.append(bundle)
    except Exception:
      continue
  return bundles


def _bundle_match_key(bundle) -> tuple[str | None, str | None, str | None]:
  return (
    getattr(bundle, "ref", None),
    getattr(bundle, "internalName", None),
    getattr(bundle, "displayName", None),
  )


def _find_runtime_upgrade(bundle, params: Params, available_bundles=None):
  if not _bundle_needs_runtime_upgrade(bundle):
    return bundle

  candidate_bundles = available_bundles if available_bundles is not None else _load_cached_manifest_bundles(params)
  ref, internal_name, display_name = _bundle_match_key(bundle)

  for candidate in candidate_bundles:
    if getattr(candidate, "ref", None) and getattr(candidate, "ref", None) == ref:
      return candidate

  for candidate in candidate_bundles:
    if getattr(candidate, "internalName", None) == internal_name:
      return candidate

  for candidate in candidate_bundles:
    if getattr(candidate, "displayName", None) == display_name:
      return candidate

  return None


def bundle_files_ready(bundle) -> bool:
  if bundle is None or not _bundle_models(bundle):
    return False

  for model in _bundle_models(bundle):
    artifact = getattr(model, "artifact", None)
    metadata = getattr(model, "metadata", None)
    for file_name in (getattr(metadata, "fileName", None), getattr(artifact, "fileName", None)):
      if file_name and not (_MODEL_ROOT / file_name).is_file():
        return False
  return True


def persist_active_bundle(params: Params, bundle) -> None:
  params.put(_ACTIVE_BUNDLE_KEY, bundle.to_dict())
  params.remove(_RUNNER_CACHE_KEY)


def _load_default_bundle_dict() -> dict:
  return json.loads(_DEFAULT_BUNDLE_JSON.read_text())


def _default_bundle_filenames(bundle_dict: dict) -> list[str]:
  names = []
  for model in bundle_dict.get("models", []):
    for artifact in (model.get("metadata"), model.get("artifact")):
      file_name = artifact.get("fileName", "") if isinstance(artifact, dict) else ""
      if file_name:
        names.append(file_name)
  return names


def is_default_bundle(bundle) -> bool:
  return bool(bundle is not None and getattr(bundle, "ref", None) == _DEFAULT_BUNDLE_REF)


def ensure_default_model_files(bundle_dict: dict = None) -> None:
  bundle_dict = bundle_dict if bundle_dict is not None else _load_default_bundle_dict()
  try:
    _MODEL_ROOT.mkdir(parents=True, exist_ok=True)
  except OSError as e:
    cloudlog.exception(f"default_model: cannot create model root: {e}")
    return
  hashes = {artifact["fileName"]: artifact.get("downloadUri", {}).get("sha256", "")
            for model in bundle_dict.get("models", [])
            for artifact in (model.get("metadata"), model.get("artifact")) if artifact and artifact.get("fileName")}
  for file_name in _default_bundle_filenames(bundle_dict):
    src = _DEFAULT_MODEL_DIR / file_name
    dst = _MODEL_ROOT / file_name
    if not src.is_file():
      cloudlog.error(f"default_model: shipped asset missing {src}")
      continue
    try:
      expected = hashes.get(file_name, "")
      if dst.is_file() and dst.stat().st_size == src.stat().st_size:
        if not expected:
          continue
        with dst.open("rb") as stream:
          if hashlib.file_digest(stream, "sha256").hexdigest() == expected.lower():
            continue
      if expected:
        with src.open("rb") as stream:
          if hashlib.file_digest(stream, "sha256").hexdigest() != expected.lower():
            cloudlog.error(f"default_model: shipped asset hash mismatch {src}")
            continue
      staged = dst.with_name(dst.name + ".staging")
      shutil.copy2(src, staged)
      os.replace(staged, dst)
      cloudlog.warning(f"default_model: staged {file_name} into model root")
    except OSError as e:
      cloudlog.exception(f"default_model: failed staging {file_name}: {e}")


def select_default_model(params: Params = None) -> None:
  params = Params() if params is None else params
  bundle_dict = _load_default_bundle_dict()
  ensure_default_model_files(bundle_dict)
  params.remove(_DOWNLOAD_INDEX_KEY)
  params.remove(_PENDING_INDEX_KEY)
  _mark_default_files_cached(bundle_dict)
  params.put(_ACTIVE_BUNDLE_KEY, bundle_dict)
  params.remove(_RUNNER_CACHE_KEY)
  params.put(_RUNNER_CACHE_KEY, _TINYGRAD_RUNNER)
  try:
    if os.path.isfile(_PENDING_MODEL_RESTORE_FILE):
      os.remove(_PENDING_MODEL_RESTORE_FILE)
  except OSError:
    pass


def _mark_default_files_cached(bundle_dict: dict) -> None:
  ready = bool(bundle_dict.get("models"))
  for model in bundle_dict.get("models", []):
    for artifact in (model.get("metadata"), model.get("artifact")):
      if not artifact or not artifact.get("fileName"):
        continue
      path = _MODEL_ROOT / artifact["fileName"]
      if path.is_file() and path.stat().st_size > 0:
        artifact["downloadProgress"] = {"status": "cached", "progress": 100, "eta": 0}
      else:
        ready = False
  if ready:
    bundle_dict["status"] = "cached"


def initialize_active_model(params: Params) -> None:
  """Prepare the shipped runtime before processes start, preserving custom selection/queues."""
  stored = params.get(_ACTIVE_BUNDLE_KEY)
  if not stored:
    seed_default_bundle_if_unset(params)
  elif isinstance(stored, dict) and stored.get("ref") == _DEFAULT_BUNDLE_REF:
    bundle_dict = _load_default_bundle_dict()
    ensure_default_model_files(bundle_dict)
    _mark_default_files_cached(bundle_dict)
    params.put(_ACTIVE_BUNDLE_KEY, bundle_dict)
  # Cache is derived from the saved bundle, never authoritative across updates/selections.
  params.remove(_RUNNER_CACHE_KEY)
  get_active_model_runner(params, force_check=True)


def seed_default_bundle_if_unset(params: Params = None) -> None:
  params = Params() if params is None else params
  if params.get(_ACTIVE_BUNDLE_KEY):
    return
  try:
    bundle_dict = _load_default_bundle_dict()
    ensure_default_model_files(bundle_dict)
    _mark_default_files_cached(bundle_dict)
    params.put(_ACTIVE_BUNDLE_KEY, bundle_dict)
    params.remove(_RUNNER_CACHE_KEY)
    cloudlog.warning("default_model: seeded Default (CD210) as active bundle")
  except Exception as e:
    cloudlog.exception(f"default_model: failed to seed default bundle: {e}")


def get_runtime_bundle_upgrade(bundle, params: Params = None, available_bundles=None):
  params = Params() if params is None else params
  return _find_runtime_upgrade(bundle, params, available_bundles)


_forced_bundle = None


def load_default_model_session():
  """Stage the shipped default model files and return a decoded default bundle
  WITHOUT persisting it as the active selection, so the next boot retries the
  user's own model instead of staying on the default forever."""
  bundle_dict = _load_default_bundle_dict()
  ensure_default_model_files(bundle_dict)
  return ModelBundle(**bundle_dict)


class forced_active_bundle:
  """Make get_active_bundle return `bundle` for the duration of the block, used to
  build a runner for a session-only fallback bundle without touching Params."""
  def __init__(self, bundle):
    self._bundle = bundle

  def __enter__(self):
    global _forced_bundle
    self._prev = _forced_bundle
    _forced_bundle = self._bundle
    return self._bundle

  def __exit__(self, *exc):
    global _forced_bundle
    _forced_bundle = self._prev
    return False


def get_active_bundle(params: Params = None):
  if _forced_bundle is not None:
    return _forced_bundle

  params = Params() if params is None else params

  try:
    active_bundle = params.get(_ACTIVE_BUNDLE_KEY) or {}
    if not active_bundle:
      return None
    is_compatible = globals().get("is_bundle_version_compatible")
    if is_compatible is not None and not is_compatible(active_bundle):
      return None
    bundle = ModelBundle(**active_bundle)
  except Exception:
    return None

  replacement = _find_runtime_upgrade(bundle, params)
  if replacement is not None and replacement is not bundle and bundle_files_ready(replacement):
    persist_active_bundle(params, replacement)
    return replacement

  return bundle


def get_active_model_runner(params: Params = None, force_check=False):
  params = Params() if params is None else params

  active_bundle = get_active_bundle(params)
  if not active_bundle:
    seed_default_bundle_if_unset(params)
    active_bundle = get_active_bundle(params)
    if not active_bundle:
      if params.get(_RUNNER_CACHE_KEY) != str(_TINYGRAD_RUNNER):
        params.put(_RUNNER_CACHE_KEY, _TINYGRAD_RUNNER)
      return _TINYGRAD_RUNNER

  cached_runner_type = params.get(_RUNNER_CACHE_KEY)
  runner_type = _coerce_runner_value(active_bundle.runner)
  if runner_type == _SNPE_RUNNER:
    replacement = _find_runtime_upgrade(active_bundle, params)
    if replacement is not None and replacement is not active_bundle and bundle_files_ready(replacement):
      persist_active_bundle(params, replacement)
      runner_type = _coerce_runner_value(replacement.runner)
    else:
      if replacement is not None and getattr(replacement, "index", None) is not None and params.get(_DOWNLOAD_INDEX_KEY) is None:
        params.put(_DOWNLOAD_INDEX_KEY, int(replacement.index))
        cloudlog.warning(f"Queued tinygrad migration for retired bundle {getattr(active_bundle, 'internalName', '<unknown>')}")
      runner_type = _TINYGRAD_RUNNER

  if cached_runner_type != runner_type:
    params.put(_RUNNER_CACHE_KEY, int(runner_type))

  return runner_type


def get_selected_model_name(params: Params = None) -> str:
  params = Params() if params is None else params
  try:
    stored = params.get(_ACTIVE_BUNDLE_KEY)
    if not stored:
      return "Default (CD210)"
    if not isinstance(stored, dict) or not stored.get("ref"):
      return "Model unavailable"
    if stored["ref"] == _DEFAULT_BUNDLE_REF:
      return stored.get("displayName") or "Default (CD210)"
    return stored.get("internalName") or stored.get("displayName") or "Model unavailable"
  except Exception:
    return "Model unavailable"


def get_cached_model_bundles(params: Params = None) -> list:
  params = Params() if params is None else params
  try:
    allow_superuser = bool(params.get_bool("Konn3ktOwnerSuperuser"))
  except Exception:
    allow_superuser = False
  try:
    return [bundle for bundle in _load_cached_manifest_bundles(params)
            if getattr(bundle, "environment", "") != "superuser" or allow_superuser]
  except Exception:
    return []
