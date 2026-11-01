"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""
import importlib

import pytest

from iqpilot.models_private_src import big_catalog

ART = {"sha256": "ab" * 32, "size": 809623713, "hf_path": "egpu/x.pkl"}


def _bundle(**artifacts):
  return {"short_name": "cinque", "model_name": "big_driving_supercombo", "source": {"sha256": "cd" * 32, "size": 1},
          "wire": {"output_len": 18452, "frame_skip": 4, "output_slices": {}}, **artifacts}


def test_the_warp_on_dock_build_reaches_the_model_meta():
  # route ed2717e2c1912f87/5 ran cinque warp-on-device at 13 Hz because this field never left the catalog
  meta = big_catalog.bundle_to_meta(_bundle(egpu_model_oob_artifact=ART, egpu_oob_artifact=dict(ART, size=1)))
  assert meta["egpu_model_oob_artifact"] == ART
  assert meta["egpu_oob_artifact"]["size"] == 1


def test_every_egpu_artifact_the_daemon_can_pick_is_carried():
  bundle = _bundle(**{field: dict(ART) for field in big_catalog.EGPU_ARTIFACT_FIELDS})
  meta = big_catalog.bundle_to_meta(bundle)
  assert all(meta.get(field) == ART for field in big_catalog.EGPU_ARTIFACT_FIELDS)


@pytest.fixture
def egpu_prefetch(monkeypatch, tmp_path):
  # the module sets XDG_CACHE_HOME=/data/.cache at import; pin it first so the setdefault cannot leak into other tests
  monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
  return importlib.import_module("iqpilot.selfdrive.iqmodeld.egpu_prefetch")


def test_the_prefetcher_wants_the_same_build_the_daemon_loads_first(egpu_prefetch):
  assert egpu_prefetch.PREFETCH_ORDER[0] == "egpu_model_oob_artifact"
  assert set(egpu_prefetch.PREFETCH_ORDER) <= set(big_catalog.EGPU_ARTIFACT_FIELDS)


def test_prefetch_downloads_the_warp_on_dock_build_when_offered(egpu_prefetch, monkeypatch, tmp_path):
  meta = dict(big_catalog.bundle_to_meta(_bundle(egpu_model_oob_artifact=ART, egpu_oob_artifact=dict(ART))), key="cinque")
  asked = []

  class P:
    def get_bool(self, k):
      return False

    def put(self, *a):
      pass

  monkeypatch.setattr(egpu_prefetch, "_selected_meta", lambda params: meta)
  monkeypatch.setattr(egpu_prefetch, "ARTIFACT_PATHS", {f: (lambda m, f=f: str(tmp_path / f)) for f in egpu_prefetch.PREFETCH_ORDER})
  monkeypatch.setattr(egpu_prefetch, "download_precompiled", lambda m, progress_cb=None, field=None: asked.append(field) or "ok")
  assert egpu_prefetch.prefetch_once(P()) is True
  assert asked == ["egpu_model_oob_artifact"]


def test_frozen_feed_warp_on_dock_builds_are_never_loaded(monkeypatch, tmp_path):
  from iqpilot.selfdrive.iqmodeld import egpu_helpers
  from iqpilot.selfdrive.iqmodeld.model_cache import is_runtime_artifact
  from iqpilot.system.hardware.hw import Paths
  monkeypatch.setattr(Paths, "model_root", staticmethod(lambda: str(tmp_path)))
  meta = {"key": "big_nav", "sha256": "8a5bc46c" + "0" * 56}
  stale = tmp_path / "egpu_big_nav_8a5bc46c_amd_model_oob.pkl"
  stale.write_bytes(b"frozen")
  live = egpu_helpers.egpu_model_oob_pkl_path(meta)
  assert live != str(stale) and live.endswith("_amd_model_oob2.pkl")
  egpu_helpers.drop_frozen_model_oob(meta)
  egpu_helpers.drop_frozen_model_oob(meta)
  assert not stale.exists()
  assert is_runtime_artifact("egpu_big_nav_8a5bc46c_amd_model_oob2.pkl") and not is_runtime_artifact(stale.name)
