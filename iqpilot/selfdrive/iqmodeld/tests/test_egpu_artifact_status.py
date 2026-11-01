"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""
import inspect

import pytest

from iqpilot.selfdrive.iqmodeld import egpu_artifact_status as status_mod
from iqpilot.selfdrive.iqmodeld.egpu_artifact_status import LOCAL_ARTIFACTS, PRECOMPILED_FIELDS, egpu_artifact_status

META = {"key": "cinque", "sha256": "ab" * 32}


@pytest.fixture(autouse=True)
def model_root(tmp_path, monkeypatch):
  from iqpilot.system.hardware.hw import Paths
  monkeypatch.setattr(Paths, "model_root", staticmethod(lambda: str(tmp_path)))
  return tmp_path


def test_nothing_hosted_or_local_compiles_on_the_dock():
  s = egpu_artifact_status(META)
  assert (s.on_device, s.precompiled, s.download_bytes) == (False, False, 0)


@pytest.mark.parametrize("path_fn", LOCAL_ARTIFACTS)
def test_any_artifact_the_daemon_loads_counts_as_on_device(path_fn):
  open(path_fn(META), "wb").close()
  assert egpu_artifact_status(META).on_device


def test_warp_on_dock_artifact_size_wins_over_older_formats():
  meta = {**META, "egpu_policy_artifact": {"size": 1}, "egpu_oob_artifact": {"size": 2},
          "egpu_model_oob_artifact": {"size": 3_200_000_000}}
  s = egpu_artifact_status(meta)
  assert s.precompiled and s.download_bytes == 3_200_000_000


@pytest.mark.parametrize("field", PRECOMPILED_FIELDS)
def test_each_hosted_format_is_precompiled(field):
  s = egpu_artifact_status({**META, field: {"size": 800_000_000}})
  assert s.precompiled and s.download_bytes == 800_000_000


def test_daemon_artifact_order_matches():
  from iqpilot.selfdrive.iqmodeld import iqegpumodeld
  src = inspect.getsource(iqegpumodeld._ensure_artifact)
  fields = [f for f in PRECOMPILED_FIELDS if f'"{f}"' in src]
  assert fields == list(PRECOMPILED_FIELDS)
  assert sorted(fields, key=lambda f: src.index(f'"{f}"')) == list(PRECOMPILED_FIELDS)
  for path_fn in LOCAL_ARTIFACTS:
    assert f"{path_fn.__name__}(meta)" in src
  assert status_mod.egpu_model_oob_pkl_path is LOCAL_ARTIFACTS[0]
