"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from iqpilot.cereal import custom
from iqpilot.selfdrive.iqmodeld.models import helpers as model_helpers
from iqpilot.selfdrive.iqmodeld.models.runners import model_runner as model_runner_mod
from iqpilot.selfdrive.iqmodeld.models.runners.tinygrad import supercombo_runner as supercombo_runner_mod
from iqpilot.selfdrive.iqmodeld.temporal_state import MODEL_INPUT_SPEC
from iqpilot.selfdrive.iqmodeld.models.runners.tinygrad.supercombo_runner import (
  TinygradSupercomboRunner,
)


class _Captured:
  def __init__(self, expected_names):
    self.expected_names = expected_names


class _FakeJit:
  def __init__(self, expected_names):
    self.captured = _Captured(expected_names)


class _RecordingJit:
  def __init__(self, expected_names, output_len=0):
    self.captured = _Captured(expected_names)
    self.output_len = output_len
    self.calls = []

  def __call__(self, **kwargs):
    from tinygrad.tensor import Tensor
    self.calls.append({name: value.numpy().copy() if hasattr(value, "numpy") else value for name, value in kwargs.items()})
    if self.output_len:
      return (Tensor(np.zeros(self.output_len, dtype=np.float32)),)
    return object()


class _Boom:
  def __init__(self, err: Exception):
    self.err = err

  def __call__(self, *args, **kwargs):
    raise self.err


class _FakeParams:
  def __init__(self, active_bundle=None):
    self.store = {}
    if active_bundle is not None:
      self.store["ModelManager_ActiveBundle"] = active_bundle

  def get(self, key):
    return self.store.get(key)

  def put(self, key, value):
    self.store[key] = value

  def remove(self, key):
    self.store.pop(key, None)


def test_verify_artifact_file_deletes_stale_cached_pkl(tmp_path: Path):
  pkl_path = tmp_path / "driving_supercombo_guard.pkl"
  pkl_path.write_bytes(b"stale-pkl")

  runner = TinygradSupercomboRunner.__new__(TinygradSupercomboRunner)
  runner._pkl_path = str(pkl_path)
  runner._expected_sha256 = hashlib.sha256(b"fresh-pkl").hexdigest()

  with pytest.raises(RuntimeError, match="SHA mismatch"):
    runner._verify_artifact_file()

  assert not pkl_path.exists()


def test_validate_jit_names_accepts_current_runtime_contract():
  runner = TinygradSupercomboRunner.__new__(TinygradSupercomboRunner)
  runner._pkl_path = "/tmp/does-not-matter.pkl"
  runner._expected_sha256 = ""
  runner._run_policy = _FakeJit(['warped', 'img_q', 'big_img_q', 'feat_q', 'desire_q', 'packed_npy_inputs'])
  runner._warp_jits = {
    (1344, 760): _FakeJit(['tfm', 'big_tfm', 'frame', 'big_frame']),
  }

  runner._validate_jit_names()


def test_validate_jit_names_raises_clear_error_for_contract_mismatch():
  runner = TinygradSupercomboRunner.__new__(TinygradSupercomboRunner)
  runner._pkl_path = "/tmp/does-not-matter.pkl"
  runner._expected_sha256 = ""
  runner._run_policy = _FakeJit(['img', 'big_img', 'feat_q', 'desire_q', 'desire', 'traffic_convention', 'action_t'])
  runner._warp_jits = {
    (1344, 760): _FakeJit(['img_q', 'big_img_q', 'tfm', 'big_tfm', 'frame', 'big_frame']),
  }

  with pytest.raises(RuntimeError, match="JIT argument mismatch"):
    runner._validate_jit_names()


def test_handle_runtime_jit_mismatch_deletes_stale_cached_pkl(tmp_path: Path):
  pkl_path = tmp_path / "driving_supercombo_guard.pkl"
  pkl_path.write_bytes(b"stale-pkl")

  runner = TinygradSupercomboRunner.__new__(TinygradSupercomboRunner)
  runner._pkl_path = str(pkl_path)
  runner._expected_sha256 = hashlib.sha256(b"fresh-pkl").hexdigest()

  with pytest.raises(RuntimeError, match="runtime JIT mismatch with stale cached SHA"):
    runner._handle_runtime_jit_mismatch(RuntimeError("args mismatch in JIT: stale bundle"))

  assert not pkl_path.exists()


def test_handle_runtime_jit_mismatch_raises_clear_error_without_sha_mismatch(tmp_path: Path):
  pkl_path = tmp_path / "driving_supercombo_guard.pkl"
  pkl_path.write_bytes(b"fresh-pkl")

  runner = TinygradSupercomboRunner.__new__(TinygradSupercomboRunner)
  runner._pkl_path = str(pkl_path)
  runner._expected_sha256 = hashlib.sha256(b"fresh-pkl").hexdigest()

  with pytest.raises(RuntimeError, match="runtime JIT mismatch"):
    runner._handle_runtime_jit_mismatch(RuntimeError("args mismatch in JIT: wrong contract"))


def test_schedule_active_bundle_redownload_sets_download_index(monkeypatch: pytest.MonkeyPatch):
  params = _FakeParams({"index": 81})
  monkeypatch.setattr(supercombo_runner_mod, "Params", lambda: params)

  runner = TinygradSupercomboRunner.__new__(TinygradSupercomboRunner)
  msg = runner._schedule_active_bundle_redownload()

  assert params.get("ModelManager_DownloadIndex") == "81"
  assert msg == "; scheduled automatic re-download of the active model"


def test_no_active_bundle_seeds_default_tinygrad(monkeypatch: pytest.MonkeyPatch):
  monkeypatch.setattr(model_helpers, "ensure_default_model_files", lambda *a, **k: None)
  params = _FakeParams()

  runner = model_helpers.get_active_model_runner(params)

  assert runner == custom.IQModelManager.Runner.tinygrad
  active = params.get("ModelManager_ActiveBundle")
  assert active is not None and active.get("ref") == "default"


def test_select_default_model_clears_custom_download_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
  pending_restore = tmp_path / "pending_model_restore"
  pending_restore.write_text("Pop")
  monkeypatch.setattr(model_helpers, "_PENDING_MODEL_RESTORE_FILE", str(pending_restore))
  monkeypatch.setattr(model_helpers, "ensure_default_model_files", lambda *a, **k: None)

  params = _FakeParams({"index": 81, "ref": "pop"})
  params.put("ModelManager_DownloadIndex", "81")
  params.put("ModelRunnerTypeCache", int(custom.IQModelManager.Runner.tinygrad))

  model_helpers.select_default_model(params)

  assert params.get("ModelManager_DownloadIndex") is None
  active = params.get("ModelManager_ActiveBundle")
  assert active is not None and active.get("ref") == "default"
  assert int(params.get("ModelRunnerTypeCache")) == int(custom.IQModelManager.Runner.tinygrad)
  assert not pending_restore.exists()


def test_seed_default_bundle_runs_while_a_download_is_queued(monkeypatch: pytest.MonkeyPatch):
  monkeypatch.setattr(model_helpers, "ensure_default_model_files", lambda *a, **k: None)

  params = _FakeParams()
  params.put("ModelManager_DownloadIndex", "81")

  model_helpers.seed_default_bundle_if_unset(params)

  active = params.get("ModelManager_ActiveBundle")
  assert active is not None and active.get("ref") == "default"
  assert params.get("ModelManager_DownloadIndex") == "81"


def test_seed_default_bundle_leaves_an_existing_active_bundle_alone(monkeypatch: pytest.MonkeyPatch):
  monkeypatch.setattr(model_helpers, "ensure_default_model_files", lambda *a, **k: None)

  params = _FakeParams({"index": 81, "ref": "pop"})
  params.put("ModelManager_DownloadIndex", "81")

  model_helpers.seed_default_bundle_if_unset(params)

  assert params.get("ModelManager_ActiveBundle").get("ref") == "pop"
  assert params.get("ModelManager_DownloadIndex") == "81"


def test_default_model_is_not_resolved_to_manifest_pop_bundle():
  pop_bundle = type("Bundle", (), {"internalName": "Pop (Default)", "displayName": "Pop (Default)"})()

  assert model_helpers.get_default_model_bundle([pop_bundle]) is None


def test_verify_artifact_file_schedules_redownload_for_stale_cached_pkl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
  params = _FakeParams({"index": 81})
  monkeypatch.setattr(supercombo_runner_mod, "Params", lambda: params)

  pkl_path = tmp_path / "driving_supercombo_guard.pkl"
  pkl_path.write_bytes(b"stale-pkl")

  runner = TinygradSupercomboRunner.__new__(TinygradSupercomboRunner)
  runner._pkl_path = str(pkl_path)
  runner._expected_sha256 = hashlib.sha256(b"fresh-pkl").hexdigest()

  with pytest.raises(RuntimeError, match="scheduled automatic re-download"):
    runner._verify_artifact_file()

  assert params.get("ModelManager_DownloadIndex") == "81"
  assert not pkl_path.exists()


def test_run_fused_converts_raw_warp_jit_mismatch_to_runtime_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
  pkl_path = tmp_path / "driving_supercombo_guard.pkl"
  pkl_path.write_bytes(b"fresh-pkl")

  runner = TinygradSupercomboRunner.__new__(TinygradSupercomboRunner)
  runner._pkl_path = str(pkl_path)
  runner._expected_sha256 = hashlib.sha256(b"fresh-pkl").hexdigest()
  runner._frame_skip = 4
  runner._cam = (1344, 760)
  runner._queues = {
    "tfm": object(),
    "big_tfm": object(),
    "img_q": object(),
    "big_img_q": object(),
    "feat_q": object(),
    "desire_q": object(),
    "packed_npy_inputs": object(),
  }
  runner._npy = {
    "tfm": [0.0],
    "big_tfm": [0.0],
    "desire": [0.0],
    "prev_feat": [0.0],
  }
  runner._prev_desire = [0.0]
  runner._warp_jits = {
    (1344, 760): _Boom(RuntimeError("args mismatch in JIT: self.captured.expected_names=['big_frame'] != ['frame']")),
  }
  runner._run_policy = _FakeJit(["warped", "img_q", "big_img_q", "feat_q", "desire_q", "packed_npy_inputs"])
  runner._hidden_slice = slice(0, 1)
  runner._slices = {"out": slice(0, 1)}
  runner._parser = type("P", (), {"parse_vision_outputs": staticmethod(lambda sliced: sliced)})()
  runner._frame_tensor = lambda *args, **kwargs: object()

  monkeypatch.setattr(TinygradSupercomboRunner, "_ensure_queues", lambda self, cam_w, cam_h: None)

  class _Buf:
    width = 1344
    height = 760
    data = memoryview(b"\x00")

  with pytest.raises(RuntimeError, match="runtime JIT mismatch"):
    runner.run_fused(
      {"img": _Buf(), "big_img": _Buf()},
      {"img": [0.0], "big_img": [0.0]},
      {},
    )


POLICY_JIT_NAMES = ["warped", "img_q", "big_img_q", "feat_q", "desire_q", "packed_npy_inputs"]
WARP_JIT_NAMES = ["tfm", "big_tfm", "frame", "big_frame"]
PACKED_STANDARD_LEN = 8 + 2 + 2 + 512
OUTPUT_LEN = 2580


class _Buf:
  width = 1928
  height = 1208
  data = memoryview(b"\x00")


def _packed_pkl_runner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extra_inputs: dict):
  input_shapes = {name: shape for name, (shape, _) in MODEL_INPUT_SPEC.items()}
  input_shapes.update(extra_inputs)
  pkl_path = tmp_path / "driving_supercombo_fake.pkl"
  with open(pkl_path, "wb") as f:
    pickle.dump({
      "metadata": {
        "input_shapes": input_shapes,
        "output_slices": {"plan": slice(917, 1907), "hidden_state": slice(2066, 2578), "pad": slice(-2, None)},
      },
      "frame_skip": 4,
      "run_policy": _RecordingJit(POLICY_JIT_NAMES, OUTPUT_LEN),
      (1928, 1208): _RecordingJit(WARP_JIT_NAMES),
    }, f)

  spec = SimpleNamespace(model=SimpleNamespace(artifact=SimpleNamespace(fileName=pkl_path.name, downloadUri=SimpleNamespace(sha256=""))))
  monkeypatch.setattr(model_runner_mod.ModelRunner, "__init__", lambda self: setattr(self, "models", {"supercombo": spec}))
  monkeypatch.setattr(supercombo_runner_mod, "CUSTOM_MODEL_PATH", str(tmp_path))
  runner = TinygradSupercomboRunner()
  runner._frame_tensor = lambda key, buf: object()
  runner._parser = type("P", (), {"parse_vision_outputs": staticmethod(lambda sliced: sliced)})()
  return runner


def _run_once(runner, numpy_inputs):
  runner.run_fused({"img": _Buf(), "big_img": _Buf()},
                   {"img": np.eye(3, dtype=np.float32), "big_img": np.eye(3, dtype=np.float32)},
                   numpy_inputs)
  return runner._run_policy.calls[-1]["packed_npy_inputs"]


def test_packed_pkl_without_extra_inputs_keeps_the_standard_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
  runner = _packed_pkl_runner(tmp_path, monkeypatch, {})

  packed = _run_once(runner, {
    "desire_pulse": np.zeros(8, dtype=np.float32),
    "traffic_convention": np.array([0.0, 1.0], dtype=np.float32),
    "action_t": np.array([0.25, 0.5], dtype=np.float32),
  })

  assert list(runner._npy) == ["tfm", "big_tfm", "desire", "traffic_convention", "action_t", "prev_feat"]
  assert packed.shape == (PACKED_STANDARD_LEN,)
  np.testing.assert_array_equal(packed[8:10], [0.0, 1.0])
  np.testing.assert_array_equal(packed[10:12], [0.25, 0.5])


def test_packed_pkl_extra_inputs_ride_the_packed_tail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
  runner = _packed_pkl_runner(tmp_path, monkeypatch, {
    "nav_intent": (1, 32), "route_line": (1, 32, 2), "drive_profile": (1, 5),
  })
  assert runner.input_shapes["nav_intent"] == (1, 32)
  assert runner.input_shapes["route_line"] == (1, 32, 2)
  assert runner.input_shapes["drive_profile"] == (1, 5)

  nav_intent = np.arange(32, dtype=np.float32).reshape(1, 32)
  route_line = np.arange(64, dtype=np.float32).reshape(1, 32, 2) / 10
  drive_profile = np.array([[0.0, 0.0, 1.0, 0.0, 0.25]], dtype=np.float32)
  packed = _run_once(runner, {
    "desire_pulse": np.zeros(8, dtype=np.float32),
    "traffic_convention": np.array([1.0, 0.0], dtype=np.float32),
    "action_t": np.array([0.25, 0.5], dtype=np.float32),
    "nav_intent": nav_intent,
    "route_line": route_line,
    "drive_profile": drive_profile,
  })

  assert packed.shape == (PACKED_STANDARD_LEN + 32 + 64 + 5,)
  np.testing.assert_array_equal(packed[8:10], [1.0, 0.0])
  np.testing.assert_array_equal(packed[PACKED_STANDARD_LEN:PACKED_STANDARD_LEN + 32], nav_intent.ravel())
  np.testing.assert_array_equal(packed[PACKED_STANDARD_LEN + 32:PACKED_STANDARD_LEN + 96], route_line.ravel())
  np.testing.assert_array_equal(packed[PACKED_STANDARD_LEN + 96:], drive_profile.ravel())

  packed = _run_once(runner, {"desire_pulse": np.zeros(8, dtype=np.float32), "nav_intent": np.zeros((1, 32), dtype=np.float32)})
  np.testing.assert_array_equal(packed[PACKED_STANDARD_LEN:PACKED_STANDARD_LEN + 32], 0.0)
  np.testing.assert_array_equal(packed[PACKED_STANDARD_LEN + 32:PACKED_STANDARD_LEN + 96], route_line.ravel())
  np.testing.assert_array_equal(packed[PACKED_STANDARD_LEN + 96:], drive_profile.ravel())


def test_wait_for_model_load_memory_raises_when_insufficient(monkeypatch):
  monkeypatch.setattr(model_runner_mod, "mem_available_mb", lambda: 100)
  with pytest.raises(RuntimeError):
    model_runner_mod.wait_for_model_load_memory(500, timeout_s=0.2)


def test_wait_for_model_load_memory_passes_when_enough(monkeypatch):
  monkeypatch.setattr(model_runner_mod, "mem_available_mb", lambda: 4096)
  model_runner_mod.wait_for_model_load_memory(500, timeout_s=0.2)


def test_load_big_model_pickle_refuses_before_loading_when_memory_short(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
  pkl = tmp_path / "big.pkl"
  pkl.write_bytes(b"x" * (1024 * 1024))
  loaded = []
  monkeypatch.setattr(model_runner_mod, "mem_available_mb", lambda: 100)
  monkeypatch.setattr(model_runner_mod, "MODEL_LOAD_MEMORY_FALLBACK_S", 0.0)
  monkeypatch.setattr(model_runner_mod.time, "sleep", lambda _s: None)
  monkeypatch.setattr(model_runner_mod, "load_pickle", lambda p: loaded.append(p))
  with pytest.raises(RuntimeError, match="insufficient memory"):
    model_runner_mod.load_big_model_pickle(str(pkl))
  assert loaded == []


def test_load_big_model_pickle_loads_when_memory_available(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
  pkl = tmp_path / "big.pkl"
  pkl.write_bytes(b"x" * (1024 * 1024))
  monkeypatch.setattr(model_runner_mod, "mem_available_mb", lambda: 8192)
  monkeypatch.setattr(model_runner_mod, "load_pickle", lambda p: {"ok": p})
  assert model_runner_mod.load_big_model_pickle(str(pkl)) == {"ok": str(pkl)}
