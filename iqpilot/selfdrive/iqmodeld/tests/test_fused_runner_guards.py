"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass

import numpy as np
import pytest

from iqpilot.selfdrive.iqmodeld.models.runners import model_runner as model_runner_mod
from iqpilot.selfdrive.iqmodeld.models.runners.model_runner import ModelType
from iqpilot.selfdrive.iqmodeld.models.runners.tinygrad import fused_runner as fused_mod


class _View:
  def __init__(self, shape):
    self.shape = shape


class _Captured:
  def __init__(self, expected_names, expected_input_info):
    self.expected_names = expected_names
    self.expected_input_info = expected_input_info


class _FakeJit:
  def __init__(self, expected_names, expected_input_info):
    self.captured = _Captured(expected_names, expected_input_info)

  def __call__(self, **kwargs):
    raise AssertionError("policy jit should not run in this test")


class _FakeTensor:
  def __init__(self, arr, device=None):
    self.shape = tuple(np.asarray(arr).shape)

  def contiguous(self):
    return self

  def realize(self):
    return self


class _FakeDevice:
  DEFAULT = "FAKE"


@dataclass
class _Type:
  raw: int


@dataclass
class _Artifact:
  fileName: str


class _Model:
  def __init__(self, file_name):
    self.type = _Type(ModelType.vision)
    self.artifact = _Artifact(file_name)
    self.metadata = None


class _Bundle:
  def __init__(self, file_name):
    self.models = [_Model(file_name)]
    self.is20hz = True


POLICY_INPUTS = ["action_t", "big_img", "desire", "desire_q", "feat_q", "img", "traffic_convention"]
POLICY_SHAPES = {
  "action_t": (1, 2), "big_img": (1, 12, 128, 256), "desire": (1, 8), "desire_q": (1, 100, 8),
  "feat_q": (1, 99, 512), "img": (1, 12, 128, 256), "traffic_convention": (1, 2),
}


def _write_fused_pkl(path, policy_inputs):
  info = [(_View(POLICY_SHAPES[n]), (), None, "NPY") for n in policy_inputs]
  role_meta = {
    "input_shapes": {"desire_pulse": (1, 100, 8), "traffic_convention": (1, 2), "features_buffer": (1, 99, 512)},
    "output_slices": {},
  }
  blob = {
    "metadata": {
      "vision": {"input_shapes": {"img": (1, 12, 128, 256), "big_img": (1, 12, 128, 256)}, "output_slices": {}},
      "on_policy": role_meta,
      "off_policy": role_meta,
    },
    "run_policy": _FakeJit(policy_inputs, info),
    "frame_skip": 4,
    (1928, 1208): _FakeJit(["frame"], [(_View((1,)), (), None, "NPY")]),
  }
  with open(path, "wb") as f:
    pickle.dump(blob, f)


@pytest.fixture
def fused_runner(tmp_path, monkeypatch):
  def _build(policy_inputs):
    name = "driving_fused_test.pkl"
    _write_fused_pkl(tmp_path / name, policy_inputs)
    monkeypatch.setattr(model_runner_mod, "_fetch_bundle", lambda params=None: _Bundle(name))
    monkeypatch.setattr(fused_mod, "CUSTOM_MODEL_PATH", str(tmp_path))
    monkeypatch.setattr(fused_mod, "_tinygrad_imports", lambda: (_FakeTensor, _FakeDevice))
    return fused_mod.TinygradFusedRunner()
  return _build


def test_action_t_allocated_when_only_the_jit_declares_it(fused_runner):
  runner = fused_runner(POLICY_INPUTS)
  assert "action_t" not in runner._on_meta["input_shapes"]

  runner._ensure_queues(1928, 1208)

  assert runner._npy_buffers["action_t"].shape == POLICY_SHAPES["action_t"]
  assert runner._npy_buffers["traffic_convention"].shape == POLICY_SHAPES["traffic_convention"]


def test_action_t_absent_when_the_jit_does_not_take_it(fused_runner):
  runner = fused_runner([n for n in POLICY_INPUTS if n != "action_t"])

  runner._ensure_queues(1928, 1208)

  assert "action_t" not in runner._npy_buffers
