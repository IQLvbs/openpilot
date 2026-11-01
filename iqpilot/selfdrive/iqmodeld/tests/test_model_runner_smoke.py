"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from tinygrad.tensor import Tensor

import iqpilot.selfdrive.iqmodeld.models.helpers as bundle_helpers
import iqpilot.selfdrive.iqmodeld.models.runners.model_runner as model_runner_mod
import iqpilot.selfdrive.iqmodeld.models.runners.tinygrad.tinygrad_runner as tinygrad_runner_mod
from iqpilot.selfdrive.iqmodeld.models.runners.model_runner import ModelType
from iqpilot.selfdrive.iqmodeld.models.runners.tinygrad.tinygrad_runner import TinygradRunner


LOCAL_MODEL_DIR = Path(__file__).resolve().parents[1] / "default_model"


@dataclass
class _TypeWrap:
  raw: int


@dataclass
class _Artifact:
  fileName: str


class _Model:
  def __init__(self, model_type: int, artifact_name: str, metadata_name: str):
    self.type = _TypeWrap(model_type)
    self.artifact = _Artifact(artifact_name)
    self.metadata = _Artifact(metadata_name)


class _Bundle:
  def __init__(self, models: list[_Model], is_20hz: bool = False):
    self.models = models
    self.is20hz = is_20hz


def _seed_runner_inputs(runner: TinygradRunner) -> None:
  for name, shape in runner.input_shapes.items():
    runner.inputs[name] = Tensor(
      np.zeros(shape, dtype=np.float32),
      device=runner.input_to_device[name],
      dtype=runner.input_to_dtype[name],
    ).realize()


@pytest.mark.tici
def test_local_tinygrad_models_execute(monkeypatch):
  bundle = _Bundle([
    _Model(ModelType.vision, "driving_vision_c210m_tinygrad.pkl", "driving_vision_c210m_metadata.pkl"),
    _Model(ModelType.policy, "driving_policy_c210m_tinygrad.pkl", "driving_policy_c210m_metadata.pkl"),
  ])

  monkeypatch.setattr(bundle_helpers, "get_active_bundle", lambda params=None: bundle, raising=False)
  monkeypatch.setattr(model_runner_mod, "_fetch_bundle", lambda params=None: bundle)
  monkeypatch.setattr(tinygrad_runner_mod, "CUSTOM_MODEL_PATH", str(LOCAL_MODEL_DIR), raising=False)
  monkeypatch.setattr(model_runner_mod, "CUSTOM_MODEL_PATH", str(LOCAL_MODEL_DIR), raising=False)

  vision_runner = TinygradRunner(ModelType.vision)
  _seed_runner_inputs(vision_runner)
  vision_outputs = vision_runner.run_model()
  assert "pose" in vision_outputs
  assert "lane_lines" in vision_outputs

  policy_runner = TinygradRunner(ModelType.policy)
  _seed_runner_inputs(policy_runner)
  policy_outputs = policy_runner.run_model()
  assert "plan" in policy_outputs
  assert "desire_state" in policy_outputs


def test_custom_named_supercombo_is_classified_by_type_not_filename():
  waypoint = _Bundle([_Model(
    ModelType.supercombo,
    "IQ_MEMSTP2g10TB7_step3000_a630_navsplit_tinygrad.pkl.iqc",
    "IQ_MEMSTP2g10TB7_step3000_a630_navsplit_metadata.pkl.iqc",
  )])
  assert model_runner_mod._is_supercombo_bundle(waypoint) is True
  assert model_runner_mod._is_fused_bundle(waypoint) is False

  conventional = _Bundle([_Model(
    ModelType.supercombo,
    "driving_supercombo_novaspark_tinygrad.pkl.iqc",
    "driving_supercombo_novaspark_metadata.pkl.iqc",
  )])
  assert model_runner_mod._is_supercombo_bundle(conventional) is True


def test_fused_artifact_carrying_supercombo_type_is_not_classified_supercombo():
  fused = _Bundle([_Model(
    ModelType.supercombo,
    "driving_fused_tinygrad.pkl",
    "driving_fused_metadata.pkl",
  )])
  assert model_runner_mod._is_fused_bundle(fused) is True
  assert model_runner_mod._is_supercombo_bundle(fused) is False
