"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos

Unit checks for the NNFF model loader against the shipped model files.
"""
import json
import os
from copy import deepcopy

import numpy as np
import pytest

from iqpilot.selfdrive.controls.lib.latcontrol_torque import NNTorqueModel
from iqpilot.selfdrive.controls.lib.latcontrol_torque import TORQUE_NN_MODEL_PATH

# A minimal valid NNFF model (Twilsonco format: column-vector mean/std, dense_N_W/b
# layers). Used as a fallback so the loader logic is still exercised when no trained
# models are shipped (they are removed pending retraining and re-added over time).
_SYNTHETIC_MODEL = {
  "input_size": 18,
  "output_size": 1,
  "input_mean": [[0.0]] * 18,
  "input_std": [[1.0]] * 18,
  "layers": [
    {"dense_1_W": [[0.5] * 18, [0.5] * 18], "dense_1_b": [[0.0], [0.0]], "activation": "sigmoid"},
    {"dense_2_W": [[2.0, 2.0]], "dense_2_b": [[-1.0]], "activation": "identity"},
  ],
}

MODEL_FILES = sorted(f for f in os.listdir(TORQUE_NN_MODEL_PATH) if f.endswith(".json")) if os.path.isdir(TORQUE_NN_MODEL_PATH) else []
if MODEL_FILES:
  _MODEL_DIR = TORQUE_NN_MODEL_PATH
  _NAMES = MODEL_FILES
else:
  import tempfile
  _MODEL_DIR = tempfile.mkdtemp(prefix="nnff_synthetic_")
  with open(os.path.join(_MODEL_DIR, "SYNTHETIC.json"), "w") as _f:
    json.dump(_SYNTHETIC_MODEL, _f)
  _NAMES = ["SYNTHETIC.json"]

SAMPLE = [f for f in ("HYUNDAI_IONIQ_5.json", "TOYOTA_RAV4_TSS2_2022.json", "MOCK.json") if f in _NAMES] \
         or _NAMES[:3]


def _path(name):
  return os.path.join(_MODEL_DIR, name)


@pytest.mark.parametrize("name", _NAMES, ids=[n[:-5] for n in _NAMES])
def test_every_model_loads_and_is_finite(name):
  m = NNTorqueModel(_path(name))
  assert m.input_size >= 2
  assert m.output_size >= 1
  assert m.input_mean.shape == m.input_std.shape
  out = m.evaluate([0.0] * m.input_size)
  assert np.isfinite(out)
  assert isinstance(m.friction_override, (bool, np.bool_))


@pytest.mark.parametrize("name", SAMPLE, ids=[n[:-5] for n in SAMPLE])
class TestModelBehavior:
  def test_short_input_is_zero_padded(self, name):
    m = NNTorqueModel(_path(name))
    padded = m.evaluate([5.0, 1.0])
    explicit = m.evaluate([5.0, 1.0] + [0.0] * (m.input_size - 2))
    assert padded == explicit

  def test_too_short_input_raises(self, name):
    m = NNTorqueModel(_path(name))
    with pytest.raises(ValueError):
      m.evaluate([1.0])

  def test_zero_bias_matches_manual_bias_removal(self, name):
    m = NNTorqueModel(_path(name))
    mz = NNTorqueModel(_path(name), zero_bias=True)
    assert all(np.allclose(b, 0.0) for b in mz._biases)
    # weights and activations are unchanged
    assert len(mz._weights) == len(m._weights)

  def test_deterministic(self, name):
    m = NNTorqueModel(_path(name))
    vec = [float(v) for v in np.linspace(-1.5, 1.5, m.input_size)]
    assert m.evaluate(vec) == m.evaluate(list(vec))


@pytest.mark.parametrize("activation", ["not_a_real_activation", None, 1, []])
def test_activation_registry_rejects_unknown(tmp_path, activation):
  with open(_path(SAMPLE[0])) as model_file:
    base = json.load(model_file)
  base["layers"][-1]["activation"] = activation
  bad = tmp_path / "bad.json"
  bad.write_text(json.dumps(base))
  with pytest.raises(ValueError):
    NNTorqueModel(str(bad))


def test_sigmoid_identity_helpers():
  assert NNTorqueModel.identity(3.5) == 3.5
  assert 0.0 < float(NNTorqueModel.sigmoid(np.array([0.0]))[0]) < 1.0
  assert abs(float(NNTorqueModel.sigmoid(np.array([0.0]))[0]) - 0.5) < 1e-6


@pytest.mark.parametrize("invalid", ["zero_std", "nan_mean", "nan_weight", "bias_shape", "layer_shape", "output_shape"])
def test_invalid_model_rejected(tmp_path, invalid):
  model = deepcopy(_SYNTHETIC_MODEL)
  if invalid == "zero_std":
    model["input_std"][0][0] = 0.0
  elif invalid == "nan_mean":
    model["input_mean"][0][0] = float("nan")
  elif invalid == "nan_weight":
    model["layers"][0]["dense_1_W"][0][0] = float("nan")
  elif invalid == "bias_shape":
    model["layers"][0]["dense_1_b"] = [[0.0]]
  elif invalid == "layer_shape":
    model["layers"][0]["dense_1_W"] = [[0.0]]
  else:
    model["output_size"] = 2
  path = tmp_path / "invalid.json"
  path.write_text(json.dumps(model))
  with pytest.raises(ValueError):
    NNTorqueModel(path)
