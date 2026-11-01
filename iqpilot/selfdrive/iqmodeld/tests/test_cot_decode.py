"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""
from __future__ import annotations

import base64
import json
import pickle
from types import SimpleNamespace

import numpy as np

from iqpilot.selfdrive.iqmodeld import metadata
from iqpilot.selfdrive.iqmodeld.models.runners.tinygrad.supercombo_runner import TinygradSupercomboRunner


def reasoning_metadata():
  return {
    "output_slices": {
      "plan": slice(0, 4),
      "pad": slice(4, 6),
      "reasoning_tokens": slice(6, 10),
      "reasoning_state": slice(10, 11),
    },
    "output_len": 11,
    "cot_decode": "rh-cot-wire-v1",
    "cot_vocab": json.dumps(["<pad>", "<bos>", "<eos>", "<unk>", "stop", "sign", "."]),
    "cot_max_length": "4",
    "reasoning_states": json.dumps(["cruise", "brake"]),
  }


def synthetic_output():
  output = np.zeros(11, dtype=np.float32)
  output[6:10] = [4, 6, 2, -1]
  output[10] = 1
  return output


def test_compiled_metadata_retains_reasoning_contract(monkeypatch):
  record = reasoning_metadata()
  encoded_slices = base64.b64encode(pickle.dumps(record["output_slices"])).decode()
  document = {
    "metadata_props": [
      {"key": "output_slices", "value": encoded_slices},
      *({"key": key, "value": str(record[key])} for key in (
        "output_len", "cot_decode", "cot_vocab", "cot_max_length", "reasoning_states"
      )),
    ],
    "graph": {
      "input": [],
      "output": [{"name": "outputs", "parsed_type": SimpleNamespace(shape=(1, 11))}],
    },
  }
  monkeypatch.setattr(metadata, "TelemetryEnvelopeParser", lambda path: SimpleNamespace(parse=lambda: document))
  compiled = metadata.build_metadata_record("synthetic.onnx")
  assert compiled["output_slices"]["reasoning_tokens"] == slice(6, 10)
  assert compiled["output_len"] == 11
  assert compiled["cot_decode"] == record["cot_decode"]
  assert compiled["cot_vocab"] == record["cot_vocab"]
  assert compiled["cot_max_length"] == record["cot_max_length"]
  assert compiled["reasoning_states"] == record["reasoning_states"]


def test_tinygrad_runner_exposes_reasoning_tail_and_metadata():
  record = reasoning_metadata()
  runner = TinygradSupercomboRunner.__new__(TinygradSupercomboRunner)
  runner._meta = record
  runner._slices = {key: value for key, value in record["output_slices"].items() if key != "pad"}
  sliced = runner._slice_flat_output(synthetic_output())
  np.testing.assert_array_equal(sliced["reasoning_tokens"], [[4, 6, 2, -1]])
  np.testing.assert_array_equal(sliced["reasoning_state"], [[1]])
  assert runner.reasoning_metadata == record

