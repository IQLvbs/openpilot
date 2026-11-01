"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""
import os

import numpy as np
import pytest

os.environ["DEV"] = "CPU"

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from onnx import TensorProto, helper

from iqpilot.selfdrive.iqmodeld.egpu_policy import STATEFUL_POLICY_FORMAT, StatefulPolicyRunner, load_bundle

H, W, Q, F = 8, 16, 4, 8
OUT_LEN = Q * F + 4
SPEC = {
  "new_img": ((2, 6, H, W), "uint8"),
  "desire": ((8,), "float32"),
  "traffic_convention": ((1, 2), "float32"),
  "action_t": ((1, 2), "float32"),
  "state_feat_q": ((Q, 1, F), "float32"),
}


def _const(name, values, dtype=TensorProto.INT64):
  return helper.make_node("Constant", [], [name], value=helper.make_tensor(name + "_v", dtype, [len(values)], values))


def _stateful_onnx(path):
  nodes = [
    _const("one", [1]), _const("big", [Q]), _const("ax0", [0]), _const("dshape", [1, 1, F]), _const("oshape", [1, Q * F]),
    helper.make_node("Cast", ["new_img"], ["img_f"], to=TensorProto.FLOAT),
    helper.make_node("ReduceMean", ["img_f"], ["img_m"], keepdims=0),
    helper.make_node("Reshape", ["desire", "dshape"], ["desire_r"]),
    helper.make_node("Add", ["desire_r", "img_m"], ["fresh"]),
    helper.make_node("Slice", ["state_feat_q", "one", "big", "ax0"], ["older"]),
    helper.make_node("Concat", ["older", "fresh"], ["next_state_feat_q"], axis=0),
    helper.make_node("Reshape", ["next_state_feat_q", "oshape"], ["flat"]),
    helper.make_node("Concat", ["flat", "traffic_convention", "action_t"], ["outputs"], axis=1),
  ]
  inputs = [helper.make_tensor_value_info(n, TensorProto.UINT8 if d == "uint8" else TensorProto.FLOAT, list(s)) for n, (s, d) in SPEC.items()]
  outputs = [helper.make_tensor_value_info("outputs", TensorProto.FLOAT, [1, OUT_LEN]),
             helper.make_tensor_value_info("next_state_feat_q", TensorProto.FLOAT, [Q, 1, F])]
  model = helper.make_model(helper.make_graph(nodes, "stateful", inputs, outputs), opset_imports=[helper.make_opsetid("", 17)])
  model.ir_version = 8
  onnx.save(model, path)


def test_stateful_policy_compiles_round_trips_and_matches_ort(tmp_path, monkeypatch):
  from iqpilot.selfdrive.iqmodeld.tools import compile_egpu_model as cem
  onnx_path, out_path = str(tmp_path / "m.onnx"), str(tmp_path / "m.pkl")
  _stateful_onnx(onnx_path)
  monkeypatch.setattr(cem, "HOST", True)
  monkeypatch.setattr(cem, "INPUT_SPEC", dict(SPEC))
  meta = {"key": "fake3", "sha256": "ab" * 32, "output_len": OUT_LEN, "frame_skip": 4, "output_slices": {}}
  assert cem.stateful_format(3, SPEC) == 5 and cem.stateful_format(2, SPEC) == 4
  cem.compile_stateful_policy(meta, onnx_path, out_path)

  bundle = load_bundle(out_path)
  assert bundle["format"] == STATEFUL_POLICY_FORMAT
  assert bundle["state_pairs"] == {"state_feat_q": "next_state_feat_q"}
  runner = StatefulPolicyRunner(bundle["run_policy"], bundle["input_spec"], bundle["state_pairs"], "CPU")

  sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
  state = np.zeros((Q, 1, F), dtype=np.float32)
  prev = np.zeros(8, dtype=np.float32)
  rng = np.random.default_rng(1)
  for i in range(7):
    img = rng.integers(0, 256, (2, 6, H, W), dtype=np.int64).astype(np.uint8)
    desire = np.zeros(8, dtype=np.float32)
    desire[2 + i % 3] = float(i % 2)
    tc, at = np.array([0.0, 1.0], dtype=np.float32), np.array([0.1 * i, 0.4], dtype=np.float32)
    got = runner.run(img, desire, tc, at)
    pulse = np.where(desire - prev > 0.99, desire, 0).astype(np.float32)
    pulse[0] = 0
    prev = desire
    out, state = sess.run(None, {"new_img": img, "desire": pulse, "traffic_convention": tc.reshape(1, 2),
                                 "action_t": at.reshape(1, 2), "state_feat_q": state})
    np.testing.assert_allclose(got, out.reshape(-1), rtol=1e-4, atol=1e-4, err_msg=f"frame {i}")


def test_format_one_refuses_stateful():
  from iqpilot.selfdrive.iqmodeld.tools import compile_egpu_model as cem
  with pytest.raises(SystemExit):
    cem.stateful_format(1, SPEC)
  assert cem.stateful_format(3, {"img": ((1, 12, 8, 16), "uint8")}) == 3


def test_stateful_warp_on_dock_compiles_and_round_trips(tmp_path, monkeypatch):
  from iqpilot.selfdrive.iqmodeld.egpu_policy import STATEFUL_MODEL_FORMAT, StatefulModelRunner
  from iqpilot.selfdrive.iqmodeld.tools import compile_egpu_model as cem
  onnx_path, out_path = str(tmp_path / "m.onnx"), str(tmp_path / "m5.pkl")
  _stateful_onnx(onnx_path)
  monkeypatch.setattr(cem, "HOST", True)
  monkeypatch.setattr(cem, "INPUT_SPEC", dict(SPEC))
  meta = {"key": "fake3", "sha256": "ab" * 32, "output_len": OUT_LEN, "frame_skip": 4, "output_slices": {}}
  res = (64, 48)
  cem.compile_stateful_model(meta, onnx_path, out_path, resolutions=(res,))
  bundle = load_bundle(out_path)
  assert bundle["format"] == STATEFUL_MODEL_FORMAT and res in bundle["run_model"]
  runner = StatefulModelRunner(bundle["run_model"][res], bundle["input_spec"], bundle["state_pairs"], "CPU", bundle["frame_copy_size"][res])
  n = runner.frame_copy_size
  rng = np.random.default_rng(4)
  eye = np.diag([res[0] / (2 * W), res[1] / (2 * H), 1.0]).astype(np.float32)
  outs = [runner.run(rng.integers(0, 256, n, dtype=np.int64).astype(np.uint8).tobytes(),
                     rng.integers(0, 256, n, dtype=np.int64).astype(np.uint8).tobytes(), eye, eye,
                     np.zeros(8, dtype=np.float32), np.array([1.0, 0.0], dtype=np.float32), np.array([0.2, 0.3], dtype=np.float32))
          for _ in range(Q + 1)]
  assert runner.warm_frames == Q + 1
  assert all(np.isfinite(o).all() and o.shape == (OUT_LEN,) for o in outs)
  assert not np.array_equal(outs[0][:Q * F], outs[-1][:Q * F])
  runner.reset()
  np.testing.assert_allclose(runner.run(b"\0" * n, b"\0" * n, eye, eye, np.zeros(8, dtype=np.float32),
                                        np.array([1.0, 0.0], dtype=np.float32), np.array([0.2, 0.3], dtype=np.float32))[:Q * F],
                             np.zeros(Q * F), atol=1e-6)
