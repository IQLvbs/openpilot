"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""
from __future__ import annotations

import os

import numpy as np

os.environ.setdefault("DEV", "CPU")

from iqpilot.selfdrive.iqmodeld.temporal_state import MODEL_INPUT_SPEC
from iqpilot.selfdrive.iqmodeld.tools.compile_supercombo import (
  POLICY_INPUTS,
  _captured_devices,
  _validate_pose_outputs,
  get_policy_npy_shapes,
  make_input_queues,
  make_run_policy,
)
from iqpilot.selfdrive.iqmodeld.models.runners.tinygrad.supercombo_runner import _captured_queue_depth

FRAME_SKIP = 4
EXTRA_INPUT_SHAPES = {"nav_intent": (1, 32), "route_line": (1, 32, 2)}


def _input_shapes(extra_inputs=()):
  shapes = {name: shape for name, (shape, _) in MODEL_INPUT_SPEC.items()}
  shapes.update((name, EXTRA_INPUT_SHAPES[name]) for name in extra_inputs)
  return shapes


class _CaptureRunner:
  def __init__(self):
    self.captured = None

  def __call__(self, inputs):
    from tinygrad.tensor import Tensor
    self.captured = {name: value.numpy().copy() for name, value in inputs.items()}
    return {"outputs": Tensor(np.zeros((1, 8), dtype=np.float32))}


class _Captured:
  def __init__(self, expected_input_info):
    self.expected_input_info = expected_input_info


class _FakeJit:
  def __init__(self, expected_input_info):
    self.captured = _Captured(expected_input_info)


def test_captured_queue_helpers_extract_depth_and_device():
  infos = [
    ("noop", (), "uchar", "QCOM"),
    ("reshape(arg=None, src=(noop, stack(arg=None, src=(const(arg=5), const(arg=6), const(arg=128), const(arg=256)))))", (), "uchar", "QCOM"),
    ("reshape(arg=None, src=(noop, const(arg=3)))", (), "float", "NPY"),
  ]
  fake_jit = _FakeJit(infos)

  assert _captured_queue_depth(fake_jit) == 5
  assert _captured_devices(fake_jit) == {"QCOM", "NPY"}


def test_validate_pose_outputs_accepts_sane_odometry_payload():
  outputs = {
    "pose": np.array([[1.0, 0.5, 0.25, 0.1, 0.2, 0.3]], dtype=np.float32),
    "pose_stds": np.array([[0.5, 0.4, 0.3, 0.2, 0.2, 0.2]], dtype=np.float32),
    "wide_from_device_euler": np.array([[0.1, 0.2, 0.3]], dtype=np.float32),
    "wide_from_device_euler_stds": np.array([[0.2, 0.2, 0.2]], dtype=np.float32),
    "road_transform": np.array([[0.5, 0.4, 0.3, 0.2, 0.1, 0.0]], dtype=np.float32),
    "road_transform_stds": np.array([[0.3, 0.3, 0.3, 0.2, 0.2, 0.2]], dtype=np.float32),
  }

  _validate_pose_outputs(outputs)


def test_policy_layout_is_unchanged_without_extra_inputs():
  shapes, sizes = get_policy_npy_shapes(_input_shapes())

  assert list(shapes) == ["desire", "traffic_convention", "action_t", "prev_feat"]
  assert sizes == [8, 2, 2, 512]


def test_run_policy_feeds_extra_inputs_from_the_packed_tail():
  from tinygrad.tensor import Tensor

  input_shapes = _input_shapes(("nav_intent", "route_line"))
  layout, sizes = get_policy_npy_shapes(input_shapes)
  assert list(layout)[4:] == ["nav_intent", "route_line"]
  assert sizes == [8, 2, 2, 512, 32, 64]

  capture = _CaptureRunner()
  run_policy = make_run_policy(capture, {"input_shapes": input_shapes}, FRAME_SKIP)
  queues, npy = make_input_queues(input_shapes, FRAME_SKIP, device="CPU")
  rng = np.random.default_rng(7)
  for name in EXTRA_INPUT_SHAPES:
    npy[name][:] = rng.standard_normal(npy[name].shape).astype(np.float32)
  npy["traffic_convention"][:] = [[1.0, 0.0]]
  warped = Tensor(np.zeros((2, 6, 128, 256), dtype=np.uint8), device="CPU")

  run_policy(**{name: queues[name] for name in POLICY_INPUTS}, warped=warped)

  assert capture.captured is not None
  assert set(capture.captured) == set(input_shapes)
  np.testing.assert_array_equal(capture.captured["traffic_convention"], [[1.0, 0.0]])
  for name, shape in EXTRA_INPUT_SHAPES.items():
    assert capture.captured[name].shape == shape
    np.testing.assert_array_equal(capture.captured[name], npy[name])
