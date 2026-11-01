"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""
import os
import pickle
import re
import shutil
import sys
import time

import numpy as np

if "JIT_BATCH_SIZE" not in os.environ:
  os.environ["JIT_BATCH_SIZE"] = "0"

from tinygrad import Context, Device, GlobalCounters, Tensor, TinyJit, dtypes
from tinygrad.helpers import DEBUG, getenv
from tinygrad.nn.onnx import OnnxRunner
from tinygrad.uop.ops import Ops


def compile_model(onnx_file, output):
  run_onnx = OnnxRunner(onnx_file)
  print("loaded model")

  input_shapes = {name: spec.shape for name, spec in run_onnx.graph_inputs.items()}
  input_types = {name: spec.dtype for name, spec in run_onnx.graph_inputs.items()}
  input_types = {key: dtypes.float32 if value is dtypes.float16 else value for key, value in input_types.items()}
  input_shapes = {key: tuple(value if isinstance(value, int) else 1 for value in shape) for key, shape in input_shapes.items()}

  Tensor.manual_seed(100)
  inputs = {
    key: Tensor(Tensor.randn(*shape, dtype=input_types[key]).mul(8).realize().numpy(), device="NPY")
    for key, shape in sorted(input_shapes.items())
  }
  if not getenv("NPY_IMG"):
    inputs = {key: Tensor(value.numpy(), device=Device.DEFAULT).realize() if "img" in key else value for key, value in inputs.items()}
  print("created tensors")

  run_onnx_jit = TinyJit(
    lambda **kwargs: next(iter(run_onnx({key: value.to(Device.DEFAULT) for key, value in kwargs.items()}).values())).cast("float32"),
    prune=True,
  )
  test_value = None
  for iteration in range(3):
    GlobalCounters.reset()
    print(f"run {iteration}")
    with Context(DEBUG=max(DEBUG.value, 2 if iteration == 2 else 1), OPENPILOT_HACKS=1):
      result = run_onnx_jit(**inputs).numpy()
    if iteration == 1:
      test_value = np.copy(result)

  kernel_asts = {Ops.PROGRAM}
  kernel_calls = [
    node for node in run_onnx_jit.captured.linear.toposort(gate=lambda value: value.op not in kernel_asts)
    if node.op is Ops.CALL and node.src[0].op in kernel_asts
  ]
  print(f"captured {len(kernel_calls)} kernels")
  np.testing.assert_equal(test_value, result, "JIT run failed")
  print("jit run validated")

  kernel_count = 0
  read_image_count = 0
  gated_read_image_count = 0
  for call in kernel_calls:
    _, _, source, _ = call.src[0].src
    rendered = source.arg
    kernel_count += 1
    read_image_count += rendered.count("read_image")
    gated_read_image_count += rendered.count("?read_image")
    for value in (match.group(1) for match in re.finditer(r"(val\d+)\s*=\s*read_imagef\(", rendered)):
      if re.search(fr"[?:]{value}\.[xyzw]", rendered):
        gated_read_image_count += 1

  print(f"{kernel_count=}, {read_image_count=}, {gated_read_image_count=}")
  expected = {
    "kernel count": (kernel_count, getenv("ALLOWED_KERNEL_COUNT", -1)),
    "read image count": (read_image_count, getenv("ALLOWED_READ_IMAGE", -1)),
    "gated read image count": (gated_read_image_count, getenv("ALLOWED_GATED_READ_IMAGE", -1)),
  }
  for name, (actual, allowed) in expected.items():
    if allowed != -1:
      assert actual == allowed, f"different {name}: {actual}, expected {allowed}"

  with open(output, "wb") as handle:
    pickle.dump(run_onnx_jit, handle)
  print(f"model size is {os.path.getsize(onnx_file) / 1e6:.2f}M")
  print(f"pkl size is {os.path.getsize(output) / 1e6:.2f}M")
  return run_onnx_jit, inputs, test_value


def test_compiled(run, inputs, test_value):
  step_times = []
  for _ in range(20):
    start = time.perf_counter()
    output = run(**inputs)
    queued = time.perf_counter()
    value = output.numpy()
    end = time.perf_counter()
    step_times.append((end - start) * 1e3)
    print(f"enqueue {(queued - start) * 1e3:6.2f} ms -- total run {step_times[-1]:6.2f} ms")

  minimum = getenv("ASSERT_MIN_STEP_TIME", 0.0)
  if minimum:
    assert min(step_times) < minimum, f"expected minimum step time below {minimum} ms, got {min(step_times)} ms"
  np.testing.assert_equal(test_value, value)
  changed_inputs = {key: Tensor(item.numpy() * 2, device=item.device) for key, item in inputs.items()}
  changed_value = run(**changed_inputs).numpy()
  np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, value, changed_value)


if __name__ == "__main__":
  model_path = sys.argv[1]
  output_path = sys.argv[2]
  if stash := os.environ.get("IQPILOT_MODEL_STASH"):
    stashed_model = os.path.join(stash, os.path.basename(output_path))
    if os.path.isfile(stashed_model) and os.path.getsize(stashed_model) > 0:
      os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
      shutil.copyfile(stashed_model, output_path)
      print(f"restored device-compiled model: {output_path}")
      sys.exit(0)
  _, input_values, expected_value = compile_model(model_path, output_path)
  with open(output_path, "rb") as compiled_file:
    compiled_model = pickle.load(compiled_file)
  test_compiled(compiled_model, input_values, expected_value)
