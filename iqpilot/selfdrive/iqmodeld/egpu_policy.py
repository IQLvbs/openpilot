"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""
from __future__ import annotations

import io
import math
import os
import pickle
import shutil
import struct
import tempfile

import numpy as np

POLICY_FORMAT = 2
MODEL_FORMAT = 3
STATEFUL_POLICY_FORMAT = 4
STATEFUL_MODEL_FORMAT = 5
STATEFUL_IMAGE_INPUT = "new_img"
STATEFUL_OUTPUT = "outputs"
STATEFUL_BASE_FEEDS = ("desire", "traffic_convention", "action_t")
OOB_MAGIC = b"IQEGPUOOB1"
QUEUE_NAMES = ("img_q", "big_img_q", "feat_q", "desire_q")
PACKED_ORDER = ("desire", "traffic_convention", "action_t", "prev_feat")
MODELD_INPUTS = (*QUEUE_NAMES, "packed_npy_inputs")


def packed_layout(input_spec: dict) -> tuple[dict[str, tuple[int, ...]], list[int]]:
  dp = input_spec["desire_pulse"][0]
  fb = input_spec["features_buffer"][0]
  shapes = {
    "desire": (dp[2],),
    "traffic_convention": tuple(input_spec["traffic_convention"][0]),
    "action_t": tuple(input_spec["action_t"][0]),
    "prev_feat": (fb[0], math.prod(fb[2:])),
  }
  return shapes, [math.prod(s) for s in shapes.values()]


def queue_shapes(input_spec: dict, frame_skip: int) -> dict[str, tuple[tuple[int, ...], str]]:
  img = input_spec["img"][0]
  fb = input_spec["features_buffer"][0]
  dp = input_spec["desire_pulse"][0]
  n_frames = img[1] // 6
  img_buf = (frame_skip * (n_frames - 1) + 1, 6, img[2], img[3])
  return {
    "img_q": (img_buf, "uint8"),
    "big_img_q": (img_buf, "uint8"),
    "feat_q": ((frame_skip * fb[1], fb[0], math.prod(fb[2:])), "float32"),
    "desire_q": ((frame_skip * dp[1], dp[0], dp[2]), "float32"),
  }


def make_queues(input_spec: dict, frame_skip: int, device: str) -> dict:
  from tinygrad.tensor import Tensor
  return {name: Tensor(np.zeros(shape, dtype=dtype), device=device).contiguous().realize()
          for name, (shape, dtype) in queue_shapes(input_spec, frame_skip).items()}


class PackedInputs:
  def __init__(self, input_spec: dict):
    from tinygrad.tensor import Tensor
    self.shapes, self.sizes = packed_layout(input_spec)
    self.array = np.zeros(sum(self.sizes), dtype=np.float32)
    parts = np.split(self.array, np.cumsum(self.sizes[:-1]))
    self.views = {name: part.reshape(shape) for (name, shape), part in zip(self.shapes.items(), parts, strict=True)}
    self.tensor = Tensor(self.array, device="NPY").realize()


def make_run_policy(model_runner, input_spec: dict, frame_skip: int, device: str):
  from tinygrad.tensor import Tensor
  shapes, sizes = packed_layout(input_spec)
  fb = input_spec["features_buffer"][0]

  def shift_and_sample(buf, new_val, sample_fn):
    buf.assign(buf[1:].cat(new_val, dim=0).contiguous())
    return sample_fn(buf)

  def sample_skip(buf):
    return buf[::frame_skip].contiguous().flatten(0, 1).unsqueeze(0)

  def sample_desire(buf):
    return buf.reshape(-1, frame_skip, *buf.shape[1:]).max(1).flatten(0, 1).unsqueeze(0)

  def run_policy(warped, img_q, big_img_q, feat_q, desire_q, packed_npy_inputs):
    packed_npy_inputs = packed_npy_inputs.to(device)
    warped = warped.to(device)
    Tensor.realize(packed_npy_inputs, warped)
    img = shift_and_sample(img_q, warped[0:1], sample_skip)
    big_img = shift_and_sample(big_img_q, warped[1:2], sample_skip)
    desire, traffic_convention, action_t, prev_feat = (t.reshape(s) for t, s in zip(packed_npy_inputs.split(sizes), shapes.values(), strict=True))
    desire_buf = shift_and_sample(desire_q, desire.reshape(1, 1, -1), sample_desire)
    feat_buf = shift_and_sample(feat_q, prev_feat.reshape(1, 1, -1), sample_skip)
    inputs = {
      "img": img,
      "big_img": big_img,
      "features_buffer": feat_buf.reshape(fb),
      "desire_pulse": desire_buf,
      "traffic_convention": traffic_convention,
      "action_t": action_t,
    }
    out = next(iter(model_runner(inputs).values())).cast("float32")
    return out.reshape(-1),

  return run_policy


class PolicyRunner:
  def __init__(self, jit, input_spec: dict, frame_skip: int, hidden_slice: slice, device: str):
    from tinygrad.tensor import Tensor
    self._Tensor = Tensor
    self._jit = jit
    self._queues = make_queues(input_spec, frame_skip, device)
    self._packed = PackedInputs(input_spec)
    self._hidden = hidden_slice
    self._prev_desire = np.zeros(input_spec["desire_pulse"][0][2], dtype=np.float32)
    self._warped_shape = (2, 6, *input_spec["img"][0][2:])

  def run(self, warped: np.ndarray, desire_pulse: np.ndarray, traffic_convention: np.ndarray,
          action_t: np.ndarray) -> np.ndarray:
    cur = desire_pulse.astype(np.float32, copy=False)
    v = self._packed.views
    v["desire"][:] = np.where(cur - self._prev_desire > 0.99, cur, 0)
    self._prev_desire[:] = cur
    v["traffic_convention"][:] = np.asarray(traffic_convention, dtype=np.float32).reshape(v["traffic_convention"].shape)
    v["action_t"][:] = np.asarray(action_t, dtype=np.float32).reshape(v["action_t"].shape)
    warped_t = self._Tensor(np.ascontiguousarray(warped, dtype=np.uint8).reshape(self._warped_shape), device="NPY").realize()
    out, = self._jit(warped=warped_t, packed_npy_inputs=self._packed.tensor, **self._queues)
    flat = out.numpy().reshape(-1)
    v["prev_feat"][:] = flat[self._hidden].reshape(v["prev_feat"].shape)
    return flat


def nv12_copy_size(stride: int, y_height: int, uv_height: int) -> int:
  return stride * (y_height + uv_height)


def frame_layout(input_spec: dict) -> tuple[dict[str, tuple[int, ...]], list[int], int]:
  policy_shapes, _ = packed_layout(input_spec)
  shapes = {"tfm": (3, 3), "big_tfm": (3, 3)} | policy_shapes
  sizes = [math.prod(s) for s in shapes.values()]
  return shapes, sizes, sum(sizes) * np.dtype(np.float32).itemsize


def model_size(input_spec: dict) -> tuple[int, int]:
  img = input_spec["img"][0]
  return img[3] * 2, img[2] * 2


def warp_perspective_tinygrad(src_flat, M_inv, dst_shape, src_shape, stride_pad, border_fill_val=None):
  from tinygrad.tensor import Tensor
  w_dst, h_dst = dst_shape
  h_src, w_src = src_shape

  x = Tensor.arange(w_dst).reshape(1, w_dst).expand(h_dst, w_dst).reshape(-1)
  y = Tensor.arange(h_dst).reshape(h_dst, 1).expand(h_dst, w_dst).reshape(-1)

  src_x = M_inv[0, 0] * x + M_inv[0, 1] * y + M_inv[0, 2]
  src_y = M_inv[1, 0] * x + M_inv[1, 1] * y + M_inv[1, 2]
  src_w = M_inv[2, 0] * x + M_inv[2, 1] * y + M_inv[2, 2]

  src_x = src_x / src_w
  src_y = src_y / src_w

  x_round = Tensor.round(src_x)
  y_round = Tensor.round(src_y)
  x_nn_clipped = x_round.clip(0, w_src - 1).cast("int")
  y_nn_clipped = y_round.clip(0, h_src - 1).cast("int")
  idx = y_nn_clipped * (w_src + stride_pad) + x_nn_clipped
  sampled = src_flat[idx]

  if border_fill_val is None:
    return sampled

  in_bounds = ((x_round >= 0) & (x_round <= w_src - 1) &
               (y_round >= 0) & (y_round <= h_src - 1)).cast(sampled.dtype)
  return sampled * in_bounds + Tensor(border_fill_val, dtype=sampled.dtype) * (1 - in_bounds)


def frames_to_tensor(frames):
  from tinygrad.tensor import Tensor
  H = (frames.shape[0] * 2) // 3
  W = frames.shape[1]
  in_img1 = Tensor.cat(frames[0:H:2, 0::2],
                       frames[1:H:2, 0::2],
                       frames[0:H:2, 1::2],
                       frames[1:H:2, 1::2],
                       frames[H:H + H // 4].reshape((H // 2, W // 2)),
                       frames[H + H // 4:H + H // 2].reshape((H // 2, W // 2)), dim=0).reshape((6, H // 2, W // 2))
  return in_img1


def make_frame_prepare(nv12: tuple[int, int, int, int, int], model_w: int, model_h: int, device: str):
  from tinygrad.helpers import Context
  from tinygrad.tensor import Tensor
  cam_w, cam_h, stride, y_height, uv_height = nv12
  uv_offset = stride * y_height
  stride_pad = stride - cam_w

  def frame_prepare_tinygrad(input_frame, M_inv):
    M_inv_uv = M_inv * Tensor([[1.0, 1.0, 0.5], [1.0, 1.0, 0.5], [2.0, 2.0, 1.0]], device=device)
    uv = input_frame[uv_offset:uv_offset + uv_height * stride].reshape(uv_height, stride)
    with Context(SPLIT_REDUCEOP=0):
      y = warp_perspective_tinygrad(input_frame[:cam_h * stride],
                                    M_inv, (model_w, model_h),
                                    (cam_h, cam_w), stride_pad).realize()
      u = warp_perspective_tinygrad(uv[:cam_h // 2, :cam_w:2].flatten(),
                                    M_inv_uv, (model_w // 2, model_h // 2),
                                    (cam_h // 2, cam_w // 2), 0).realize()
      v = warp_perspective_tinygrad(uv[:cam_h // 2, 1:cam_w:2].flatten(),
                                    M_inv_uv, (model_w // 2, model_h // 2),
                                    (cam_h // 2, cam_w // 2), 0).realize()
    yuv = y.cat(u).cat(v).reshape((model_h * 3 // 2, model_w))
    return frames_to_tensor(yuv)
  return frame_prepare_tinygrad


def make_warp(nv12: tuple[int, int, int, int, int], model_w: int, model_h: int, device: str):
  from tinygrad.tensor import Tensor
  frame_prepare = make_frame_prepare(nv12, model_w, model_h, device)

  def warp(tfm, big_tfm, frame, big_frame):
    tfm = tfm.to(device)
    big_tfm = big_tfm.to(device)
    frame = frame.to(device)
    big_frame = big_frame.to(device)
    Tensor.realize(tfm, big_tfm, frame, big_frame)

    warped_frame = frame_prepare(frame, tfm).unsqueeze(0)
    warped_big_frame = frame_prepare(big_frame, big_tfm).unsqueeze(0)
    return Tensor.cat(warped_frame, warped_big_frame)

  return warp


def make_run_model(warp, run_policy, input_spec: dict, frame_copy_size: int, device: str):
  from tinygrad.tensor import Tensor
  _, policy_sizes = packed_layout(input_spec)
  _, _, packed_npy_size = frame_layout(input_spec)

  def run_model(img_q, big_img_q, feat_q, desire_q, packed_npy_inputs):
    packed_input = packed_npy_inputs.to(device)
    Tensor.realize(packed_input)
    packed_npy_inputs = packed_input[:packed_npy_size].bitcast("float32")
    frame = packed_input[packed_npy_size:packed_npy_size + frame_copy_size]
    big_frame = packed_input[packed_npy_size + frame_copy_size:]
    tfm, big_tfm, policy_inputs = packed_npy_inputs.split([9, 9, sum(policy_sizes)])
    warped = warp(tfm.reshape(3, 3), big_tfm.reshape(3, 3), frame, big_frame)
    return run_policy(warped, img_q, big_img_q, feat_q, desire_q, policy_inputs)

  return run_model


class PackedFrames:
  def __init__(self, input_spec: dict, frame_copy_size: int):
    from tinygrad.tensor import Tensor
    self.shapes, self.sizes, npy_bytes = frame_layout(input_spec)
    self.frame_copy_size = frame_copy_size
    self.array = np.zeros(npy_bytes + 2 * frame_copy_size, dtype=np.uint8)
    npy = self.array[:npy_bytes].view(np.float32)
    self.views = dict(zip(self.shapes, [v.reshape(s) for s, v in zip(self.shapes.values(), np.split(npy, np.cumsum(self.sizes[:-1])), strict=True)],
                          strict=True))
    frames = self.array[npy_bytes:]
    self.frames = {"img": frames[:frame_copy_size], "big_img": frames[frame_copy_size:]}
    self.tensor = Tensor(self.array, device="NPY").realize()


def make_model_queues(input_spec: dict, frame_skip: int, device: str, frame_copy_size: int) -> tuple[dict, PackedFrames]:
  packed = PackedFrames(input_spec, frame_copy_size)
  return {**make_queues(input_spec, frame_skip, device), "packed_npy_inputs": packed.tensor}, packed


class ModelRunner:
  def __init__(self, jit, input_spec: dict, frame_skip: int, hidden_slice: slice, device: str, frame_copy_size: int):
    self._jit = jit
    self._queues, self._packed = make_model_queues(input_spec, frame_skip, device, frame_copy_size)
    self._hidden = hidden_slice
    self._prev_desire = np.zeros(input_spec["desire_pulse"][0][2], dtype=np.float32)
    self.frame_copy_size = frame_copy_size

  def run(self, main_frame, extra_frame, tfm: np.ndarray, big_tfm: np.ndarray, desire_pulse: np.ndarray,
          traffic_convention: np.ndarray, action_t: np.ndarray) -> np.ndarray:
    n = self.frame_copy_size
    v = self._packed.views
    f = self._packed.frames
    np.copyto(f["img"], np.frombuffer(main_frame, dtype=np.uint8, count=n))
    np.copyto(f["big_img"], np.frombuffer(extra_frame, dtype=np.uint8, count=n))
    v["tfm"][:, :] = tfm
    v["big_tfm"][:, :] = big_tfm
    cur = desire_pulse.astype(np.float32, copy=False)
    v["desire"][:] = np.where(cur - self._prev_desire > 0.99, cur, 0)
    self._prev_desire[:] = cur
    v["traffic_convention"][:] = np.asarray(traffic_convention, dtype=np.float32).reshape(v["traffic_convention"].shape)
    v["action_t"][:] = np.asarray(action_t, dtype=np.float32).reshape(v["action_t"].shape)
    out, = self._jit(**self._queues)
    flat = out.numpy().reshape(-1)
    v["prev_feat"][:] = flat[self._hidden].reshape(v["prev_feat"].shape)
    return flat


def state_pairs(input_spec: dict, declared: dict | None = None) -> dict[str, str]:
  if declared:
    return {name: declared[name] for name in input_spec if name in declared}
  return {name: f"next_{name}" for name in input_spec if name.startswith("state_")}


def is_stateful_spec(input_spec: dict) -> bool:
  return STATEFUL_IMAGE_INPUT in input_spec and bool(state_pairs(input_spec))


def stateful_packed_layout(input_spec: dict, pairs: dict[str, str]) -> tuple[dict[str, tuple[int, ...]], list[int]]:
  shapes = {name: tuple(shape) for name, (shape, _) in input_spec.items() if name != STATEFUL_IMAGE_INPUT and name not in pairs}
  return shapes, [math.prod(s) for s in shapes.values()]


def make_state_buffers(input_spec: dict, pairs: dict[str, str], device: str) -> tuple[dict, dict]:
  from tinygrad.tensor import Tensor

  def zeros():
    return {name: Tensor(np.zeros(input_spec[name][0], dtype=input_spec[name][1]), device=device).contiguous().realize() for name in pairs}

  return zeros(), zeros()


class StatefulPackedInputs:
  def __init__(self, input_spec: dict, pairs: dict[str, str]):
    from tinygrad.tensor import Tensor
    self.shapes, self.sizes = stateful_packed_layout(input_spec, pairs)
    self.array = np.zeros(sum(self.sizes), dtype=np.float32)
    parts = np.split(self.array, np.cumsum(self.sizes[:-1]))
    self.views = {name: part.reshape(shape) for (name, shape), part in zip(self.shapes.items(), parts, strict=True)}
    self.tensor = Tensor(self.array, device="NPY").realize()


def _state_kwargs(cur: dict, nxt: dict) -> dict:
  return {**cur, **{f"next_{name}": t for name, t in nxt.items()}}


def make_run_stateful(model_runner, input_spec: dict, pairs: dict[str, str], device: str):
  from tinygrad.tensor import Tensor
  shapes, sizes = stateful_packed_layout(input_spec, pairs)
  dtypes = {name: input_spec[name][1] for name in shapes}
  img_shape = tuple(input_spec[STATEFUL_IMAGE_INPUT][0])

  def run_stateful(warped, packed_npy_inputs, **states):
    packed_npy_inputs = packed_npy_inputs.to(device)
    warped = warped.to(device)
    Tensor.realize(packed_npy_inputs, warped)
    inputs = {STATEFUL_IMAGE_INPUT: warped.reshape(img_shape)}
    inputs.update((name, t.reshape(shape).cast(dtypes[name])) for name, t, shape in zip(shapes, packed_npy_inputs.split(sizes), shapes.values(),
                                                                                       strict=True))
    inputs.update((name, states[name]) for name in pairs)
    outs = model_runner(inputs)
    # next state lands in the other buffer of the pair: writing the queue it is read from races inside the shift kernel
    written = [states[f"next_{name}"].assign(outs[nxt].cast(states[name].dtype).contiguous()) for name, nxt in pairs.items()]
    out = outs[STATEFUL_OUTPUT].cast("float32").reshape(-1).contiguous()
    Tensor.realize(out, *written)
    return out,

  return run_stateful


class _StatefulState:
  def _init_state(self, input_spec: dict, pairs: dict[str, str], device: str) -> None:
    self.pairs = pairs
    self._cur, self._nxt = make_state_buffers(input_spec, pairs, device)
    self._prev_desire = np.zeros(input_spec["desire"][0], dtype=np.float32) if "desire" in input_spec else None
    self._extra_names = tuple(name for name in input_spec
                              if name not in pairs and name not in (STATEFUL_IMAGE_INPUT, *STATEFUL_BASE_FEEDS))
    self.warm_frames = 0

  def reset(self) -> None:
    for bufs in (self._cur, self._nxt):
      for t in bufs.values():
        t.assign(t.zeros_like()).realize()
    if self._prev_desire is not None:
      self._prev_desire[:] = 0
    self.warm_frames = 0

  def _pulse(self, desire_pulse: np.ndarray) -> np.ndarray:
    cur = np.asarray(desire_pulse, dtype=np.float32).reshape(-1)
    pulse = np.where(cur - self._prev_desire > 0.99, cur, 0)
    pulse[0] = 0
    self._prev_desire[:] = cur
    return pulse

  def _fill(self, views: dict, desire_pulse, traffic_convention, action_t, extras: dict | None = None) -> None:
    if "desire" in views:
      views["desire"][:] = self._pulse(desire_pulse).reshape(views["desire"].shape)
    views["traffic_convention"][:] = np.asarray(traffic_convention, dtype=np.float32).reshape(views["traffic_convention"].shape)
    views["action_t"][:] = np.asarray(action_t, dtype=np.float32).reshape(views["action_t"].shape)
    for name in self._extra_names:
      value = extras.get(name) if extras else None
      views[name][:] = 0 if value is None else np.asarray(value, dtype=np.float32).reshape(views[name].shape)

  def _step(self, jit, **kw) -> np.ndarray:
    out, = jit(**kw, **_state_kwargs(self._cur, self._nxt))
    self._cur, self._nxt = self._nxt, self._cur
    self.warm_frames += 1
    return out.numpy().reshape(-1)


class StatefulPolicyRunner(_StatefulState, PolicyRunner):
  def __init__(self, jit, input_spec: dict, pairs: dict[str, str], device: str):
    from tinygrad.tensor import Tensor
    self._Tensor = Tensor
    self._jit = jit
    self._packed = StatefulPackedInputs(input_spec, pairs)
    self._warped_shape = tuple(input_spec[STATEFUL_IMAGE_INPUT][0])
    self._init_state(input_spec, pairs, device)

  def run(self, warped: np.ndarray, desire_pulse: np.ndarray, traffic_convention: np.ndarray,
          action_t: np.ndarray, extras: dict | None = None) -> np.ndarray:
    self._fill(self._packed.views, desire_pulse, traffic_convention, action_t, extras)
    warped_t = self._Tensor(np.ascontiguousarray(warped, dtype=np.uint8).reshape(self._warped_shape), device="NPY").realize()
    return self._step(self._jit, warped=warped_t, packed_npy_inputs=self._packed.tensor)


def stateful_frame_layout(input_spec: dict, pairs: dict[str, str]) -> tuple[dict[str, tuple[int, ...]], list[int], int]:
  policy_shapes, _ = stateful_packed_layout(input_spec, pairs)
  shapes = {"tfm": (3, 3), "big_tfm": (3, 3)} | policy_shapes
  sizes = [math.prod(s) for s in shapes.values()]
  return shapes, sizes, sum(sizes) * np.dtype(np.float32).itemsize


def stateful_model_size(input_spec: dict) -> tuple[int, int]:
  img = input_spec[STATEFUL_IMAGE_INPUT][0]
  return img[-1] * 2, img[-2] * 2


class StatefulPackedFrames(PackedFrames):
  def __init__(self, input_spec: dict, pairs: dict[str, str], frame_copy_size: int):
    from tinygrad.tensor import Tensor
    self.shapes, self.sizes, npy_bytes = stateful_frame_layout(input_spec, pairs)
    self.frame_copy_size = frame_copy_size
    self.array = np.zeros(npy_bytes + 2 * frame_copy_size, dtype=np.uint8)
    npy = self.array[:npy_bytes].view(np.float32)
    self.views = dict(zip(self.shapes, [v.reshape(s) for s, v in zip(self.shapes.values(), np.split(npy, np.cumsum(self.sizes[:-1])), strict=True)],
                          strict=True))
    frames = self.array[npy_bytes:]
    self.frames = {"img": frames[:frame_copy_size], "big_img": frames[frame_copy_size:]}
    self.tensor = Tensor(self.array, device="NPY").realize()


def make_run_stateful_model(warp, run_stateful, input_spec: dict, pairs: dict[str, str], frame_copy_size: int, device: str):
  from tinygrad.tensor import Tensor
  _, policy_sizes = stateful_packed_layout(input_spec, pairs)
  _, _, packed_npy_size = stateful_frame_layout(input_spec, pairs)

  def run_model(packed_npy_inputs, **states):
    packed_input = packed_npy_inputs.to(device)
    Tensor.realize(packed_input)
    npy = packed_input[:packed_npy_size].bitcast("float32")
    frame = packed_input[packed_npy_size:packed_npy_size + frame_copy_size]
    big_frame = packed_input[packed_npy_size + frame_copy_size:]
    tfm, big_tfm, policy_inputs = npy.split([9, 9, sum(policy_sizes)])
    warped = warp(tfm.reshape(3, 3), big_tfm.reshape(3, 3), frame, big_frame)
    return run_stateful(warped, policy_inputs, **states)

  return run_model


class StatefulModelRunner(_StatefulState, ModelRunner):
  def __init__(self, jit, input_spec: dict, pairs: dict[str, str], device: str, frame_copy_size: int):
    self._jit = jit
    self._packed = StatefulPackedFrames(input_spec, pairs, frame_copy_size)
    self.frame_copy_size = frame_copy_size
    self._init_state(input_spec, pairs, device)

  def run(self, main_frame, extra_frame, tfm: np.ndarray, big_tfm: np.ndarray, desire_pulse: np.ndarray,
          traffic_convention: np.ndarray, action_t: np.ndarray, extras: dict | None = None) -> np.ndarray:
    n = self.frame_copy_size
    v = self._packed.views
    f = self._packed.frames
    np.copyto(f["img"], np.frombuffer(main_frame, dtype=np.uint8, count=n))
    np.copyto(f["big_img"], np.frombuffer(extra_frame, dtype=np.uint8, count=n))
    v["tfm"][:, :] = tfm
    v["big_tfm"][:, :] = big_tfm
    self._fill(v, desire_pulse, traffic_convention, action_t, extras)
    return self._step(self._jit, packed_npy_inputs=self._packed.tensor)


def dump_oob(obj, f) -> None:
  # Out-of-band pickle buffers keep the host peak at one tensor while the weights stream to the
  # dock; a plain pickle keeps every weight referenced in the memo until load() returns (~1.7GB).
  f.write(OOB_MAGIC)
  with tempfile.TemporaryFile(dir=os.path.dirname(os.path.abspath(f.name)) or ".") as tmp:
    def buffer_callback(pb: pickle.PickleBuffer):
      m = pb.raw()
      tmp.write(struct.pack("<q", m.nbytes))
      tmp.write(m)
      pb.release()
    stream = io.BytesIO()
    pickle.Pickler(stream, protocol=5, buffer_callback=buffer_callback).dump(obj)
    opcodes = stream.getvalue()
    f.write(struct.pack("<q", len(opcodes)))
    f.write(opcodes)
    tmp.seek(0)
    shutil.copyfileobj(tmp, f)


def is_oob(path: str) -> bool:
  with open(path, "rb") as f:
    return f.read(len(OOB_MAGIC)) == OOB_MAGIC


def load_oob(f):
  if f.read(len(OOB_MAGIC)) != OOB_MAGIC:
    raise ValueError("not an out-of-band bundle")
  opcodes = f.read(struct.unpack("<q", f.read(8))[0])

  def buffers():
    while (h := f.read(8)):
      pb = pickle.PickleBuffer(bytearray(struct.unpack("<q", h)[0]))
      f.readinto(pb)
      yield pb

  return pickle.load(io.BytesIO(opcodes), buffers=buffers())


def load_bundle(path: str):
  with open(path, "rb") as f:
    if f.read(len(OOB_MAGIC)) == OOB_MAGIC:
      f.seek(0)
      return load_oob(f)
    f.seek(0)
    return pickle.load(f)
