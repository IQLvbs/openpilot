# Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos
from iqpilot._proprietary_loader import ProprietaryModuleMissing, load_private_module

try:
  load_private_module(__name__, "iqpilot_private.context.temporal_state")
except ProprietaryModuleMissing:
  import math

  import numpy as np

  DEFAULT_FRAME_SKIP = 4

  MODEL_INPUT_SPEC = {
    "img": ((1, 12, 128, 256), "uint8"),
    "big_img": ((1, 12, 128, 256), "uint8"),
    "desire_pulse": ((1, 25, 8), "float32"),
    "traffic_convention": ((1, 2), "float32"),
    "features_buffer": ((1, 24, 512), "float32"),
    "action_t": ((1, 2), "float32"),
  }

  def extra_policy_inputs(input_shapes):
    return [name for name in input_shapes if name not in MODEL_INPUT_SPEC]

  def packed_policy_layout(input_shapes):
    desire = input_shapes["desire_pulse"]
    features = input_shapes["features_buffer"]
    layout = {
      "desire": (desire[2],),
      "traffic_convention": tuple(input_shapes["traffic_convention"]),
      "action_t": tuple(input_shapes["action_t"]),
      "prev_feat": (features[0], features[2]),
    }
    layout.update((name, tuple(input_shapes[name])) for name in extra_policy_inputs(input_shapes))
    return layout

  def spec_from_meta(meta):
    shapes = meta.get("input_shapes")
    if not shapes:
      return None
    declared = meta.get("input_dtypes", {})
    return {name: (tuple(shape), np.dtype(declared.get(name, "uint8" if name in ("img", "big_img") else "float32")).name)
            for name, shape in shapes.items()}

  def packed_input_casts(meta):
    return {}

  def copy_policy_input(target, value, dtype="float32"):
    target[:] = np.asarray(value)

  def supercombo_metadata(metadata):
    return metadata

  def _fit_columns(vec, width):
    flat = np.asarray(vec, dtype=np.float32).reshape(-1)
    if flat.shape[0] >= width:
      return flat[:width]
    return np.concatenate([flat, np.zeros(width - flat.shape[0], dtype=np.float32)])

  class TemporalInputState:
    def __init__(self, frame_skip, spec=MODEL_INPUT_SPEC):
      self.frame_skip = frame_skip
      img = spec["img"][0]
      fb = spec["features_buffer"][0]
      dp = spec["desire_pulse"][0]
      self.n_frames = img[1] // 6
      img_q_shape = (frame_skip * (self.n_frames - 1) + 1, 6, img[2], img[3])
      self._img_shape = img
      self._fb_shape = fb
      self._dp_shape = dp
      feat_dim = math.prod(fb[2:])
      self.img_q = np.zeros(img_q_shape, dtype=np.uint8)
      self.big_img_q = np.zeros(img_q_shape, dtype=np.uint8)
      self.feat_q = np.zeros((frame_skip * fb[1], fb[0], feat_dim), dtype=np.float32)
      self.desire_q = np.zeros((frame_skip * dp[1], dp[0], dp[2]), dtype=np.float32)
      self.prev_desire = np.zeros(dp[2], dtype=np.float32)
      self.prev_feat = np.zeros((fb[0], feat_dim), dtype=np.float32)
      self._traffic_shape = spec["traffic_convention"][0]
      self._action_shape = spec["action_t"][0] if "action_t" in spec else (1, 2)
      self.traffic_q = (np.zeros((frame_skip * self._traffic_shape[1], self._traffic_shape[2]), dtype=np.float32)
                        if len(self._traffic_shape) == 3 else None)
      self._traffic_primed = False

    @staticmethod
    def _shift_append(q, new_val):
      q[:-1] = q[1:]
      q[-1] = new_val

    def _materialize_traffic(self, traffic_convention):
      shape = self._traffic_shape
      if self.traffic_q is None:
        return _fit_columns(traffic_convention, shape[-1]).reshape(shape)
      cur = _fit_columns(traffic_convention, shape[2])
      if not self._traffic_primed:
        self.traffic_q[:] = cur
        self._traffic_primed = True
      self._shift_append(self.traffic_q, cur)
      return np.ascontiguousarray(self.traffic_q[self.frame_skip - 1::self.frame_skip]).reshape(shape)

    def _materialize_action_t(self, action_t):
      shape = self._action_shape
      cur = _fit_columns(action_t, shape[-1])
      if len(shape) == 3:
        return np.broadcast_to(cur, (shape[1], shape[2])).reshape(shape).copy()
      return cur.reshape(shape)

    def push_and_materialize(self, warped, desire_pulse, traffic_convention, action_t):
      fs = self.frame_skip
      cur = desire_pulse.astype(np.float32).copy()
      cur[0] = 0
      pulse = np.where(cur - self.prev_desire > 0.99, cur, 0).astype(np.float32)
      self.prev_desire[:] = cur
      self._shift_append(self.img_q, warped[0])
      self._shift_append(self.big_img_q, warped[1])
      self._shift_append(self.desire_q, pulse.reshape(self._dp_shape[0], self._dp_shape[2]))
      self._shift_append(self.feat_q, self.prev_feat)
      dp = self._dp_shape
      return {
        "img": np.ascontiguousarray(self.img_q[::fs]).reshape(self._img_shape),
        "big_img": np.ascontiguousarray(self.big_img_q[::fs]).reshape(self._img_shape),
        "features_buffer": np.ascontiguousarray(self.feat_q[::fs]).reshape(self._fb_shape),
        "desire_pulse": self.desire_q.reshape(dp[1], fs, dp[0], dp[2]).max(axis=1).reshape(dp),
        "traffic_convention": self._materialize_traffic(traffic_convention),
        "action_t": self._materialize_action_t(action_t),
      }

    def note_hidden_state(self, model_output, hidden_slice):
      self.prev_feat[:] = model_output[hidden_slice].reshape(self.prev_feat.shape)

  class SplitTemporalState:
    def __init__(self, frame_skip, img_shape, feature_shape, desire_shape, traffic_shape=(1, 2), action_shape=(1, 2)):
      self.frame_skip = frame_skip
      self._img_shape = tuple(img_shape)
      self._fb_shape = tuple(feature_shape)
      self._dp_shape = tuple(desire_shape)
      self._traffic_shape = tuple(traffic_shape)
      self._action_shape = tuple(action_shape)
      self.traffic_q = (np.zeros((frame_skip * self._traffic_shape[1], self._traffic_shape[2]), dtype=np.float32)
                        if len(self._traffic_shape) == 3 else None)
      self._traffic_primed = False
      n_frames = img_shape[1] // 6
      img_q_shape = (frame_skip * (n_frames - 1) + 1, 6, img_shape[2], img_shape[3])
      self.img_q = np.zeros(img_q_shape, dtype=np.uint8)
      self.big_img_q = np.zeros(img_q_shape, dtype=np.uint8)
      self.feat_q = np.zeros((frame_skip * (feature_shape[1] - 1) + 1, feature_shape[0], feature_shape[2]), dtype=np.float32)
      self.desire_q = np.zeros((frame_skip * desire_shape[1], desire_shape[0], desire_shape[2]), dtype=np.float32)
      self.prev_desire = np.zeros(desire_shape[2], dtype=np.float32)

    def materialize_vision(self, warped, desire):
      fs = self.frame_skip
      cur = desire.astype(np.float32).copy()
      cur[0] = 0
      pulse = np.where(cur - self.prev_desire > 0.99, cur, 0).astype(np.float32)
      self.prev_desire[:] = cur
      TemporalInputState._shift_append(self.img_q, warped[0])
      TemporalInputState._shift_append(self.big_img_q, warped[1])
      TemporalInputState._shift_append(self.desire_q, pulse.reshape(self._dp_shape[0], self._dp_shape[2]))
      return {
        "img": np.ascontiguousarray(self.img_q[::fs]).reshape(self._img_shape),
        "big_img": np.ascontiguousarray(self.big_img_q[::fs]).reshape(self._img_shape),
      }

    def materialize_policy(self, vision_feature, traffic_convention, action_t=None):
      fs = self.frame_skip
      TemporalInputState._shift_append(self.feat_q, vision_feature.reshape(self._fb_shape[0], self._fb_shape[2]))
      dp = self._dp_shape
      out = {
        "features_buffer": np.ascontiguousarray(self.feat_q[::fs]).reshape(self._fb_shape),
        "desire_pulse": self.desire_q.reshape(dp[1], fs, dp[0], dp[2]).max(axis=1).reshape(dp),
        "traffic_convention": TemporalInputState._materialize_traffic(self, traffic_convention),
      }
      if action_t is not None:
        out["action_t"] = TemporalInputState._materialize_action_t(self, action_t)
      return out
