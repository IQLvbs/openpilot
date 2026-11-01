# Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos
from __future__ import annotations

import math

import numpy as np

from iqpilot.selfdrive.iqmodeld.temporal_state import copy_policy_input, packed_input_casts, packed_policy_layout

NumpyDict = dict[str, np.ndarray]


def _tinygrad_imports():
    from tinygrad.tensor import Tensor
    from tinygrad.device import Device
    return Tensor, Device


class TinygradSupercomboRuntime:
    def _frame_tensor(self, key: str, buf):
        Tensor, Device = _tinygrad_imports()
        arr = np.frombuffer(buf.data, dtype=np.uint8)
        ck = (key, arr.ctypes.data)
        t = self._blob_cache.get(ck)
        if t is None:
            t = Tensor.from_blob(arr.ctypes.data, (arr.size,), dtype='uint8', device=Device.DEFAULT)
            self._blob_cache[ck] = t
        return t

    def _slice_flat_output(self, flat: np.ndarray) -> NumpyDict:
        return {name: flat[np.newaxis, span] for name, span in self._slices.items()}

    def _ensure_queues(self, cam_w: int, cam_h: int) -> None:
        if self._queues is not None and self._cam == (cam_w, cam_h):
            return
        if (cam_w, cam_h) not in self._warp_jits:
            raise RuntimeError(f"no warp JIT for {cam_w}x{cam_h}; have {sorted(self._warp_jits)}")

        Tensor, Device = _tinygrad_imports()
        fs = self._frame_skip
        img = self._ish['img']
        n_frames = img[1] // 6
        img_buf = (fs * (n_frames - 1) + 1, 6, img[2], img[3])
        fb = self._ish['features_buffer']
        dp = self._ish['desire_pulse']

        def zeros_u8(shape):
            return Tensor(np.zeros(shape, dtype=np.uint8), device=Device.DEFAULT).contiguous().realize()

        def zeros_f32(shape):
            return Tensor(np.zeros(shape, dtype=np.float32), device=Device.DEFAULT).contiguous().realize()

        shapes = packed_policy_layout(self._ish)
        sizes = [math.prod(s) for s in shapes.values()]
        packed = np.zeros(sum(sizes), dtype=np.float32)
        views = {k: v.reshape(s) for (k, s), v in zip(shapes.items(), np.split(packed, np.cumsum(sizes[:-1])), strict=True)}

        self._npy = {'tfm': np.zeros((3, 3), dtype=np.float32), 'big_tfm': np.zeros((3, 3), dtype=np.float32), **views}
        self._queues = {
            'img_q':     zeros_u8(img_buf),
            'big_img_q': zeros_u8(img_buf),
            'feat_q':    zeros_f32((fs * fb[1], fb[0], fb[2])),
            'desire_q':  zeros_f32((fs * dp[1], dp[0], dp[2])),
            'tfm':       Tensor(self._npy['tfm'], device='NPY'),
            'big_tfm':   Tensor(self._npy['big_tfm'], device='NPY'),
            'packed_npy_inputs': Tensor(packed, device='NPY'),
        }
        self._cam = (cam_w, cam_h)

    def run_fused(self, bufs: dict, transforms: dict[str, np.ndarray], numpy_inputs: NumpyDict) -> NumpyDict:
        Tensor, Device = _tinygrad_imports()
        main_buf = bufs['img']
        self._ensure_queues(main_buf.width, main_buf.height)
        assert self._queues is not None and self._npy is not None

        self._npy['tfm'][:] = transforms['img']
        self._npy['big_tfm'][:] = transforms['big_img']

        desire_key = next((k for k in numpy_inputs if k.startswith('desire')), None)
        cur = numpy_inputs[desire_key].copy() if desire_key is not None else np.zeros_like(self._prev_desire)
        cur[0] = 0
        self._npy['desire'][:] = np.where(cur - self._prev_desire > .99, cur, 0)
        self._prev_desire[:] = cur
        casts = packed_input_casts(getattr(self, '_meta', {}))
        for name in casts:
            if name not in numpy_inputs:
                self._npy[name].fill(0)
        for name, value in numpy_inputs.items():
            if name in self._npy and name != desire_key:
                copy_policy_input(self._npy[name], value, casts.get(name, 'float32'))

        frame = self._frame_tensor('img', bufs['img'])
        big_frame = self._frame_tensor('big_img', bufs['big_img'])

        warp = self._warp_jits[self._cam]
        try:
            warped = warp(tfm=self._queues['tfm'], big_tfm=self._queues['big_tfm'], frame=frame, big_frame=big_frame)
            out, = self._run_policy(warped=warped, img_q=self._queues['img_q'], big_img_q=self._queues['big_img_q'],
                                    feat_q=self._queues['feat_q'], desire_q=self._queues['desire_q'],
                                    packed_npy_inputs=self._queues['packed_npy_inputs'])
        except Exception as err:
            self._handle_runtime_jit_mismatch(err)
            raise
        flat = out.numpy().flatten()

        self._npy['prev_feat'][:] = flat[self._hidden_slice].reshape(self._npy['prev_feat'].shape)

        sliced = self._slice_flat_output(flat)
        return self._parser.parse_vision_outputs(sliced)

    def _handle_runtime_jit_mismatch(self, error):
        raise RuntimeError('Private supercombo inference failed') from error
