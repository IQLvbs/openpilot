"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""
from __future__ import annotations

import hashlib
import os
import re
from typing import Any

import numpy as np

from iqpilot.common.params import Params
from iqpilot.selfdrive.iqmodeld.models.runners.model_runner import CUSTOM_MODEL_PATH, NumpyDict, ShapeDict, SliceDict, _META_MAX_BYTES, load_big_model_pickle
from iqpilot.selfdrive.iqmodeld.models.runners.model_runner import ModelRunner
from iqpilot.selfdrive.iqmodeld.models.split_model_constants import SplitModelConstants
from iqpilot.selfdrive.iqmodeld.parser import PhaseParser
from iqpilot.selfdrive.iqmodeld.models.runners.tinygrad.supercombo_runtime import TinygradSupercomboRuntime
from iqpilot.selfdrive.iqmodeld.models.iqc import load_pickle


def _captured_queue_depth(warp_jit: Any) -> int | None:
    captured = getattr(warp_jit, "captured", None)
    infos = getattr(captured, "expected_input_info", None)
    if not infos or len(infos) < 2:
        return None

    view_repr = repr(infos[1][0])
    dims = [int(val) for val in re.findall(r"arg=(\d+)", view_repr)]
    return dims[0] if len(dims) >= 4 else None


def _captured_devices(warp_jit: Any) -> set[str]:
    captured = getattr(warp_jit, "captured", None)
    infos = getattr(captured, "expected_input_info", None)
    if not infos:
        return set()

    devices: set[str] = set()
    for info in infos:
        if isinstance(info, tuple) and len(info) >= 4 and isinstance(info[3], str):
            devices.add(info[3])
    return devices


def _captured_expected_names(jit_obj: Any) -> list[str]:
    captured = getattr(jit_obj, "captured", None)
    names = getattr(captured, "expected_names", None)
    return list(names) if names else []


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_jit_arg_mismatch(err: BaseException) -> bool:
    return "args mismatch in JIT" in str(err)


class TinygradSupercomboRunner(TinygradSupercomboRuntime, ModelRunner):
    uses_opencl_warp: bool = False

    def __init__(self):
        super().__init__()
        self._constants = SplitModelConstants
        self._parser = PhaseParser()

        if len(self.models) != 1:
            raise ValueError(f"supercombo bundle must have exactly one artifact, got {list(self.models)}")
        self._model_data = next(iter(self.models.values()))

        pkl_path = os.path.join(CUSTOM_MODEL_PATH, self._model_data.model.artifact.fileName)
        self._pkl_path = pkl_path
        self._expected_sha256 = getattr(getattr(self._model_data.model.artifact, "downloadUri", None), "sha256", "") or ""
        self._verify_artifact_file()
        self._m: dict[Any, Any] = load_big_model_pickle(pkl_path)

        self._meta = self._load_meta()
        self._ish = self._meta['input_shapes']
        self._slices = {k: v for k, v in self._meta['output_slices'].items() if k != 'pad'}
        self._hidden_slice = self._meta['output_slices']['hidden_state']
        self._run_policy = self._m['run_policy']
        self._warp_jits: dict[tuple[int, int], Any] = {k: v for k, v in self._m.items() if isinstance(k, tuple)}
        if not self._warp_jits:
            raise ValueError("supercombo pkl has no warp JITs")
        self._frame_skip = int(self._m.get('frame_skip', 4))
        self._validate_warp_jits(pkl_path)
        self._validate_jit_names()

        self._queues: dict[str, Any] | None = None
        self._npy: dict[str, np.ndarray] | None = None
        self._cam: tuple[int, int] | None = None
        self._prev_desire = np.zeros(self._ish['desire_pulse'][2], dtype=np.float32)
        self._blob_cache: dict[tuple[str, int], Any] = {}

    def _load_meta(self) -> dict[Any, Any]:
        if 'metadata' in self._m:
            return self._m['metadata']
        meta_art = getattr(self._model_data.model, "metadata", None)
        meta_name = getattr(meta_art, "fileName", "") or ""
        if not meta_name:
            raise KeyError("supercombo model has no embedded 'metadata' and no metadata artifact")
        meta_path = os.path.join(CUSTOM_MODEL_PATH, meta_name)
        meta_size = os.path.getsize(meta_path)
        if meta_size > _META_MAX_BYTES:
            raise ValueError(f"supercombo metadata artifact {meta_name} is artifact-sized ({meta_size}B); refusing to unpickle")
        expected = getattr(getattr(meta_art, "downloadUri", None), "sha256", "") or ""
        if expected:
            actual = _file_sha256(meta_path)
            if actual != expected:
                try:
                    os.remove(meta_path)
                except OSError:
                    pass
                redownload_msg = self._schedule_active_bundle_redownload()
                raise RuntimeError(
                    "supercombo metadata SHA mismatch: "
                    f"expected {expected}, got {actual} for {meta_path}. "
                    f"Deleted the stale cached file{redownload_msg}."
                )
        meta = load_pickle(meta_path)
        if not isinstance(meta, dict):
            raise ValueError(f"supercombo metadata artifact {meta_name} is not a dict")
        return meta

    def _verify_artifact_file(self) -> None:
        if not self._expected_sha256:
            return

        actual_sha256 = _file_sha256(self._pkl_path)
        if actual_sha256 == self._expected_sha256:
            return

        try:
            os.remove(self._pkl_path)
        except OSError:
            pass
        redownload_msg = self._schedule_active_bundle_redownload()

        raise RuntimeError(
            "supercombo artifact SHA mismatch: "
            f"expected {self._expected_sha256}, got {actual_sha256} for {self._pkl_path}. "
            f"Deleted the stale cached file{redownload_msg}."
        )

    def _validate_warp_jits(self, pkl_path: str) -> None:
        img = self._ish['img']
        n_frames = img[1] // 6
        expected_depth = self._frame_skip * (n_frames - 1) + 1
        expected_device = os.getenv('DEV')

        mismatches: list[str] = []
        for cam, warp_jit in sorted(self._warp_jits.items()):
            captured_depth = _captured_queue_depth(warp_jit)
            captured_devices = _captured_devices(warp_jit)
            if captured_depth is not None and captured_depth != expected_depth:
                mismatches.append(
                    f"{cam[0]}x{cam[1]} queue-depth captured={captured_depth} expected={expected_depth}"
                )
            if expected_device and captured_devices and expected_device not in captured_devices:
                mismatches.append(
                    f"{cam[0]}x{cam[1]} device captured={sorted(captured_devices)} expected={expected_device}"
                )

        if mismatches:
            details = "; ".join(mismatches)
            raise RuntimeError(
                "supercombo warp JIT compatibility mismatch: "
                f"{details}. Bundle {pkl_path} was compiled with the wrong backend, frame_skip, or queue shape; "
                "re-download or rebuild this model artifact."
            )

    def _validate_jit_names(self) -> None:
        expected_warp_names = ['big_frame', 'big_tfm', 'frame', 'tfm']
        expected_policy_names = ['big_img_q', 'desire_q', 'feat_q', 'img_q', 'packed_npy_inputs', 'warped']

        mismatches: list[str] = []

        policy_names = sorted(_captured_expected_names(self._run_policy))
        if policy_names and policy_names != expected_policy_names:
            mismatches.append(f"run_policy captured={policy_names} expected={expected_policy_names}")

        for cam, warp_jit in sorted(self._warp_jits.items()):
            warp_names = sorted(_captured_expected_names(warp_jit))
            if warp_names and warp_names != expected_warp_names:
                mismatches.append(f"{cam[0]}x{cam[1]} warp captured={warp_names} expected={expected_warp_names}")

        if mismatches:
            details = "; ".join(mismatches)
            actual_sha = None
            try:
                actual_sha = _file_sha256(self._pkl_path)
            except OSError:
                pass

            if actual_sha and self._expected_sha256 and actual_sha != self._expected_sha256:
                try:
                    os.remove(self._pkl_path)
                except OSError:
                    pass
                redownload_msg = self._schedule_active_bundle_redownload()
                raise RuntimeError(
                    "supercombo artifact contract mismatch with stale cached SHA: "
                    f"{details}. Expected SHA {self._expected_sha256}, got {actual_sha}. "
                    f"Deleted the stale cached file{redownload_msg}."
                )

            raise RuntimeError(
                "supercombo artifact JIT argument mismatch: "
                f"{details}. This model file does not match the current IQPilot runtime contract. "
                "Re-download or rebuild this model artifact."
            )

    def _handle_runtime_jit_mismatch(self, err: BaseException) -> None:
        if not _is_jit_arg_mismatch(err):
            raise err

        actual_sha = None
        try:
            actual_sha = _file_sha256(self._pkl_path)
        except OSError:
            pass

        if actual_sha and self._expected_sha256 and actual_sha != self._expected_sha256:
            try:
                os.remove(self._pkl_path)
            except OSError:
                pass
            redownload_msg = self._schedule_active_bundle_redownload()
            raise RuntimeError(
                "supercombo artifact runtime JIT mismatch with stale cached SHA: "
                f"expected {self._expected_sha256}, got {actual_sha} for {self._pkl_path}. "
                f"Deleted the stale cached file{redownload_msg}."
            ) from err

        raise RuntimeError(
            "supercombo artifact runtime JIT mismatch: "
            f"{err}. This model file does not match the current IQPilot runtime contract. "
            "Re-download or rebuild this model artifact."
        ) from err

    def _schedule_active_bundle_redownload(self) -> str:
        try:
            params = Params()
            active_bundle = params.get("ModelManager_ActiveBundle") or {}
            index = active_bundle.get("index") if isinstance(active_bundle, dict) else None
            if isinstance(index, str) and index.isdigit():
                index = int(index)
            if isinstance(index, int) and index >= 0:
                params.put("ModelManager_DownloadIndex", str(index))
                params.remove("ModelRunnerTypeCache")
                return "; scheduled automatic re-download of the active model"
        except Exception:
            pass

        return "; unable to schedule automatic re-download"

    @property
    def vision_input_names(self) -> list[str]:
        return ['img', 'big_img']

    @property
    def input_shapes(self) -> ShapeDict:
        return dict(self._ish)

    @property
    def input_dtypes(self) -> dict:
        from iqpilot.selfdrive.iqmodeld.temporal_state import spec_from_meta
        return {name: dtype for name, (_, dtype) in spec_from_meta(self._meta).items()}

    @property
    def output_slices(self) -> SliceDict:
        return dict(self._slices)

    @property
    def reasoning_metadata(self) -> dict:
        return dict(self._meta)

    def prepare_inputs(self, imgs_cl, numpy_inputs, frames):
        raise RuntimeError("supercombo runner has no OpenCL path; use run_fused()")

    def _run_model(self) -> NumpyDict:
        raise RuntimeError("supercombo path goes through run_fused(), not _run_model()")
