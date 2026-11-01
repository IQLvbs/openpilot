"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""
from __future__ import annotations

import os
os.environ.setdefault("XDG_CACHE_HOME", "/data/.cache")
import subprocess
import sys
import time

from iqpilot.system.hardware import TICI

os.environ.setdefault("GMMU", "0")
from iqpilot.selfdrive.iqmodeld.egpu_helpers import apply_dock_power_limit
apply_dock_power_limit()
if TICI:
  os.environ.setdefault("DEV", "QCOM")
else:
  os.environ.setdefault("DEV", "CPU")

import numpy as np
from setproctitle import setproctitle

import iqpilot.cereal.messaging as messaging
from iqpilot.cereal import car, log
from iqpilot.cereal.messaging import SubMaster
from iqpilot.cereal.services import SERVICE_LIST
from iqdbc.car.car_helpers import get_demo_car_params

from iqpilot.common.params import Params
from iqpilot.common.realtime import DT_MDL
from iqpilot.common.swaglog import cloudlog
from iqpilot.selfdrive.controls.lib.action_engine import ActionEngine
from iqpilot.system import sentry

from iqpilot.common.steer_delay import lateral_action_delay
from iqpilot.selfdrive.iqmodeld.daemon import CalibrationAtlas, CameraIngress, FrameDropMeter
from iqpilot.selfdrive.iqmodeld.driving_action import (
  DESIRE_LEN, LAT_SMOOTH_SECONDS, LONG_SMOOTH_SECONDS, get_action_from_model,
)
from iqpilot.selfdrive.iqmodeld.egpu_sched import MODEL_CORE, CpuBudgetGuard, enter_realtime, enter_setup
from iqpilot.selfdrive.iqmodeld.egpu_helpers import (
  die_with_parent, download_onnx, download_precompiled, drop_frozen_model_oob, egpu_model_oob_pkl_path, egpu_oob_pkl_path, egpu_pkl_path,
  egpu_policy_pkl_path, egpu_present_consented, egpu_selected, free_dock_lock, local_onnx,
  patch_tinygrad_fetch_fw, patch_usb_signal_wait, quarantine_artifact, resolve_backend, usbgpu_present,
)
from iqpilot.selfdrive.iqmodeld.egpu_model import resolve_egpu_model
from iqpilot.selfdrive.iqmodeld.aux_inputs import AuxModelInputs
from iqpilot.selfdrive.iqmodeld.iqmodel_io import publish_outputs
from iqpilot.selfdrive.iqmodeld.egpu_pipeline import EgpuOutputInvalid, EgpuPipeline, EgpuPipelineError, make_big_channel_payload
from iqpilot.selfdrive.iqmodeld.egpu_telemetry import EgpuDockTelemetry
from iqpilot.selfdrive.iqmodeld.messaging import DrivePacketMemory, populate_drive_messages, populate_odometry_message
from iqpilot.selfdrive.iqmodeld.metadata import Meta20hz
from iqpilot.selfdrive.iqmodeld.model_channel import BIG_CHANNEL, ModelChannel
from iqpilot.selfdrive.iqmodeld.egpu_policy import (
  MODEL_FORMAT, POLICY_FORMAT, STATEFUL_MODEL_FORMAT, STATEFUL_POLICY_FORMAT, ModelRunner, PolicyRunner, StatefulModelRunner,
  StatefulPolicyRunner, load_bundle,
)
from iqpilot.selfdrive.iqmodeld.model_warp import FrameWarp
from iqpilot.selfdrive.iqmodeld.parser import PhaseParser

PROCESS_NAME = "iqpilot.selfdrive.iqmodeld.iqegpumodeld"

PRESENCE_POLL_S = 5.0
COMPILE_TIMEOUT_S = 3600
LINK_UP_TIMEOUT_S = 10.0
SETUP_EXIT_AFTER = 3
MIN_LOAD_AVAIL_MB = 350
MEMORY_WAIT_S = 90.0
SETUP_RETRY_BASE_S = 3.0
SETUP_RETRY_MAX_S = 30.0
MAX_INVALID_STREAK = 20
STATE_GAP_RESET_FRAMES = 20
STATEFUL_WARMUP_FRAMES = 128
DOCK_MIN_SUPPLY_MV = 5000
DOCK_POWER_STABLE_POLLS = 4
DOCK_POWER_POLL_S = 0.1
DOCK_POWER_TIMEOUT_S = 30.0


class DockBusy(RuntimeError):
  pass


def park(reason: str) -> None:
  cloudlog.warning(f"iqegpumodeld parked: {reason}")
  params = Params()
  params.put_bool("UsbGpuFailed", True)
  params.put("UsbGpuLastError", reason[:512])
  while True:
    time.sleep(1)


def _wait_for_egpu(params: Params) -> None:
  was_absent = False
  while not usbgpu_present():
    if not was_absent:
      was_absent = True
      params.put_bool("UsbGpuPresent", False)
      params.put_bool("UsbGpuLoading", False)
      params.put("UsbGpuLastError", "eGPU dock not detected; check its USB cable and power")
    time.sleep(PRESENCE_POLL_S)
  params.put_bool("UsbGpuPresent", True)
  if was_absent:
    params.put_bool("UsbGpuLoading", True)
  try:
    from iqpilot.system.hardware.egpu_dock.flash import link_up
  except Exception:
    return
  deadline = time.monotonic() + LINK_UP_TIMEOUT_S
  while time.monotonic() < deadline:
    try:
      if link_up():
        break
    except Exception:
      return
    time.sleep(0.5)
  _wait_for_dock_power(params)


def dock_power_ready(reading) -> bool:
  voltage, _current, fault = reading
  return int(voltage) >= DOCK_MIN_SUPPLY_MV and not fault


def _wait_for_dock_power(params: Params) -> None:
  telemetry = EgpuDockTelemetry(None, big=False)
  deadline = time.monotonic() + DOCK_POWER_TIMEOUT_S
  stable = 0
  warned = False
  try:
    while time.monotonic() < deadline:
      try:
        reading = telemetry._read_ina()
      except Exception:
        return
      if reading is None:
        return
      stable = stable + 1 if dock_power_ready(reading) else 0
      if stable >= DOCK_POWER_STABLE_POLLS:
        return
      if stable == 0 and not warned:
        warned = True
        cloudlog.warning(f"iqegpumodeld dock supply not ready {reading}; waiting for a stable 5V rail")
        params.put("UsbGpuLastError", f"dock supply not ready (voltage={reading[0]}mV fault={reading[2]}); waiting")
      time.sleep(DOCK_POWER_POLL_S)
    cloudlog.warning("iqegpumodeld dock supply never stabilised; continuing")
  finally:
    handle = getattr(telemetry, "_asm_usb", None)
    if handle is not None:
      try:
        handle.close()
      except Exception:
        pass


def _compile_child_setup() -> None:
  os.nice(20)
  # an orphaned compile keeps the dock's flock, so the next iqegpumodeld could never open it
  die_with_parent()


def _compile_in_subprocess(meta: dict, onnx_path: str, pkl_path: str, cam_size: tuple[int, int]) -> None:
  cmd = [sys.executable, "-m", "iqpilot.selfdrive.iqmodeld.tools.compile_egpu_model",
         "--model", meta["key"], "--onnx", onnx_path, "--output", pkl_path,
         "--format", str(MODEL_FORMAT), "--camera-resolutions", f"{cam_size[0]}x{cam_size[1]}",
         "--progress-param", "UsbGpuSetupProgress", "--progress-base", "0.5", "--progress-span", "0.48"]
  compile_env = {**os.environ, "DEV": "USB+AMD:LLVM", "FLOAT16": "1",
                 "JIT_BATCH_SIZE": "0", "GMMU": "0", "TC_OPT": "2"}
  proc = subprocess.run(cmd, timeout=COMPILE_TIMEOUT_S, capture_output=True, text=True,
                        env=compile_env, preexec_fn=_compile_child_setup)
  if proc.returncode != 0:
    tail = (proc.stderr or proc.stdout or "").strip()[-800:]
    raise RuntimeError(f"eGPU model compile failed (rc={proc.returncode}): {tail}")


_precompiled_tried = False
_model_precompiled_tried = False


def _ensure_artifact(params: Params, meta: dict, cam_size: tuple[int, int]) -> str:
  global _precompiled_tried, _model_precompiled_tried
  drop_frozen_model_oob(meta)
  model_path = egpu_model_oob_pkl_path(meta)
  if os.path.isfile(model_path):
    return model_path
  if meta.get("egpu_model_oob_artifact") and not _model_precompiled_tried:
    _model_precompiled_tried = True
    params.put_bool("UsbGpuCompiled", False)
    params.put_bool("UsbGpuReady", False)
    params.put("UsbGpuSetupProgress", "0.0")
    model_last = [-1.0]

    def _model_prog(p: float) -> None:
      if p - model_last[0] >= 0.02 or p >= 1.0:
        model_last[0] = p
        params.put("UsbGpuSetupProgress", f"{p:.3f}")

    try:
      size_mb = int(meta["egpu_model_oob_artifact"].get("size", 0)) / 1e6
      cloudlog.warning(f"iqegpumodeld downloading precompiled {meta['key']} (warp-on-dock, {size_mb:.0f}MB)")
      precompiled = download_precompiled(meta, progress_cb=_model_prog, field="egpu_model_oob_artifact")
      if precompiled is not None:
        cloudlog.warning(f"iqegpumodeld precompiled ready -> {precompiled}")
        return precompiled
    except Exception as e:
      cloudlog.warning(f"iqegpumodeld warp-on-dock artifact unavailable ({e}); falling back")

  oob_path = egpu_oob_pkl_path(meta)
  if os.path.isfile(oob_path):
    return oob_path
  policy_path = egpu_policy_pkl_path(meta)
  legacy_path = egpu_pkl_path(meta)

  params.put_bool("UsbGpuCompiled", False)
  params.put_bool("UsbGpuReady", False)

  if meta.get("egpu_oob_artifact") and not _precompiled_tried:
    _precompiled_tried = True
    params.put("UsbGpuSetupProgress", "0.0")
    oob_last = [-1.0]

    def _oob_prog(p: float) -> None:
      if p - oob_last[0] >= 0.02 or p >= 1.0:
        oob_last[0] = p
        params.put("UsbGpuSetupProgress", f"{p:.3f}")

    try:
      cloudlog.warning(f"iqegpumodeld downloading precompiled {meta['key']} (streamable) "
                       f"({int(meta['egpu_oob_artifact'].get('size', 0)) / 1e6:.0f}MB)")
      precompiled = download_precompiled(meta, progress_cb=_oob_prog, oob=True)
      if precompiled is not None:
        cloudlog.warning(f"iqegpumodeld precompiled ready -> {precompiled}")
        return precompiled
    except Exception as e:
      cloudlog.warning(f"iqegpumodeld streamable artifact unavailable ({e}); falling back")

  if os.path.isfile(policy_path):
    return policy_path

  if meta.get("egpu_policy_artifact") and not _precompiled_tried:
    _precompiled_tried = True
    params.put("UsbGpuSetupProgress", "0.0")
    dl_last = [-1.0]

    def _dl_prog(p: float) -> None:
      if p - dl_last[0] >= 0.02 or p >= 1.0:
        dl_last[0] = p
        params.put("UsbGpuSetupProgress", f"{p:.3f}")

    try:
      cloudlog.warning(f"iqegpumodeld downloading precompiled {meta['key']} policy "
                       f"({int(meta['egpu_policy_artifact'].get('size', 0)) / 1e6:.0f}MB)")
      precompiled = download_precompiled(meta, progress_cb=_dl_prog, policy=True)
      if precompiled is not None:
        cloudlog.warning(f"iqegpumodeld precompiled ready -> {precompiled}")
        return precompiled
    except Exception as e:
      cloudlog.warning(f"iqegpumodeld precompiled policy unavailable ({e}); falling back")

  if os.path.isfile(legacy_path):
    cloudlog.warning(f"iqegpumodeld using legacy per-tensor artifact {legacy_path}; policy artifact not hosted yet")
    return legacy_path

  onnx_path = local_onnx(meta)
  if onnx_path is None:
    params.put("UsbGpuSetupProgress", "0.0")
    cloudlog.warning(f"iqegpumodeld downloading {meta['key']} onnx ({meta.get('download', {}).get('size', 0) / 1e6:.0f}MB)")
    last = [-1.0]

    def _prog(p: float) -> None:
      if p - last[0] >= 0.02 or p >= 1.0:
        last[0] = p
        params.put("UsbGpuSetupProgress", f"{p * 0.5:.3f}")

    onnx_path = download_onnx(meta, progress_cb=_prog)

  cloudlog.warning(f"iqegpumodeld compiling {meta['key']} for USB-AMD ({cam_size[0]}x{cam_size[1]}, one-time, can take minutes)")
  _compile_in_subprocess(meta, onnx_path, model_path, cam_size)
  cloudlog.warning(f"iqegpumodeld compiled -> {model_path}")
  return model_path


def _mem_available_mb() -> int:
  try:
    with open("/proc/meminfo") as f:
      for line in f:
        if line.startswith("MemAvailable:"):
          return int(line.split()[1]) // 1024
  except OSError:
    pass
  return 1 << 20


def _wait_for_memory(need_mb: int) -> None:
  deadline = time.monotonic() + MEMORY_WAIT_S
  avail = _mem_available_mb()
  while avail < need_mb and time.monotonic() < deadline:
    cloudlog.warning(f"iqegpumodeld waiting for memory: {avail}MB available, need {need_mb}MB")
    time.sleep(5.0)
    avail = _mem_available_mb()
  if avail < need_mb:
    raise RuntimeError(f"insufficient memory to load the dock model: {avail}MB available, need {need_mb}MB")


def _load_infer_fn(pkl_path: str, meta: dict, cam_size: tuple[int, int]):
  patch_tinygrad_fetch_fw()
  patch_usb_signal_wait()
  from tinygrad.tensor import Tensor

  _wait_for_memory(MIN_LOAD_AVAIL_MB)
  bundle = load_bundle(pkl_path)
  if bundle.get("model_sha256") != meta["sha256"]:
    quarantine_artifact(pkl_path, "pkl model sha mismatch")
    raise RuntimeError(f"artifact model sha {bundle.get('model_sha256')} != {meta['sha256']}")
  if int(bundle.get("output_len", -1)) != int(meta["output_len"]):
    quarantine_artifact(pkl_path, "pkl output_len mismatch")
    raise RuntimeError(f"artifact output_len {bundle.get('output_len')} != {meta['output_len']}")
  if bundle.get("format") == STATEFUL_MODEL_FORMAT:
    jits = bundle["run_model"]
    if cam_size not in jits:
      have = ", ".join(f"{w}x{h}" for w, h in sorted(jits))
      raise RuntimeError(f"artifact has no warp for the {cam_size[0]}x{cam_size[1]} camera (bundled: {have})")
    runner = StatefulModelRunner(jits[cam_size], bundle["input_spec"], bundle["state_pairs"], bundle.get("input_device", "AMD"),
                                 int(bundle["frame_copy_size"][cam_size]))
    return runner, bundle["input_spec"]
  if bundle.get("format") == STATEFUL_POLICY_FORMAT:
    runner = StatefulPolicyRunner(bundle["run_policy"], bundle["input_spec"], bundle["state_pairs"], bundle.get("input_device", "AMD"))
    return runner, bundle["input_spec"]
  if bundle.get("format") == MODEL_FORMAT:
    jits = bundle["run_model"]
    if cam_size not in jits:
      have = ", ".join(f"{w}x{h}" for w, h in sorted(jits))
      raise RuntimeError(f"artifact has no warp for the {cam_size[0]}x{cam_size[1]} camera (bundled: {have})")
    runner = ModelRunner(jits[cam_size], bundle["input_spec"], int(bundle["frame_skip"]), meta["output_slices"]["hidden_state"],
                         bundle.get("input_device", "AMD"), int(bundle["frame_copy_size"][cam_size]))
    return runner, bundle["input_spec"]
  if bundle.get("format") == POLICY_FORMAT:
    runner = PolicyRunner(bundle["run_policy"], bundle["input_spec"], int(bundle["frame_skip"]),
                          meta["output_slices"]["hidden_state"], bundle.get("input_device", "AMD"))
    return runner, bundle["input_spec"]
  jit = bundle["run_model"]
  input_dev = bundle.get("input_device", "AMD")
  input_spec = bundle["input_spec"]

  def infer(inputs: dict[str, np.ndarray]) -> np.ndarray:
    tensors = {name: Tensor(np.ascontiguousarray(inputs[name]), device=input_dev).realize()
               for name in input_spec}
    out, = jit(**tensors)
    return out.numpy().reshape(-1)

  return infer, input_spec


def _warmup(infer_fn, input_spec: dict, output_len: int) -> float:
  zeros = {name: np.zeros(shape, dtype=dtype) for name, (shape, dtype) in input_spec.items()}
  t0 = time.perf_counter()
  if isinstance(infer_fn, StatefulModelRunner):
    n = infer_fn.frame_copy_size
    eye = np.eye(3, dtype=np.float32)
    out = infer_fn.run(np.zeros(n, dtype=np.uint8), np.zeros(n, dtype=np.uint8), eye, eye,
                       np.zeros(input_spec["desire"][0][0], dtype=np.float32), np.zeros(2, dtype=np.float32), np.zeros(2, dtype=np.float32))
    infer_fn.reset()
  elif isinstance(infer_fn, StatefulPolicyRunner):
    out = infer_fn.run(np.zeros(input_spec["new_img"][0], dtype=np.uint8), np.zeros(input_spec["desire"][0][0], dtype=np.float32),
                       np.zeros(2, dtype=np.float32), np.zeros(2, dtype=np.float32))
    infer_fn.reset()
  elif isinstance(infer_fn, ModelRunner):
    n = infer_fn.frame_copy_size
    eye = np.eye(3, dtype=np.float32)
    out = infer_fn.run(np.zeros(n, dtype=np.uint8), np.zeros(n, dtype=np.uint8), eye, eye,
                       np.zeros(input_spec["desire_pulse"][0][2], dtype=np.float32), np.zeros(2, dtype=np.float32), np.zeros(2, dtype=np.float32))
  elif isinstance(infer_fn, PolicyRunner):
    img = input_spec["img"][0]
    out = infer_fn.run(np.zeros((2, 6, img[2], img[3]), dtype=np.uint8), np.zeros(input_spec["desire_pulse"][0][2], dtype=np.float32),
                       np.zeros(2, dtype=np.float32), np.zeros(2, dtype=np.float32))
  else:
    out = infer_fn(zeros)
  dt = time.perf_counter() - t0
  if out.shape[0] != output_len or not np.isfinite(out).all():
    raise RuntimeError(f"warmup produced invalid output (len={out.shape[0]})")
  return dt


def main(demo: bool = False) -> None:
  cloudlog.warning("iqegpumodeld init")
  sentry.set_tag("daemon", PROCESS_NAME)
  cloudlog.bind(daemon=PROCESS_NAME)
  setproctitle(PROCESS_NAME)
  enter_setup()

  params = Params()
  backend = resolve_backend(params.get_bool("IQEmacEnabled"), egpu_selected(params), egpu_present_consented(params))
  if backend != "egpu":
    park(f"backend resolution is {backend!r}, not egpu; refusing to own the big channel")

  channel = ModelChannel(BIG_CHANNEL, create=True)

  cloudlog.warning("iqegpumodeld waiting for camerad")
  cameras = CameraIngress(None)
  layout = cameras.layout

  _wait_for_egpu(params)
  params.put_bool("UsbGpuLoading", True)
  attempt = 0
  opened_dock = False
  while True:
    try:
      meta = resolve_egpu_model(params)
      if meta is None:
        raise RuntimeError("selected big model is not in the catalog; check connectivity or pick another model")
      if meta.get("split"):
        params.put_bool("UsbGpuLoading", False)
        park(f"model {meta['key']} needs the Mac backend; the eGPU runs fused models only")
      cam_size = (int(cameras._primary.width), int(cameras._primary.height))
      pkl_path = _ensure_artifact(params, meta, cam_size)
      held = free_dock_lock()
      if held:
        raise DockBusy("eGPU dock is held by another process: " + ", ".join(f"pid {pid} {cmd[:60]}" for pid, cmd in held))
      opened_dock = True
      infer_fn, input_spec = _load_infer_fn(pkl_path, meta, cam_size)
      warp = None if isinstance(infer_fn, ModelRunner) else FrameWarp(cam_size[0], cam_size[1], meta["frame_skip"])
      warm_s = _warmup(infer_fn, input_spec, meta["output_len"])
      break
    except Exception as e:
      attempt += 1
      subs = "; ".join(f"{type(x).__name__}: {x}" for x in (getattr(e, "exceptions", None) or []))
      params.put("UsbGpuLastError", (f"{e} [{subs}]" if subs else str(e))[:512])
      cloudlog.warning(f"iqegpumodeld setup attempt {attempt} failed: {e}; {subs}; retrying")
      if opened_dock:
        # a failed device init keeps this process's flock on the dock, so every in-process retry would fail on
        # our own lock and overwrite the real error above; restart clean and keep the first error visible
        cloudlog.error("iqegpumodeld dock init failed; exiting for a clean restart")
        sys.exit(1)
      if attempt >= SETUP_EXIT_AFTER:
        # tinygrad keeps the dock's flock in a failed device init, so a stale process can never
        # reopen it; exit and let the manager respawn a clean one.
        cloudlog.error(f"iqegpumodeld giving up after {attempt} setup failures; exiting for a clean restart")
        sys.exit(1)
      if not usbgpu_present():
        from iqpilot.system.hardware.usb import ensure_host_role
        if ensure_host_role():
          cloudlog.warning("iqegpumodeld: Type-C controller was out of host mode; restored")
          time.sleep(2.0)
        _wait_for_egpu(params)
      time.sleep(min(SETUP_RETRY_MAX_S, SETUP_RETRY_BASE_S * attempt))

  params.put_bool("UsbGpuLoading", False)
  params.put_bool("UsbGpuCompiled", True)
  params.put_bool("UsbGpuReady", True)
  params.put("UsbGpuSetupProgress", "1.0")
  cloudlog.warning(f"iqegpumodeld model: {meta['key']} ({meta['model_name']})")
  cloudlog.warning(f"iqegpumodeld model up (warmup {warm_s * 1e3:.0f}ms, {'warp on dock' if warp is None else 'warp on device'})")

  pipeline = EgpuPipeline(meta, infer_fn)
  aux_inputs = AuxModelInputs(input_spec)
  warmup_frames = int(meta.get("warmup_frames") or STATEFUL_WARMUP_FRAMES)
  telemetry_pm = messaging.PubMaster(["egpuDockState"])
  telemetry = EgpuDockTelemetry(telemetry_pm, big=True)
  telemetry_every = max(1, round((1.0 / DT_MDL) / SERVICE_LIST["egpuDockState"].frequency))

  sub = SubMaster(["deviceState", "carState", "roadCameraState", "extrinsicsCalibration",
                   "driverMonitoringState", "carControl", "controlsState", "lateralDelay", "iqNavState", "radarState"])
  if demo:
    CP = get_demo_car_params()
  else:
    CP = messaging.log_from_bytes(params.get("CarParams", block=True), car.CarParams)
  long_delay = CP.longitudinalActuatorDelay + LONG_SMOOTH_SECONDS

  parser = PhaseParser()
  memory = DrivePacketMemory()
  desire_logic = ActionEngine()
  frame_meter = FrameDropMeter(20.0)
  warps = CalibrationAtlas()
  prev_action = log.ModelDataV2.Action()
  slices = {k: v for k, v in meta["output_slices"].items() if k != "pad"}

  produced = 0
  stats: dict[str, list[float]] = {k: [] for k in ("pull", "warp", "infer", "publish", "loop")}
  iter_count = 0
  skip_count = 0
  invalid_streak = 0
  last_pulled_fid = -1
  last_frame_mono = time.monotonic()
  t_loop = time.perf_counter()
  budget = CpuBudgetGuard() if enter_realtime() else None
  cloudlog.warning(f"iqegpumodeld starting ({'realtime' if budget else 'normal priority'} on core {MODEL_CORE})")

  while True:
    if budget is not None:
      budget.tick()
    frame_pair = cameras.pull()
    t_pull = time.perf_counter()
    if frame_pair is None:
      if time.monotonic() - last_frame_mono > 2.0:
        cloudlog.warning("iqegpumodeld camera stream silent >2s; reconnecting VisionIPC")
        cameras = CameraIngress(None)
        last_frame_mono = time.monotonic()
      continue
    last_frame_mono = time.monotonic()
    main_buf, extra_buf, main_stamp, extra_stamp = frame_pair

    stats["pull"].append(t_pull - t_loop)
    stats["loop"].append(time.perf_counter() - t_loop)
    t_loop = time.perf_counter()
    if last_pulled_fid >= 0 and main_stamp.frame_id > last_pulled_fid + 1:
      skip_count += main_stamp.frame_id - last_pulled_fid - 1
    if pipeline.stateful and last_pulled_fid >= 0 and not 0 <= main_stamp.frame_id - last_pulled_fid <= STATE_GAP_RESET_FRAMES:
      cloudlog.warning(f"iqegpumodeld frame gap {last_pulled_fid}->{main_stamp.frame_id}: resetting model state")
      pipeline.reset_state()
    last_pulled_fid = main_stamp.frame_id
    iter_count += 1
    if iter_count % 200 == 0:
      pcts = {k: {"p50": round(sorted(v)[len(v) // 2] * 1e3, 1),
                  "p90": round(sorted(v)[int(len(v) * 0.9)] * 1e3, 1)}
              for k, v in stats.items() if v}
      cloudlog.event("iqegpu_stats", **pcts, cam_skips=skip_count, window=iter_count)
      msg = " ".join(f"{k}=p50:{v['p50']:.0f}/p90:{v['p90']:.0f}ms" for k, v in pcts.items())
      cloudlog.warning(f"iqegpumodeld stages: {msg} cam_skips={skip_count} over {iter_count}")
      for v in stats.values():
        v.clear()
      skip_count = 0

    sub.update(0)

    v_ego = max(sub["carState"].vEgo, 0.0)
    lat_delay = lateral_action_delay(params, CP, sub["lateralDelay"].lateralDelay) + LAT_SMOOTH_SECONDS
    main_tfm, extra_tfm, live_calib_seen = warps.refresh(sub, layout.main_is_wide, layout.dual_camera)
    dropped_frames, frame_drop_ratio, _ = frame_meter.sample(main_stamp.frame_id)

    traffic = np.zeros(2, dtype=np.float32)
    traffic[int(sub["driverMonitoringState"].isRHD)] = 1
    desire_vec = np.zeros(DESIRE_LEN, dtype=np.float32)
    if 0 <= desire_logic.desire < DESIRE_LEN:
      desire_vec[desire_logic.desire] = 1

    frame_delay = DT_MDL
    action_delay = DT_MDL / 2
    lat_action_t = lat_delay + frame_delay + action_delay
    long_action_t = long_delay + frame_delay + action_delay
    action_t = np.array([lat_action_t, long_action_t], dtype=np.float32)

    aux_inputs.poll()
    aux_extras = aux_inputs.extras()

    started_at = time.perf_counter()
    t_warp = started_at
    try:
      if warp is None:
        output = pipeline.run_frames(main_buf.data, extra_buf.data, main_tfm, extra_tfm, desire_vec, traffic, action_t, aux_extras)
      else:
        try:
          warped = warp.run(main_buf, extra_buf, main_tfm, extra_tfm)
        except Exception as e:
          park(f"warp run failed: {e}")
        t_warp = time.perf_counter()
        output = pipeline.run(warped, desire_vec, traffic, action_t, aux_extras)
    except EgpuOutputInvalid as e:
      invalid_streak += 1
      if invalid_streak == 1 or invalid_streak % MAX_INVALID_STREAK == 0:
        cloudlog.warning(f"iqegpumodeld dropping frame {main_stamp.frame_id}: {e} (streak {invalid_streak})")
      if invalid_streak >= MAX_INVALID_STREAK:
        params.put("UsbGpuLastError", f"{e} for {invalid_streak} consecutive frames"[:512])
        cloudlog.error(f"iqegpumodeld output invalid for {invalid_streak} frames; exiting for a clean restart")
        sys.exit(1)
      frame_meter.commit(main_stamp.frame_id)
      continue
    except EgpuPipelineError as e:
      park(str(e))
    except Exception as e:
      park(f"eGPU inference failed: {e}")
    invalid_streak = 0
    t_infer = time.perf_counter()
    stats["warp"].append(t_warp - started_at)
    stats["infer"].append(t_infer - t_warp)
    if pipeline.warm_frames < warmup_frames:
      frame_meter.commit(main_stamp.frame_id)
      continue

    execution_time = time.perf_counter() - started_at
    sliced = {k: output[np.newaxis, sl] for k, sl in slices.items()}
    outputs = parser.parse_vision_outputs(sliced)

    action = get_action_from_model(outputs, prev_action, v_ego, float(lat_action_t), float(long_action_t),
                                   lat_smooth_seconds=meta.get("lat_smooth_seconds"),
                                   lat_active=sub["carControl"].latActive, current_curvature=sub["controlsState"].curvature)
    prev_action = action

    model_msg = messaging.new_message("modelV2")
    driving_msg = messaging.new_message("drivingModelData")
    pose_msg = messaging.new_message("cameraOdometry")
    iq_msg = messaging.new_message("iqDriveModelData")

    populate_drive_messages(
      driving_msg, model_msg, outputs, action, memory,
      main_stamp.frame_id, extra_stamp.frame_id, sub["roadCameraState"].frameId,
      frame_drop_ratio, main_stamp.timestamp_eof, execution_time,
      live_calib_seen, Meta20hz,
    )

    model_msg.modelV2.big = True
    driving_msg.drivingModelData.big = True

    desire_state = model_msg.modelV2.meta.desireState
    lane_change_prob = desire_state[log.Desire.laneChangeLeft] + desire_state[log.Desire.laneChangeRight]
    desire_logic.update(sub["carState"], sub["carControl"].latActive, lane_change_prob,
                        sub["iqNavState"], model_msg.modelV2, sub["radarState"])
    model_msg.modelV2.meta.laneChangeState = desire_logic.lane_change_state
    model_msg.modelV2.meta.laneChangeDirection = desire_logic.lane_change_direction
    driving_msg.drivingModelData.meta.laneChangeState = desire_logic.lane_change_state
    driving_msg.drivingModelData.meta.laneChangeDirection = desire_logic.lane_change_direction
    iq_msg.iqDriveModelData.turnSignalDirection = desire_logic.lane_turn_direction
    publish_outputs(model_msg.modelV2, iq_msg.iqDriveModelData, outputs)

    populate_odometry_message(pose_msg, outputs, main_stamp.frame_id, dropped_frames,
                              main_stamp.timestamp_eof, live_calib_seen)

    channel.write(main_stamp.frame_id, make_big_channel_payload(
      main_stamp.frame_id, live_calib_seen, execution_time, (t_infer - t_warp) * 1e3, {
        "modelV2": model_msg.to_bytes(),
        "drivingModelData": driving_msg.to_bytes(),
        "cameraOdometry": pose_msg.to_bytes(),
        "iqDriveModelData": iq_msg.to_bytes(),
      }))
    stats["publish"].append(time.perf_counter() - t_infer)
    produced += 1
    if produced == 1 or produced % 100 == 0:
      infer_ms = (t_infer - t_warp) * 1e3
      cloudlog.warning(f"iqegpumodeld producing: frame={main_stamp.frame_id} total={execution_time * 1e3:.0f}ms infer={infer_ms:.0f}ms count={produced}")

    if produced % telemetry_every == 0:
      telemetry.send()

    frame_meter.commit(main_stamp.frame_id)


if __name__ == "__main__":
  try:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    main(demo=args.demo)
  except KeyboardInterrupt:
    cloudlog.warning("iqegpumodeld got SIGINT")
  except Exception:
    import traceback
    sentry.capture_exception()
    cloudlog.exception("iqegpumodeld crashed, parking")
    park(f"crashed: {traceback.format_exc(limit=8)}")
