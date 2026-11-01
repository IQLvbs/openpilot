"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections import deque

from iqpilot.cereal.messaging import PubMaster, log_from_bytes
from setproctitle import setproctitle

from iqpilot.common.filter_simple import FirstOrderFilter
from iqpilot.common.params import Params
from iqpilot.common.realtime import config_realtime_process
from iqpilot.common.swaglog import cloudlog
from iqpilot.selfdrive.iqmodeld.model_channel import BIG_CHANNEL, SMALL_CHANNEL, ModelChannel

PROCESS_NAME = "iqpilot.selfdrive.iqmodeld.modeld_selector"

BIG_MODEL_DEADLINE = float(os.getenv("IQEMAC_BIG_DEADLINE_MS", "45")) / 1000.0
BIG_MAX_LAG_FRAMES = int(os.getenv("IQEMAC_MAX_BIG_LAG_FRAMES", "6"))
BIG_FUTURE_ACCEPT = int(os.getenv("IQEMAC_BIG_FUTURE_ACCEPT", "2"))
BIG_ANCHOR_MS = float(os.getenv("IQEMAC_BIG_ANCHOR_MS", "90"))
BIG_WAIT_FLOOR_S = 0.002
BIG_WAIT_CEIL_S = float(os.getenv("IQEMAC_BIG_WAIT_CEIL_MS", "58")) / 1000.0
BIG_MISS_LIMIT = int(os.getenv("IQEMAC_BIG_MISS_LIMIT", "80"))
ACTIVATE_WINDOW = int(os.getenv("IQEMAC_ACTIVATE_WINDOW", "50"))
ACTIVATE_FRAC = float(os.getenv("IQEMAC_ACTIVATE_FRAC", "0.7"))
REARM_LIMIT = int(os.getenv("IQEMAC_REARM_LIMIT", "2"))
MODEL_FREQ = 20.0
WARMUP_FRAMES = 40
STATUS_WINDOW = int(os.getenv("IQEMAC_STATUS_EVERY", "20"))

SELECTOR_SERVICES = ["modelV2", "drivingModelData", "cameraOdometry", "iqDriveModelData"]

EMAC_STATUS_KEYS = {
  "active": "MacModelActive", "failed": "MacModelFailed", "last_error": "MacModelLastError",
  "latency_ms": "MacModelLatencyMs", "status": "MacModelStatus",
  "reachable": "MacModelReachable", "progress": "MacModelDownloadProgress",
}
EGPU_STATUS_KEYS = {
  "active": "UsbGpuActive", "failed": "UsbGpuFailed", "last_error": "UsbGpuLastError",
  "latency_ms": "UsbGpuLatencyMs", "status": "UsbGpuStatus",
  "reachable": "UsbGpuPresent", "progress": "UsbGpuSetupProgress",
}


def backend_status_keys(emac_enabled: bool, egpu_enabled: bool, egpu_present: bool = False) -> dict[str, str]:
  from iqpilot.selfdrive.iqmodeld.egpu_helpers import resolve_backend
  return EGPU_STATUS_KEYS if resolve_backend(emac_enabled, egpu_enabled, egpu_present) == "egpu" else EMAC_STATUS_KEYS


def resolve_status_keys(params):
  from iqpilot.selfdrive.iqmodeld.egpu_helpers import egpu_present_consented, egpu_selected
  return backend_status_keys(params.get_bool("IQEmacEnabled"), egpu_selected(params), egpu_present_consented(params))


def resolve_model_name(params, keys) -> str:
  if keys is EGPU_STATUS_KEYS:
    from iqpilot.selfdrive.iqmodeld.egpu_model import DEFAULT_EGPU_MODEL, resolve_egpu_model
    resolved = resolve_egpu_model(params, allow_refresh=False)
    return resolved["key"] if resolved else DEFAULT_EGPU_MODEL
  if params.get_bool("IQEmacSmallModel"):
    from iqpilot.selfdrive.iqmodeld.models.helpers import get_active_bundle
    bundle = get_active_bundle(params)
    if bundle is not None and (bundle.internalName or bundle.displayName):
      return bundle.internalName or bundle.displayName
  name = params.get("IQEmacModel") or b"lebrowski"
  return name.decode() if isinstance(name, bytes) else name


class AsyncParamWriter:

  def __init__(self, params: Params):
    self._params = params
    self._pending: dict[str, object] = {}
    self._lock = threading.Lock()
    self._event = threading.Event()
    threading.Thread(target=self._drain, daemon=True).start()

  def put(self, key: str, value) -> None:
    with self._lock:
      self._pending[key] = value
    self._event.set()

  def put_bool(self, key: str, value: bool) -> None:
    self.put(key, bool(value))

  def _drain(self) -> None:
    while True:
      self._event.wait()
      self._event.clear()
      with self._lock:
        batch, self._pending = self._pending, {}
      for key, value in batch.items():
        try:
          if isinstance(value, bool):
            self._params.put_bool(key, value)
          else:
            self._params.put(key, value)
        except Exception:
          cloudlog.exception(f"async param write failed: {key}")


def wait_for_big(big_channel, target: int, deadline: float, min_frame: int = -1,
                 max_lag_frames: int = BIG_MAX_LAG_FRAMES) -> tuple[dict | None, int | None]:
  big_peek = None
  grab_at = deadline - 0.004
  while time.perf_counter() < deadline:
    bfid, writing = big_channel.peek()
    if writing:
      time.sleep(0.0002)
      continue
    big_peek = bfid
    if bfid == target - 1 and time.perf_counter() < grab_at:
      time.sleep(0.0005)
      continue
    if bfid is not None and min_frame < bfid <= target + BIG_FUTURE_ACCEPT and target - bfid <= max_lag_frames:
      got = big_channel.read()
      if got is not None and got[0] == bfid:
        return got[1], big_peek
      time.sleep(0.0002)
      continue
    if bfid is None or bfid <= min_frame or bfid > target + BIG_FUTURE_ACCEPT or target - bfid > max_lag_frames:
      break
    time.sleep(0.0005)
  return None, big_peek


def hold_big_plan(active: bool, last_big_payload: dict | None, last_big_published: int, target: int,
                  max_lag_frames: int = BIG_MAX_LAG_FRAMES) -> dict | None:
  # a big model slower than 20 Hz left gaps the small model filled, so two different models steered on alternate frames
  if not active or last_big_payload is None or last_big_published < 0:
    return None
  return last_big_payload if 0 <= target - last_big_published <= max_lag_frames else None


class BigLatch:

  def __init__(self, miss_limit: int = BIG_MISS_LIMIT, activate_window: int = ACTIVATE_WINDOW,
               activate_frac: float = ACTIVATE_FRAC, rearm_limit: int = REARM_LIMIT):
    self.miss_limit = miss_limit
    self.activate_window = activate_window
    self.activate_need = int(round(activate_window * activate_frac))
    self.rearm_limit = rearm_limit
    self.active = False
    self.done = False
    self._miss = 0
    self._window: deque[bool] = deque(maxlen=activate_window)
    self._retires = 0

  def update(self, used_big: bool) -> tuple[bool, bool]:
    if self.done:
      return False, False
    if not self.active:
      self._window.append(used_big)
      if len(self._window) >= self.activate_window and sum(self._window) >= self.activate_need:
        self.active = True
        self._miss = 0
        self._window.clear()
        return True, False
    if used_big:
      self._miss = 0
    elif self.active:
      self._miss += 1
      if self._miss >= self.miss_limit:
        self.active = False
        self._miss = 0
        self._window.clear()
        self._retires += 1
        self.done = self._retires > self.rearm_limit
        return False, True
    return False, False


BIG_SOURCES = frozenset({"egpu_big", "mac_big"})


def _patch_and_send(pm: PubMaster, payload: dict, frame_drop_perc: float, selector_dropped: int,
                    target: int, source_lag: int, mismatch: bool | None = None, pose_payload: dict | None = None,
                    backup_payload: dict | None = None) -> None:
  msgs = payload["msgs"]
  if mismatch is None:
    mismatch = source_lag > 0

  big = bool(payload.get("big", payload.get("source") in BIG_SOURCES))
  model_msg = log_from_bytes(msgs["modelV2"]).as_builder()
  if mismatch:
    model_msg.modelV2.frameId = target
    model_msg.modelV2.frameAge = max(model_msg.modelV2.frameAge, source_lag)
  model_msg.modelV2.frameDropPerc = frame_drop_perc
  model_msg.modelV2.big = big
  pm.send("modelV2", model_msg)

  driving_msg = log_from_bytes(msgs["drivingModelData"]).as_builder()
  if mismatch:
    driving_msg.drivingModelData.frameId = target
  driving_msg.drivingModelData.frameDropPerc = frame_drop_perc
  driving_msg.drivingModelData.big = big
  pm.send("drivingModelData", driving_msg)

  # a stale big plan still beats the fallback, but its pose is for another frame; paramsd needs this frame's
  pose_src, pose_mismatch = payload, mismatch
  if mismatch and pose_payload is not None:
    pose_src, pose_mismatch = pose_payload, False
  pose_msg = log_from_bytes(pose_src["msgs"]["cameraOdometry"]).as_builder()
  if pose_mismatch:
    pose_msg.cameraOdometry.frameId = target
  pose_msg.valid = bool(pose_src["live_calib_seen"]) and selector_dropped < 1 and not pose_mismatch
  pm.send("cameraOdometry", pose_msg)

  if not big or backup_payload is None:
    pm.send("iqDriveModelData", msgs["iqDriveModelData"])
    return
  iq_msg = log_from_bytes(msgs["iqDriveModelData"]).as_builder()
  backup = log_from_bytes(backup_payload["msgs"]["modelV2"]).modelV2.position
  path = iq_msg.iqDriveModelData.backupPath
  path.x, path.y, path.z = list(backup.x), list(backup.y), list(backup.z)
  pm.send("iqDriveModelData", iq_msg)


def _read_float(params, key: str, default: float) -> float:
  v = params.get(key)
  try:
    return float(v) if v is not None else default
  except (TypeError, ValueError):
    return default


def main() -> None:
  cloudlog.warning("modeld_selector init")
  cloudlog.bind(daemon=PROCESS_NAME)
  setproctitle(PROCESS_NAME)
  config_realtime_process([0, 1, 2, 3], 54)

  params = Params()
  keys = resolve_status_keys(params)
  pwriter = AsyncParamWriter(params)
  pwriter.put_bool(keys["active"], False)
  pwriter.put_bool(keys["failed"], False)
  pm = PubMaster(SELECTOR_SERVICES)

  small_channel: ModelChannel | None = None
  big_channel: ModelChannel | None = None
  latch = BigLatch()
  big_used_count = 0
  run_count = 0
  last_published = -1
  last_big_published = -1
  last_big_payload: dict | None = None
  held_count = 0
  frame_dropped_filter = FirstOrderFilter(0.0, 10.0, 1.0 / MODEL_FREQ)
  recent_big = deque(maxlen=STATUS_WINDOW)
  model_name = resolve_model_name(params, keys)
  last_backend_check = 0.0
  last_latency_ms = 0.0
  last_source_lag = 0
  miss_reasons = {"no_head": 0, "already_used": 0, "far_future": 0,
                  "too_stale": 0, "head_prev_timeout": 0, "read_race": 0}

  cloudlog.warning(f"modeld_selector starting (max_big_lag_frames={BIG_MAX_LAG_FRAMES})")
  while True:
    if small_channel is None:
      try:
        small_channel = ModelChannel(SMALL_CHANNEL, create=False)
      except OSError:
        time.sleep(0.05)
        continue
    if big_channel is None:
      try:
        big_channel = ModelChannel(BIG_CHANNEL, create=False)
      except OSError:
        big_channel = None

    fid = small_channel.peek_frame_id()
    if fid is None or fid == last_published:
      time.sleep(0.0005)
      continue
    if last_published >= 0 and fid < last_published - 1:
      cloudlog.warning(f"modeld_selector frame reset {last_published} -> {fid}; re-arming")
      last_published = -1
      last_big_published = -1
      last_big_payload = None
      big_used_count = 0
      run_count = 0
      latch = BigLatch()
      pwriter.put_bool(keys["active"], False)
      pwriter.put_bool(keys["failed"], False)

    now_mono = time.monotonic()
    if now_mono - last_backend_check > 1.0:
      last_backend_check = now_mono
      new_keys = resolve_status_keys(params)
      if new_keys is not keys:
        cloudlog.warning(f"modeld_selector backend changed {keys['active']} -> {new_keys['active']}; re-arming")
        pwriter.put_bool(keys["active"], False)
        pwriter.put_bool(keys["failed"], False)
        keys = new_keys
        model_name = resolve_model_name(params, keys)
        recent_big.clear()
        last_big_published = -1
        last_big_payload = None
        big_used_count = 0
        run_count = 0
        latch = BigLatch()
        pwriter.put_bool(keys["active"], False)
        pwriter.put_bool(keys["failed"], False)
    target = fid
    t_start = time.perf_counter()

    small_payload = None
    got = small_channel.read()
    if got is not None and got[0] == target:
      small_payload = got[1]

    payload = None
    used_big = False
    held = False
    big_peek = None
    if big_channel is not None and not latch.done:
      deadline = t_start + BIG_MODEL_DEADLINE
      sof_ns = (small_payload or {}).get("timestamp_sof")
      if sof_ns:
        remaining = (BIG_ANCHOR_MS / 1000.0) - (time.clock_gettime(time.CLOCK_BOOTTIME) - sof_ns / 1e9)
        deadline = t_start + min(max(remaining, BIG_WAIT_FLOOR_S), BIG_WAIT_CEIL_S)
      payload, big_peek = wait_for_big(big_channel, target, deadline,
                                       last_big_published, BIG_MAX_LAG_FRAMES)
      used_big = payload is not None
      if not used_big:
        if big_peek is None:
          miss_reasons["no_head"] += 1
        elif big_peek <= last_big_published:
          miss_reasons["already_used"] += 1
        elif big_peek > target + BIG_FUTURE_ACCEPT:
          miss_reasons["far_future"] += 1
        elif target - big_peek > BIG_MAX_LAG_FRAMES:
          miss_reasons["too_stale"] += 1
        elif big_peek == target - 1:
          miss_reasons["head_prev_timeout"] += 1
        else:
          miss_reasons["read_race"] += 1
        payload = hold_big_plan(latch.active, last_big_payload, last_big_published, target)
        used_big = held = payload is not None

    if payload is None:
      payload = small_payload
      if payload is None:
        got = small_channel.read()
        if got is not None and got[0] == target:
          payload = got[1]

    activated_now, failed_now = latch.update(used_big)
    if activated_now:
      pwriter.put_bool(keys["active"], True)
      pwriter.put_bool(keys["failed"], False)
      cloudlog.warning(f"modeld_selector switched to BIG model at frame {target}")
    elif failed_now:
      pwriter.put_bool(keys["active"], False)
      pwriter.put_bool(keys["failed"], latch.done)
      pwriter.put(keys["last_error"], "big model stalled onroad; local fallback latched"
                 if latch.done else "big model stalled onroad; small active, big may re-arm")
      if latch.done:
        cloudlog.warning(f"modeld_selector big stalled, staying on small until next ignition (frame {target})")
      else:
        cloudlog.warning(f"modeld_selector big stalled, small active; big may re-arm after a clean streak (frame {target})")

    if payload is not None:
      selector_dropped = max(0, target - last_published - 1) if last_published >= 0 else 0
      frames_dropped = frame_dropped_filter.update(min(selector_dropped, 10))
      if run_count < WARMUP_FRAMES:
        frame_dropped_filter.x = 0.0
        frames_dropped = 0.0
      run_count += 1
      recent_big.append(used_big)
      if held:
        held_count += 1
      elif used_big:
        big_used_count += 1
        last_big_payload = payload
        big_fid = int(payload.get("frame_id", big_peek if big_peek is not None else target))
        last_big_published = min(big_fid, target)
        last_latency_ms = float(payload.get("model_execution_time", 0.0)) * 1e3
      source_lag = max(0, target - int(payload.get("frame_id", target)))
      frame_mismatch = int(payload.get("frame_id", target)) != target
      last_source_lag = source_lag
      if run_count % STATUS_WINDOW == 0:
        hit_rate = (sum(recent_big) / len(recent_big)) if recent_big else 0.0
        pwriter.put(keys["latency_ms"], last_latency_ms)
        pwriter.put(keys["status"], json.dumps({
          "active": latch.active,
          "failed": latch.done,
          "hit_rate": round(hit_rate, 3),
          "latency_ms": round(last_latency_ms, 1),
          "source_lag_frames": last_source_lag,
          "model": model_name,
          "reachable": params.get_bool(keys["reachable"]),
          "download_progress": _read_float(params, keys["progress"], 1.0),
          "ts_mono": round(time.monotonic(), 1),
        }))
      if run_count % 100 == 0:
        cloudlog.warning(f"modeld_selector misses: {miss_reasons}")
        pwriter.put_bool(keys["active"], latch.active)
        cloudlog.warning(f"modeld_selector: big_used={big_used_count}/{run_count} held={held_count} "
                         + f"last_big_peek={big_peek} target={target} active={latch.active} "
                         + f"max_big_lag={BIG_MAX_LAG_FRAMES}")

      frame_drop_perc = 100.0 * frames_dropped / (1.0 + frames_dropped)
      _patch_and_send(pm, payload, frame_drop_perc, selector_dropped, target, source_lag, frame_mismatch,
                      pose_payload=small_payload if used_big else None,
                      backup_payload=small_payload if used_big else None)
      last_published = target


if __name__ == "__main__":
  try:
    main()
  except KeyboardInterrupt:
    cloudlog.warning("modeld_selector got SIGINT")
