"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""
from __future__ import annotations

import statistics
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from enum import IntEnum

DEFAULT_SERVER = "https://speed.cloudflare.com"
STAGE_SECONDS = 6.0
LATENCY_SAMPLES = 5
CHUNK = 64 * 1024
TIMEOUT_S = 10.0
CAPS = {False: (250_000_000, 60_000_000), True: (25_000_000, 8_000_000)}
REQUEST_BYTES = 25_000_000
STAGE_SPAN = {"latency": (0.0, 0.1), "download": (0.1, 0.6), "upload": (0.6, 1.0)}
SEPARATOR = " \u2022 "


class Phase(IntEnum):
  IDLE = 0
  LATENCY = 1
  DOWNLOAD = 2
  UPLOAD = 3
  DONE = 4
  FAILED = 5


@dataclass(frozen=True)
class SpeedTestState:
  phase: Phase = Phase.IDLE
  progress: float = 0.0
  latency_ms: float | None = None
  download_mbps: float | None = None
  upload_mbps: float | None = None
  live_mbps: float = 0.0
  metered: bool = False
  error: str = ""

  @property
  def running(self) -> bool:
    return self.phase in (Phase.LATENCY, Phase.DOWNLOAD, Phase.UPLOAD)


class Cancelled(Exception):
  pass


def mbps(nbytes: int, seconds: float) -> float:
  return nbytes * 8 / max(seconds, 1e-6) / 1e6


def _session():
  import requests
  session = requests.Session()
  session.headers["User-Agent"] = "IQ.Pilot speed test"
  return session


class SpeedTest:
  def __init__(self, server: str = DEFAULT_SERVER, stage_seconds: float = STAGE_SECONDS,
               session_factory: Callable = _session, clock: Callable[[], float] = time.monotonic):
    self._server = server.rstrip("/")
    self._stage_seconds = stage_seconds
    self._session_factory = session_factory
    self._clock = clock
    self._lock = threading.Lock()
    self._state = SpeedTestState()
    self._cancel = threading.Event()
    self._thread: threading.Thread | None = None

  @property
  def state(self) -> SpeedTestState:
    with self._lock:
      return self._state

  def _update(self, **changes) -> None:
    with self._lock:
      self._state = replace(self._state, **changes)

  def start(self, metered: bool = False) -> bool:
    if self._thread is not None and self._thread.is_alive():
      return False
    self._cancel.clear()
    with self._lock:
      self._state = SpeedTestState(Phase.LATENCY, metered=metered)
    self._thread = threading.Thread(target=self._run, args=(metered,), daemon=True, name="speed-test")
    self._thread.start()
    return True

  def cancel(self) -> None:
    self._cancel.set()

  def join(self, timeout: float | None = None) -> None:
    if self._thread is not None:
      self._thread.join(timeout)

  def _stage_progress(self, stage: str, fraction: float) -> float:
    lo, hi = STAGE_SPAN[stage]
    return lo + (hi - lo) * max(0.0, min(1.0, fraction))

  def _check_cancel(self) -> None:
    if self._cancel.is_set():
      raise Cancelled

  def _run(self, metered: bool) -> None:
    down_cap, up_cap = CAPS[metered]
    try:
      with self._session_factory() as session:
        latency = self._latency(session)
        self._update(phase=Phase.DOWNLOAD, latency_ms=latency, progress=self._stage_progress("download", 0.0))
        download = self._download(session, down_cap)
        self._update(phase=Phase.UPLOAD, download_mbps=download, live_mbps=0.0, progress=self._stage_progress("upload", 0.0))
        upload = self._upload(session, up_cap)
      self._update(phase=Phase.DONE, upload_mbps=upload, live_mbps=0.0, progress=1.0)
    except Cancelled:
      self._update(phase=Phase.IDLE, live_mbps=0.0, progress=0.0)
    except Exception as e:
      self._update(phase=Phase.FAILED, live_mbps=0.0, error=_describe(e))

  def _latency(self, session) -> float:
    samples = []
    for i in range(LATENCY_SAMPLES):
      self._check_cancel()
      start = self._clock()
      response = session.get(f"{self._server}/__down", params={"bytes": 0}, timeout=TIMEOUT_S)
      response.raise_for_status()
      samples.append((self._clock() - start) * 1000.0)
      self._update(progress=self._stage_progress("latency", (i + 1) / LATENCY_SAMPLES))
    return statistics.median(samples)

  def _stage_done(self, start: float, moved: int, cap: int) -> bool:
    return self._clock() - start >= self._stage_seconds or moved >= cap

  def _download(self, session, cap: int) -> float:
    start = self._clock()
    received = 0
    while not self._stage_done(start, received, cap):
      size = min(REQUEST_BYTES, cap - received)
      with session.get(f"{self._server}/__down", params={"bytes": size}, stream=True, timeout=TIMEOUT_S) as response:
        response.raise_for_status()
        for chunk in response.iter_content(CHUNK):
          self._check_cancel()
          received += len(chunk)
          elapsed = self._clock() - start
          self._update(live_mbps=mbps(received, elapsed),
                       progress=self._stage_progress("download", max(elapsed / self._stage_seconds, received / cap)))
          if self._stage_done(start, received, cap):
            break
    if received == 0:
      raise RuntimeError("no data received")
    return mbps(received, self._clock() - start)

  def _upload(self, session, cap: int) -> float:
    start = self._clock()
    sent = 0
    block = bytes(CHUNK)

    def body(limit: int) -> Iterator[bytes]:
      nonlocal sent
      while sent < limit and not self._stage_done(start, sent, cap):
        self._check_cancel()
        piece = block[:min(CHUNK, limit - sent)]
        sent += len(piece)
        elapsed = self._clock() - start
        self._update(live_mbps=mbps(sent, elapsed),
                     progress=self._stage_progress("upload", max(elapsed / self._stage_seconds, sent / cap)))
        yield piece

    while not self._stage_done(start, sent, cap):
      response = session.post(f"{self._server}/__up", data=body(min(cap, sent + REQUEST_BYTES)), timeout=TIMEOUT_S)
      self._check_cancel()
      response.raise_for_status()
    return mbps(sent, self._clock() - start)


def _describe(e: Exception) -> str:
  name = type(e).__name__
  if "Timeout" in name:
    return "timed out"
  if "Connection" in name:
    return "no connection"
  if "HTTPError" in name:
    return "server error"
  return str(e) or name


def format_mbps(value: float | None) -> str:
  if value is None:
    return "--"
  if value >= 1000:
    return f"{value / 1000:.2f} G"
  if value >= 100:
    return f"{value:.0f}"
  return f"{value:.1f}"


def format_latency(value: float | None) -> str:
  return "--" if value is None else f"{value:.0f}"


def summary(state: SpeedTestState) -> str:
  if state.phase == Phase.FAILED:
    return state.error
  if state.download_mbps is None:
    return ""
  parts = [f"down {format_mbps(state.download_mbps)} Mbps", f"up {format_mbps(state.upload_mbps)} Mbps"]
  if state.latency_ms is not None:
    parts.append(f"ping {format_latency(state.latency_ms)} ms")
  return SEPARATOR.join(parts)


_shared: SpeedTest | None = None


def shared_speed_test() -> SpeedTest:
  global _shared
  if _shared is None:
    _shared = SpeedTest()
  return _shared


def live_text(state: SpeedTestState) -> str:
  if state.phase == Phase.LATENCY:
    return "measuring ping"
  if state.phase == Phase.DOWNLOAD:
    return f"download {format_mbps(state.live_mbps)} Mbps"
  if state.phase == Phase.UPLOAD:
    return f"upload {format_mbps(state.live_mbps)} Mbps"
  return summary(state)


def compact_text(state: SpeedTestState) -> str:
  if state.phase in (Phase.DONE, Phase.IDLE) and state.download_mbps is not None:
    return f"{format_mbps(state.download_mbps)} / {format_mbps(state.upload_mbps)} Mbps"
  return live_text(state)
