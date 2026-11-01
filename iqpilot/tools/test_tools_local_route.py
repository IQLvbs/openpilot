"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos
"""
import json
import os
import pty
import select
import shutil
import subprocess
import time
from pathlib import Path

import pytest

import iqpilot.cereal.messaging as messaging
from iqpilot.cereal import log

TOOLS_DIR = Path(__file__).parent
CABANA_BIN = TOOLS_DIR / "cabana" / "_cabana"
JOTPLUGGLER_BIN = TOOLS_DIR / "jotpluggler" / "jotpluggler"

DONGLE_ID = "0000000000000000"
TIMESTAMP = "2024-01-01--00-00-00"
ROUTE = f"{DONGLE_ID}|{TIMESTAMP}"


def write_rlog(path: Path, n_frames: int = 200):
  with open(path, "wb") as f:
    cp = messaging.new_message('carParams')
    cp.carParams.carFingerprint = "TOYOTA_RAV4_TSS2"
    cp.carParams.brand = "toyota"
    f.write(cp.to_bytes())

    for i in range(n_frames):
      msg = messaging.new_message('can', 2)
      msg.logMonoTime = int(i * 1e7)
      for j, addr in enumerate((0x1D2, 0x260)):
        msg.can[j].address = addr
        msg.can[j].src = 0
        msg.can[j].dat = bytes([i % 256] * 8)
      f.write(msg.to_bytes())


def write_video_rlog(path: Path, n_frames: int):
  with open(path, "wb") as f:
    cp = messaging.new_message('carParams')
    cp.logMonoTime = 1_000_000_000
    cp.carParams.carFingerprint = "TOYOTA_RAV4_TSS2"
    cp.carParams.brand = "toyota"
    f.write(cp.to_bytes())

    for i in range(n_frames):
      timestamp = 1_000_000_000 + i * 50_000_000
      msg = messaging.new_message('can', 1)
      msg.logMonoTime = timestamp
      msg.can[0].address = 0x1D2
      msg.can[0].src = 0
      msg.can[0].dat = bytes([i % 256] * 8)
      f.write(msg.to_bytes())

      idx = messaging.new_message('roadEncodeIdx')
      idx.logMonoTime = timestamp
      idx.roadEncodeIdx.frameId = i
      idx.roadEncodeIdx.type = 'fullHEVC'
      idx.roadEncodeIdx.encodeId = i
      idx.roadEncodeIdx.segmentNum = 0
      idx.roadEncodeIdx.segmentId = i
      idx.roadEncodeIdx.segmentIdEncode = i
      idx.roadEncodeIdx.timestampSof = timestamp
      idx.roadEncodeIdx.timestampEof = timestamp + 10_000_000
      f.write(idx.to_bytes())


@pytest.fixture(scope="module")
def local_route(tmp_path_factory):
  data_dir = tmp_path_factory.mktemp("routes")
  for seg in range(2):
    seg_dir = data_dir / f"{DONGLE_ID}|{TIMESTAMP}--{seg}"
    seg_dir.mkdir()
    write_rlog(seg_dir / "rlog")
  return data_dir


@pytest.fixture(scope="module")
def local_video_route(tmp_path_factory):
  data_dir = tmp_path_factory.mktemp("video_routes")
  seg_dir = data_dir / f"{DONGLE_ID}|{TIMESTAMP}--0"
  seg_dir.mkdir()
  frame_count = 20
  result = subprocess.run([
    "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
    "-i", "testsrc=size=320x180:rate=20", "-frames:v", str(frame_count),
    "-pix_fmt", "yuv420p", "-c:v", "libx265", "-preset", "ultrafast",
    "-x265-params", "pools=1:frame-threads=1:log-level=error", "-f", "hevc",
    str(seg_dir / "fcamera.hevc"),
  ], capture_output=True, text=True)
  if result.returncode != 0:
    pytest.skip(result.stderr)
  write_video_rlog(seg_dir / "rlog", frame_count)
  return data_dir


def run(cmd, timeout=180):
  return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=os.environ.copy(),
                        cwd=TOOLS_DIR.parent)


def cabana_command(*args):
  command = [str(CABANA_BIN), *args]
  if os.uname().sysname == "Linux" and shutil.which("xvfb-run"):
    command = ["xvfb-run", "-a", *command]
  return command


def cabana_output_until(args, expected, timeout=60):
  master, slave = pty.openpty()
  proc = subprocess.Popen(cabana_command(*args), stdout=slave, stderr=slave,
                          env=os.environ.copy(), cwd=TOOLS_DIR.parent)
  os.close(slave)
  output = bytearray()
  deadline = time.monotonic() + timeout
  try:
    while time.monotonic() < deadline:
      ready, _, _ = select.select([master], [], [], min(1, deadline - time.monotonic()))
      if not ready:
        if proc.poll() is not None:
          break
        continue
      try:
        output.extend(os.read(master, 4096))
      except OSError:
        break
      if expected.encode() in output:
        break
  finally:
    proc.kill()
    proc.wait()
    os.close(master)
  return output.decode(errors="replace")


def test_jotpluggler_renders_a_local_route(local_route, tmp_path, monkeypatch):
  monkeypatch.setenv("HOME", str(tmp_path))
  assert JOTPLUGGLER_BIN.exists(), "jotpluggler not built"
  out = tmp_path / "plot.png"
  result = run([str(JOTPLUGGLER_BIN), "--data-dir", str(local_route),
                "--sync-load", "--output", str(out), ROUTE])
  assert result.returncode == 0, result.stdout + result.stderr
  assert out.is_file(), result.stdout + result.stderr
  assert out.stat().st_size > 5000, f"suspiciously small render: {out.stat().st_size} bytes"
  history = tmp_path / ".iq/jotpluggler/recent_routes.json"
  assert json.loads(history.read_text()) == [{"route": ROUTE, "data_dir": str(local_route)}]
  startup = tmp_path / "startup.png"
  result = run([str(JOTPLUGGLER_BIN), "--output", str(startup)])
  assert result.returncode == 0, result.stdout + result.stderr
  assert startup.stat().st_size > 5000
  assert json.loads(history.read_text()) == [{"route": ROUTE, "data_dir": str(local_route)}]


def test_cabana_loads_a_local_route(local_route):
  assert CABANA_BIN.exists(), "cabana not built"
  loaded = f"loaded route {ROUTE} with 2 valid segments"
  out = cabana_output_until(("--data_dir", str(local_route), "--no-vipc", ROUTE), loaded)
  assert "failed to load route" not in out, out
  assert "invalid route format" not in out, out
  assert loaded in out, out


def test_cabana_replays_local_video(local_video_route):
  expected = "camera[0] vipc send #1"
  out = cabana_output_until(("--data_dir", str(local_video_route), ROUTE), expected)
  assert "failed to get frame" not in out, out
  assert expected in out, out


def test_replay_logreader_reports_load_stats(local_route):
  assert shutil.which("python3") is not None
  header = (TOOLS_DIR / "replay" / "logreader.h").read_text()
  for accessor in ("compressed_size", "decompressed_size", "download_seconds",
                   "decompress_seconds", "parse_seconds"):
    assert f"{accessor}() const" in header


def test_can_capnp_field_matches_extractor_codegen():
  assert 'busTimeDEPRECATED' in log.CanData.schema.fields
  gen = (TOOLS_DIR / "jotpluggler" / "generate_event_extractors.py").read_text()
  assert "getBusTimeDEPRECATED()" in gen
