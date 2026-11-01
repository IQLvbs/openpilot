"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""
import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

META = {"key": "test", "sha256": "0" * 64}
TC_OFF_OUTPUT = np.linspace(-1.0, 1.0, 16, dtype=np.float32)


@pytest.fixture
def m(monkeypatch):
  saved = dict(os.environ)
  module = importlib.import_module("iqpilot.selfdrive.iqmodeld.tools.compile_egpu_model")
  os.environ.clear()
  os.environ.update(saved)
  for key in ("IQ_EGPU_REFERENCE", "IQ_EGPU_SKIP_PARITY", "IQ_EGPU_TC_OFF", "BEAM"):
    monkeypatch.delenv(key, raising=False)
  monkeypatch.setenv("TC_OPT", "2")
  monkeypatch.setattr(module, "HOST", False)
  monkeypatch.setattr(module, "get_egpu_model", lambda *_: dict(META))
  monkeypatch.setattr(module, "set_input_spec", lambda _: None)
  return module


def output_names(fmt, resolutions):
  return ["policy"] if fmt == 2 else [f"{w}x{h}" for w, h in resolutions]


class Harness:
  def __init__(self, m, monkeypatch, tmp_path, fmt, tc_output=TC_OFF_OUTPUT):
    self.m, self.fmt, self.tc_output = m, fmt, tc_output
    self.onnx = tmp_path / "model.onnx"
    self.onnx.touch()
    self.out = tmp_path / "models" / "compiled.pkl"
    self.out.parent.mkdir()
    self.resolutions = ((1928, 1208), (1344, 760))
    self.events, self.children, self.references = [], [], []
    monkeypatch.setattr(sys, "argv", ["compile", "--onnx", str(self.onnx), "--output", str(self.out), "--format", str(fmt)])
    monkeypatch.setattr(subprocess, "run", self.child)
    monkeypatch.setattr(m, "compile_policy_model", self.build)
    monkeypatch.setattr(m, "compile_model_v3", self.build)
    monkeypatch.setattr(m, "compile_model", self.build)

  def child(self, cmd, **kw):
    self.events.append("reference")
    self.children.append((cmd, kw["env"]))
    Path(cmd[cmd.index("--output") + 1]).write_bytes(b"tc-off")
    np.savez(kw["env"]["IQ_EGPU_REFERENCE"], **dict.fromkeys(output_names(self.fmt, self.resolutions), TC_OFF_OUTPUT))
    return SimpleNamespace(returncode=0, stderr="")

  def build(self, meta, onnx_path, out_path, *args, reference=None):
    self.events.append("build")
    self.references.append(reference)
    tmp = out_path + ".part"
    Path(tmp).write_bytes(b"tc")
    return self.m._publish(meta, tmp, out_path, dict.fromkeys(output_names(self.fmt, self.resolutions), self.tc_output), reference)

  @property
  def work_dir(self):
    return Path(str(self.out) + ".reference")


@pytest.mark.parametrize("fmt", [2, 3])
def test_reference_compiles_in_a_child_before_the_dock_is_opened(m, monkeypatch, tmp_path, fmt):
  monkeypatch.setenv("BEAM", "3")
  h = Harness(m, monkeypatch, tmp_path, fmt)
  m.main()

  assert h.events == ["reference", "build"]
  cmd, env = h.children[0]
  assert "--tc-off" in cmd and env["TC_OPT"] == "0" and "BEAM" not in env
  assert Path(cmd[cmd.index("--output") + 1]).parent == h.work_dir
  assert ("--camera-resolutions" in cmd) == (fmt == 3)
  assert set(h.references[0]) == set(output_names(fmt, h.resolutions))
  assert h.out.read_bytes() == b"tc"
  assert not h.work_dir.exists()


@pytest.mark.parametrize("fmt", [2, 3])
def test_parity_failure_ships_the_validated_tc_off_build(m, monkeypatch, tmp_path, fmt):
  h = Harness(m, monkeypatch, tmp_path, fmt, tc_output=TC_OFF_OUTPUT + 0.5)
  m.main()

  assert h.events == ["reference", "build"]
  assert h.out.read_bytes() == b"tc-off"
  assert not Path(str(h.out) + ".part").exists()
  assert not h.work_dir.exists()


@pytest.mark.parametrize("fmt", [2, 3])
def test_reference_child_publishes_its_validated_artifact(m, monkeypatch, tmp_path, fmt):
  ref = tmp_path / "ref.npz"
  monkeypatch.setenv("TC_OPT", "0")
  monkeypatch.setenv("IQ_EGPU_REFERENCE", str(ref))
  h = Harness(m, monkeypatch, tmp_path, fmt)
  m.main()

  assert h.events == ["build"]
  assert h.references == [None]
  assert h.out.read_bytes() == b"tc"
  with np.load(ref) as archive:
    assert sorted(archive.files) == sorted(output_names(fmt, h.resolutions))


@pytest.mark.parametrize("setup", ["host", "skip", "tc_off", "format1"])
def test_parity_reference_is_skipped_when_not_gated(m, monkeypatch, tmp_path, setup):
  h = Harness(m, monkeypatch, tmp_path, 1 if setup == "format1" else 3)
  if setup == "host":
    monkeypatch.setattr(m, "HOST", True)
  elif setup == "skip":
    monkeypatch.setenv("IQ_EGPU_SKIP_PARITY", "1")
  elif setup == "tc_off":
    monkeypatch.setenv("TC_OPT", "0")
  m.main()

  assert h.events == ["build"]
  assert h.references == [None]


def test_progress_span_is_split_between_reference_and_tc_build(m, monkeypatch, tmp_path):
  h = Harness(m, monkeypatch, tmp_path, 3)
  sampled = []
  monkeypatch.setattr(m, "_progress_sampler", lambda param, base, span, stop: sampled.append((param, base, span)))
  monkeypatch.setattr(sys, "argv", sys.argv + ["--progress-param", "UsbGpuSetupProgress", "--progress-base", "0.5", "--progress-span", "0.48"])
  m.main()

  cmd, _ = h.children[0]
  assert cmd[cmd.index("--progress-base") + 1] == "0.5"
  assert float(cmd[cmd.index("--progress-span") + 1]) == pytest.approx(0.24)
  assert sampled == [("UsbGpuSetupProgress", pytest.approx(0.74), pytest.approx(0.24))]


def test_reference_arrays_outlive_the_work_dir(m, monkeypatch, tmp_path):
  h = Harness(m, monkeypatch, tmp_path, 3)
  work_dir = tmp_path / "work"
  work_dir.mkdir()
  pkl, reference = m._tc_off_reference(META, str(h.onnx), 3, h.resolutions, str(work_dir))
  assert pkl == str(work_dir / "reference.pkl")
  shutil.rmtree(work_dir)
  np.testing.assert_array_equal(reference["1928x1208"], TC_OFF_OUTPUT)


def test_reference_failure_does_not_publish(m, monkeypatch, tmp_path):
  h = Harness(m, monkeypatch, tmp_path, 3)
  monkeypatch.setattr(subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=1, stderr="dock lock held"))
  with pytest.raises(RuntimeError, match="dock lock held"):
    m.main()
  assert h.events == []
  assert not h.out.exists()
  assert not h.work_dir.exists()
