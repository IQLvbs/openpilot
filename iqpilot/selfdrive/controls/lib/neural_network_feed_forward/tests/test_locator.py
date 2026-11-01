"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos

Candidate-ladder selection checks for get_nn_model_path, driven by a synthetic
model directory so the assertions don't depend on which cars ship a model.
"""

import pytest

from iqdbc.car import structs
import iqpilot.selfdrive.controls.lib.latcontrol_torque as locator


def test_packaged_substitute_table():
  assert locator.TORQUE_NN_MODEL_SUBSTITUTE_PATH.is_file()
  assert locator._substitute_for("MAZDA_3") == "MAZDA_CX9_2021"
  assert locator._substitute_for("UNKNOWN") == "UNKNOWN"


@pytest.fixture
def model_dir(tmp_path, monkeypatch):
  """A fake model directory with known fingerprints + a substitute table."""
  models = tmp_path / "models"
  models.mkdir()
  for stem in ("HONDA_CIVIC", "HONDA_CIVIC 12345", "TOYOTA_RAV4_TSS2", "MOCK"):
    (models / f"{stem}.json").write_text("{}")

  sub = tmp_path / "substitute.toml"
  sub.write_text('"CHEVROLET_XX" = "TOYOTA_RAV4_TSS2"\n')

  monkeypatch.setattr(locator, "TORQUE_NN_MODEL_PATH", str(models))
  monkeypatch.setattr(locator, "TORQUE_NN_MODEL_SUBSTITUTE_PATH", str(sub))
  monkeypatch.setattr(locator, "MOCK_MODEL_PATH", str(models / "MOCK.json"))
  return models


def make_cp(fingerprint, eps_fw=b"", angle=False):
  cp = structs.CarParams()
  cp.carFingerprint = fingerprint
  if eps_fw:
    fw = structs.CarParams.CarFw()
    fw.ecu = "eps"
    fw.fwVersion = eps_fw
    cp.carFw = [fw]
  if angle:
    cp.steerControlType = structs.CarParams.SteerControlType.angle
  return cp


def test_exact_fingerprint_match(model_dir):
  path, name, exact = locator.get_nn_model_path(make_cp("TOYOTA_RAV4_TSS2"))
  assert name == "TOYOTA_RAV4_TSS2"
  assert exact is True


def test_fingerprint_plus_eps_fw_prefers_specific_file(model_dir):
  # eps fw steers selection toward the fw-specific file. Note the resolved match
  # is fuzzy, not exact: fwVersion is bytes and the candidate stringifies it as
  # b'12345', so it never scores a perfect 1.0 against the "... 12345" filename.
  path, name, exact = locator.get_nn_model_path(make_cp("HONDA_CIVIC", eps_fw=b"12345"))
  assert name == "HONDA_CIVIC 12345"
  assert exact is False


def test_unmapped_fingerprint_does_not_select_similar_vehicle(model_dir):
  path, name, exact = locator.get_nn_model_path(make_cp("TOYOTA_RAV4_TSS2_XYZ"))
  assert name == "MOCK"
  assert path == locator.MOCK_MODEL_PATH
  assert exact is False


@pytest.mark.parametrize("fingerprint", ["UNKNOWN", "HONDA", "HONDA_CIVIC_OTHER", "TOYOTA_RAV4"])
def test_unmatched_vehicle_uses_mock(model_dir, fingerprint):
  path, name, exact = locator.get_nn_model_path(make_cp(fingerprint))
  assert (path, name, exact) == (locator.MOCK_MODEL_PATH, "MOCK", False)


def test_missing_substitute_does_not_select_unrelated_vehicle(model_dir):
  (model_dir / "TOYOTA_RAV4_TSS2.json").unlink()
  path, name, exact = locator.get_nn_model_path(make_cp("CHEVROLET_XX"))
  assert (path, name, exact) == (locator.MOCK_MODEL_PATH, "MOCK", False)


def test_substitute_fallback(model_dir):
  # unknown fingerprint that the substitute table redirects
  path, name, exact = locator.get_nn_model_path(make_cp("CHEVROLET_XX"))
  assert name == "TOYOTA_RAV4_TSS2"
  assert exact is False


def test_angle_steer_is_always_mock(model_dir):
  path, name, exact = locator.get_nn_model_path(make_cp("TOYOTA_RAV4_TSS2", angle=True))
  assert name == "MOCK"
  assert path == locator.MOCK_MODEL_PATH
  assert exact is False


def test_short_eps_fw_ignored(model_dir):
  # a 3-char-or-less fw string is not used to build the candidate
  path, name, _ = locator.get_nn_model_path(make_cp("HONDA_CIVIC", eps_fw=b"ab"))
  assert name == "HONDA_CIVIC"


def test_missing_model_directory_falls_back(model_dir, tmp_path, monkeypatch):
  missing = tmp_path / "missing"
  monkeypatch.setattr(locator, "TORQUE_NN_MODEL_PATH", str(missing))
  monkeypatch.setattr(locator, "MOCK_MODEL_PATH", str(missing / "MOCK.json"))
  path, name, exact = locator.get_nn_model_path(make_cp("HONDA_CIVIC"))
  assert path == locator.MOCK_MODEL_PATH
  assert name == "MOCK"
  assert exact is False
