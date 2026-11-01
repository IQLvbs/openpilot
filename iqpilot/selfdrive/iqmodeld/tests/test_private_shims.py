# Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
import ast
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from iqpilot import _proprietary_loader

ROOT = Path(__file__).resolve().parents[4]
SHIMS = {
  "iqpilot.selfdrive.controls.lib.action_engine": "iqpilot_private.navd.action_engine",
  "iqpilot.selfdrive.iqmodeld.cot_decode": "iqpilot_private.context.cot_decode",
  "iqpilot.selfdrive.iqmodeld.drive_profile": "iqpilot_private.context.drive_profile",
  "iqpilot.selfdrive.iqmodeld.iqmodel_io": "iqpilot_private.context.iqmodel_io",
  "iqpilot.selfdrive.iqmodeld.temporal_state": "iqpilot_private.context.temporal_state",
  "iqpilot.ui.reasoning_relay": "iqpilot_private.context.reasoning_relay",
}
CONSUMER_ROOTS = ("iqpilot/selfdrive", "iqpilot/ui", "iqpilot/tools", "iqpilot/models_private_src")


def load_fallback(monkeypatch, public_name):
  calls = []

  def missing(public, private):
    calls.append(private)
    raise _proprietary_loader.ProprietaryModuleMissing("missing context bundle")

  monkeypatch.setattr(_proprietary_loader, "load_private_module", missing)
  path = ROOT / Path(*public_name.split(".")).with_suffix(".py")
  spec = importlib.util.spec_from_file_location(public_name + "_fallback_test", path)
  module = importlib.util.module_from_spec(spec)
  monkeypatch.setitem(sys.modules, spec.name, module)
  spec.loader.exec_module(module)
  assert calls == [SHIMS[public_name]]
  return module


def consumer_imports():
  wanted = {}
  for base in CONSUMER_ROOTS:
    for path in sorted((ROOT / base).rglob("*.py")):
      if "tests" in path.parts:
        continue
      text = path.read_text(encoding="utf-8", errors="ignore")
      if not any(name.rsplit(".", 1)[1] in text for name in SHIMS):
        continue
      for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.ImportFrom) and node.module in SHIMS:
          wanted.setdefault(node.module, set()).update(alias.name for alias in node.names)
  return wanted


@pytest.mark.parametrize("public_name", sorted(SHIMS))
def test_shim_source_holds_no_private_logic(public_name):
  source = (ROOT / Path(*public_name.split(".")).with_suffix(".py")).read_text()
  assert f'load_private_module(__name__, "{SHIMS[public_name]}")' in source
  assert len(source.splitlines()) < 200


def test_every_consumer_import_resolves_in_the_fallback(monkeypatch):
  wanted = consumer_imports()
  assert set(wanted) == set(SHIMS)
  for public_name, names in wanted.items():
    module = load_fallback(monkeypatch, public_name)
    assert not sorted(name for name in names if not hasattr(module, name)), public_name


def test_reasoning_fallbacks_are_inert(monkeypatch, tmp_path):
  cot = load_fallback(monkeypatch, "iqpilot.selfdrive.iqmodeld.cot_decode")
  wire = cot.ReasoningWire({"output_slices": {"reasoning_tokens": slice(0, 4)}})
  assert not wire.enabled
  assert wire.update({"_reasoning_envelope": "mrv1:sealed"}, 1, 2) is None
  assert cot.decode_reasoning(np.zeros(4), {}) is None
  assert not cot.validate_reasoning_wire({"cot_decode": "rh-cot-wire-v2"})

  relay = load_fallback(monkeypatch, "iqpilot.ui.reasoning_relay")
  with pytest.raises(_proprietary_loader.ProprietaryModuleMissing):
    relay.RasterEndpoint(tmp_path)


def test_feature_fallbacks_feed_zeros_only_to_declared_inputs(monkeypatch):
  profile = load_fallback(monkeypatch, "iqpilot.selfdrive.iqmodeld.drive_profile")
  inputs, state = profile.compose_drive_profile_inputs({"drive_profile": np.ones((1, 5))}, None, None, 0, 10.0)
  np.testing.assert_array_equal(inputs["drive_profile"], np.zeros((1, 5), np.float32))
  assert state == profile.DriveProfileState()
  assert profile.compose_drive_profile_inputs({}, None, None, 0, 10.0)[0] == {}
  assert not profile.active_model_declares_drive_profile(None)
  assert profile.drive_profile_name(None) == "neutral"


def test_temporal_fallback_keeps_the_stock_policy_contract(monkeypatch):
  temporal = load_fallback(monkeypatch, "iqpilot.selfdrive.iqmodeld.temporal_state")
  shapes = {name: shape for name, (shape, _) in temporal.MODEL_INPUT_SPEC.items()}
  assert list(temporal.packed_policy_layout(shapes)) == ["desire", "traffic_convention", "action_t", "prev_feat"]
  assert list(temporal.packed_policy_layout({**shapes, "nav_intent": (1, 32)}))[-1] == "nav_intent"
  assert temporal.spec_from_meta({"input_shapes": {"img": (1, 12, 128, 256)}}) == {"img": ((1, 12, 128, 256), "uint8")}
  state = temporal.TemporalInputState(temporal.DEFAULT_FRAME_SKIP)
  warped = np.zeros((2, 6, 128, 256), np.uint8)
  inputs = state.push_and_materialize(warped, np.zeros(8, np.float32), np.array([1.0, 0.0]), np.zeros(2))
  assert {name: value.shape for name, value in inputs.items()} == shapes
