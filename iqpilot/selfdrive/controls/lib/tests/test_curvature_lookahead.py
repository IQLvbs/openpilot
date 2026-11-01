from types import SimpleNamespace

import pytest

from iqpilot.cereal import log
from iqpilot.selfdrive.controls.lib.curvature_lookahead import Selection, select_lateral_curvature


def make_model(source="plan", contract="planOrientationV1", curvature=0.002, horizon=0.6):
  return log.ModelDataV2.new_message(action={
    "desiredCurvature": -0.0012,
    "lateralActionSource": source,
    "legacyLookahead": {"contract": contract, "desiredCurvature": curvature, "horizonSeconds": horizon},
  })


def select(model, enabled=True, eligible=True, active=True, maneuver=None):
  return select_lateral_curvature(model, active, 0.0003, maneuver, enabled, eligible)


@pytest.mark.parametrize("source", ["unknown", "nativeAction", "nativeCurvature", "plan", 65535])
def test_disabled_does_not_consume_candidate(source):
  model = make_model(source, curvature=float("nan"))
  assert select(model, enabled=False) == (model.action.desiredCurvature, Selection.lookaheadDisabled)


@pytest.mark.parametrize("source", ["unknown", "nativeAction", "nativeCurvature", 65535])
def test_only_explicit_legacy_plan_can_override(source):
  model = make_model(source)
  assert select(model) == (model.action.desiredCurvature, Selection.modelUnsupported)


@pytest.mark.parametrize("contract", ["unsupported", 65535])
def test_unrecognized_contract_preserves_action(contract):
  model = make_model(contract=contract)
  assert select(model) == (model.action.desiredCurvature, Selection.modelUnsupported)


def test_ineligible_controller_preserves_action():
  model = make_model()
  assert select(model, eligible=False) == (model.action.desiredCurvature, Selection.controllerUnsupported)


@pytest.mark.parametrize(("curvature", "horizon"), [
  (float("nan"), 0.6), (float("inf"), 0.6), (-float("inf"), 0.6),
  (0.002, float("nan")), (0.002, float("inf")), (0.002, 0.0), (0.002, -0.1),
])
def test_invalid_candidate_preserves_action(curvature, horizon):
  model = make_model(curvature=curvature, horizon=horizon)
  assert select(model) == (model.action.desiredCurvature, Selection.invalidLookahead)


def test_supported_candidate_is_selected_before_control_limits():
  model = make_model()
  assert select(model) == (model.action.legacyLookahead.desiredCurvature, Selection.legacyLookahead)


def test_inactive_and_maneuver_do_not_read_model():
  model = SimpleNamespace()
  assert select(model, active=False, maneuver=-0.02) == (0.0003, Selection.inactive)
  assert select(model, maneuver=-0.02) == (-0.02, Selection.maneuver)


def test_disabled_does_not_require_metadata():
  model = SimpleNamespace(action=SimpleNamespace(desiredCurvature=-0.0012))
  assert select(model, enabled=False) == (-0.0012, Selection.lookaheadDisabled)


def test_model_change_does_not_reuse_legacy_candidate():
  legacy = make_model()
  native = make_model("nativeAction")
  old_producer = log.ModelDataV2.new_message(action={"desiredCurvature": -0.004})
  assert select(legacy)[1] == Selection.legacyLookahead
  for model in (native, old_producer, native, legacy):
    expected = Selection.legacyLookahead if model is legacy else Selection.modelUnsupported
    assert select(model)[1] == expected


def test_old_action_wire_format_defaults_to_unsupported():
  payload = bytes.fromhex("0000000003000000000000000200000052499dba0000803e0100000000000000")
  with log.ModelDataV2.Action.from_bytes(payload) as action:
    assert action.desiredCurvature == pytest.approx(-0.0012)
    assert action.desiredAcceleration == 0.25
    assert action.shouldStop
    assert action.lateralActionSource == "unknown"
    assert action.legacyLookahead.contract == "unsupported"
    model = log.ModelDataV2.new_message(action=action)
    assert select(model) == (action.desiredCurvature, Selection.modelUnsupported)
