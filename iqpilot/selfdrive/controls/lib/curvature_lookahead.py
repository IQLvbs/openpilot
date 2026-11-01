"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos
"""
import math

from iqpilot.cereal import log


Selection = log.ControlsState.LateralActionSelection
ActionSource = log.ModelDataV2.Action.LateralActionSource
LookaheadContract = log.ModelDataV2.Action.LegacyLookahead.Contract


def select_lateral_curvature(model_v2, active: bool, current_curvature: float, maneuver_curvature: float | None,
                             lookahead_enabled: bool, controller_eligible: bool) -> tuple[float, int]:
  if not active:
    return current_curvature, Selection.inactive
  if maneuver_curvature is not None:
    return maneuver_curvature, Selection.maneuver

  action = model_v2.action
  curvature = action.desiredCurvature
  if not lookahead_enabled:
    return curvature, Selection.lookaheadDisabled
  if not controller_eligible:
    return curvature, Selection.controllerUnsupported
  if action.lateralActionSource.raw != ActionSource.plan:
    return curvature, Selection.modelUnsupported

  candidate = action.legacyLookahead
  if candidate.contract.raw != LookaheadContract.planOrientationV1:
    return curvature, Selection.modelUnsupported
  if not math.isfinite(candidate.desiredCurvature) or not math.isfinite(candidate.horizonSeconds) or candidate.horizonSeconds <= 0.0:
    return curvature, Selection.invalidLookahead
  return candidate.desiredCurvature, Selection.legacyLookahead
