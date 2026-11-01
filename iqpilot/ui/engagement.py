# Copyright (c) 2026 IQ.Lvbs. All rights reserved.
from dataclasses import dataclass

from iqpilot.cereal import custom, log

_CONTROL_STATES = log.SelfdriveState.OpenpilotState
_LATERAL_STATES = custom.AlwaysOnLateral.AlwaysOnLateralState
_AXIS_STATUS = {
  (False, False): "disengaged",
  (True, False): "lat_only",
  (False, True): "long_only",
  (True, True): "engaged",
}


@dataclass(frozen=True, slots=True)
class EngagementDisplay:
  status: str
  phase: str


def resolve_engagement_display(controls, iq_state, events) -> EngagementDisplay:
  guidance = iq_state.aol
  driver_intervention = controls.state == _CONTROL_STATES.overriding and (
    not guidance.available or any(event.overrideLongitudinal for event in events)
  )
  suspended = (
    controls.state == _CONTROL_STATES.preEnabled,
    driver_intervention,
    guidance.state in (_LATERAL_STATES.paused, _LATERAL_STATES.overriding),
  )
  if any(suspended):
    return EngagementDisplay("override", "standby")

  longitudinal = bool(controls.enabled)
  lateral = bool(guidance.enabled) if guidance.available else longitudinal
  phase = "active" if longitudinal or guidance.enabled else "standby"
  return EngagementDisplay(_AXIS_STATUS[lateral, longitudinal], phase)
