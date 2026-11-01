# Copyright (c) 2026 IQ.Lvbs. All rights reserved.
from types import SimpleNamespace

import pytest

from iqpilot.cereal import custom, log
from iqpilot.ui.engagement import EngagementDisplay, resolve_engagement_display

CONTROL = log.SelfdriveState.OpenpilotState
LATERAL = custom.AlwaysOnLateral.AlwaysOnLateralState


@pytest.mark.parametrize("available,lateral,longitudinal,status,phase", [
  (False, False, False, "disengaged", "standby"),
  (False, False, True, "engaged", "active"),
  (False, True, False, "disengaged", "active"),
  (False, True, True, "engaged", "active"),
  (True, False, False, "disengaged", "standby"),
  (True, False, True, "long_only", "active"),
  (True, True, False, "lat_only", "active"),
  (True, True, True, "engaged", "active"),
])
def test_engagement_axes(available, lateral, longitudinal, status, phase):
  controls = SimpleNamespace(state=CONTROL.enabled, enabled=longitudinal)
  iq = SimpleNamespace(aol=SimpleNamespace(state=LATERAL.enabled, available=available, enabled=lateral))
  assert resolve_engagement_display(controls, iq, []) == EngagementDisplay(status, phase)


@pytest.mark.parametrize("control,lateral,available,long_override,expected", [
  (CONTROL.preEnabled, LATERAL.enabled, True, False, "override"),
  (CONTROL.overriding, LATERAL.enabled, True, False, "engaged"),
  (CONTROL.overriding, LATERAL.enabled, False, False, "override"),
  (CONTROL.overriding, LATERAL.enabled, True, True, "override"),
  (CONTROL.enabled, LATERAL.paused, True, False, "override"),
  (CONTROL.enabled, LATERAL.overriding, True, False, "override"),
  (CONTROL.enabled, LATERAL.enabled, True, True, "engaged"),
  (CONTROL.softDisabling, LATERAL.softDisabling, True, False, "engaged"),
])
def test_intervention_precedence(control, lateral, available, long_override, expected):
  controls = SimpleNamespace(state=control, enabled=True)
  iq = SimpleNamespace(aol=SimpleNamespace(state=lateral, available=available, enabled=True))
  events = [SimpleNamespace(overrideLongitudinal=long_override)]
  result = resolve_engagement_display(controls, iq, events)
  assert result.status == expected
  assert result.phase == ("standby" if expected == "override" else "active")
