# Copyright (c) 2026 IQ.Lvbs. All rights reserved.
from types import SimpleNamespace

import pytest

from iqpilot.selfdrive.ui.mici.onroad import confidence_ball as ball_module
from iqpilot.ui.onroad import lead_confidence as confidence_module


@pytest.fixture
def shared_confidence(monkeypatch):
  class Messages(dict):
    frame = 1

  predictions = SimpleNamespace(brakeDisengageProbs=[0.2], steerOverrideProbs=[0.4])
  messages = Messages(modelV2=SimpleNamespace(meta=SimpleNamespace(disengagePredictions=predictions)))
  state = SimpleNamespace(sm=messages, status=ball_module.UIStatus.LAT_ONLY)
  shared = confidence_module.DrivingConfidence()
  monkeypatch.setattr(confidence_module, 'ui_state', state)
  monkeypatch.setattr(ball_module, 'ui_state', state)
  monkeypatch.setattr(ball_module, 'driving_confidence', shared)
  return state, shared


@pytest.mark.parametrize('status,target', [('disengaged', -0.5), ('lat_only', 0.6), ('long_only', 0.8),
                                         ('engaged', 0.48), ('override', 0.48)])
def test_ball_and_other_overlays_share_one_update_per_frame(shared_confidence, status, target):
  state, shared = shared_confidence
  state.status = getattr(ball_module.UIStatus, status.upper())
  values = []
  shared._filter.update = values.append
  ball = ball_module.ConfidenceBall()
  ball._update_state()
  shared.update()
  ball._update_state()
  assert values == pytest.approx([target])


def test_demo_does_not_advance_shared_confidence(shared_confidence):
  _, shared = shared_confidence
  ball = ball_module.ConfidenceBall(demo=True)
  ball.update_filter(0.9)
  ball._update_state()
  assert shared._last_frame == -1
  assert ball._confidence_filter.x > -0.5


def test_partial_engagement_keeps_iq_colors_and_shared_height(shared_confidence, monkeypatch):
  _, shared = shared_confidence
  shared._filter.x = 0.75
  ball = ball_module.ConfidenceBall()
  ball._rect = ball_module.rl.Rectangle(0, 0, 500, 300)
  draws = []
  monkeypatch.setattr(ball_module, 'draw_circle_gradient', lambda *args: draws.append(args))
  ball._render(None)
  assert draws[0][1] == pytest.approx((1 - 0.75) * (300 - 48) + 24)
  assert (draws[0][3].r, draws[0][3].g, draws[0][3].b) == (0x22, 0xB8, 0xB9)
  assert (draws[0][4].r, draws[0][4].g, draws[0][4].b) == (0x0C, 0x94, 0x96)
