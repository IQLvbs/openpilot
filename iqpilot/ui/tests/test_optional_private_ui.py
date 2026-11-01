"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos
"""
import threading
from types import SimpleNamespace

import pyray as rl

from iqpilot.ui import optional_private
from iqpilot.ui.onroad.hud_renderer import REASONING_BAND_HEIGHT, _reasoning_rect
from iqpilot.ui.optional_private import OptionalPrivateUI


def render_big_hud(monkeypatch, endpoint):
  from iqpilot.ui.onroad import hud_renderer
  from iqpilot.system.ui.lib.application import gui_app

  draws = []

  class Surface:
    def __init__(self, *args, **kwargs):
      pass

    def render(self, *args):
      draws.append("base onroad")

    def layout_under(self, rect):
      pass

  def base_init(self):
    self.is_cruise_available = False
    self._exp_button = Surface()

  monkeypatch.setattr(hud_renderer.HudRenderer, "__init__", base_init)
  monkeypatch.setattr(hud_renderer, "optional_private_ui", endpoint)
  sm = {"carState": SimpleNamespace(cruiseFaultLateralMode=False), "carControl": SimpleNamespace(latActive=False)}
  monkeypatch.setattr(hud_renderer, "ui_state", SimpleNamespace(developer_ui=0, torque_bar=False, sm=sm))
  monkeypatch.setattr(gui_app, "set_modal_overlay", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("error dialog")))
  for name in ("IQDevMetricsOverlay", "EmacStatusRenderer", "NavMapPanel", "RoadNameRenderer", "SpeedLimitState",
               "IQTurnSignalOverlay", "IQSpeedOverlay", "SoftWarningRenderer", "TorqueBar", "ReportButtons"):
    monkeypatch.setattr(hud_renderer, name, Surface)
  monkeypatch.setattr(hud_renderer.rl, "draw_rectangle_gradient_v", lambda *args: None)
  monkeypatch.setattr(hud_renderer.IQHudRenderer, "_draw_limit_sign", lambda self, rect: None)
  hud = hud_renderer.IQHudRenderer()
  hud._render(rl.Rectangle(0, 0, 2160, 1080))
  assert len(draws) == 8
  assert endpoint.get() is None


def test_missing_bundle_ui_startup_and_onroad(monkeypatch):
  def missing(*args):
    raise FileNotFoundError("deleted bundle")

  monkeypatch.setattr(optional_private, "_create_endpoint", missing)
  endpoint = optional_private.OptionalPrivateUI()
  endpoint._load()
  endpoint._started = True
  render_big_hud(monkeypatch, endpoint)


def test_onroad_does_not_wait_for_loader(monkeypatch):
  entered, release = threading.Event(), threading.Event()

  def pending(*args):
    entered.set()
    release.wait(5)
    raise ImportError("stale bundle")

  monkeypatch.setattr(optional_private, "_create_endpoint", pending)
  endpoint = optional_private.OptionalPrivateUI()
  try:
    assert endpoint.get() is None
    assert entered.wait(1)
    render_big_hud(monkeypatch, endpoint)
    assert not release.is_set()
  finally:
    release.set()


def test_reasoning_band_is_centred():
  rect = rl.Rectangle(0.0, 0.0, 2160.0, 1080.0)
  band = _reasoning_rect(rect)
  assert abs((band.x + band.width / 2.0) - (rect.x + rect.width / 2.0)) < 1e-3
  assert abs((band.y + band.height / 2.0) - (rect.y + rect.height / 2.0)) < 1e-3
  assert 0.0 < band.width < rect.width
  assert band.height == REASONING_BAND_HEIGHT


def test_reasoning_band_never_exceeds_a_short_screen():
  band = _reasoning_rect(rl.Rectangle(10.0, 20.0, 800.0, 100.0))
  assert band.height == 100.0
  assert band.y == 20.0


def test_missing_bundle_leaves_the_onroad_render_silent():
  private = OptionalPrivateUI()
  private._started = True
  private._endpoint = None
  private.render_onroad(rl.Rectangle(0.0, 0.0, 100.0, 100.0))
  assert private.attach(object()) is False
