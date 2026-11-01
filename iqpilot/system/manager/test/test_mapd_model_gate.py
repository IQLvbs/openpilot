# Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
from types import SimpleNamespace

from iqpilot.system.manager import process_config
from iqpilot.system.manager.process_config import iqmapd_onroad, iqmodeld_model_ready, mapd_onroad, managed_processes


def params(**values):
  return SimpleNamespace(get_bool=lambda key: values.get(key, False))


def _tinygrad(monkeypatch, value):
  monkeypatch.setattr(process_config, "is_tinygrad_model", lambda *a: value)


def test_mapd_waits_for_the_model_load_on_tinygrad(monkeypatch):
  _tinygrad(monkeypatch, True)
  wants_maps = params(ShowSpeedLimits=True)
  assert not iqmodeld_model_ready(True, wants_maps, None)
  assert not mapd_onroad(True, wants_maps, None)


def test_mapd_runs_once_the_model_is_loaded(monkeypatch):
  _tinygrad(monkeypatch, True)
  loaded = params(ShowSpeedLimits=True, IqModelLoaded=True)
  assert iqmodeld_model_ready(True, loaded, None)
  assert mapd_onroad(True, loaded, None)


def test_mapd_not_gated_when_iqmodeld_is_not_the_loader(monkeypatch):
  _tinygrad(monkeypatch, False)
  wants_maps = params(ShowSpeedLimits=True)
  assert iqmodeld_model_ready(True, wants_maps, None)
  assert mapd_onroad(True, wants_maps, None)


def test_mapd_still_requires_started_and_maps_wanted(monkeypatch):
  _tinygrad(monkeypatch, True)
  loaded = params(ShowSpeedLimits=True, IqModelLoaded=True)
  assert not mapd_onroad(False, loaded, None)
  assert not mapd_onroad(True, params(IqModelLoaded=True), None)


def test_iqmapd_nav_bundle_also_waits_for_the_model(monkeypatch):
  _tinygrad(monkeypatch, True)
  nav = params(ShowSpeedLimits=True, NavigationEnabled=True)
  assert not iqmapd_onroad(True, nav, None)
  nav_loaded = params(ShowSpeedLimits=True, NavigationEnabled=True, IqModelLoaded=True)
  assert iqmapd_onroad(True, nav_loaded, None)


def test_mapd_gate_wired_in_managed_processes():
  assert managed_processes["mapd"].should_run is mapd_onroad
