"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""
from iqpilot.selfdrive.iqmodeld.models import helpers
from iqpilot.selfdrive.iqmodeld.models.helpers import (
  forced_active_bundle,
  get_active_bundle,
  load_default_model_session,
  is_default_bundle,
)


class _FakeParams:
  def __init__(self, store):
    self.store = store
  def get(self, key):
    return self.store.get(key)
  def put(self, key, value):
    self.store["_puts"] = self.store.get("_puts", 0) + 1
    self.store[key] = value
  def remove(self, key):
    self.store.pop(key, None)


def test_forced_bundle_overrides_get_active_bundle_only_inside_the_block(monkeypatch):
  monkeypatch.setattr(helpers, "_forced_bundle", None)
  default_sentinel = object()
  with forced_active_bundle(default_sentinel):
    assert get_active_bundle() is default_sentinel
  assert helpers._forced_bundle is None


def test_forced_bundle_does_not_persist_anything_to_params(monkeypatch):
  store = {}
  monkeypatch.setattr(helpers, "_forced_bundle", None)
  monkeypatch.setattr(helpers, "Params", lambda *a, **k: _FakeParams(store))
  default_sentinel = object()
  with forced_active_bundle(default_sentinel):
    assert get_active_bundle() is default_sentinel
  assert store.get("_puts", 0) == 0
  assert "ModelManager_ActiveBundle" not in store


def test_forced_bundle_restores_the_previous_override(monkeypatch):
  monkeypatch.setattr(helpers, "_forced_bundle", None)
  outer, inner = object(), object()
  with forced_active_bundle(outer):
    with forced_active_bundle(inner):
      assert get_active_bundle() is inner
    assert get_active_bundle() is outer
  assert helpers._forced_bundle is None


def test_load_default_model_session_returns_default_bundle_and_stages_files(monkeypatch):
  staged = {"n": 0}
  monkeypatch.setattr(helpers, "ensure_default_model_files", lambda *a, **k: staged.__setitem__("n", staged["n"] + 1))
  bundle = load_default_model_session()
  assert staged["n"] == 1
  assert is_default_bundle(bundle)
