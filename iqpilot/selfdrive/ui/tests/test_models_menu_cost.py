"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""

import threading
import time

import pytest

from iqpilot.selfdrive.iqmodeld import emac_model_meta
from iqpilot.selfdrive.ui.mici.layouts.settings import models


@pytest.fixture
def catalog_calls(monkeypatch):
  calls = []

  def fake_big_models(params=None):
    calls.append(1)
    return [("cinque", "Cinque")]

  monkeypatch.setattr(emac_model_meta, "big_models", fake_big_models)
  monkeypatch.setattr(models, "_big_options_cache", (0.0, []))
  return calls


def test_big_label_does_not_reparse_catalog_every_frame(catalog_calls):
  for _ in range(600):
    assert models._big_label("cinque") == "Cinque"
  assert len(catalog_calls) == 1


def test_stale_catalog_reloads_off_the_calling_thread(catalog_calls, monkeypatch):
  models._big_options()
  monkeypatch.setattr(models, "_big_options_cache", (time.monotonic() - models._BIG_OPTIONS_TTL - 1.0, [("cinque", "Cinque")]))
  caller_thread_calls = []
  real_load = models._load_big_options

  def tracking_load():
    caller_thread_calls.append(threading.current_thread() is threading.main_thread())
    return real_load()

  monkeypatch.setattr(models, "_load_big_options", tracking_load)
  assert models._big_options() == [("cinque", "Cinque")]
  deadline = time.monotonic() + 2.0
  while not caller_thread_calls and time.monotonic() < deadline:
    time.sleep(0.01)
  assert caller_thread_calls == [False]


def test_fresh_request_reparses_synchronously(catalog_calls):
  models._big_options()
  models._big_options(fresh=True)
  assert len(catalog_calls) == 2
