"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos
"""
import ast
import copy
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from iqpilot.selfdrive.iqmodeld.models import helpers


class Params:
  def __init__(self, values=None):
    self.values = values or {}

  def get(self, key):
    return self.values.get(key)

  def put(self, key, value):
    self.values[key] = value

  def remove(self, key):
    self.values.pop(key, None)

  def get_bool(self, key):
    return bool(self.get(key))


@pytest.fixture
def saved():
  bundle = helpers._load_default_bundle_dict()
  bundle.update(ref="custom", internalName="Saved custom", displayName="Saved Custom Model")
  return Params({"ModelManager_ActiveBundle": bundle})


def test_both_ui_names_without_manager(saved):
  before = copy.deepcopy(saved.values)
  for relative, cls, method in [
    ("ui/layouts/settings/iq_panels.py", "ModelsLayout", "_update_state"),
    ("selfdrive/ui/mici/layouts/settings/models.py", "ModelsLayoutMici", "_active_model_name"),
  ]:
    path = Path(__file__).resolve().parents[3] / relative
    tree = ast.parse(path.read_text())
    node = next(n for c in tree.body if isinstance(c, ast.ClassDef) and c.name == cls
                for n in c.body if isinstance(n, ast.FunctionDef) and n.name == method)
    env = {"tr": lambda text: text, "get_selected_model_name": helpers.get_selected_model_name,
           "time": time, "active_model_declares_drive_profile": lambda params: False,
           "drive_profile_name": lambda params: "neutral",
           "ui_state": SimpleNamespace(params=saved, sm={"iqModelManager": None}, is_offroad=lambda: True)}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), env)
    values = []
    item = SimpleNamespace(set_value=values.append, set_enabled=lambda value: None)
    ui = SimpleNamespace(_update_refresh_state=lambda: None, _handle_bundle_download_progress=lambda: None,
                         current_model_item=SimpleNamespace(action_item=item, set_description=lambda value: None),
                         drive_profile_item=SimpleNamespace(action_item=item, set_description=lambda value: None),
                         desired_arrival_item=SimpleNamespace(action_item=item, set_visible=lambda value: None),
                         _last_drive_profile_check_t=0.0, _drive_profile_supported=False,
                         redownload_item=SimpleNamespace(action_item=item), _can_redownload=lambda: False,
                         _redownload_target_bundle=lambda: None, _private_ui_attached=True,
                         big_model_item=SimpleNamespace(action_item=SimpleNamespace(set_value=lambda value: None)),
                         _big_model_value=lambda: "Off")
    result = env[method](ui)
    assert (result if method == "_active_model_name" else values[0]) == "Saved custom"
  assert saved.values == before


@pytest.mark.parametrize("stored,expected", [(None, "Default (CD210)"), ({"ref": "default"}, "Default (CD210)"),
                                           ({"internalName": "Broken"}, "Model unavailable"), ("bad", "Model unavailable")])
def test_saved_name_states(stored, expected):
  assert helpers.get_selected_model_name(Params({"ModelManager_ActiveBundle": stored})) == expected

