# Copyright (c) 2026 IQ.Lvbs. All rights reserved.

import ast
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


def test_remote_language_refresh_notifies_without_writing_back():
  source = ast.parse((Path(__file__).parents[1] / "lib" / "multilang.py").read_text())
  node = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "Multilang")
  namespace = {"time": SimpleNamespace(monotonic=lambda: 100.0), "cloudlog": Mock(), "Callable": Callable}
  exec(compile(ast.Module(body=[node], type_ignores=[]), "language_refresh", "exec"), namespace)
  language = namespace["Multilang"].__new__(namespace["Multilang"])
  language._params = Mock()
  language._params.get.return_value = b"main_pl"
  language._next_refresh = 0.0
  language._language = "en"
  language.codes = {"en": "English", "pl": "Polski"}
  callback = Mock()
  language._change_callbacks = [callback]
  language.setup = Mock()
  language.refresh()
  assert language.language == "pl"
  callback.assert_called_once()
  language.setup.assert_called_once()
  language._params.put.assert_not_called()
  language.refresh()
  language._params.get.assert_called_once()


def test_invalid_remote_language_is_ignored():
  source = ast.parse((Path(__file__).parents[1] / "lib" / "multilang.py").read_text())
  node = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "Multilang")
  namespace = {"time": SimpleNamespace(monotonic=lambda: 100.0), "cloudlog": Mock(), "Callable": Callable}
  exec(compile(ast.Module(body=[node], type_ignores=[]), "language_refresh", "exec"), namespace)
  language = namespace["Multilang"].__new__(namespace["Multilang"])
  language._params = Mock()
  language._params.get.return_value = "invalid"
  language._next_refresh = 0.0
  language._language = "en"
  language.codes = {"en": "English", "pl": "Polski"}
  language._apply_language = Mock()
  language.refresh()
  language._apply_language.assert_not_called()
