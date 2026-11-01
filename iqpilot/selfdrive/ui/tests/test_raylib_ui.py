import time
import pytest
from iqpilot.selfdrive.test.helpers import with_processes
from iqpilot.system.ui.lib import application
from iqpilot.system.ui.lib.utils import gui_style_color


@pytest.mark.linux
@with_processes(["ui"])
def test_raylib_ui():
  """Test initialization of the UI widgets is successful."""
  time.sleep(1)


def test_style_colors_match_gui_style_abi(monkeypatch):
  calls = []
  monkeypatch.setattr(application.rl, "gui_set_style", lambda *args: calls.append(args))
  monkeypatch.setattr(application, "gui_style_color", lambda color: 27)
  application.GuiApplication._set_styles(None)
  assert [call[2] for call in calls[-3:]] == [27, 27, 27]


def test_gui_style_color_uses_binding_value_type():
  color = application.rl.Color(229, 229, 229, 255)
  value_type = application.rl.ffi.typeof(application.rl.raylib.GuiSetStyle).args[2]
  expected = int(application.rl.ffi.cast(value_type, application.rl.color_to_int(color)))
  assert gui_style_color(color) == expected
