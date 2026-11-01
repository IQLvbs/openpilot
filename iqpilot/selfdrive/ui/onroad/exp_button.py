import time
import pyray as rl
from iqpilot.common.params import Params
from iqpilot.selfdrive.ui.ui_state import ui_state
from iqpilot.system.ui.lib.application import gui_app
from iqpilot.system.ui.widgets import Widget


class ExpButton(Widget):
  def __init__(self, button_size: int, icon_size: int):
    super().__init__()
    self._params = Params()
    self._experimental_mode: bool = False
    self._engageable: bool = False

    self._hold_duration = 2.0
    self._held_mode: bool | None = None
    self._hold_end_time: float | None = None

    self._mode_icons = {
      "STOCK ACC": gui_app.texture("icons/chffr_wheel.png", icon_size, icon_size),
      "IQ.CHILL": gui_app.texture("icons_mici/iqstandard_mode_tizi.png", icon_size, icon_size),
      "IQ.PILOT": gui_app.texture("icons_mici/experimental_mode_tizi.png", icon_size, icon_size),
    }
    self._icon_size = icon_size
    self._rect = rl.Rectangle(0, 0, button_size, button_size)

  def _update_state(self) -> None:
    selfdrive_state = ui_state.sm["selfdriveState"]
    self._experimental_mode = selfdrive_state.experimentalMode
    self._engageable = selfdrive_state.engageable or selfdrive_state.enabled

  def _handle_mouse_release(self, _):
    super()._handle_mouse_release(_)
    if not self._is_toggle_allowed():
      return

    new_exp = not self._current_mode()

    self._params.put_bool("ExperimentalMode", new_exp)
    self._held_mode = new_exp
    self._hold_end_time = time.monotonic() + self._hold_duration

  def _render(self, rect: rl.Rectangle) -> None:
    exp = self._current_mode()
    if not ui_state.has_longitudinal_control:
      label = "STOCK ACC"
    elif exp:
      label = "IQ.PILOT"
    else:
      label = "IQ.CHILL"

    alpha = 170 if not self._engageable or self.is_pressed else 255
    icon_rect = rl.Rectangle(
      rect.x + (rect.width - self._icon_size) / 2,
      rect.y + (rect.height - self._icon_size) / 2,
      self._icon_size,
      self._icon_size,
    )
    texture = self._mode_icons[label]
    rl.draw_texture_pro(texture, rl.Rectangle(0, 0, texture.width, texture.height), icon_rect,
                        rl.Vector2(0, 0), 0, rl.Color(255, 255, 255, alpha))

  def _current_mode(self) -> bool:
    now = time.monotonic()
    if self._hold_end_time and now < self._hold_end_time:
      return bool(self._held_mode)
    if self._hold_end_time and now >= self._hold_end_time:
      self._hold_end_time = self._held_mode = None
    return self._experimental_mode

  def _is_toggle_allowed(self):
    if not self._params.get_bool("ExperimentalModeConfirmed"):
      return False
    return ui_state.has_longitudinal_control
