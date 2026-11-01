"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""

import pyray as rl

from iqpilot.common.params import Params
from iqpilot.system.ui.widgets.scroller import NavScroller
from iqpilot.selfdrive.ui.mici.widgets.stock_button import BigButton
from iqpilot.selfdrive.ui.mici.layouts.settings.toggles import TogglesLayoutMici
from iqpilot.selfdrive.ui.mici.layouts.settings.steering import SteeringLayoutMici
from iqpilot.selfdrive.ui.mici.layouts.settings.cruise import CruiseLayoutMici
from iqpilot.selfdrive.ui.mici.layouts.settings.visuals import VisualsLayoutMici
from iqpilot.selfdrive.ui.mici.layouts.settings.models import ModelsLayoutMici
from iqpilot.selfdrive.ui.mici.layouts.settings.display import DisplayLayoutMici
from iqpilot.selfdrive.ui.mici.layouts.settings.vehicle import VehicleLayoutMici
from iqpilot.selfdrive.ui.mici.layouts.settings.dashcam import DashcamLayoutMici
from iqpilot.selfdrive.ui.mici.layouts.settings.network.network_layout import NetworkLayoutMici
from iqpilot.selfdrive.ui.mici.layouts.settings.device import DeviceLayoutMici, PairBigButton
from iqpilot.selfdrive.ui.mici.layouts.settings.developer import DeveloperLayoutMici
from iqpilot.selfdrive.ui.mici.layouts.settings.software import SoftwareLayoutMici
from iqpilot.system.ui.lib.application import gui_app, FontWeight
from iqpilot.system.ui.lib.multilang import tr


class SettingsBigButton(BigButton):
  _ICON_SIZE = 72
  _icon_bounds = {}

  def __init__(self, text, value="", icon=None, *, icon_scale=1.0, icon_offset_y=0, icon_offset_x=0):
    super().__init__(text, value, icon)
    self._icon_scale = icon_scale
    self._icon_offset_y = icon_offset_y
    self._icon_offset_x = icon_offset_x

  def _draw_icon(self, btn_x: float, btn_y: float, btn_width: float):
    icon = self._txt_icon
    if icon is None:
      return
    if icon.id not in self._icon_bounds:
      image = rl.load_image_from_texture(icon)
      self._icon_bounds[icon.id] = rl.get_image_alpha_border(image, 0.05)
      rl.unload_image(image)
    source = self._icon_bounds[icon.id]
    if source.width <= 0 or source.height <= 0:
      return
    scale = min(68 / source.height, 90 / source.width) * self._icon_scale
    width, height = source.width * scale, source.height * scale
    center_x = btn_x + btn_width - 30 - SettingsBigButton._ICON_SIZE / 2 + self._icon_offset_x
    center_y = btn_y + 30 + SettingsBigButton._ICON_SIZE / 2 + self._icon_offset_y
    dest = rl.Rectangle(center_x - width / 2, center_y - height / 2, width, height)
    alpha = int(255 * (0.9 if self.enabled else 0.35))
    rl.draw_texture_pro(icon, source, dest, rl.Vector2(0, 0), 0, rl.Color(255, 255, 255, alpha))

  def _get_label_font_size(self):
    return 64


class CruiseModeButton(SettingsBigButton):
  """Cruise menu button whose icon reflects the active longitudinal mode."""
  _ICON_SIZE = 60

  def __init__(self):
    super().__init__(tr("cruise"), "", gui_app.texture("icons_mici/speedometer.png", self._ICON_SIZE, self._ICON_SIZE))
    self._p = Params()
    self._icons = [
      gui_app.texture("icons_mici/speedometer.png", self._ICON_SIZE, self._ICON_SIZE),
      gui_app.texture("icons_mici/iqstandard_mode_mici_60.png", self._ICON_SIZE, self._ICON_SIZE),
      gui_app.texture("icons_mici/experimental_mode_mici.png", self._ICON_SIZE, self._ICON_SIZE),
    ]

  def _mode_index(self) -> int:
    if not self._p.get_bool("AlphaLongitudinalEnabled"):
      return 0
    if not self._p.get_bool("ExperimentalMode"):
      return 1
    return 2

  def _update_state(self):
    super()._update_state()
    self.set_icon(self._icons[self._mode_index()])


class SettingsLayout(NavScroller):
  def __init__(self):
    super().__init__()
    self._params = Params()

    toggles_panel = TogglesLayoutMici()
    toggles_btn = SettingsBigButton(tr("toggles"), "", gui_app.texture("offroad/icon_toggle.png", 64, 64, keep_aspect_ratio=True), icon_scale=0.92)
    toggles_btn.set_click_callback(lambda: gui_app.push_widget(toggles_panel))

    steering_panel = SteeringLayoutMici()
    steering_btn = SettingsBigButton(tr("steering"), "", gui_app.texture("icons_mici/wheel.png", 64, 64))
    steering_btn.set_click_callback(lambda: gui_app.push_widget(steering_panel))

    cruise_panel = CruiseLayoutMici()
    cruise_btn = CruiseModeButton()
    cruise_btn.set_click_callback(lambda: gui_app.push_widget(cruise_panel))

    visuals_panel = VisualsLayoutMici()
    visuals_btn = SettingsBigButton(tr("visuals"), "", gui_app.texture("offroad/icon_visuals.png", 64, 64, keep_aspect_ratio=True))
    visuals_btn.set_click_callback(lambda: gui_app.push_widget(visuals_panel))

    models_panel = ModelsLayoutMici()
    models_btn = SettingsBigButton(tr("models"), "", gui_app.texture("offroad/icon_models.png", 64, 64, keep_aspect_ratio=True))
    models_btn.set_click_callback(lambda: gui_app.push_widget(models_panel))

    display_panel = DisplayLayoutMici()
    display_btn = SettingsBigButton(tr("display"), "", gui_app.texture("icons_mici/settings/brightness.png", 62, 62))
    display_btn.set_click_callback(lambda: gui_app.push_widget(display_panel))

    vehicle_panel = VehicleLayoutMici()
    vehicle_btn = SettingsBigButton(tr("vehicle"), "", gui_app.texture("offroad/icon_vehicle.png", 64, 64, keep_aspect_ratio=True), icon_scale=0.92)
    vehicle_btn.set_click_callback(lambda: gui_app.push_widget(vehicle_panel))

    dashcam_panel = DashcamLayoutMici()
    dashcam_btn = SettingsBigButton(tr("dashcam"), "", gui_app.texture("icons_mici/settings/camera.png", 64, 56), icon_scale=0.72, icon_offset_y=-6)
    dashcam_btn.set_click_callback(lambda: gui_app.push_widget(dashcam_panel))

    network_panel = NetworkLayoutMici(back_callback=lambda: gui_app.pop_widget())
    network_btn = SettingsBigButton(tr("network"), "", gui_app.texture("icons/network.png", 64, 64, keep_aspect_ratio=True), icon_scale=0.85)
    network_btn.set_click_callback(lambda: gui_app.push_widget(network_panel))


    device_panel = DeviceLayoutMici()
    device_btn = SettingsBigButton(tr("device"), "", gui_app.texture("icons_mici/settings/device_icon.png", 72, 58, keep_aspect_ratio=True), icon_scale=0.9)
    device_btn.set_click_callback(lambda: gui_app.push_widget(device_panel))

    software_panel = SoftwareLayoutMici()
    software_btn = SettingsBigButton(tr("software"), "", gui_app.texture("offroad/icon_software.png", 64, 64, keep_aspect_ratio=True), icon_scale=0.85)
    software_btn.set_click_callback(lambda: gui_app.push_widget(software_panel))

    developer_panel = DeveloperLayoutMici()
    developer_btn = SettingsBigButton(tr("developer"), "", gui_app.texture("icons/shell.png", 64, 64, keep_aspect_ratio=True),
                                      icon_scale=0.62, icon_offset_y=-4, icon_offset_x=2)
    developer_btn.set_click_callback(lambda: gui_app.push_widget(developer_panel))

    self._scroller.add_widgets([
      device_btn,
      network_btn,
      PairBigButton(),
      models_btn,
      software_btn,
      steering_btn,
      cruise_btn,
      visuals_btn,
      display_btn,
      dashcam_btn,
      vehicle_btn,
      toggles_btn,
      developer_btn,
    ])

    self._font_medium = gui_app.font(FontWeight.MEDIUM)
