"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""

import pyray as rl
from enum import IntEnum
from collections.abc import Callable

from iqpilot.system.ui.widgets.scroller import Scroller, draw_scroller_edge_fades, draw_scroller_page_slider
from iqpilot.selfdrive.ui.mici.layouts.settings.network.wifi_ui import WifiUIMici
from iqpilot.selfdrive.ui.mici.layouts.settings.network.esim_ui import EsimUIMici
from iqpilot.selfdrive.ui.mici.layouts.settings.network.speed_test_ui import SpeedTestMici
from iqpilot.selfdrive.ui.mici.widgets.stock_button import BigButton, BigParamControl, BigToggle
from iqpilot.selfdrive.ui.mici.widgets.stock_dialog import BigInputDialog
from iqpilot.selfdrive.ui.ui_state import ui_state
from iqpilot.system.ui.lib.application import gui_app
from iqpilot.system.ui.widgets.nav_widget import NavWidget
from iqpilot.system.ui.lib.wifi_manager import WifiManager, Network, MeteredType
from iqpilot.system.hardware.tici.esim_manager import get_esim_manager
from iqpilot.system.ui.lib.multilang import tr
from iqpilot.system.ui.lib.cellular_info import CellularInfoPoller
from iqpilot.system.ui.lib.speed_test import compact_text, shared_speed_test
from iqpilot.system.wireguard import tunnel as wireguard


class NetworkPanelType(IntEnum):
  NONE = 0
  WIFI = 1
  ESIM = 2


class NetworkLayoutMici(NavWidget):
  CALLBACK_INTERVAL_FRAMES = 3

  def __init__(self, back_callback: Callable):
    super().__init__()

    self._current_panel = NetworkPanelType.WIFI
    self._callback_frame = 0
    self._esim_profile_count: str | None = None
    self._esim_profile_frame = 0

    self._wifi_manager = WifiManager()
    self._wifi_manager.set_active(False)
    self._wifi_ui = WifiUIMici(self._wifi_manager)
    self._esim_ui = EsimUIMici(back_callback=lambda: self._switch_to_panel(NetworkPanelType.NONE))
    self._esim_manager = get_esim_manager()

    self._wifi_manager.add_callbacks(
      networks_updated=self._on_network_updated,
    )

    # ******** Tethering ********
    def tethering_toggle_callback(checked: bool):
      self._tethering_toggle_btn.set_enabled(False)
      self._network_metered_btn.set_enabled(False)
      self._wifi_manager.set_tethering_active(checked)

    self._tethering_checked = False
    self._tethering_toggle_btn = BigButton(tr("tethering"), tr("disabled"))
    self._tethering_toggle_btn.set_click_callback(lambda: self._on_tethering_clicked(tethering_toggle_callback))

    def tethering_password_callback(password: str):
      if password:
        self._wifi_manager.set_tethering_password(password)

    def tethering_password_clicked():
      tethering_password = self._wifi_manager.tethering_password
      dlg = BigInputDialog(tr("enter password..."), tethering_password, minimum_length=8,
                           confirm_callback=tethering_password_callback)
      gui_app.push_widget(dlg)

    self._tethering_password_btn = BigButton(tr("tethering password"))
    self._tethering_password_btn.set_click_callback(tethering_password_clicked)

    # ******** IP Address ********
    self._ip_address_btn = BigButton(tr("IP Address"), tr("Not connected"))

    self._cellular = CellularInfoPoller()
    self._carrier_btn = BigButton(tr("carrier"), tr("no modem"))
    self._cell_ip_btn = BigButton(tr("cellular IP"), tr("not connected"))
    self._cell_ipv6_btn = BigButton(tr("cellular IPv6"), "")
    self._cell_ipv6_btn.set_visible(lambda: bool(self._cellular.info.ipv6))

    self._speed_test = shared_speed_test()
    self._speed_test_ui = SpeedTestMici(self._speed_test)
    self._speed_test_btn = BigButton(tr("speed test"), tr("tap to run"))
    self._speed_test_btn.set_click_callback(lambda: gui_app.push_widget(self._speed_test_ui))

    self._wireguard_btn = BigToggle(tr("wireguard"), "", initial_state=wireguard.enabled(),
                                    toggle_callback=lambda on: wireguard.set_enabled(on))
    self._wireguard_installed = wireguard.installed()
    self._wireguard_btn.set_visible(lambda: self._wireguard_installed)

    def network_metered_callback(enabled: bool):
      self._network_metered_btn.set_enabled(False)
      self._wifi_manager.set_current_network_metered(MeteredType.YES if enabled else MeteredType.NO)

    self._network_metered_btn = BigToggle(tr("Wi-Fi Metered"), toggle_callback=network_metered_callback)
    self._network_metered_btn.set_enabled(False)

    wifi_button = BigButton(tr("wi-fi"))
    wifi_button.set_click_callback(lambda: gui_app.push_widget(self._wifi_ui))
    self._esim_button = BigButton(tr("eSIM"), tr("manage profiles"))
    self._esim_button.set_click_callback(lambda: self._switch_to_panel(NetworkPanelType.ESIM))
    self._esim_button.set_visible(lambda: self._esim_manager.is_supported())

    # ******** Advanced settings ********
    # ******** Roaming toggle ********
    self._roaming_btn = BigParamControl(tr("enable roaming"), "GsmRoaming", toggle_callback=self._toggle_roaming)

    # ******** APN settings ********
    self._apn_btn = BigButton(tr("apn settings"))
    self._apn_btn.set_click_callback(self._edit_apn)

    # ******** Cellular metered toggle ********
    self._cellular_metered_btn = BigParamControl(tr("cellular metered"), "GsmMetered", toggle_callback=self._toggle_cellular_metered)

    # Main scroller ----------------------------------
    self._scroller = Scroller([
      wifi_button,
      self._esim_button,
      self._speed_test_btn,
      self._wireguard_btn,
      self._network_metered_btn,
      self._tethering_toggle_btn,
      self._tethering_password_btn,
      # /* Advanced settings
      self._roaming_btn,
      self._apn_btn,
      self._cellular_metered_btn,
      # */
      self._ip_address_btn,
      self._carrier_btn,
      self._cell_ip_btn,
      self._cell_ipv6_btn,
    ], snap_items=False)

    # Set initial config
    roaming_enabled = ui_state.params.get_bool("GsmRoaming")
    metered = ui_state.params.get_bool("GsmMetered")
    self._wifi_manager.update_gsm_settings(roaming_enabled, ui_state.params.get("GsmApn") or "", metered)

    # Set up back navigation
    self.set_back_callback(back_callback)

  def _back_enabled(self) -> bool:
    # Only allow swipe-to-dismiss back to settings when no sub-panel (eSIM) is open.
    return self._current_panel == NetworkPanelType.NONE

  def _update_state(self):
    super()._update_state()

    # konn3kt has no managed cellular SIM, so always expose the GSM/APN settings.
    show_cell_settings = True
    self._wifi_manager.set_ipv4_forward(show_cell_settings)
    self._roaming_btn.set_visible(show_cell_settings)
    self._apn_btn.set_visible(show_cell_settings)
    self._cellular_metered_btn.set_visible(show_cell_settings)

    self._update_status_cards()

    self._esim_profile_frame += 1
    if self._esim_profile_frame % 30 == 0:
      esim_profiles = (self._esim_manager.get_state().profiles or []) if self._esim_manager.is_supported() else []
      count = f"{len(esim_profiles)} profiles"
      if count != self._esim_profile_count:
        self._esim_profile_count = count
        self._esim_button.set_value(count)

  @staticmethod
  def _set_value(button: BigButton, value: str) -> None:
    if button.get_value() != value:
      button.set_value(value)

  def _update_status_cards(self):
    info = self._cellular.poll()
    self._set_value(self._carrier_btn, info.carrier_text or tr("no modem"))
    self._set_value(self._cell_ip_btn, info.ipv4 or tr("not connected"))
    self._set_value(self._cell_ipv6_btn, info.ipv6)
    self._set_value(self._speed_test_btn, compact_text(self._speed_test.state) or tr("tap to run"))
    if self._esim_profile_frame % 30 == 0:
      self._wireguard_installed = wireguard.installed()
    if self._esim_profile_frame % 30 == 0 and self._wireguard_installed:
      self._wireguard_btn.set_checked(wireguard.enabled())
      self._set_value(self._wireguard_btn, wireguard.describe_compact(wireguard.read_status()))

  def show_event(self):
    super().show_event()
    self._current_panel = NetworkPanelType.NONE
    self._esim_profile_frame = 0
    self._esim_profile_count = None
    self._roaming_btn.refresh()
    self._cellular_metered_btn.refresh()
    self._scroller.show_event()

  def hide_event(self):
    super().hide_event()
    if self._current_panel == NetworkPanelType.ESIM:
      self._esim_ui.hide_event()

  def _toggle_roaming(self, checked: bool):
    self._wifi_manager.update_gsm_settings(checked, ui_state.params.get("GsmApn") or "", ui_state.params.get_bool("GsmMetered"))

  def _edit_apn(self):
    def update_apn(apn: str):
      apn = apn.strip()
      if apn == "":
        ui_state.params.remove("GsmApn")
      else:
        ui_state.params.put("GsmApn", apn)

      self._wifi_manager.update_gsm_settings(ui_state.params.get_bool("GsmRoaming"), apn, ui_state.params.get_bool("GsmMetered"))

    current_apn = ui_state.params.get("GsmApn") or ""
    dlg = BigInputDialog(tr("enter APN"), current_apn, minimum_length=0, confirm_callback=update_apn)
    gui_app.push_widget(dlg)

  def _toggle_cellular_metered(self, checked: bool):
    self._wifi_manager.update_gsm_settings(ui_state.params.get_bool("GsmRoaming"), ui_state.params.get("GsmApn") or "", checked)

  def _on_tethering_clicked(self, toggle_callback):
    self._tethering_checked = not self._tethering_checked
    self._tethering_toggle_btn.set_value(tr("enabled") if self._tethering_checked else tr("disabled"))
    toggle_callback(self._tethering_checked)

  def _on_network_updated(self, networks: list[Network]):
    # Update tethering state
    tethering_active = self._wifi_manager.is_tethering_active()
    self._tethering_toggle_btn.set_enabled(True)
    self._network_metered_btn.set_enabled(lambda: not tethering_active and bool(self._wifi_manager.ipv4_address))
    self._tethering_checked = tethering_active
    self._tethering_toggle_btn.set_value(tr("enabled") if tethering_active else tr("disabled"))

    # Update IP address
    self._ip_address_btn.set_value(self._wifi_manager.ipv4_address or tr("Not connected"))

    self._network_metered_btn.set_checked(self._wifi_manager.current_network_metered == MeteredType.YES)

  def _switch_to_panel(self, panel_type: NetworkPanelType):
    if panel_type == NetworkPanelType.ESIM:
      if not self._esim_manager.is_supported():
        return
      self._esim_ui.show_event()
    elif self._current_panel == NetworkPanelType.ESIM:
      self._esim_ui.hide_event()

    self._current_panel = panel_type

  def _render(self, rect: rl.Rectangle):
    rl.draw_rectangle_rec(rect, rl.BLACK)
    if self._callback_frame % self.CALLBACK_INTERVAL_FRAMES == 0:
      self._wifi_manager.process_callbacks()
    self._callback_frame += 1

    if self._current_panel == NetworkPanelType.ESIM:
      self._esim_ui.render(rect)
    else:
      self._scroller.render(rect)
      draw_scroller_edge_fades(rect)
      draw_scroller_page_slider(self._scroller, rect)
