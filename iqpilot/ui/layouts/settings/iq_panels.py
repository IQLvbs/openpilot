"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos

IQ.Pilot settings panels.
"""
import datetime
import os
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, IntEnum
from functools import partial
from pathlib import Path

import pyray as rl
from iqdbc.car.subaru.values import CAR as SUBARU_CAR
from iqdbc.car.subaru.values import SubaruFlags
from iqdbc.car.volkswagen.values import CAR as VOLKSWAGEN_CAR
from iqdbc.car.volkswagen.values import VolkswagenFlags

from iqpilot.cereal import custom
from iqpilot.common.params import Params
from iqpilot.common.turn_guard_settings import get_turn_guards_enabled, reset_turn_guards, set_turn_guards_enabled
from iqpilot.common.ui_settings import DISPLAY_BRIGHTNESS_VALUES, NO_NUDGE_WARNING, SLEEP_TIMER_VALUES
from iqpilot.common.swaglog import cloudlog
from iqpilot.selfdrive.car.vehicle_catalog import load_catalog
from iqpilot.selfdrive.iqmodeld.model_cache import bundle_matches, clear_model_cache, model_cache_size, remove_bundle_files
from iqpilot.selfdrive.iqmodeld.drive_profile import (
  DRIVE_PROFILE_LABELS,
  DRIVE_PROFILE_PARAM_VALUES,
  active_model_declares_drive_profile,
  arrival_time_choices,
  desired_arrival_label,
  drive_profile_name,
  profile_label,
)
from iqpilot.selfdrive.iqmodeld.models.helpers import get_cached_model_bundles, get_selected_model_name, is_default_bundle, select_default_model
from iqpilot.selfdrive.iqmodeld.models.runners.model_runner import CUSTOM_MODEL_PATH
from iqpilot.selfdrive.ui.layouts.settings import settings as OP
from iqpilot.selfdrive.ui.layouts.settings.developer import DeveloperLayout
from iqpilot.selfdrive.ui.layouts.settings.device import DeviceLayout
from iqpilot.selfdrive.ui.layouts.settings.software import SoftwareLayout
from iqpilot.selfdrive.ui.layouts.settings.toggles import TogglesLayout
from iqpilot.selfdrive.ui.lib.nnff_settings import nnff_availability
from iqpilot.selfdrive.ui.lib.turn_guard_settings import TURN_GUARDS_DESCRIPTION, turn_guards_availability
from iqpilot.selfdrive.ui.ui_state import device, ui_state
from iqpilot.system.hardware import HARDWARE
from iqpilot.system.hardware.hw import Paths
from iqpilot.system.ui.iqwidgets.lib.styles import ink, metrics, style
from iqpilot.system.ui.iqwidgets.lib.utils import WideButtonAction
from iqpilot.system.ui.iqwidgets.widgets.helpers.status_pill import StatusPill
from iqpilot.system.ui.iqwidgets.widgets.list_view import (
  IQLineSeparator,
  IQListItem,
  NoticeModal,
  PickerDialog,
  PickerGroup,
  PickerItem,
  multiple_button_item,
  option_item,
  progress_item,
  toggle_item,
  toggle_item_iq,
)
from iqpilot.system.ui.iqwidgets.widgets.list_view import (
  dual_button_item as dual_button_item,
)
from iqpilot.system.ui.lib.application import FontWeight, MousePos, gui_app
from iqpilot.system.ui.lib.multilang import tr, tr_noop
from iqpilot.system.ui.lib.text_measure import measure_text_cached
from iqpilot.system.ui.lib.wifi_manager import WifiManager
from iqpilot.system.ui.widgets import DialogResult, Widget
from iqpilot.system.ui.widgets.button import Button, ButtonStyle
from iqpilot.system.ui.widgets.confirm_dialog import ConfirmDialog
from iqpilot.system.ui.widgets.list_view import ButtonAction, button_item, text_item
from iqpilot.system.ui.widgets.network import TEAL, NetworkUI
from iqpilot.system.ui.widgets.network import PanelType as NetworkPanelType
from iqpilot.system.ui.widgets.option_dialog import MultiOptionDialog
from iqpilot.system.ui.widgets.scroller_tici import LineSeparator, Scroller
from iqpilot.ui.onroad.offline_tiles import offline_map_root
from iqpilot.ui.optional_private import optional_private_ui


def _fmt_seconds(value) -> str:
  return f"{value} s" if value < 60 else f"{int(value / 60)} m"

_HUD_TOGGLES = {
  "IQBlindSpotAlerts": (
    lambda: tr("Blind Spot Alerts"),
    lambda: tr("Flashes a side warning whenever the car reports something sitting in your blind spot (BSM-equipped cars only)."),
  ),
  "IQRoadNameOverlay": (
    lambda: tr("Road Name Overlay"),
    lambda: tr("Show the current road's name over the driving view."
               "<br>Requires offline map data for your region to be installed."),
  ),
  "IQBlinkerIndicators": (
    lambda: tr("Blinker Indicators"),
    lambda: tr("Mirror the car's blinkers as arrows on the driving screen."),
  ),
}

class VisualsLayout(Widget):
  def __init__(self):
    super().__init__()
    self._params = Params()
    self._scroller = Scroller(self._build_rows(), line_separator=True, spacing=0)

  def _build_rows(self):
    self._toggles = {
      key: toggle_item(title=title, description=desc, param=key,
                       initial_state=ui_state.params.get_bool(key))
      for key, (title, desc) in _HUD_TOGGLES.items()
    }

    self._chevron_info = toggle_item(
      title=lambda: tr("Lead Vehicle Stats"),
      description=lambda: tr("Show distance, speed, and time gap below the lead vehicle marker."),
      initial_state=bool(ui_state.params.get("IQLeadReadouts", return_default=True)),
      callback=lambda enabled: ui_state.params.put("IQLeadReadouts", 4 if enabled else 0),
    )
    self._dev_ui_info = toggle_item(
      title=lambda: tr("Onroad Developer UI"),
      description=lambda: tr("Overlay a bar of live control metrics (steering, lateral accel, lead data) while driving."),
      initial_state=bool(int(ui_state.params.get("IQDevUIInfo", return_default=True))),
      callback=lambda on: ui_state.params.put("IQDevUIInfo", 3 if on else 0),
    )

    return [
      option_item(
        param="Brightness", title=lambda: tr("Display Brightness"), description="",
        min_value=0, max_value=20, value_change_step=1, value_map=DISPLAY_BRIGHTNESS_VALUES,
        label_callback=lambda v: tr("Default") if v == 0 else f"{v} %", inline=True,
      ),
      option_item(
        param="InteractivityTimeout", title=lambda: tr("Sleep Timer"), description="",
        min_value=0, max_value=3, value_change_step=1, value_map=SLEEP_TIMER_VALUES,
        label_callback=lambda v: tr("Default") if not v else _fmt_seconds(v), inline=True,
      ),
      IQLineSeparator(60),
      *self._toggles.values(),
      self._chevron_info,
      self._dev_ui_info,
    ]

  def _update_state(self):
    super()._update_state()
    for key, row in self._toggles.items():
      row.action_item.set_state(self._params.get_bool(key))
    self._chevron_info.action_item.set_state(bool(self._params.get("IQLeadReadouts", return_default=True)))
    self._dev_ui_info.action_item.set_state(bool(int(ui_state.params.get("IQDevUIInfo", return_default=True))))

  def _render(self, rect):
    self._scroller.render(rect)

  def show_event(self):
    self._scroller.show_event()

_IDLE_BG = rl.Color(52, 55, 62, 255)

class ScanPhase(Enum):
  IDLE = 0
  RUNNING = 1
  FAILED = 2

class IQNetworkUI(NetworkUI):
  def __init__(self, wifi_manager):
    super().__init__(wifi_manager)
    self._phase = ScanPhase.IDLE
    self.scan_button = Button(lambda: tr("Searching...") if self._phase == ScanPhase.RUNNING else tr("Scan"), self._start_scan,
                              button_style=ButtonStyle.TRANSPARENT_WHITE_TEXT,
                              font_size=60, border_radius=30)
    self.scan_button.set_rect(rl.Rectangle(0, 0, 400, 100))
    self._wifi_manager.add_callbacks(networks_updated=self._on_networks_updated)

  def _set_phase(self, phase: ScanPhase):
    self._phase = phase
    running = phase == ScanPhase.RUNNING
    self.scan_button.set_enabled(not running)
    self._wifi_manager._scanning = running

  def _start_scan(self):
    self._set_phase(ScanPhase.RUNNING)
    threading.Thread(target=self._scan_worker, daemon=True).start()

  def _scan_worker(self):
    try:
      self._wifi_manager._update_networks()
      self._wifi_manager._request_scan()
      self._wifi_manager._last_network_update = time.monotonic()
    except Exception:
      cloudlog.exception("IQNetworkUI scan failed")
      self._phase = ScanPhase.FAILED

  def _on_networks_updated(self, networks):
    if self._phase == ScanPhase.RUNNING:
      self._set_phase(ScanPhase.IDLE)

  def _render(self, rect: rl.Rectangle):
    super()._render(rect)

    if self._phase == ScanPhase.FAILED:
      self._set_phase(ScanPhase.IDLE)

    if self._current_panel == NetworkPanelType.WIFI:
      self.scan_button.set_position(self._rect.x, self._rect.y + 20)
      r = self.scan_button.rect
      bg = TEAL if self.scan_button.enabled else _IDLE_BG
      rl.draw_rectangle_rounded(r, 30 / (min(r.width, r.height) / 2), 10, bg)
      self.scan_button.render()

MAP_PATH = Path(Paths.mapd_root()) / "offline"
OFFLINE_TILES_PATH = offline_map_root() / "regions"

_BOUNDS_BASE_URL = "https://raw.githubusercontent.com/pfeiferj/openpilot-mapd/main/"
_REGION_PARAMS = ("OsmDownloadedDate", "OsmLocal", "OsmLocationName", "OsmLocationTitle", "OsmStateName", "OsmStateTitle")


SAB_BRAKE_RESPONSE_OPTIONS = [
  (tr_noop("Stay Engaged"), tr_noop("Remain Active: braking never interrupts steering assistance.")),
  (tr_noop("Standby"), tr_noop("Standby: braking parks steering assistance; it rejoins once you're off the pedal.")),
  (tr_noop("Disengage"), tr_noop("Disengage: braking shuts steering assistance off entirely.")),
]

_MODE_DISENGAGE = 2
SAB_STEERING_OVERRIDE_DESC = tr(
  "Steering assistance stands by while you steer yourself and rejoins the moment you let go. It stays armed the whole time; only the torque stops."
)
class SteeringLayout(Widget):
  def __init__(self):
    super().__init__()

    items = self._initialize_items()
    self._scroller = Scroller(items, line_separator=False, spacing=0)

    self._delay_control.action_item.set_value(int(float(ui_state.params.get("IQSoftwareSteerDelay", return_default=True)) * 100))

  def _initialize_items(self):
    self._aol_base_desc = tr("Enable Always on Lateral (AOL). Disable this toggle to return to stock IQ.Pilot steering engagement behavior.")
    self._aol_toggle = toggle_item(
      param="AolEnabled",
      title=lambda: tr("Always on Lateral (AOL)"),
      description=self._aol_base_desc,
    )
    self._steering_mode = multiple_button_item(
      param="AolSteeringMode",
      title=lambda: tr("Brake Response Mode"),
      description="",
      buttons=[lambda label=label: tr(label) for label, _ in SAB_BRAKE_RESPONSE_OPTIONS],
      inline=False,
      button_width=350,
    )
    self._pause_on_steering_override_toggle = toggle_item(
      title=lambda: tr("Pause Steering on Override"),
      description=SAB_STEERING_OVERRIDE_DESC,
      param="AolPauseOnSteeringOverride",
    )
    self._lane_change_setting = multiple_button_item(
      title=lambda: tr("Blinker-Initiated Lane Change"),
      description=lambda: tr("Choose whether a steering nudge is required before a blinker-initiated lane change."),
      buttons=[lambda: tr("Confirmation (Nudge)"), lambda: tr("No Confirmation (No Nudge)")],
      button_width=480,
      selected_index=int(ui_state.params.get("IQLaneChangeTimer", return_default=True)),
      callback=self._set_lane_change_mode,
    )
    self._lane_turn_desire_toggle = toggle_item(
      lambda: tr("Blinker-Initiated Turn Planning"),
      lambda: tr("If you're driving at 20 mph (32 km/h) or below and have your blinker on,"
                 " the car will plan a turn in that direction at the nearest drivable path."
                 " This prevents situations (like at red lights) where the car might plan the wrong turn direction."),
      param="IQLaneTurnDesire"
    )
    self._edge_guard_toggle = toggle_item(
      title=lambda: tr("Lane Edge Guard"),
      description=lambda: tr("Blocks lane changes when a road edge is detected on the target side."),
      param="IQEdgeGuard",
    )
    self._turn_guards_toggle = toggle_item(
      title=lambda: tr("Turn Guards (Experimental)"),
      description=lambda: tr(TURN_GUARDS_DESCRIPTION),
      initial_state=get_turn_guards_enabled(ui_state.params),
      callback=self._on_turn_guards,
    )
    self._refresh_turn_guards()
    self._steer_delay_toggle = toggle_item(lambda: tr("Self-Tuning Steer Delay"), "", param="IQLiveSteerDelay")
    self._delay_control = option_item(
      tr("Manual Delay Offset"), "IQSoftwareSteerDelay", 5, 50,
      tr("How much lead time to add on top of the car's own steering rack delay while self-tuning is off. Default is 0.2 s."),
      1, None, True, "", style.BUTTON_ACTION_WIDTH, None, True, lambda v: f"{float(v):.2f}s"
    )

    items = [
      self._aol_toggle,
      self._steering_mode,
      self._pause_on_steering_override_toggle,
      IQLineSeparator(40),
      self._lane_change_setting,
      IQLineSeparator(40),
      self._lane_turn_desire_toggle,
      IQLineSeparator(40),
      self._edge_guard_toggle,
      IQLineSeparator(40),
      self._turn_guards_toggle,
      IQLineSeparator(40),
      self._steer_delay_toggle,
      self._delay_control,
    ]
    return items

  def _refresh_turn_guards(self):
    available, reason = turn_guards_availability(ui_state.CP, ui_state.params)
    saved = get_turn_guards_enabled(ui_state.params)
    offroad = ui_state.is_offroad()
    self._turn_guards_toggle.action_item.set_state(saved)
    self._turn_guards_toggle.action_item.set_enabled(offroad and (available or saved))
    description = tr(TURN_GUARDS_DESCRIPTION)
    if not available:
      description += "<br><br>" + reason
    if not offroad:
      description += "<br><br>" + tr("Park the vehicle to change this setting.")
    self._turn_guards_toggle.set_description(description)

  def _on_turn_guards(self, enabled: bool):
    if not ui_state.is_offroad() or (enabled and not turn_guards_availability(ui_state.CP, ui_state.params)[0]):
      self._refresh_turn_guards()
      return
    try:
      set_turn_guards_enabled(ui_state.params, enabled)
    except OSError:
      cloudlog.exception("Failed to save turn guards")
      self._refresh_turn_guards()
      return
    ui_state.params.put_bool("OnroadCycleRequested", True)
    self._refresh_turn_guards()

  def _set_lane_change_mode(self, selected: int):
    current = int(ui_state.params.get("IQLaneChangeTimer", return_default=True))
    if selected == current:
      return
    if selected == 0:
      ui_state.params.put("IQLaneChangeTimer", 0)
      return

    self._lane_change_setting.action_item.set_selected_button(current)

    def confirm(result):
      if result == DialogResult.CONFIRM:
        ui_state.params.put("IQLaneChangeTimer", 1)
        self._lane_change_setting.action_item.set_selected_button(1)

    dialog = ConfirmDialog(tr(NO_NUDGE_WARNING), tr("Enable No Nudge"), large=True, title=tr("Are you sure?"))
    gui_app.set_modal_overlay(dialog, callback=confirm)

  @staticmethod
  def _has_limited_sab_options() -> bool:
    brand = ""
    if ui_state.is_offroad() and (bundle := ui_state.params.get("CarPlatformBundle")):
      brand = bundle.get("brand", "")
    if not brand and ui_state.CP:
      brand = ui_state.CP.brand
    return brand == "rivian"

  def _update_state(self):
    super()._update_state()

    self._refresh_turn_guards()
    self._aol_toggle.action_item.set_enabled(ui_state.is_offroad())
    aol_enabled = self._aol_toggle.action_item.get_state()
    limited = self._has_limited_sab_options()
    if limited:
      self._steering_mode.action_item.set_selected_button(_MODE_DISENGAGE)
    self._steering_mode.action_item.set_enabled(ui_state.is_offroad() and aol_enabled and not limited)
    self._steering_mode.set_description("")
    self._pause_on_steering_override_toggle.action_item.set_enabled(ui_state.is_offroad() and aol_enabled)

    self._lane_change_setting.action_item.set_selected_button(int(ui_state.params.get("IQLaneChangeTimer", return_default=True)))

    turn_desire = ui_state.params.get_bool("IQLaneTurnDesire")
    live_delay = ui_state.params.get_bool("IQLiveSteerDelay")
    self._lane_turn_desire_toggle.action_item.set_state(turn_desire)
    self._edge_guard_toggle.action_item.set_state(ui_state.params.get_bool("IQEdgeGuard"))
    self._steer_delay_toggle.action_item.set_state(live_delay)
    self._delay_control.set_visible(not live_delay)
    delay_desc = tr("Let IQ.Pilot measure how long your steering takes to respond and keep that figure up to date. "
                    "Switch it off to pin the timing yourself.")
    if live_delay:
      measured = ui_state.measured_steer_delay()
      if measured is None:
        delay_desc += f"<br>{tr('Measured:')} {tr('not yet, drive to calibrate')}"
      else:
        delay_desc += f"<br>{tr('Measured:')} {measured:.3f} s"
    elif ui_state.CP:
      sw = float(ui_state.params.get("IQSoftwareSteerDelay", "0.2"))
      cp = ui_state.CP.steerActuatorDelay
      delay_desc += f"<br>{tr('Rack:')} {cp:.2f} s + {tr('Offset:')} {sw:.2f} s = {tr('Total:')} {cp + sw:.2f} s"
    self._steer_delay_toggle.set_description(delay_desc)

  def _render(self, rect):
    self._scroller.render(rect)

  def show_event(self):
    self._refresh_turn_guards()
    self._scroller.show_event()


class IQDeveloperLayout(DeveloperLayout):
  def __init__(self):
    super().__init__()
    self.error_log_path = os.path.join(Paths.crash_log_root(), "error.log")
    self._is_release_branch: bool = self._is_release or ui_state.params.get_bool("IsReleaseIqBranch")
    self._is_development_branch: bool = ui_state.params.get_bool("IsTestedBranch") or ui_state.params.get_bool("IsDevelopmentBranch")
    self._initialize_items()

    for item in self.items:
      self._scroller.add_widget(item)

  def _initialize_items(self):
    self.error_log_btn = button_item(lambda: tr("Crash Log"), lambda: tr("VIEW"),
                                     lambda: tr("View the error log for IQ.Pilot crashes."), callback=self._on_error_log_clicked)

    self.items: list = [self.error_log_btn]

  def _on_delete_confirm(self, result):
    if result == DialogResult.CONFIRM:
      if os.path.exists(self.error_log_path):
        os.remove(self.error_log_path)

  def _on_error_log_closed(self, result, log_exists):
    if result == DialogResult.CONFIRM and log_exists:
      dialog2 = ConfirmDialog(tr("Delete this log?"), tr("Yes"), tr("No"), rich=False)
      gui_app.set_modal_overlay(dialog2, callback=self._on_delete_confirm)

  def _on_error_log_clicked(self):
    text = ""
    if os.path.exists(self.error_log_path):
      text = f"<b>{datetime.datetime.fromtimestamp(os.path.getmtime(self.error_log_path)).strftime('%d-%b-%Y %H:%M:%S').upper()}</b><br><br>"
      try:
        with open(self.error_log_path) as file:
          text += file.read()
      except Exception:
        pass
    dialog = NoticeModal(text=text, callback=lambda result: self._on_error_log_closed(result, os.path.exists(self.error_log_path)))
    gui_app.set_modal_overlay(dialog)

  def _update_state(self):
    self.error_log_btn.set_visible(not self._is_release_branch)

UPDATES_DESCRIPTIONS = {
  'disable_updates_offroad': tr_noop(
    "Turns off over-the-air update checks entirely.<br><b>Reboot for this to take effect.</b>"
  ),
  'disable_updates_onroad': tr_noop(
    "Put the device in Always Offroad, or shut the car down, before touching these."
  ),
}

INSTALL_MODE_DOWNLOAD_ONLY = "download_only"
INSTALL_MODE_DOWNLOAD_AND_INSTALL = "download_and_install"
class IQSoftwareLayout(SoftwareLayout):
  def __init__(self):
    super().__init__()
    self._download_btn.set_title(lambda: tr("Update"))
    self.disable_updates_toggle = toggle_item(
      lambda: tr("Disable Updates"),
      description="",
      initial_state=ui_state.params.get_bool("DisableUpdates"),
      callback=self._on_disable_updates_toggled,
    )
    install_mode = ui_state.params.get("UpdaterInstallMode") or INSTALL_MODE_DOWNLOAD_AND_INSTALL
    self.preinstall_updates_toggle = toggle_item(
      lambda: tr("Pre-install Updates"),
      description="",
      initial_state=install_mode == INSTALL_MODE_DOWNLOAD_AND_INSTALL,
      callback=lambda enabled: ui_state.params.put(
        "UpdaterInstallMode", INSTALL_MODE_DOWNLOAD_AND_INSTALL if enabled else INSTALL_MODE_DOWNLOAD_ONLY
      ),
    )
    try:
      os_version = HARDWARE.get_os_version() or "unknown"
    except Exception:
      os_version = "unknown"
    self.iqos_version_item = text_item(lambda: tr("IQ.OS Version"), os_version)
    self._scroller.add_widget(self.iqos_version_item)
    self._scroller.add_widget(self.preinstall_updates_toggle)
    self._scroller.add_widget(self.disable_updates_toggle)

  def _handle_reboot(self, result):
    if result == DialogResult.CONFIRM:
      ui_state.params.put_bool("DisableUpdates", self.disable_updates_toggle.action_item.get_state())
      ui_state.params.put_bool("DoReboot", True)
    else:
      self.disable_updates_toggle.action_item.set_state(ui_state.params.get_bool("DisableUpdates"))

  def _on_disable_updates_toggled(self, enabled):
    dialog = ConfirmDialog(tr("This needs a reboot to apply. Restart now?"), tr("Reboot"))
    gui_app.set_modal_overlay(dialog, callback=self._handle_reboot)

  def _on_uninstall(self):
    def handle_uninstall_confirmation(result):
      if result == DialogResult.CONFIRM:
        ui_state.params.put_bool("DoUninstall", True)
        gui_app.request_close()

    dialog = ConfirmDialog(tr("Are you sure you want to uninstall?"), tr("Uninstall"))
    gui_app.set_modal_overlay(dialog, callback=handle_uninstall_confirmation)

  def _on_select_branch(self):
    current_git_branch = ui_state.params.get("GitBranch") or ""
    branches_str = ui_state.params.get("UpdaterAvailableBranches") or ""
    branches = [b for b in branches_str.split(",") if b]
    current_target = ui_state.params.get("UpdaterTargetBranch") or ""
    top_level_branches = [current_git_branch, "release-mici", "release-tizi", "staging", "dev", "master"]

    if HARDWARE.get_device_type() == "tici":
      top_level_branches = ["release-new", "release-tici", "staging-tici"]
      branches = [b for b in branches if b in ("release-new", "beta") or b.endswith("-tici")]
    pinned_nodes = [PickerItem(b, {'display_name': b}) for b in top_level_branches if b in branches]
    other_nodes = [PickerItem(b, {'display_name': b}) for b in sorted(branches)
                   if b not in top_level_branches and not b.endswith("-prebuilt")]

    folders = [PickerGroup("", pinned_nodes + other_nodes)]

    def _on_branch_selected(result):
      if result == DialogResult.CONFIRM and self._branch_dialog is not None:
        selection = self._branch_dialog.selection_ref
        if selection:
          ui_state.params.put("UpdaterTargetBranch", selection)
          self._branch_btn.action_item.set_value(selection)
          os.system("pkill -SIGUSR1 -f system.updated.updated")
      self._branch_dialog = None

    self._branch_dialog = PickerDialog(tr("Select a branch"), folders, current_target, "",
                                           on_exit=_on_branch_selected)

    gui_app.set_modal_overlay(self._branch_dialog, callback=_on_branch_selected)

  def _update_state(self):
    super()._update_state()
    self.disable_updates_toggle.action_item.set_enabled(ui_state.is_offroad())
    self.disable_updates_toggle.set_visible(True)
    self.preinstall_updates_toggle.action_item.set_enabled(ui_state.is_offroad())
    self.preinstall_updates_toggle.action_item.set_state(
      (ui_state.params.get("UpdaterInstallMode") or INSTALL_MODE_DOWNLOAD_AND_INSTALL) == INSTALL_MODE_DOWNLOAD_AND_INSTALL
    )

    disable_updates_desc = tr(UPDATES_DESCRIPTIONS["disable_updates_offroad"] if ui_state.is_offroad() else UPDATES_DESCRIPTIONS["disable_updates_onroad"])
    self.disable_updates_toggle.set_description(disable_updates_desc)
    waiting = getattr(self, "_waiting_for_updater", False)
    self._download_btn.action_item.set_loading(waiting)
    if waiting:
      self._download_btn.action_item.set_enabled(False)

class IQDeviceLayout(DeviceLayout):
  def __init__(self):
    DeviceLayout.__init__(self)
    self._scroller._line_separator = None

  def _initialize_items(self):
    DeviceLayout._initialize_items(self)
    self._change_language_btn = button_item(lambda: tr("Change Language"), lambda: tr("CHANGE"), callback=self._show_language_dialog)
    self._driver_camera_btn = button_item(lambda: tr("Driver Camera Preview"), lambda: tr("PREVIEW"), callback=self._show_driver_camera)
    self._reg_and_training = self._left_button(lambda: tr("Regulatory"), self._on_regulatory)
    self._onroad_uploads_and_reset_settings = dual_button_item(
      left_text=lambda: tr("Upload While Driving"),
      left_callback=lambda: ui_state.params.put_bool("OnroadUploads", not ui_state.params.get_bool("OnroadUploads")),
      right_text=lambda: tr("Restore Defaults"),
      right_callback=self._reset_settings
    )

    items = [
      text_item(lambda: tr("Dongle ID"), self._params.get("DongleId") or (lambda: tr("N/A"))),
      LineSeparator(),
      text_item(lambda: tr("Serial"), self._params.get("HardwareSerial") or (lambda: tr("N/A"))),
      LineSeparator(),
      self._pair_device_btn,
      LineSeparator(),
      self._reset_calib_btn,
      LineSeparator(),
      self._driver_camera_btn,
      LineSeparator(),
      self._change_language_btn,
      LineSeparator(),
      self._reg_and_training,
      LineSeparator(),
      self._onroad_uploads_and_reset_settings,
    ]

    return items

  def _left_button(self, text, callback):
    item = dual_button_item(left_text=text, left_callback=callback, right_text="", right_callback=None)
    item.action_item.right_button.set_visible(False)
    return item

  @staticmethod
  def _reset_settings():
    def _do_reset(result: int):
      if result == DialogResult.CONFIRM:
        try:
          reset_turn_guards(ui_state.params)
        except OSError:
          cloudlog.exception("Failed to reset turn guards")
          return
        for _key in ui_state.params.all_keys():
          ui_state.params.remove(_key)
        HARDWARE.reboot()

    def _second_confirm(result: int):
      if result == DialogResult.CONFIRM:
        gui_app.set_modal_overlay(ConfirmDialog(
          text=tr("There's no undo once this runs — last chance to back out."),
          confirm_text=tr("Confirm")
        ), callback=_do_reset)

    gui_app.set_modal_overlay(ConfirmDialog(
      text=tr("Are you sure you want to reset all IQ.Pilot settings to default? Once the settings are reset, there is no going back."),
      confirm_text=tr("Reset")
    ), callback=_second_confirm)

  def _update_state(self):
    super()._update_state()
    self._onroad_uploads_and_reset_settings.action_item.left_button.set_button_style(
      ButtonStyle.PRIMARY if ui_state.params.get_bool("OnroadUploads") else ButtonStyle.NORMAL
    )

    self._driver_camera_btn.set_enabled(ui_state.is_offroad())
    self._reg_and_training.action_item.left_button.set_enabled(ui_state.is_offroad())
    self._onroad_uploads_and_reset_settings.action_item.right_button.set_enabled(ui_state.is_offroad())


if gui_app.iqpilot_ui():
  from iqpilot.system.ui.iqwidgets.widgets.list_view import button_item as button_item

_ACTIVE_BUNDLE_KEY = "ModelManager_ActiveBundle"
_DOWNLOAD_INDEX_KEY = "ModelManager_DownloadIndex"
_RUNNER_CACHE_KEY = "ModelRunnerTypeCache"
MODEL_READY_COLOR = rl.Color(16, 185, 169, 255)

def _big_model_options() -> list[tuple[str, str]]:
  try:
    from iqpilot.selfdrive.iqmodeld.emac_model_meta import big_models
    return big_models(ui_state.params)
  except Exception:
    return []


def _big_model_label(key: str) -> str:
  for name, display in _big_model_options():
    if name == key:
      return display
  return key


def _dock_present() -> bool:
  return bool(getattr(ui_state.sm["deviceState"], "egpuDockPresent", False))


def _format_download_size(size: int) -> str:
  if size <= 0:
    return ""
  return f"{size / 1e9:.1f} GB" if size >= 1e9 else f"{round(size / 1e6)} MB"


def _big_model_decoration(key: str) -> tuple[StatusPill | None, str]:
  try:
    from iqpilot.selfdrive.iqmodeld.egpu_artifact_status import egpu_artifact_status
    from iqpilot.selfdrive.iqmodeld.egpu_model import resolve_egpu_model
    meta = resolve_egpu_model(ui_state.params, key, allow_refresh=False)
    if meta is None:
      return None, ""
    status = egpu_artifact_status(meta)
  except Exception:
    return None, ""
  if status.on_device:
    return StatusPill(tr("on device"), ink.STATUS_GOOD, "check"), ""
  if status.precompiled:
    return StatusPill(tr("precompiled"), ink.STATUS_INFO, "download"), _format_download_size(status.download_bytes)
  return StatusPill(tr("compile on dock"), ink.STATUS_WARN, "chip"), ""


def _big_setup_progress() -> float | None:
  p = ui_state.params
  if p.get_bool("IQEmacEnabled"):
    raw, loading = p.get("MacModelDownloadProgress"), not p.get_bool("MacModelReady")
  else:
    raw, loading = p.get("UsbGpuSetupProgress"), p.get_bool("UsbGpuLoading") and not p.get_bool("UsbGpuCompiled")
  if not loading:
    return None
  try:
    return max(0.0, min(1.0, float(raw)))
  except (TypeError, ValueError):
    return None


def _refresh_big_catalog() -> None:
  def worker():
    try:
      from iqpilot.selfdrive.iqmodeld.emac_model_meta import refresh_catalog
      refresh_catalog(ui_state.params)
    except Exception:
      pass
  threading.Thread(target=worker, daemon=True).start()


def _request_full_model_refresh() -> None:
  ui_state.params.put("ModelManager_LastSyncTime", "0")
  ui_state.params.put_bool("ModelManager_RefreshRequest", True)
  _refresh_big_catalog()


class ModelsLayout(Widget):
  def __init__(self):
    super().__init__()
    self._private_ui_attached = False
    optional_private_ui.get()
    self.model_manager = None
    self.download_status = None
    self.prev_download_status = None
    self.model_dialog = None
    self._big_model_dialog = None
    self._drive_profile_dialog = None
    self._desired_arrival_dialog = None
    self._drive_profile_supported = False
    self._last_drive_profile_check_t = 0.0
    self.last_cache_calc_time = 0

    self._initialize_items()

    self.clear_cache_item.action_item.set_value(f"{self._calculate_cache_size():.2f} MB")
    self._scroller = Scroller(self.items, line_separator=True, spacing=0)

  def _initialize_items(self):
    self.current_model_item = IQListItem(
      title=lambda: tr("Active Model"),
      description="",
      action_item=WideButtonAction(lambda: tr("SELECT")),
      callback=self._handle_current_model_clicked
    )

    self.big_model_item = button_item(
      lambda: tr("Big Model"),
      lambda: tr("CHANGE"),
      tr("Only works with external compute connected over USB."),
      self._handle_big_model_clicked,
    )
    self.big_model_item.action_item.set_value(self._big_model_value())

    self.small_on_mac_item = toggle_item_iq(
      lambda: tr("Active Model on eMac"),
      tr("Run the selected small model on the Mac instead of a big model."),
      initial_state=ui_state.params.get_bool("IQEmacSmallModel"),
      callback=self._on_small_on_mac_toggled,
      param="IQEmacSmallModel",
    )

    self.drive_profile_item = IQListItem(
      title=lambda: tr("Drive profile"),
      description="",
      description_visible=True,
      action_item=WideButtonAction(lambda: tr(profile_label(drive_profile_name(ui_state.params)))),
      callback=self._handle_drive_profile_clicked,
    )

    self.desired_arrival_item = IQListItem(
      title=lambda: tr("Desired arrival"),
      description="",
      action_item=WideButtonAction(lambda: desired_arrival_label(ui_state.params)),
      callback=self._handle_desired_arrival_clicked,
    )

    self.supercombo_label = progress_item(tr("Combined Model"))
    self.vision_label = progress_item(tr("Vision Weights"))
    self.policy_label = progress_item(tr("Policy Weights"))

    self.clear_cache_item = IQListItem(
      title=lambda: tr("Model Cache"),
      description="",
      action_item=WideButtonAction(lambda: tr("CLEAR")),
      callback=self._clear_cache
    )

    self.cancel_download_item = button_item(tr("Stop Download"), tr("Cancel"), "", self._cancel_model_request)

    self.items = [self.current_model_item, self.big_model_item, self.small_on_mac_item,
                  self.drive_profile_item, self.desired_arrival_item,
                  self.cancel_download_item, self.supercombo_label, self.vision_label,
                  self.policy_label, self.clear_cache_item]

  def _is_downloading(self):
    return (self.model_manager and self.model_manager.selectedBundle and
            self.model_manager.selectedBundle.status == custom.IQModelManager.DownloadStatus.downloading)

  @staticmethod
  def _has_download_request() -> bool:
    try:
      return int(ui_state.params.get(_DOWNLOAD_INDEX_KEY)) >= 0
    except (TypeError, ValueError):
      return False

  @staticmethod
  def _has_active_bundle_param() -> bool:
    return bool(ui_state.params.get(_ACTIVE_BUNDLE_KEY))

  def _has_model_request(self) -> bool:
    return self._has_download_request()

  @staticmethod
  def _calculate_cache_size():
    return model_cache_size(CUSTOM_MODEL_PATH) / (1024 ** 2)

  def _clear_cache(self):
    def _callback(response):
      if response == DialogResult.CONFIRM:
        keep = set()
        active = getattr(self.model_manager, "activeBundle", None)
        for model in getattr(active, "models", []) or []:
          for artifact in (getattr(model, "metadata", None), getattr(model, "artifact", None)):
            if artifact is not None and getattr(artifact, "fileName", ""):
              keep.add(artifact.fileName)
        clear_model_cache(CUSTOM_MODEL_PATH, keep)
        self.clear_cache_item.action_item.set_value(f"{self._calculate_cache_size():.2f} MB")

    gui_app.set_modal_overlay(ConfirmDialog(tr("This will delete ALL downloaded models from the cache except the currently active model. Are you sure?"),
                                            tr("Clear Cache")), callback=_callback)

  def _redownload_target_bundle(self):
    if not self.model_manager:
      return None
    selected = self.model_manager.selectedBundle
    if selected and selected.status == custom.IQModelManager.DownloadStatus.failed:
      return selected
    active = self.model_manager.activeBundle
    if self._has_active_bundle_param() and active and active.ref:
      return active
    return None

  def _redownload_target_index(self) -> int | None:
    target = self._redownload_target_bundle()
    if not target:
      return None
    try:
      return int(target.index)
    except (TypeError, ValueError):
      pass

    for bundle in self.model_manager.availableBundles:
      if bundle.ref and bundle.ref == target.ref:
        return int(bundle.index)
      if bundle.internalName and bundle.internalName == target.internalName:
        return int(bundle.index)
    return None

  def _can_redownload(self) -> bool:
    return bool(ui_state.is_offroad() and not self._is_downloading() and not self._has_model_request() and self._redownload_target_index() is not None)

  def _cancel_model_request(self):
    ui_state.params.remove(_DOWNLOAD_INDEX_KEY)

  def _redownload_model(self):
    index = self._redownload_target_index()
    if index is None:
      return

    def _callback(response):
      if response == DialogResult.CONFIRM:
        target = self._redownload_target_bundle()
        if target is not None:
          remove_bundle_files(CUSTOM_MODEL_PATH, target)
          if bundle_matches(getattr(self.model_manager, "activeBundle", None), target):
            ui_state.params.remove(_ACTIVE_BUNDLE_KEY)
            ui_state.params.remove(_RUNNER_CACHE_KEY)
        ui_state.params.put(_DOWNLOAD_INDEX_KEY, index)

    gui_app.set_modal_overlay(ConfirmDialog(tr("Clear the selected model cache and download it again?"),
                                            tr("Redownload")), callback=_callback)

  def _handle_bundle_download_progress(self):
    labels = {custom.IQModelManager.Model.Type.supercombo: self.supercombo_label,
              custom.IQModelManager.Model.Type.vision: self.vision_label,
              custom.IQModelManager.Model.Type.policy: self.policy_label}
    for label in labels.values():
      label.set_visible(False)
    self.cancel_download_item.set_visible(False)

    if not self.model_manager or (not self.model_manager.selectedBundle and (not self._has_active_bundle_param() or not self.model_manager.activeBundle)):
      return

    bundle = self.model_manager.selectedBundle if self._is_downloading() or (
      self.model_manager.selectedBundle and self.model_manager.selectedBundle.status == custom.IQModelManager.DownloadStatus.failed
    ) else (self.model_manager.activeBundle if self._has_active_bundle_param() else None)
    if not bundle:
      return

    self.download_status = bundle.status
    status_changed = self.prev_download_status != self.download_status
    self.prev_download_status = self.download_status

    self.cancel_download_item.set_visible(bool(self.model_manager.selectedBundle) and self._has_download_request())

    if (current_time := time.monotonic()) - self.last_cache_calc_time > 0.5:
      self.last_cache_calc_time = current_time
      self.clear_cache_item.action_item.set_value(f"{self._calculate_cache_size():.2f} MB")

    if self.download_status == custom.IQModelManager.DownloadStatus.downloading:
      device._reset_interactive_timeout()

    DS = custom.IQModelManager.DownloadStatus
    bundle_downloading = bundle.status == DS.downloading
    if bundle.ref == "default" and not bundle_downloading and bundle.status != DS.failed:
      if not hasattr(self, "_default_files_ready") or current_time - getattr(self, "_default_files_checked", 0) > 1.0:
        from iqpilot.selfdrive.iqmodeld.models.helpers import bundle_files_ready
        self._default_files_ready = bundle_files_ready(bundle)
        self._default_files_checked = current_time

    for model in bundle.models:
      label = labels.get(getattr(model.type, 'raw', model.type))
      if label is None:
        continue
      label.set_visible(True)
      p = model.artifact.downloadProgress
      if bundle.ref == "default" and not bundle_downloading and bundle.status != DS.failed and getattr(self, "_default_files_ready", False):
        text, show, color, indeterminate = f"{bundle.displayName} - {tr('ready')}", False, MODEL_READY_COLOR, False
      else:
        text, show, color, indeterminate = self._model_label_state(p, bundle, bundle_downloading, status_changed)
      label.action_item.update(p.progress, text, show, color, indeterminate=indeterminate, gradient=False)

  def _model_label_state(self, p, bundle, bundle_downloading, status_changed):
    DS = custom.IQModelManager.DownloadStatus
    name = bundle.displayName
    live = p.status == DS.downloading or (bundle_downloading and p.status not in (DS.downloaded, DS.cached, DS.failed))
    if live:
      if p.progress > 0:
        return f"{int(p.progress)}% - {name}", True, rl.GRAY, False
      return f"{tr('downloading')} - {name}", False, rl.GRAY, False
    if p.status in (DS.downloaded, DS.cached):
      status_text = tr("from cache" if p.status == DS.cached else "downloaded")
      return f"{name} - {status_text if status_changed else tr('ready')}", False, MODEL_READY_COLOR, False
    if p.status == DS.failed:
      return f"download failed - {name}", False, rl.RED, False
    return f"pending - {name}", False, rl.GRAY, False

  @staticmethod
  def _show_reset_params_dialog():
    def _callback(response):
      if response == DialogResult.CONFIRM:
        ui_state.params.remove("CalibrationParams")
        ui_state.params.remove("LiveTorqueParameters")
    msg = tr("The selected model changed. We suggest resetting calibration. Would you like to do that now?")
    gui_app.set_modal_overlay(ConfirmDialog(msg, tr("Reset Calibration")), callback=_callback)

  def _on_model_selected(self, result):
    if result != DialogResult.CONFIRM:
      return
    selected_ref = self.model_dialog.selection_ref
    if selected_ref == "Default":
      active = self.model_manager.activeBundle if self.model_manager else None
      had_custom_model = bool(active and active.ref and not is_default_bundle(active))
      select_default_model(ui_state.params)
      if had_custom_model:
        self._show_reset_params_dialog()
    elif selected_bundle := next((bundle for bundle in self.model_manager.availableBundles if bundle.ref == selected_ref), None):
      ui_state.params.put(_DOWNLOAD_INDEX_KEY, selected_bundle.index)
      if self.model_manager.activeBundle and selected_bundle.generation != self.model_manager.activeBundle.generation:
        self._show_reset_params_dialog()
    self.model_dialog = None

  @staticmethod
  def _bundle_to_node(bundle):
    return PickerItem(bundle.ref, {'display_name': bundle.displayName, 'short_name': bundle.internalName})

  def _get_folders(self, _favorites=None):
    self.model_manager = ui_state.sm["iqModelManager"]
    bundles = list(self.model_manager.availableBundles or get_cached_model_bundles(ui_state.params))
    bundles.sort(key=lambda bundle: bundle.index, reverse=True)

    def is_notre_dame(bundle):
      name = " ".join((bundle.displayName or "", bundle.internalName or "", bundle.ref or ""))
      return "notre dame" in re.sub(r"[-_]", " ", name).lower()

    cutoff = next((bundle.index for bundle in bundles if is_notre_dame(bundle)), None)
    world_models = [bundle for bundle in bundles if cutoff is None or bundle.index > cutoff]
    legacy_models = [bundle for bundle in bundles if cutoff is not None and bundle.index <= cutoff]

    return [
      PickerGroup("", [PickerItem("Default", {'display_name': tr("Default (CD210)"), 'short_name': "Default"})]),
      PickerGroup(tr("World Models"), [self._bundle_to_node(bundle) for bundle in world_models]),
      PickerGroup(tr("Legacy Models"), [self._bundle_to_node(bundle) for bundle in legacy_models]),
    ]

  def _handle_current_model_clicked(self):
    _request_full_model_refresh()
    folders_list = self._get_folders()

    active_ref = self.model_manager.activeBundle.ref if self._has_active_bundle_param() and self.model_manager.activeBundle else "Default"
    self.model_dialog = PickerDialog(tr("Choose a Model"), folders_list, active_ref, "",
                                     get_folders_fn=self._get_folders, on_exit=self._on_model_selected,
                                     show_leaf_accent=False, header_action_text=lambda: tr("Redownload Model"),
                                     header_action=self._redownload_model, header_action_enabled=self._can_redownload,
                                     pin_current=False)
    gui_app.set_modal_overlay(self.model_dialog, callback=self._on_model_selected)

  def _on_small_on_mac_toggled(self, state: bool) -> None:
    p = ui_state.params
    p.put_bool("IQEmacSmallModel", bool(state))
    p.put_bool("IQEmacEnabled", True if state else bool(p.get("IQEmacModel")))
    self.big_model_item.action_item.set_value(self._big_model_value())

  @staticmethod
  def _big_model_value() -> str:
    p = ui_state.params
    if not p.get_bool("IQEmacEnabled") and not _dock_present():
      return tr("Off")
    if p.get_bool("IQEmacEnabled") and p.get_bool("IQEmacSmallModel"):
      value = tr("Active model (eMac)")
    else:
      key = p.get("IQEmacModel")
      key = key.decode() if isinstance(key, bytes) else (key or "")
      if key not in [n for n, _ in _big_model_options()]:
        return tr("Off")
      value = _big_model_label(key)
    progress = _big_setup_progress()
    return f"{value} {int(progress * 100)}%" if progress is not None and progress < 1.0 else value

  def _handle_big_model_clicked(self):
    _refresh_big_catalog()

    def folders(_favorites=None):
      items = [PickerItem("", {'display_name': tr("Off")})]
      dock = _dock_present()
      for key, display in _big_model_options():
        pill, detail = _big_model_decoration(key) if dock else (None, "")
        items.append(PickerItem(key, {'display_name': display, 'status_pill': pill, 'detail_text': detail}))
      return [PickerGroup("", items)]

    def handle_selection(result):
      if result == DialogResult.CONFIRM and self._big_model_dialog is not None:
        selected = self._big_model_dialog.selection_ref
        if not selected:
          ui_state.params.put_bool("IQEmacEnabled", ui_state.params.get_bool("IQEmacSmallModel"))
        else:
          ui_state.params.put("IQEmacModel", selected)
          ui_state.params.put_bool("IQEmacEnabled", True)
        self.big_model_item.action_item.set_value(self._big_model_value())
      self._big_model_dialog = None

    current = ui_state.params.get("IQEmacModel") if ui_state.params.get_bool("IQEmacEnabled") else ""
    current = current.decode() if isinstance(current, bytes) else (current or "")
    self._big_model_dialog = PickerDialog(tr("Big Model"), folders(), current, get_folders_fn=folders,
                                         on_exit=handle_selection, show_leaf_accent=False, pin_current=False)
    gui_app.set_modal_overlay(self._big_model_dialog, callback=handle_selection)

  def _handle_drive_profile_clicked(self):
    labels = [tr(label) for label in DRIVE_PROFILE_LABELS]
    current = tr(profile_label(drive_profile_name(ui_state.params)))
    self._drive_profile_dialog = MultiOptionDialog(tr("Drive profile"), labels, current)

    def handle_selection(result):
      if result == DialogResult.CONFIRM and self._drive_profile_dialog is not None:
        selected = self._drive_profile_dialog.selection
        if selected in labels:
          ui_state.params.put("IQDriveProfile", DRIVE_PROFILE_PARAM_VALUES[labels.index(selected)])
      self._drive_profile_dialog = None

    gui_app.set_modal_overlay(self._drive_profile_dialog, callback=handle_selection)

  def _handle_desired_arrival_clicked(self):
    choices = arrival_time_choices()
    labels = [label for label, _ in choices]
    current = desired_arrival_label(ui_state.params)
    self._desired_arrival_dialog = MultiOptionDialog(
      tr("Desired arrival"), labels, current if current in labels else labels[0]
    )

    def handle_selection(result):
      if result == DialogResult.CONFIRM and self._desired_arrival_dialog is not None:
        selected = self._desired_arrival_dialog.selection
        value = next((value for label, value in choices if label == selected), "")
        if value:
          ui_state.params.put("IQDriveDesiredArrival", value)
        else:
          ui_state.params.remove("IQDriveDesiredArrival")
      self._desired_arrival_dialog = None

    gui_app.set_modal_overlay(self._desired_arrival_dialog, callback=handle_selection)

  def _update_state(self):
    self.model_manager = ui_state.sm["iqModelManager"]
    if not self._private_ui_attached:
      self._private_ui_attached = optional_private_ui.attach(self._scroller)
    self._handle_bundle_download_progress()
    now = time.monotonic()
    if now - self._last_drive_profile_check_t > 1.0:
      self._last_drive_profile_check_t = now
      self._drive_profile_supported = active_model_declares_drive_profile(ui_state.params)
      self.big_model_item.action_item.set_value(self._big_model_value())
    self.drive_profile_item.action_item.set_enabled(self._drive_profile_supported)
    self.drive_profile_item.set_description(
      "" if self._drive_profile_supported else tr("The active model does not support drive profiles.")
    )
    self.desired_arrival_item.set_visible(drive_profile_name(ui_state.params) == "eta")
    self.desired_arrival_item.action_item.set_enabled(self._drive_profile_supported)
    self.current_model_item.action_item.set_value(tr(get_selected_model_name(ui_state.params)))
    if not ui_state.is_offroad():
      self.current_model_item.action_item.set_enabled(False)
      self.current_model_item.set_description(tr("Reachable only with the car switched off or Always Offroad turned on."))
    else:
      self.current_model_item.action_item.set_enabled(True)
      self.current_model_item.set_description("")

  def _render(self, rect):
    self._scroller.render(rect)

  def show_event(self):
    self._scroller.show_event()

OP.PANEL_COLOR = rl.Color(10, 10, 10, 255)
ICON_SIZE = 70

OP.PanelType = IntEnum(
  "PanelType",
  [es.name for es in OP.PanelType] + [
    "MODELS",
    "CRUISE",
    "STEERING",
    "VISUALS",
    "NAVIGATION",
    "VEHICLE",
  ],
  start=0,
)

@dataclass
class PanelInfo(OP.PanelInfo):
  icon: str = ""

class SidebarEntry(Widget):
  def __init__(self, hub, panel_type, panel_info):
    super().__init__()
    self._hub = hub
    self.panel_type = panel_type
    self.panel_info = panel_info

  @staticmethod
  def _draw_active_pill(rect: rl.Rectangle, left_x: float):
    pill = rl.Rectangle(left_x - 50, rect.y, OP.SIDEBAR_WIDTH - 50, OP.NAV_BTN_HEIGHT)
    rl.draw_rectangle_rounded(pill, 0.2, 5, OP.CLOSE_BTN_COLOR)

  def _render(self, rect):
    active = self.panel_type == self._hub._current_panel
    x = rect.x + 90
    if active:
      self._draw_active_pill(rect, x)

    if self.panel_info.icon:
      icon = gui_app.texture(self.panel_info.icon, ICON_SIZE, ICON_SIZE, keep_aspect_ratio=True)
      rl.draw_texture(icon, int(x), int(rect.y + (OP.NAV_BTN_HEIGHT - icon.height) / 2), rl.WHITE)
      x += ICON_SIZE + 20

    label_h = measure_text_cached(self._hub._font_medium, self.panel_info.name, 65).y
    rl.draw_text_ex(self._hub._font_medium, self.panel_info.name,
                    rl.Vector2(x, rect.y + (OP.NAV_BTN_HEIGHT - label_h) / 2), 55, 0,
                    OP.TEXT_SELECTED if active else OP.TEXT_NORMAL)

    self.panel_info.button_rect = rect

class IQSettingsLayout(OP.SettingsLayout):
  def __init__(self):
    OP.SettingsLayout.__init__(self)
    self._nav_items: list[Widget] = []
    self._sidebar_scroller = Scroller([], spacing=0, line_separator=False, pad_end=False)

    wifi_manager = WifiManager()
    wifi_manager.set_active(False)

    iq_asset = "offroad"
    self._panels = {
      OP.PanelType.DEVICE: PanelInfo(tr_noop("Device"), IQDeviceLayout(), icon=f"{iq_asset}/icon_home.png"),
      OP.PanelType.NETWORK: PanelInfo(tr_noop("Network"), IQNetworkUI(wifi_manager), icon="icons/network.png"),
      OP.PanelType.TOGGLES: PanelInfo(tr_noop("Toggles"), TogglesLayout(), icon=f"{iq_asset}/icon_toggle.png"),
      OP.PanelType.SOFTWARE: PanelInfo(tr_noop("Software"), IQSoftwareLayout(), icon=f"{iq_asset}/icon_software.png"),
      OP.PanelType.MODELS: PanelInfo(tr_noop("Models"), ModelsLayout(), icon=f"{iq_asset}/icon_models.png"),
      OP.PanelType.STEERING: PanelInfo(tr_noop("Steering"), SteeringLayout(), icon="icons_mici/wheel.png"),
      OP.PanelType.VISUALS: PanelInfo(tr_noop("Visuals"), VisualsLayout(), icon=f"{iq_asset}/icon_visuals.png"),
      OP.PanelType.VEHICLE: PanelInfo(tr_noop("Vehicle"), VehicleLayout(), icon=f"{iq_asset}/icon_vehicle.png"),
      OP.PanelType.DEVELOPER: PanelInfo(tr_noop("Developer"), IQDeveloperLayout(), icon="icons/shell.png"),
    }

  def _populate_sidebar(self, rect: rl.Rectangle):
    for panel_type, panel_info in self._panels.items():
      entry = SidebarEntry(self, panel_type, panel_info)
      entry.rect.width = rect.width - 100
      entry.rect.height = OP.NAV_BTN_HEIGHT
      self._nav_items.append(entry)
      self._sidebar_scroller.add_widget(entry)

  def _draw_close_button(self, rect: rl.Rectangle) -> rl.Rectangle:
    btn = rl.Rectangle(rect.x + metrics.GUTTER * 3, rect.y + metrics.GUTTER * 2,
                       metrics.CLOSE_BTN, metrics.CLOSE_BTN)
    pressed = (rl.is_mouse_button_down(rl.MouseButton.MOUSE_BUTTON_LEFT) and
               rl.check_collision_point_rec(rl.get_mouse_position(), btn))
    rl.draw_rectangle_rounded(btn, 1.0, 20, OP.CLOSE_BTN_PRESSED if pressed else OP.CLOSE_BTN_COLOR)

    icon = self._close_icon
    dest = rl.Rectangle(btn.x + (btn.width - icon.width) / 2, btn.y + (btn.height - icon.height) / 2,
                        icon.width, icon.height)
    tint = rl.Color(220, 220, 220, 255) if pressed else rl.WHITE
    rl.draw_texture_pro(icon, rl.Rectangle(0, 0, icon.width, icon.height), dest, rl.Vector2(0, 0), 0, tint)
    return btn

  def _draw_sidebar(self, rect: rl.Rectangle):
    rl.draw_rectangle_rec(rect, OP.SIDEBAR_COLOR)
    self._close_btn_rect = self._draw_close_button(rect)

    if not self._nav_items:
      self._populate_sidebar(rect)

    nav_rect = rl.Rectangle(rect.x, self._close_btn_rect.height + metrics.GUTTER * 4,
                            rect.width, rect.height - 300)
    self._sidebar_scroller.render(nav_rect)

  def _handle_mouse_release(self, mouse_pos: MousePos) -> bool:
    if rl.check_collision_point_rec(mouse_pos, self._close_btn_rect):
      if self._close_callback:
        self._close_callback()
      return True

    if self._sidebar_scroller.scroll_panel.is_touch_valid():
      for panel_type, panel_info in self._panels.items():
        if rl.check_collision_point_rec(mouse_pos, panel_info.button_rect):
          self.set_current_panel(panel_type)
          return True
    return False

  def show_event(self):
    super().show_event()
    self._panels[self._current_panel].instance.show_event()
    self._sidebar_scroller.show_event()

class BrandPanel:
  def __init__(self):
    self.items: list = []
  def update_settings(self) -> None:
    pass

_UNSUPPORTED_FLAGS = SubaruFlags.GLOBAL_GEN2 | SubaruFlags.HYBRID

class SubaruSettings(BrandPanel):
  def __init__(self):
    super().__init__()
    self._supported = False
    self.stop_and_go_toggle = toggle_item(tr("Creep from Standstill (Beta)"), "", param="IQSubaruCreepAssist",
                                          callback=lambda _: self.update_settings())
    self.stop_and_go_manual_parking_brake_toggle = toggle_item(
      tr("Creep from Standstill — Manual Handbrake (Beta)"), "",
      param="IQSubaruCreepAssistManualBrake", callback=lambda _: self.update_settings())
    self.items = [self.stop_and_go_toggle, self.stop_and_go_manual_parking_brake_toggle]

  def _platform_flags(self) -> int:
    if bundle := ui_state.params.get("CarPlatformBundle"):
      return SUBARU_CAR[bundle.get("platform")].config.flags
    if ui_state.CP:
      return ui_state.CP.flags
    return 0

  def _blocker_text(self) -> str:
    if not self._supported:
      return tr("Not available on this Subaru platform.")
    if not ui_state.is_offroad():
      return tr("Flip on Always Offroad from the Device panel, or power the car down, to change this.")
    return ""

  def update_settings(self):
    self._supported = not (self._platform_flags() & _UNSUPPORTED_FLAGS)
    blocker = self._blocker_text()
    usable = self._supported and ui_state.is_offroad()

    rows = (
      (self.stop_and_go_toggle,
       tr("Automatically resume from a stop while following traffic, on Subaru platforms where "
          "the beta implementation applies.")),
      (self.stop_and_go_manual_parking_brake_toggle,
       tr("Stop-and-go variant for Subaru Global cars with a manual handbrake. Leave this off on "
          "cars with an electric parking brake. Thanks to martinl for this implementation!")),
    )
    for row, body in rows:
      row.action_item.set_enabled(usable)
      row.set_description(f"<b>{blocker}</b><br><br>{body}" if blocker else body)

COOP_STEERING_MIN_KMH = 23
OEM_STEERING_MIN_KMH = 48
KM_TO_MILE = 0.621371

def _speed_text(kmh: int) -> str:
  if ui_state.is_metric:
    return f"{kmh} km/h"
  return f"{round(kmh * KM_TO_MILE)} mph"

class TeslaSettings(BrandPanel):
  def __init__(self):
    super().__init__()
    self.torque_blend_toggle = toggle_item(tr("VTB (Virtual Torque Blending)"), "", param="IQTeslaTorqueBlend")
    self.fsd_visualization_toggle = toggle_item(tr("FSD Visualization"), "", param="IQTeslaFsdVisualization")
    self.items = [self.torque_blend_toggle, self.fsd_visualization_toggle]

  def update_settings(self):
    caution = tr("Warning: steering may oscillate in turns below {}; turn this off if you feel it.").format(
      _speed_text(OEM_STEERING_MIN_KMH))
    body = (f"<b>{caution}</b><br><br>"
            f"{tr('Lets you nudge the wheel while engaged without fully disengaging steering.')}<br>"
            f"{tr('Active above {} only.').format(_speed_text(COOP_STEERING_MIN_KMH))}")

    if not ui_state.is_offroad():
      blocker = tr("Flip on Always Offroad from the Device panel, or power the car down, to change this.")
      body = f"<b>{blocker}</b><br><br>{body}"

    self.torque_blend_toggle.set_description(body)
    visualization_body = tr("Shows Tesla's FSD / Autosteer visualization while IQ.Pilot controls steering. " +
                            "Requires an offroad restart or reboot.")
    if not ui_state.is_offroad():
      visualization_body = f"<b>{blocker}</b><br><br>{visualization_body}"
    self.fsd_visualization_toggle.set_description(visualization_body)
    for row in self.items:
      row.action_item.set_enabled(ui_state.is_offroad())

class ToyotaSettings(BrandPanel):
  def __init__(self):
    super().__init__()
    self.enforce_stock_longitudinal = toggle_item(
      lambda: tr("Keep Factory Gas and Brake"),
      description=lambda: tr("Keeps gas and brakes with the factory Toyota system; IQ.Pilot steers only."),
      initial_state=ui_state.params.get_bool("IQToyotaFactoryLong"),
      callback=self._on_toggled,
      enabled=lambda: not ui_state.engaged,
    )
    self.items = [self.enforce_stock_longitudinal]

  @staticmethod
  def _apply(enabled: bool):
    ui_state.params.put_bool("IQToyotaFactoryLong", enabled)
    if enabled and ui_state.params.get_bool("AlphaLongitudinalEnabled"):
      ui_state.params.put_bool("AlphaLongitudinalEnabled", False)
    ui_state.params.put_bool("OnroadCycleRequested", True)

  def _on_toggled(self, state: bool):
    if not state:
      self._apply(False)
      return

    def after_confirm(result: int):
      if result == DialogResult.CONFIRM:
        self._apply(True)
      else:
        self.enforce_stock_longitudinal.action_item.set_state(False)

    row = self.enforce_stock_longitudinal
    prompt = f"<h1>{row.title}</h1><br><p>{row.description}</p>"
    gui_app.set_modal_overlay(ConfirmDialog(prompt, tr("Enable"), rich=True), callback=after_confirm)

DESCRIPTIONS = {
  'pqhca5or7Toggle': tr_noop(
    'Use HCA Status 7 instead of Status 5 for steering control on PQ platform vehicles. '
    'This may help with compatibility on some older Volkswagen models.'
  ),
  'AllowLateralWhenLongUnavailable': tr_noop(
    'Allow lateral control (steering) to remain active even when longitudinal control (gas/brake) '
    'is temporarily unavailable due to a cruise control fault.'
  ),
  'iqMqbAccResume': tr_noop(
    'Allow IQ.Pilot to use MQB ACC resume behavior on supported FtS with Extended Hold without Auto Resume Volkswagen MQB vehicles.'
  ),
  'iqMqbSteeringLockout': tr_noop(
    'Enable MQB steering lockout handling on Volkswagen MQB vehicles with low speed LKAS faults.'
  ),
  'EnableCurvatureController': tr_noop(
    'Close the steering loop on MEB and MQB Evo vehicles. IQ.Pilot corrects its steering command against '
    'the curvature the car is actually holding, instead of sending the planned curvature straight through.'
  ),
  'EnableSmoothSteer': tr_noop(
    'Low-pass the planned curvature before it reaches the rack on MEB and MQB Evo vehicles, which damps '
    'lateral wobble. On by default for these cars.'
  ),
}

class VolkswagenSettings(BrandPanel):
  def __init__(self):
    super().__init__()

    self.pq_hca_toggle = toggle_item(
      lambda: tr("PQ HCA Status 7 Mode"),
      description=lambda: tr(DESCRIPTIONS["pqhca5or7Toggle"]),
      initial_state=ui_state.params.get_bool("pqhca5or7Toggle"),
      callback=self._on_pq_hca_toggle,
      enabled=lambda: not ui_state.engaged,
    )

    self.lateral_when_long_unavailable = toggle_item(
      lambda: tr("Lateral Control When Cruise Faulted"),
      description=lambda: tr(DESCRIPTIONS["AllowLateralWhenLongUnavailable"]),
      initial_state=ui_state.params.get_bool("AllowLateralWhenLongUnavailable"),
      callback=self._on_lateral_when_long_unavailable,
      enabled=lambda: not ui_state.engaged,
    )

    self.mqb_acc_resume = toggle_item(
      lambda: tr("MQB ACC Resume"),
      description=lambda: tr(DESCRIPTIONS["iqMqbAccResume"]),
      initial_state=ui_state.params.get_bool("iqMqbAccResume"),
      callback=self._on_mqb_acc_resume,
      enabled=lambda: not ui_state.engaged,
    )

    self.mqb_steering_lockout = toggle_item(
      lambda: tr("MQB Steering Lockout"),
      description=lambda: tr(DESCRIPTIONS["iqMqbSteeringLockout"]),
      initial_state=ui_state.params.get_bool("iqMqbSteeringLockout"),
      callback=self._on_mqb_steering_lockout,
      enabled=lambda: not ui_state.engaged,
    )

    self.curvature_controller = toggle_item(
      lambda: tr("Curvature Controller"),
      description=lambda: tr(DESCRIPTIONS["EnableCurvatureController"]),
      initial_state=ui_state.params.get_bool("EnableCurvatureController"),
      callback=self._on_curvature_controller,
      enabled=lambda: not ui_state.engaged,
    )

    self.smooth_steer = toggle_item(
      lambda: tr("Smooth Steering"),
      description=lambda: tr(DESCRIPTIONS["EnableSmoothSteer"]),
      initial_state=ui_state.params.get_bool("EnableSmoothSteer"),
      callback=self._on_smooth_steer,
      enabled=lambda: not ui_state.engaged,
    )

    self.items = [self.pq_hca_toggle, self.lateral_when_long_unavailable, self.mqb_acc_resume, self.mqb_steering_lockout,
                  self.curvature_controller, self.smooth_steer]

  def _flags(self) -> VolkswagenFlags:
    bundle = ui_state.params.get("CarPlatformBundle")
    if bundle:
      platform = bundle.get("platform")
      if platform:
        try:
          return VOLKSWAGEN_CAR[platform].config.flags
        except (KeyError, AttributeError):
          return VolkswagenFlags(0)
    elif ui_state.CP:
      return ui_state.CP.flags
    return VolkswagenFlags(0)

  def _uses_hca_status_toggle(self) -> bool:
    return bool(self._flags() & (VolkswagenFlags.PQ | VolkswagenFlags.MLB))

  def _is_mqb(self) -> bool:
    flags = self._flags()
    return not bool(flags & (VolkswagenFlags.PQ | VolkswagenFlags.MLB | VolkswagenFlags.MEB | VolkswagenFlags.MEB_GEN2 | VolkswagenFlags.MQB_EVO))

  def _supports_lateral_when_faulted(self) -> bool:
    return not bool(self._flags() & VolkswagenFlags.MLB)

  def _is_curvature_car(self) -> bool:
    return bool(self._flags() & (VolkswagenFlags.MEB | VolkswagenFlags.MQB_EVO))

  def _on_pq_hca_toggle(self, state: bool):
    ui_state.params.put_bool("pqhca5or7Toggle", state)

  def _on_lateral_when_long_unavailable(self, state: bool):
    ui_state.params.put_bool("AllowLateralWhenLongUnavailable", state)

  def _on_mqb_acc_resume(self, state: bool):
    ui_state.params.put_bool("iqMqbAccResume", state)

  def _on_mqb_steering_lockout(self, state: bool):
    ui_state.params.put_bool("iqMqbSteeringLockout", state)

  def _on_curvature_controller(self, state: bool):
    ui_state.params.put_bool("EnableCurvatureController", state)

  def _on_smooth_steer(self, state: bool):
    ui_state.params.put_bool("EnableSmoothSteer", state)

  def update_settings(self):
    self.pq_hca_toggle.set_visible(self._uses_hca_status_toggle())
    self.lateral_when_long_unavailable.set_visible(self._supports_lateral_when_faulted())
    is_mqb = self._is_mqb()
    self.mqb_acc_resume.set_visible(is_mqb)
    self.mqb_steering_lockout.set_visible(is_mqb)
    is_curvature_car = self._is_curvature_car()
    self.curvature_controller.set_visible(is_curvature_car)
    self.smooth_steer.set_visible(is_curvature_car)

_REGISTRY: dict[str, type[BrandPanel]] = {
  "subaru": SubaruSettings,
  "tesla": TeslaSettings,
  "toyota": ToyotaSettings,
  "volkswagen": VolkswagenSettings,
}


def brand_settings_for(brand: str) -> BrandPanel | None:
  cls = _REGISTRY.get(brand)
  return cls() if cls else None


class FingerprintStatus(IntEnum):
  NONE = 0      # nothing identified, nothing forced
  AUTO = 1      # car identified itself over CAN
  FORCED = 2    # user pinned a platform manually


STATUS_COLORS = {
  FingerprintStatus.NONE: ink.STATUS_WARN,
  FingerprintStatus.AUTO: ink.STATUS_GOOD,
  FingerprintStatus.FORCED: ink.STATUS_INFO,
}


class VehicleSelection:
  def __init__(self):
    self.platforms: dict = load_catalog()

  @staticmethod
  def forced_bundle():
    return ui_state.params.get("CarPlatformBundle")

  def status(self) -> FingerprintStatus:
    if self.forced_bundle():
      return FingerprintStatus.FORCED
    if ui_state.CP and ui_state.CP.carFingerprint != "MOCK":
      return FingerprintStatus.AUTO
    return FingerprintStatus.NONE

  def display_name(self) -> str:
    if bundle := self.forced_bundle():
      return bundle.get("name", "")
    if ui_state.CP and ui_state.CP.carFingerprint != "MOCK":
      return ui_state.CP.carFingerprint
    return tr("No vehicle set")

  def force(self, platform_name: str) -> bool:
    data = self.platforms.get(platform_name)
    if not data:
      return False
    ui_state.params.put("CarPlatformBundle", {**data, "name": platform_name})
    return True

  @staticmethod
  def clear():
    ui_state.params.remove("CarPlatformBundle")

  def picker_folders(self) -> list[PickerGroup]:
    def node_for(name: str) -> PickerItem:
      info = self.platforms[name]
      years = ' '.join(map(str, info.get('year', [])))
      return PickerItem(name, {
        'display_name': name,
        'search_tags': f"{name} {info.get('make')} {years} {info.get('model', name)}",
      })

    names = sorted(self.platforms)
    makes = sorted({self.platforms[n].get('make') for n in names})
    return [PickerGroup(make, [node_for(n) for n in names if self.platforms[n].get('make') == make])
            for make in makes]


class VehiclePicker(Button):
  def __init__(self, on_platform_change: Callable[[], None] | None = None):
    super().__init__(tr("Vehicle"), self._on_clicked, button_style=ButtonStyle.NORMAL)
    self.set_rect(rl.Rectangle(0, 0, 0, 120))
    self.selection = VehicleSelection()
    self._on_platform_change = on_platform_change
    self.refresh()

  @property
  def text(self):
    return self._label._text

  @property
  def status(self) -> FingerprintStatus:
    return self.selection.status()

  @property
  def color(self) -> rl.Color:
    return STATUS_COLORS[self.status]

  def set_parent_rect(self, parent_rect):
    super().set_parent_rect(parent_rect)
    self._rect.width = parent_rect.width

  def refresh(self):
    self.set_text(self.selection.display_name())
    self.set_enabled(True)

  def _notify(self):
    self.refresh()
    if self._on_platform_change:
      self._on_platform_change()

  def _on_clicked(self):
    if self.selection.forced_bundle():
      self.selection.clear()
      self._notify()
    else:
      self._open_picker()

  def _open_picker(self):
    dialog = PickerDialog(
      tr("Pick your vehicle"),
      self.selection.picker_folders(),
      search_prompt=tr("Search make or model"),
      search_title=tr("Find your vehicle"),
      search_subtitle=tr("Type a year then a model — for example, 2021 Toyota Corolla:"),
      search_funcs=[lambda node: node.data.get('display_name', ''), lambda node: node.data.get('search_tags', '')],
    )
    done = partial(self._on_picked, dialog)
    dialog.on_exit = done
    gui_app.set_modal_overlay(dialog, callback=done)

  def _on_picked(self, dialog, res):
    if res != DialogResult.CONFIRM or not dialog.selection_ref:
      return
    when = tr("Applies right away.") if ui_state.is_offroad else \
           tr("Applies the next time the device goes offroad.")

    def confirm(result):
      if result == DialogResult.CONFIRM and self.selection.force(dialog.selection_ref):
        self._notify()

    gui_app.set_modal_overlay(ConfirmDialog(when, tr("Confirm")), callback=confirm)


_LEGEND = (
  (FingerprintStatus.AUTO, lambda: tr("Detected automatically")),
  (FingerprintStatus.FORCED, lambda: tr("Chosen manually")),
  (FingerprintStatus.NONE, lambda: tr("Undetected and unset")),
)

class LegendWidget(Widget):
  def __init__(self, platform_selector: VehiclePicker):
    super().__init__()
    self.set_rect(rl.Rectangle(0, 0, 0, 350))
    self._selector = platform_selector
    self._font = gui_app.font(FontWeight.NORMAL)
    self._bold_font = gui_app.font(FontWeight.BOLD)

  def _render(self, rect):
    x = rect.x + 20
    y = rect.y + 20
    rl.draw_text_ex(self._font, tr("Pin a vehicle to skip auto-detection."), rl.Vector2(x, y), 40, 0, ink.CAPTION)
    y += 80
    rl.draw_text_ex(self._font, tr("Colour key for fingerprint state:"), rl.Vector2(x, y), 40, 0, ink.CAPTION)
    y += 80

    active = self._selector.status
    for status, label in _LEGEND:
      font = self._bold_font if status == active else self._font
      text_color = rl.WHITE if status == active else ink.CAPTION
      text = f"- {label()}"
      ts = measure_text_cached(font, text, 40)
      chip_cy = (y - 7) + ts.y / 2
      rl.draw_rectangle_rounded(rl.Rectangle(x, chip_cy - 18, 36, 36), 0.45, 10, STATUS_COLORS[status])
      rl.draw_text_ex(font, text, rl.Vector2(x + 56, y - 7), 40, 0, text_color)
      y += 50

_STATUS_BADGES = {
  FingerprintStatus.AUTO: lambda: tr("AUTO"),
  FingerprintStatus.FORCED: lambda: tr("MANUAL"),
  FingerprintStatus.NONE: lambda: tr("NONE"),
}

class VehicleLayout(Widget):
  def __init__(self):
    super().__init__()
    self._brand_settings = None
    self._current_brand = None
    self._platform_selector = VehiclePicker(self._on_vehicle_changed)
    self._vehicle_item = IQListItem(title=self._platform_selector.text, action_item=ButtonAction(text=tr("SELECT")),
                                    callback=self._platform_selector._on_clicked)
    self._legend_widget = LegendWidget(self._platform_selector)
    self._refresh_vehicle_row()
    self._nnff_available = False
    self._nnff_status = ""
    self._nnff_vehicle_key = None
    self._nnff = toggle_item(
      title=lambda: tr("NNFF Steering"),
      description=lambda: tr("Use a vehicle-specific neural network for steering feedforward.") + "<br><br>" + self._nnff_status,
      initial_state=ui_state.params.get_bool("NeuralNetworkFeedForward"),
      callback=self._on_nnff,
    )
    self._refresh_nnff(force=True)

    self.items = [self._vehicle_item, self._legend_widget, self._nnff]
    self._scroller = Scroller(self.items, line_separator=True, spacing=0)

  @staticmethod
  def get_brand():
    bundle = ui_state.params.get("CarPlatformBundle")
    if bundle:
      return bundle.get("brand", "")
    fingerprinted = ui_state.CP and ui_state.CP.carFingerprint != "MOCK"
    return ui_state.CP.brand if fingerprinted else ""

  def _refresh_vehicle_row(self):
    status = self._platform_selector.status
    self._vehicle_item._title = self._platform_selector.text
    self._vehicle_item.title_color = rl.WHITE
    self._vehicle_item.title_badge = (_STATUS_BADGES[status](), STATUS_COLORS[status])
    forced = ui_state.params.get("CarPlatformBundle") is not None
    self._vehicle_item.action_item.set_text(tr("REMOVE") if forced else tr("SELECT"))

  def _sync_brand_panel(self):
    brand = self.get_brand()
    if brand == self._current_brand:
      return
    self._current_brand = brand
    self._brand_settings = brand_settings_for(brand)
    brand_rows = self._brand_settings.items if self._brand_settings else []
    self.items = [self._vehicle_item, self._legend_widget, self._nnff, *brand_rows]
    self._scroller = Scroller(self.items, line_separator=True, spacing=0)

  def _refresh_nnff(self, force=False):
    cp = ui_state.CP
    key = (id(cp), cp.carFingerprint if cp else None, cp.brand if cp else None, cp.steerControlType if cp else None,
           cp.lateralTuning.which() if cp else None, cp.flags if cp else None,
           ui_state.params.get("CarPlatformBundle"))
    if force or key != self._nnff_vehicle_key:
      self._nnff_vehicle_key = key
      self._nnff_available, self._nnff_status = nnff_availability(cp, ui_state.params)
    self._sync_nnff_state()

  def _sync_nnff_state(self):
    saved = ui_state.params.get_bool("NeuralNetworkFeedForward")
    self._nnff.action_item.set_state(saved)
    self._nnff.action_item.set_enabled(ui_state.is_offroad() and (self._nnff_available or saved))

  def _on_nnff(self, enabled):
    if not ui_state.is_offroad():
      self._sync_nnff_state()
      return
    if enabled:
      self._refresh_nnff(force=True)
      if not self._nnff_available:
        return
      ui_state.params.put_bool("IQHkgReducedTorqueFeedback", False)
    ui_state.params.put_bool("NeuralNetworkFeedForward", enabled)
    ui_state.params.put_bool("OnroadCycleRequested", True)
    self._sync_nnff_state()

  def _on_vehicle_changed(self):
    self._refresh_vehicle_row()
    self._sync_brand_panel()
    self._refresh_nnff()

  def _update_state(self):
    self._on_vehicle_changed()
    if self._brand_settings:
      self._brand_settings.update_settings()
    self._platform_selector.refresh()

  def _render(self, rect):
    self._scroller.render(rect)

  def show_event(self):
    self._refresh_nnff(force=True)
    self._scroller.show_event()
