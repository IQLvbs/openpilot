"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""

import re
import threading
import time

import pyray as rl

from iqpilot.cereal import custom

from iqpilot.system.ui.iqwidgets.widgets.helpers.glyphs import draw_star

from iqpilot.selfdrive.ui.ui_state import ui_state
from iqpilot.system.ui.lib.application import gui_app
from iqpilot.system.ui.widgets.scroller import NavScroller
from iqpilot.selfdrive.ui.mici.widgets.stock_button import BigButton, BigParamControl, GreyBigButton
from iqpilot.selfdrive.ui.mici.widgets.stock_dialog import BigConfirmationDialog
from iqpilot.selfdrive.ui.mici.widgets.dialog import BigMultiOptionDialog
from iqpilot.selfdrive.ui.mici.layouts.settings.iq_widgets import MappedParamToggle

from iqpilot.selfdrive.iqmodeld.drive_profile import (
  DRIVE_PROFILE_LABELS,
  DRIVE_PROFILE_PARAM_VALUES,
  active_model_declares_drive_profile,
  arrival_time_choices,
  desired_arrival_label,
  drive_profile_name,
  profile_label,
)
from iqpilot.selfdrive.iqmodeld.models.helpers import select_default_model, is_default_bundle, get_selected_model_name, get_cached_model_bundles
from iqpilot.selfdrive.iqmodeld.models.runners.model_runner import CUSTOM_MODEL_PATH
from iqpilot.selfdrive.iqmodeld.model_cache import bundle_matches, clear_model_cache, model_cache_size, remove_bundle_files
from iqpilot.system.ui.lib.multilang import tr

_DELAY_OPTIONS = [f"{v / 100:.2f}s" for v in range(5, 51)]
_DELAY_VALUES = [v / 100 for v in range(5, 51)]


_DL = custom.IQModelManager.DownloadStatus
_ACTIVE_BUNDLE_KEY = "ModelManager_ActiveBundle"
_DOWNLOAD_INDEX_KEY = "ModelManager_DownloadIndex"
_RUNNER_CACHE_KEY = "ModelRunnerTypeCache"


def _display_model_name(bundle) -> str:
  return bundle.internalName if getattr(bundle, "internalName", "") else bundle.displayName


_BIG_OPTIONS_TTL = 3.0
_big_options_cache: tuple[float, list[tuple[str, str]]] = (0.0, [])
_big_options_reload = threading.Lock()


def _load_big_options() -> list[tuple[str, str]]:
  global _big_options_cache
  try:
    from iqpilot.selfdrive.iqmodeld.emac_model_meta import big_models
    options = big_models(ui_state.params)
  except Exception:
    options = []
  _big_options_cache = (time.monotonic(), options)
  return options


def _reload_big_options_async() -> None:
  if not _big_options_reload.acquire(blocking=False):
    return

  def worker():
    try:
      _load_big_options()
    finally:
      _big_options_reload.release()
  threading.Thread(target=worker, daemon=True).start()


def _big_options(fresh: bool = False) -> list[tuple[str, str]]:
  cached_at, cached = _big_options_cache
  if fresh or not cached_at:
    return _load_big_options()
  if time.monotonic() - cached_at >= _BIG_OPTIONS_TTL:
    _reload_big_options_async()
  return cached


def _big_label(key: str) -> str:
  for name, display in _big_options():
    if name == key:
      return display
  return key or "lebrowski"


def _refresh_big_catalog() -> None:
  def worker():
    try:
      from iqpilot.selfdrive.iqmodeld.emac_model_meta import refresh_catalog
      refresh_catalog(ui_state.params)
    except Exception:
      pass
  threading.Thread(target=worker, daemon=True).start()


class _ModelSelectPanel(NavScroller):
  """A throwaway scroller panel (folder list or bundle list) pushed onto the nav stack."""
  def __init__(self, items):
    super().__init__()
    self._scroller.add_widgets(items)


class _ModelButton(BigButton):
  """A bundle in the model list: single tap selects (download), double tap toggles favorite.

  A golden star is drawn in the corner when the model is favorited.
  """
  THRESHOLD = 0.4
  _STAR_GOLD = rl.Color(0xFF, 0xC1, 0x07, 255)

  def __init__(self, bundle, on_select, on_favorite, is_favorite):
    super().__init__(bundle.displayName)
    self._bundle = bundle
    self._on_select = on_select
    self._on_favorite = on_favorite
    self._is_favorite = is_favorite
    self._pending_t = 0.0
    self._pending_pos = None

  def _handle_mouse_release(self, mouse_pos):
    now = time.monotonic()
    if self._pending_pos is not None and now - self._pending_t < self.THRESHOLD:
      self._pending_pos = None
      self._pending_t = 0.0
      self._is_favorite = self._on_favorite(self._bundle)
      return
    self._pending_t = now
    self._pending_pos = mouse_pos

  def _update_state(self):
    super()._update_state()
    if self._pending_pos is not None and time.monotonic() - self._pending_t >= self.THRESHOLD:
      self._pending_pos = None
      self._on_select(self._bundle)

  def _render(self, _):
    super()._render(_)
    if self._is_favorite:
      cx = self._rect.x + self._rect.width - 46
      cy = self._rect.y + 46
      draw_star(cx, cy, 24, True, self._STAR_GOLD)


class ModelsLayoutMici(NavScroller):
  REFRESH_PERIOD = 0.25

  def __init__(self):
    super().__init__()
    self._last_refresh_t = 0.0
    self._redownload_ready = False
    self._last_cache_t = 0.0
    self._last_drive_profile_check_t = 0.0
    self._drive_profile_supported = False
    self._download_status = None
    self._prev_download_status = None
    self._clear_icon = gui_app.texture("icons_mici/settings/developer_icon.png", 56, 56)
    self._redownload_icon = gui_app.texture("icons_mici/settings/device/update.png", 56, 56, keep_aspect_ratio=True)
    self._reset_icon = gui_app.texture("icons_mici/wheel.png", 56, 56)

    self._current = BigButton(tr("active model"))
    self._current.set_click_callback(self._show_folders)

    self._big = BigButton(tr("big model"))
    self._big.set_click_callback(self._show_big_models)

    self._small_on_mac = BigParamControl(tr("active model on eMac"), "IQEmacSmallModel", toggle_callback=self._small_on_mac_toggled)

    self._drive_profile = BigButton(tr("Drive profile"), scroll=True)
    self._drive_profile.set_click_callback(self._show_drive_profiles)

    self._desired_arrival = BigButton(tr("Desired arrival"), scroll=True)
    self._desired_arrival.set_click_callback(self._show_arrival_times)

    self._cancel = BigButton(tr("stop download"))
    self._cancel.set_click_callback(self._cancel_model_request)
    self._cancel.set_visible(self._is_downloading)

    self._redownload = BigButton(tr("redownload model"))
    self._redownload.set_click_callback(self._confirm_redownload_model)
    self._redownload.set_enabled(lambda: self._redownload_ready)

    self._supercombo = GreyBigButton(tr("combined model"))
    self._supercombo.set_visible(False)
    self._vision = GreyBigButton(tr("vision weights"))
    self._vision.set_visible(False)
    self._policy = GreyBigButton(tr("policy weights"))
    self._policy.set_visible(False)

    self._clear = BigButton(tr("purge model cache"))
    self._clear.set_click_callback(self._confirm_clear_cache)
    self._clear.set_enabled(lambda: ui_state.is_offroad())

    self._steer_delay = BigParamControl(tr("self-tuning steer delay"), "IQLiveSteerDelay")
    self._sw_delay = MappedParamToggle(tr("manual delay offset"), "IQSoftwareSteerDelay", _DELAY_OPTIONS, _DELAY_VALUES)
    self._sw_delay.set_visible(lambda: not self._steer_delay._checked)

    self._lane_turn = BigParamControl(tr("low-speed turn planning"), "IQLaneTurnDesire")

    self._main_items = [self._current, self._big, self._small_on_mac, self._drive_profile, self._desired_arrival,
                        self._cancel, self._supercombo, self._vision, self._policy,
                        self._redownload, self._clear,
                        self._steer_delay, self._sw_delay, self._lane_turn]
    self._scroller.add_widgets(self._main_items)

  @property
  def model_manager(self):
    return ui_state.sm["iqModelManager"]

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

  def _is_downloading(self) -> bool:
    try:
      return bool(self.model_manager.selectedBundle and self.model_manager.selectedBundle.status == _DL.downloading)
    except Exception:
      return False

  @staticmethod
  def _calculate_cache_size() -> float:
    return model_cache_size(CUSTOM_MODEL_PATH) / (1024 ** 2)

  def _group_folders(self, bundles):
    folders: dict = {}
    for bundle in bundles:
      folder = next((ov.value for ov in bundle.overrides if ov.key == "folder"), "")
      folders.setdefault(folder, []).append(bundle)
    return folders

  @staticmethod
  def _read_favorites() -> set:
    favs = ui_state.params.get("IQModelFavorites")
    return set(favs.split(';')) if favs else set()

  def _toggle_favorite(self, bundle) -> bool:
    favs = self._read_favorites()
    if bundle.ref in favs:
      favs.discard(bundle.ref)
    else:
      favs.add(bundle.ref)
    ui_state.params.put("IQModelFavorites", ';'.join(sorted(favs)))
    return bundle.ref in favs

  def _confirm_clear_cache(self):
    gui_app.push_widget(BigConfirmationDialog(tr("slide to\nclear cache"), self._clear_icon,
                                              self._clear_model_cache,
                                              red=True))

  def _clear_model_cache(self) -> None:
    keep = set()
    active = getattr(self.model_manager, "activeBundle", None)
    for model in getattr(active, "models", []) or []:
      for artifact in (getattr(model, "metadata", None), getattr(model, "artifact", None)):
        if artifact is not None and getattr(artifact, "fileName", ""):
          keep.add(artifact.fileName)
    clear_model_cache(CUSTOM_MODEL_PATH, keep)
    self._clear.set_value(f"{self._calculate_cache_size():.2f} MB")

  def _redownload_target_bundle(self):
    try:
      selected = self.model_manager.selectedBundle
      if selected and selected.status == _DL.failed:
        return selected
      active = self.model_manager.activeBundle
      if self._has_active_bundle_param() and active and active.ref:
        return active
    except Exception:
      pass
    return None

  def _redownload_target_index(self) -> int | None:
    target = self._redownload_target_bundle()
    if not target:
      return None

    try:
      return int(target.index)
    except (TypeError, ValueError):
      pass

    try:
      for bundle in self.model_manager.availableBundles:
        if bundle.ref and bundle.ref == target.ref:
          return int(bundle.index)
        if bundle.internalName and bundle.internalName == target.internalName:
          return int(bundle.index)
    except Exception:
      pass
    return None

  def _can_redownload(self) -> bool:
    return bool(ui_state.is_offroad() and not self._is_downloading() and not self._has_model_request() and self._redownload_target_index() is not None)

  def _cancel_model_request(self):
    ui_state.params.remove(_DOWNLOAD_INDEX_KEY)

  def _confirm_redownload_model(self):
    index = self._redownload_target_index()
    if index is None:
      return

    def _redownload():
      target = self._redownload_target_bundle()
      if target is not None:
        remove_bundle_files(CUSTOM_MODEL_PATH, target)
        if bundle_matches(getattr(self.model_manager, "activeBundle", None), target):
          ui_state.params.remove(_ACTIVE_BUNDLE_KEY)
          ui_state.params.remove(_RUNNER_CACHE_KEY)
      ui_state.params.put(_DOWNLOAD_INDEX_KEY, index)
      self._redownload.set_value(tr("queued"))

    gui_app.push_widget(BigConfirmationDialog(tr("slide to\nredownload"), self._redownload_icon, _redownload, red=True))

  def _show_folders(self):
    self._reload_model_lists()
    bundles = list(self.model_manager.availableBundles) or get_cached_model_bundles(ui_state.params)
    favorites = self._read_favorites()
    btns = []

    default_btn = BigButton(tr("Default (CD210)"))
    default_btn.set_click_callback(self._select_default)
    btns.append(default_btn)

    if favorites and (fav_bundles := [b for b in bundles if b.ref in favorites]):
      fav_btn = BigButton(tr("Favorites"), str(len(fav_bundles)))
      fav_btn.set_click_callback(lambda fb=fav_bundles: self._show_bundles(fb))
      btns.append(fav_btn)

    folders = self._group_folders(bundles)
    for folder in sorted(folders, key=lambda f: max((b.index for b in folders[f]), default=-1), reverse=True):
      name = folder if folder else "Other"
      folder_bundles = sorted(folders[folder], key=lambda b: b.index, reverse=True)
      if folder_bundles and (m := re.search(r'\(([^)]*)\)[^(]*$', folder_bundles[0].displayName)):
        name += f" ({m.group(1)})"
      btn = BigButton(name)
      btn.set_click_callback(lambda fb=folder_bundles: self._show_bundles(fb))
      btns.append(btn)

    gui_app.push_widget(_ModelSelectPanel(btns))

  def _show_bundles(self, bundles):
    favorites = self._read_favorites()
    btns = [_ModelButton(b, self._select_model, self._toggle_favorite, b.ref in favorites) for b in bundles]
    gui_app.push_widget(_ModelSelectPanel(btns))

  def _reload_model_lists(self):
    ui_state.params.put("ModelManager_LastSyncTime", 0)
    _refresh_big_catalog()

  def _show_big_models(self):
    _refresh_big_catalog()
    options = _big_options(fresh=True)
    off = BigButton(tr("Off"))
    off.set_click_callback(lambda: self._select_big(None))
    btns = [off]
    for key, display in options:
      btn = BigButton(display)
      btn.set_click_callback(lambda k=key: self._select_big(k))
      btns.append(btn)
    gui_app.push_widget(_ModelSelectPanel(btns))

  def _select_big(self, key):
    if key is None:
      ui_state.params.put_bool("IQEmacEnabled", ui_state.params.get_bool("IQEmacSmallModel"))
    else:
      ui_state.params.put("IQEmacModel", key)
      ui_state.params.put_bool("IQEmacEnabled", True)
    gui_app.pop_widgets_to(self)

  def _big_setup_progress(self) -> float | None:
    p = ui_state.params
    if p.get_bool("IQEmacEnabled"):
      raw = p.get("MacModelDownloadProgress")
      loading = not p.get_bool("MacModelReady")
    else:
      raw = p.get("UsbGpuSetupProgress")
      loading = p.get_bool("UsbGpuLoading") and not p.get_bool("UsbGpuCompiled")
    if not loading:
      return None
    try:
      return max(0.0, min(1.0, float(raw)))
    except (TypeError, ValueError):
      return None

  def _small_on_mac_toggled(self, checked: bool) -> None:
    p = ui_state.params
    if checked:
      p.put_bool("IQEmacEnabled", True)
    else:
      p.put_bool("IQEmacEnabled", bool(p.get("IQEmacModel")))

  def _small_on_mac_value(self) -> str:
    try:
      active = self.model_manager.activeBundle
      name = _display_model_name(active) if active and active.ref else ""
    except Exception:
      name = ""
    return f"{name} ({tr('eMac')})" if name else tr("active model")

  def _big_model_value(self) -> str:
    p = ui_state.params
    dock = bool(getattr(ui_state.sm["deviceState"], "egpuDockPresent", False))
    if p.get_bool("IQEmacEnabled") and p.get_bool("IQEmacSmallModel"):
      progress = self._big_setup_progress()
      value = self._small_on_mac_value()
      return f"{value} {int(progress * 100)}%" if progress is not None and progress < 1.0 else value
    if not p.get_bool("IQEmacEnabled") and not dock:
      return tr("Off")
    key = p.get("IQEmacModel")
    key = key.decode() if isinstance(key, bytes) else (key or "")
    label = _big_label(key)
    progress = self._big_setup_progress()
    if progress is not None and progress < 1.0:
      return f"{label} {int(progress * 100)}%"
    return label

  def _generation_changed(self, bundle) -> bool:
    try:
      active = self.model_manager.activeBundle
      return bool(active and active.ref and bundle.generation != active.generation)
    except Exception:
      return False

  def _select_model(self, bundle):
    ui_state.params.put(_DOWNLOAD_INDEX_KEY, bundle.index)
    cb = self._show_reset_calibration_prompt if self._generation_changed(bundle) else lambda: None
    gui_app.pop_widgets_to(self, callback=cb)

  def _select_default(self):
    try:
      active = self.model_manager.activeBundle
      had_custom_model = bool(active and active.ref and not is_default_bundle(active))
    except Exception:
      had_custom_model = False

    select_default_model(ui_state.params)
    gui_app.pop_widgets_to(self, callback=self._show_reset_calibration_prompt if had_custom_model else (lambda: None))

  def _show_reset_calibration_prompt(self):
    def _reset():
      ui_state.params.remove("CalibrationParams")
      ui_state.params.remove("LiveTorqueParameters")
    gui_app.push_widget(BigConfirmationDialog(tr("slide to\nreset calibration"), self._reset_icon, _reset))

  def _show_drive_profiles(self):
    labels = [tr(label) for label in DRIVE_PROFILE_LABELS]
    current = tr(profile_label(drive_profile_name(ui_state.params)))
    dialog = BigMultiOptionDialog(
      labels,
      current,
      right_btn="check",
      right_btn_callback=lambda: self._set_drive_profile(labels, dialog.get_selected_option()),
    )
    gui_app.set_modal_overlay(dialog)

  def _set_drive_profile(self, labels: list[str], selected: str) -> None:
    if selected in labels:
      ui_state.params.put("IQDriveProfile", DRIVE_PROFILE_PARAM_VALUES[labels.index(selected)])

  def _show_arrival_times(self):
    choices = arrival_time_choices()
    labels = [label for label, _ in choices]
    current = desired_arrival_label(ui_state.params)
    dialog = BigMultiOptionDialog(
      labels,
      current if current in labels else labels[0],
      right_btn="check",
      right_btn_callback=lambda: self._set_desired_arrival(choices, dialog.get_selected_option()),
    )
    gui_app.set_modal_overlay(dialog)

  @staticmethod
  def _set_desired_arrival(choices: list[tuple[str, str]], selected: str) -> None:
    value = next((value for label, value in choices if label == selected), "")
    if value:
      ui_state.params.put("IQDriveDesiredArrival", value)
    else:
      ui_state.params.remove("IQDriveDesiredArrival")

  def _update_state(self):
    super()._update_state()

    now = time.monotonic()
    if now - self._last_refresh_t < self.REFRESH_PERIOD:
      return
    self._last_refresh_t = now
    self._refresh_values(now)

  def _refresh_values(self, now: float):
    self._handle_bundle_download_progress()
    self._current.set_value(self._current_model_value())
    self._current.set_enabled(ui_state.is_offroad())
    self._big.set_value(self._big_model_value())
    target = self._redownload_target_bundle()
    self._redownload.set_value(_display_model_name(target) if target else "")
    self._redownload_ready = self._can_redownload()

    if now - self._last_drive_profile_check_t > 1.0:
      self._last_drive_profile_check_t = now
      self._drive_profile_supported = active_model_declares_drive_profile(ui_state.params)
    self._drive_profile.set_enabled(self._drive_profile_supported)
    self._drive_profile.set_value(
      tr(profile_label(drive_profile_name(ui_state.params))) if self._drive_profile_supported
      else tr("The active model does not support drive profiles.")
    )
    self._desired_arrival.set_visible(drive_profile_name(ui_state.params) == "eta")
    self._desired_arrival.set_enabled(self._drive_profile_supported)
    self._desired_arrival.set_value(desired_arrival_label(ui_state.params))
    if now - self._last_cache_t > 1.0:
      self._last_cache_t = now
      self._clear.set_value(f"{self._calculate_cache_size():.1f} MB")

    self._update_steer_delay_subtext()

  def _progress_target_bundle(self):
    try:
      selected = self.model_manager.selectedBundle
      active = self.model_manager.activeBundle
    except Exception:
      return None

    if selected and (selected.status == _DL.downloading or selected.status == _DL.failed):
      return selected
    return active if self._has_active_bundle_param() else None

  def _handle_bundle_download_progress(self):
    labels = {
      custom.IQModelManager.Model.Type.supercombo: self._supercombo,
      custom.IQModelManager.Model.Type.vision: self._vision,
      custom.IQModelManager.Model.Type.policy: self._policy,
    }
    for label in labels.values():
      label.set_visible(False)
      label.set_value("")

    self._cancel.set_visible(False)

    bundle = self._progress_target_bundle()
    if not bundle:
      self._download_status = None
      self._prev_download_status = None
      return

    self._download_status = bundle.status
    status_changed = self._download_status != self._prev_download_status
    self._prev_download_status = self._download_status

    self._cancel.set_visible(bool(getattr(self.model_manager, "selectedBundle", None)) and self._has_download_request())

    if self._download_status not in (_DL.downloading, _DL.failed):
      return

    if self._download_status == _DL.downloading:
      try:
        from iqpilot.selfdrive.ui.ui_state import device
        device._reset_interactive_timeout()
      except Exception:
        pass

    for model in bundle.models:
      label = labels.get(getattr(model.type, "raw", model.type))
      if label is None:
        continue
      label.set_visible(True)
      label.set_value(self._download_label_text(bundle, model, status_changed))

  def _download_label_text(self, bundle, model, status_changed: bool) -> str:
    p = model.artifact.downloadProgress
    if p.status == _DL.downloading:
      return f"{int(p.progress)}% downloading {_display_model_name(bundle)}"
    if p.status in (_DL.downloaded, _DL.cached):
      if self._download_status == _DL.downloading:
        return f"{_display_model_name(bundle)} ready"
      return f"{_display_model_name(bundle)} {'downloaded' if status_changed else 'ready'}"
    if p.status == _DL.failed:
      return f"download failed {_display_model_name(bundle)}"
    return f"pending {_display_model_name(bundle)}"

  def _current_model_value(self) -> str:
    bundle = self._progress_target_bundle()
    if not bundle:
      return self._active_model_name()

    if self._download_status == _DL.downloading:
      return self._download_progress_text(bundle)
    if self._download_status == _DL.failed:
      return f"failed: {_display_model_name(bundle)}"
    return self._active_model_name()

  def _update_steer_delay_subtext(self):
    if self._steer_delay._checked:
      try:
        measured = ui_state.measured_steer_delay()
        self._steer_delay.set_value("calibrating" if measured is None else f"measured {measured:.3f} s")
      except Exception:
        self._steer_delay.set_value("")
      return
    try:
      sw = float(ui_state.params.get("IQSoftwareSteerDelay", return_default=True))
    except (TypeError, ValueError):
      sw = 0.2
    if ui_state.CP is not None:
      self._steer_delay.set_value(f"total {ui_state.CP.steerActuatorDelay + sw:.2f} s")
    else:
      self._steer_delay.set_value(f"+{sw:.2f} s offset")

  def _active_model_name(self) -> str:
    return tr(get_selected_model_name(ui_state.params))

  def _download_progress_text(self, bundle=None) -> str:
    bundle = bundle or getattr(self.model_manager, "selectedBundle", None)
    if not bundle:
      return "downloading..."
    try:
      parts = []
      for model in bundle.models:
        p = model.artifact.downloadProgress
        if p.status == _DL.downloading:
          parts.append(f"{int(p.progress)}%")
        elif p.status in (_DL.downloaded, _DL.cached):
          parts.append("ready")
        elif p.status == _DL.failed:
          parts.append("failed")
      return f"{_display_model_name(bundle)} {' '.join(parts)}".strip() or "downloading..."
    except Exception:
      return "downloading..."

  def show_event(self):
    super().show_event()
    self._last_refresh_t = 0.0
    _reload_big_options_async()
    for w in (self._steer_delay, self._sw_delay, self._lane_turn, self._small_on_mac):
      w.refresh()
