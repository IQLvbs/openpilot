from iqpilot.cereal import log
from iqpilot.common.params import Params, UnknownKeyName
from iqpilot.common.ui_settings import LONGITUDINAL_MODE_VALUES
from iqpilot.selfdrive.longitudinal_settings import apply_longitudinal_mode
from iqpilot.system.ui.widgets import Widget
from iqpilot.system.ui.widgets.list_view import multiple_button_item, toggle_item
from iqpilot.system.ui.widgets.scroller_tici import Scroller
from iqpilot.system.ui.widgets.confirm_dialog import ConfirmDialog
from iqpilot.system.ui.lib.application import gui_app
from iqpilot.system.ui.lib.multilang import tr, tr_noop
from iqpilot.system.ui.widgets import DialogResult
from iqpilot.selfdrive.ui.ui_state import ui_state

if gui_app.iqpilot_ui():
  from iqpilot.system.ui.iqwidgets.widgets.list_view import toggle_item
  from iqpilot.system.ui.iqwidgets.widgets.list_view import multiple_button_item

PERSONALITY_TO_INT = log.LongitudinalPersonality.schema.enumerants
PERSONALITY_DISPLAY_TO_PARAM = [PERSONALITY_TO_INT["relaxed"], PERSONALITY_TO_INT["standard"], PERSONALITY_TO_INT["aggressive"]]
PERSONALITY_PARAM_TO_DISPLAY = {param: idx for idx, param in enumerate(PERSONALITY_DISPLAY_TO_PARAM)}

# Description constants
DESCRIPTIONS = {
  "OpenpilotEnabledToggle": tr_noop(
    "Use the IQ.Pilot system for adaptive cruise control and lane keep driver assistance. " +
    "Your attention is required at all times to use this feature."
  ),
  "LongitudinalPersonality": tr_noop(
    "Standard is recommended. In aggressive mode, IQ.Pilot will follow lead cars closer and be more aggressive with the gas and brake. " +
    "In relaxed mode IQ.Pilot will stay further away from lead cars. On supported cars, you can cycle through these personalities with " +
    "your steering wheel distance button."
  ),
  "IQSpeedAssistMode": tr_noop("Visual displays detected limits. Control also adjusts the set speed to follow them."),
  "IsLdwEnabled": tr_noop(
    "Receive alerts to steer back into the lane when your vehicle drifts over a detected lane line " +
    "without a turn signal activated while driving over 31 mph (50 km/h)."
  ),
  "AlwaysOnDM": tr_noop("Enable driver monitoring even when IQ.Pilot is not engaged."),
  "DashcamMode": tr_noop(
    "Keep IQ.Pilot off the harness and let the stock ADAS handle lateral and longitudinal control. " +
    "IQ.Pilot will not engage and the road view shows Dashcam Mode."
  ),
  "DashcamEnabled": tr_noop("Record and upload driving data and video. Disabling this stops all recording! No logs, no video, no audio."),
  'RecordFront': tr_noop("Upload data from the driver facing camera and help improve the driver monitoring algorithm."),
  "IsMetric": tr_noop("Display speed in km/h instead of mph."),
  "IQAutoUnits": tr_noop(
    "Set the units from the device location. Speeds switch to km/h everywhere except the United States, " +
    "the United Kingdom and Liberia, and are re-checked when you cross a border."
  ),
  "RecordAudio": tr_noop("Record and store microphone audio while driving. The audio will be included in the dashcam video in Konn3kt."),
  "LongitudinalControlMode": tr_noop(
    "Choose longitudinal behavior: IQ.Pilot (IQ longitudinal + end-to-end), " +
    "IQ.Chill (IQ longitudinal + traffic-aware cruise), or Stock ACC."
  ),
  "IQTelemetryEnabled": tr_noop(
    "Once a day, sends settings, versions, car model and coarse usage counts under a random install ID. " +
    "Never sends your dongle ID, VIN, serials, location, routes or anything you typed."
  ),
}


class TogglesLayout(Widget):
  def __init__(self):
    super().__init__()
    self._params = Params()

    # param, title, desc, icon, needs_restart
    self._toggle_defs = {
      "IsLdwEnabled": (
        lambda: tr("Enable Lane Departure Warnings"),
        DESCRIPTIONS["IsLdwEnabled"],
        "warning.png",
        False,
      ),
      "DashcamMode": (
        lambda: tr("Dashcam Mode"),
        DESCRIPTIONS["DashcamMode"],
        "monitoring.png",
        True,
      ),
      "RecordFront": (
        lambda: tr("Record and Upload Driver Camera"),
        DESCRIPTIONS["RecordFront"],
        "monitoring.png",
        True,
      ),
      "RecordAudio": (
        lambda: tr("Record and Upload Microphone Audio"),
        DESCRIPTIONS["RecordAudio"],
        "microphone.png",
        True,
      ),
      "IQGasOverrideBoost": (
        lambda: tr("Gas Override Boost"),
        tr_noop("Off by default. When the end-to-end model limits acceleration, holding the gas above 10 mph while engaged " +
                "adds up to 0.05 m/s² per press, at 0.025 m/s² per second, up to 0.2 m/s² total. " +
                "Boost fades during stronger braking, bleeds off below 10 mph, and resets when you disengage. " +
                "Takes effect on the next drive."),
        "disengage_on_accelerator.png",
        False,
      ),
      "IsMetric": (
        lambda: tr("Force Metric Units"),
        DESCRIPTIONS["IsMetric"],
        "metric.png",
        False,
      ),
      "IQTelemetryEnabled": (
        lambda: tr("Share Anonymous Usage Statistics"),
        DESCRIPTIONS["IQTelemetryEnabled"],
        "network.png",
        False,
      ),
    }

    self._long_personality_setting = multiple_button_item(
      lambda: tr("Driving Personality"),
      lambda: tr(DESCRIPTIONS["LongitudinalPersonality"]),
      buttons=[lambda: tr("Relaxed"), lambda: tr("Standard"), lambda: tr("Aggressive")],
      button_width=280,
      callback=self._set_longitudinal_personality,
      selected_index=PERSONALITY_PARAM_TO_DISPLAY.get(self._params.get("LongitudinalPersonality", return_default=True), 1),
      icon="speed_limit.png"
    )
    self._speed_limit_mode_setting = multiple_button_item(
      lambda: tr("Speed Limit"),
      lambda: tr(DESCRIPTIONS["IQSpeedAssistMode"]),
      buttons=[lambda: tr("Visual"), lambda: tr("Control")],
      button_width=280,
      callback=self._set_speed_limit_mode,
      selected_index=1 if self._params.get("IQSpeedAssistMode", return_default=True) == 3 else 0,
      icon="speed_limit.png",
    )
    self._longitudinal_control_mode_setting = multiple_button_item(
      lambda: tr("Longitudinal Control"),
      lambda: tr(DESCRIPTIONS["LongitudinalControlMode"]),
      buttons=[lambda: tr("Stock ACC"), lambda: tr("IQ.Chill"), lambda: tr("IQ.Pilot")],
      button_width=280,
      callback=self._set_longitudinal_control_mode,
      selected_index=self._get_longitudinal_control_mode_index(),
      icon="experimental_white.png",
    )

    self._toggles = {}
    self._locked_toggles = set()
    self._toggles["LongitudinalControlMode"] = self._longitudinal_control_mode_setting
    self._toggles["LongitudinalPersonality"] = self._long_personality_setting
    self._toggles["IQSpeedAssistMode"] = self._speed_limit_mode_setting

    for param, (title, desc, icon, needs_restart) in self._toggle_defs.items():
      initial_state = self._params.get_bool(param)
      toggle = toggle_item(
        title,
        desc,
        initial_state,
        callback=lambda state, p=param: self._toggle_callback(state, p),
        icon=icon,
      )

      try:
        locked = self._params.get_bool(param + "Lock")
      except UnknownKeyName:
        locked = False
      toggle.action_item.set_enabled(not locked)

      # Make description callable for live translation
      additional_desc = ""
      if needs_restart and not locked:
        additional_desc = tr("Changing this setting will restart IQ.Pilot if the car is powered on.")
      toggle.set_description(lambda og_desc=toggle.description, add_desc=additional_desc: tr(og_desc) + (" " + tr(add_desc) if add_desc else ""))

      # track for engaged state updates
      if locked:
        self._locked_toggles.add(param)

      self._toggles[param] = toggle

    self._scroller = Scroller(list(self._toggles.values()), line_separator=True, spacing=0)

    ui_state.add_engaged_transition_callback(self._update_toggles)

  def _update_state(self):
    if ui_state.sm.updated["selfdriveState"]:
      personality = PERSONALITY_TO_INT[ui_state.sm["selfdriveState"].personality]
      if personality != ui_state.personality and ui_state.started:
        self._long_personality_setting.action_item.set_selected_button(PERSONALITY_PARAM_TO_DISPLAY.get(personality, 1))
      ui_state.personality = personality
    self._speed_limit_mode_setting.action_item.set_selected_button(1 if self._params.get("IQSpeedAssistMode", return_default=True) == 3 else 0)

  def show_event(self):
    self._scroller.show_event()
    self._update_toggles()

  def _update_toggles(self):
    ui_state.update_params()

    e2e_description = tr(
      "Longitudinal Control modes:<br>" +
      "IQ.Pilot features are listed below:<br>" +
      "<h4>IQ.Pilot End-to-End Longitudinal Control</h4><br>" +
      "Let the driving model control the gas and brakes. IQ.Pilot will drive as it thinks a human would, including stopping for red lights and stop signs. " +
      "Since the driving model decides the speed to drive, the set speed will only act as an upper bound. This feature is still being improved; " +
      "mistakes should be expected.<br>" +
      "<h4>IQ.Chill</h4><br>" +
      "Uses standard traffic-aware cruise behavior for longitudinal control.<br>" +
      "<h4>New Driving Visualization</h4><br>" +
      "The driving visualization will transition to the road-facing wide-angle camera at low speeds to better show some turns. " +
      "The IQ.Pilot logo will also be shown in the top right corner."
    )

    alpha_available = bool(ui_state.CP is not None and ui_state.CP.alphaLongitudinalAvailable)
    alpha_requested = self._params.get_bool("AlphaLongitudinalEnabled")
    toyota_stock_long_forced = bool(
      ui_state.CP is not None and
      ui_state.CP.brand == "toyota" and
      self._params.get_bool("IQToyotaFactoryLong")
    )
    iq_modes_selectable = alpha_available or alpha_requested or toyota_stock_long_forced
    availability_note = ""
    if ui_state.CP is None:
      availability_note = tr("Vehicle longitudinal capability has not been detected yet. Start the car once to detect support.")
    elif toyota_stock_long_forced:
      availability_note = tr("Factory Toyota longitudinal control is currently enforced. Choose an IQ longitudinal mode to disable it.")
    elif not alpha_available:
      availability_note = tr("IQ longitudinal modes are unavailable for this vehicle. Stock ACC is the only available option.")

    self._toggles["LongitudinalControlMode"].set_visible(True)
    self._long_personality_setting.set_visible(True)

    mode_index = self._get_longitudinal_control_mode_index()
    longitudinal_control_item = self._toggles["LongitudinalControlMode"]
    longitudinal_control_item.action_item.set_selected_button(mode_index)
    longitudinal_control_item.action_item.set_enabled(not ui_state.engaged)
    longitudinal_control_item.action_item.set_enabled_buttons([True, iq_modes_selectable, iq_modes_selectable])

    description = tr(DESCRIPTIONS["LongitudinalControlMode"]) + "<br><br>" + e2e_description
    if availability_note:
      description += "<br><br><i>" + availability_note + "</i>"
    longitudinal_control_item.set_description(description)

    personality_enabled = iq_modes_selectable and mode_index != 0
    self._long_personality_setting.action_item.set_enabled(personality_enabled)

    # TODO: make a param control list item so we don't need to manage internal state as much here
    # refresh toggles from params to mirror external changes
    for param in self._toggle_defs:
      self._toggles[param].action_item.set_state(self._params.get_bool(param))

    # these toggles need restart, block while engaged
    for toggle_def in self._toggle_defs:
      if self._toggle_defs[toggle_def][3] and toggle_def not in self._locked_toggles:
        self._toggles[toggle_def].action_item.set_enabled(not ui_state.engaged)

  def _render(self, rect):
    self._scroller.render(rect)

  def _get_longitudinal_control_mode_index(self) -> int:
    if not self._params.get_bool("AlphaLongitudinalEnabled"):
      return 0  # Stock ACC
    if not self._params.get_bool("ExperimentalMode"):
      return 1  # IQ.Chill
    return 2

  def _apply_longitudinal_control_mode(self, button_index: int):
    # 0 = Stock ACC, 1 = IQ.Chill, 2 = IQ.Pilot
    previous_alpha = self._params.get_bool("AlphaLongitudinalEnabled")
    previous_toyota_stock_long = self._params.get_bool("IQToyotaFactoryLong")

    apply_longitudinal_mode(self._params, LONGITUDINAL_MODE_VALUES[button_index])

    if button_index != 0 and previous_toyota_stock_long:
      self._params.put_bool("IQToyotaFactoryLong", False)

    if previous_alpha != self._params.get_bool("AlphaLongitudinalEnabled") or previous_toyota_stock_long != self._params.get_bool("IQToyotaFactoryLong"):
      self._params.put_bool("OnroadCycleRequested", True)

  def _toggle_callback(self, state: bool, param: str):
    self._params.put_bool(param, state)
    if param == "IQAutoUnits" and state:
      self._params.remove("IQAutoUnitsRegion")
    if self._toggle_defs[param][3]:
      self._params.put_bool("OnroadCycleRequested", True)

  def _set_longitudinal_personality(self, button_index: int):
    self._params.put("LongitudinalPersonality", PERSONALITY_DISPLAY_TO_PARAM[button_index])

  def _set_speed_limit_mode(self, button_index: int):
    self._params.put("IQSpeedAssistMode", 3 if button_index else 1)

  def _set_longitudinal_control_mode(self, button_index: int):
    # 0 = Stock ACC, 1 = IQ.Chill, 2 = IQ.Pilot
    if button_index == self._get_longitudinal_control_mode_index():
      return

    if button_index == 2 and not self._params.get_bool("ExperimentalModeConfirmed"):
      def confirm_callback(result: int):
        if result == DialogResult.CONFIRM:
          self._apply_longitudinal_control_mode(button_index)
          self._params.put_bool("ExperimentalModeConfirmed", True)
        else:
          self._toggles["LongitudinalControlMode"].action_item.set_selected_button(self._get_longitudinal_control_mode_index())
        self._update_toggles()

      content = (f"<h1>{self._toggles['LongitudinalControlMode'].title}</h1><br>" +
                 f"<p>{self._toggles['LongitudinalControlMode'].description}</p>")
      dlg = ConfirmDialog(content, tr("Enable"), rich=True)
      gui_app.set_modal_overlay(dlg, callback=confirm_callback)
    else:
      self._apply_longitudinal_control_mode(button_index)
      self._update_toggles()
