#!/usr/bin/env python3
import math

from iqpilot.cereal import log, car
import iqpilot.cereal.messaging as messaging
from iqpilot.common.constants import CV
from iqpilot.common.realtime import DT_CTRL
from iqpilot.selfdrive.locationd.calibrationd import MIN_SPEED_FILTER
from iqpilot.selfdrive.selfdrived.helpers import CRITICAL_SERVICES, unresponsive_critical_services
from iqpilot.system.micd import SAMPLE_RATE, SAMPLE_BUFFER
from iqpilot.selfdrive.ui.feedback.feedbackd import FEEDBACK_MAX_DURATION
from iqpilot.system.hardware import HARDWARE

from iqpilot.common.atlas_alerts import EventBook as EventsBase, Tier as Priority, Tags as ET, AlertCard as Alert, \
  NoEntryCard as NoEntryAlert, GentleDisableCard as SoftDisableAlert, PendingDisableCard as UserSoftDisableAlert, \
  HardDisableCard as ImmediateDisableAlert, ChimeCard as EngagementAlert, BannerCard as NormalPermanentAlert, \
  BootCard as StartupAlert, AlertFactory as AlertCallbackType, car_mode_entry_alert as wrong_car_mode_alert

AlertSize = log.SelfdriveState.AlertSize
AlertStatus = log.SelfdriveState.AlertStatus
VisualAlert = car.CarControl.HUDControl.VisualAlert
AudibleAlert = car.CarControl.HUDControl.AudibleAlert
EventName = log.OnroadEvent.EventName

EVENT_NAME = {v: k for k, v in EventName.schema.enumerants.items()}

class Events(EventsBase):
  def __init__(self):
    super().__init__()
    self.event_counters = dict.fromkeys(EVENTS.keys(), 0)

  def get_events_mapping(self) -> dict[int, dict[str, Alert | AlertCallbackType]]:
    return EVENTS

  def get_event_name(self, event: int):
    return EVENT_NAME[event]

  def get_event_msg_type(self):
    return log.OnroadEvent

def get_display_speed(speed_ms: float, metric: bool) -> str:
  speed = int(round(speed_ms * (CV.MS_TO_KPH if metric else CV.MS_TO_MPH)))
  unit = 'km/h' if metric else 'mph'
  return f"{speed} {unit}"

def soft_disable_alert(alert_text_2: str) -> AlertCallbackType:
  def func(CP: car.CarParams, CS: car.CarState, sm: messaging.SubMaster, metric: bool, soft_disable_time: int, personality) -> Alert:
    if soft_disable_time < int(0.5 / DT_CTRL):
      return ImmediateDisableAlert(alert_text_2)
    return SoftDisableAlert(alert_text_2)
  return func

def user_soft_disable_alert(alert_text_2: str) -> AlertCallbackType:
  def func(CP: car.CarParams, CS: car.CarState, sm: messaging.SubMaster, metric: bool, soft_disable_time: int, personality) -> Alert:
    if soft_disable_time < int(0.5 / DT_CTRL):
      return ImmediateDisableAlert(alert_text_2)
    return UserSoftDisableAlert(alert_text_2)
  return func

def below_engage_speed_alert(CP: car.CarParams, CS: car.CarState, sm: messaging.SubMaster, metric: bool, soft_disable_time: int, personality) -> Alert:
  return NoEntryAlert(f"Drive above {get_display_speed(CP.minEnableSpeed, metric)} to engage")


def below_steer_speed_alert(CP: car.CarParams, CS: car.CarState, sm: messaging.SubMaster, metric: bool, soft_disable_time: int, personality) -> Alert:
  return Alert(
    f"Steer Assist Unavailable Below {get_display_speed(CP.minSteerSpeed, metric)}",
    "",
    AlertStatus.userPrompt, AlertSize.small,
    Priority.LOW, VisualAlert.none, AudibleAlert.prompt, 0.4)


def calibration_incomplete_alert(CP: car.CarParams, CS: car.CarState, sm: messaging.SubMaster, metric: bool, soft_disable_time: int, personality) -> Alert:
  first_word = 'Recalibrating' if sm['extrinsicsCalibration'].calStatus == log.ExtrinsicsCalibration.Status.recalibrating else 'Calibrating'
  return Alert(
    f"{first_word}: {sm['extrinsicsCalibration'].calPerc:.0f}%",
    f"Drive Above {get_display_speed(MIN_SPEED_FILTER, metric)}",
    AlertStatus.normal, AlertSize.mid,
    Priority.LOWEST, VisualAlert.none, AudibleAlert.none, .2)


def audio_feedback_alert(CP: car.CarParams, CS: car.CarState, sm: messaging.SubMaster, metric: bool, soft_disable_time: int, personality) -> Alert:
  duration = FEEDBACK_MAX_DURATION - ((sm['audioFeedback'].blockNum + 1) * SAMPLE_BUFFER / SAMPLE_RATE)
  return NormalPermanentAlert(
    "Recording Audio Feedback",
    f"{round(duration)} second{'s' if round(duration) != 1 else ''} remaining. Press again to save early.",
    priority=Priority.LOW)

def out_of_space_alert(CP: car.CarParams, CS: car.CarState, sm: messaging.SubMaster, metric: bool, soft_disable_time: int, personality) -> Alert:
  full_perc = round(100. - sm['deviceState'].freeSpacePercent)
  return NormalPermanentAlert("Out of Storage", f"{full_perc}% full")


def posenet_invalid_alert(CP: car.CarParams, CS: car.CarState, sm: messaging.SubMaster, metric: bool, soft_disable_time: int, personality) -> Alert:
  if sm.frame * DT_CTRL < 10.:
    return NoEntryAlert("IQModel is starting up", alert_text_1="Please Wait")
  mdl = sm['modelV2'].velocity.x[0] if len(sm['modelV2'].velocity.x) else math.nan
  err = CS.vEgo - mdl
  msg = f"Speed Error: {err:.1f} m/s"
  return NoEntryAlert(msg, alert_text_1="Posenet Speed Invalid")


def unresponsive_reason(sm: messaging.SubMaster) -> str | None:
  # a restarted or hung process can keep a running PID while its output is gone
  unresponsive = unresponsive_critical_services(sm)
  return f"{CRITICAL_SERVICES[unresponsive[0]]} Not Responding" if unresponsive else None


def process_not_running_alert(CP: car.CarParams, CS: car.CarState, sm: messaging.SubMaster, metric: bool, soft_disable_time: int, personality) -> Alert:
  not_running = [p.name for p in sm['managerState'].processes if not p.running and p.shouldBeRunning]
  if not not_running and (reason := unresponsive_reason(sm)):
    return NoEntryAlert(reason)
  msg = ', '.join(not_running)
  return NoEntryAlert(msg, alert_text_1="Process Not Running")


def process_not_running_disable_alert(CP: car.CarParams, CS: car.CarState, sm: messaging.SubMaster, metric: bool, soft_disable_time: int, personality) -> Alert:
  reason = unresponsive_reason(sm) or "Process Not Running"
  return soft_disable_alert(reason)(CP, CS, sm, metric, soft_disable_time, personality)


def camera_malfunction_alert(CP: car.CarParams, CS: car.CarState, sm: messaging.SubMaster, metric: bool, soft_disable_time: int, personality) -> Alert:
  all_cams = ('roadCameraState', 'driverCameraState', 'wideRoadCameraState')
  bad_cams = [s.replace('State', '') for s in all_cams if s in sm.data.keys() and not sm.all_checks([s, ])]
  return NormalPermanentAlert("Camera Malfunction", ', '.join(bad_cams))


def calibration_invalid_alert(CP: car.CarParams, CS: car.CarState, sm: messaging.SubMaster, metric: bool, soft_disable_time: int, personality) -> Alert:
  rpy = sm['extrinsicsCalibration'].rpyCalib
  yaw = math.degrees(rpy[2] if len(rpy) == 3 else math.nan)
  pitch = math.degrees(rpy[1] if len(rpy) == 3 else math.nan)
  angles = f"Remount Device (Pitch: {pitch:.1f}°, Yaw: {yaw:.1f}°)"
  return NormalPermanentAlert("Calibration Invalid", angles)


def paramsd_invalid_alert(CP: car.CarParams, CS: car.CarState, sm: messaging.SubMaster, metric: bool, soft_disable_time: int, personality) -> Alert:
  if not sm['vehicleParameters'].angleOffsetValid:
    angle_offset_deg = sm['vehicleParameters'].angleOffsetDeg
    title = "Steering misalignment detected"
    text = f"Angle offset too high (Offset: {angle_offset_deg:.1f}°)"
  elif not sm['vehicleParameters'].steerRatioValid:
    steer_ratio = sm['vehicleParameters'].steerRatio
    title = "Steer ratio mismatch"
    text = f"Steering rack geometry may be off (Ratio: {steer_ratio:.1f})"
  elif not sm['vehicleParameters'].stiffnessFactorValid:
    stiffness_factor = sm['vehicleParameters'].stiffnessFactor
    title = "Abnormal tire stiffness"
    text = f"Check tires, pressure, or alignment (Factor: {stiffness_factor:.1f})"
  else:
    return NoEntryAlert("paramsd Temporary Error")

  return NoEntryAlert(alert_text_1=title, alert_text_2=text)

def overheat_alert(CP: car.CarParams, CS: car.CarState, sm: messaging.SubMaster, metric: bool, soft_disable_time: int, personality) -> Alert:
  cpu = max(sm['deviceState'].cpuTempC, default=0.)
  gpu = max(sm['deviceState'].gpuTempC, default=0.)
  temp = max((cpu, gpu, sm['deviceState'].memoryTempC))
  return NormalPermanentAlert("System Overheated", f"{temp:.0f} °C")


def low_memory_alert(CP: car.CarParams, CS: car.CarState, sm: messaging.SubMaster, metric: bool, soft_disable_time: int, personality) -> Alert:
  return NormalPermanentAlert("Low Memory", f"{sm['deviceState'].memoryUsagePercent}% used")


def modeld_lagging_alert(CP: car.CarParams, CS: car.CarState, sm: messaging.SubMaster, metric: bool, soft_disable_time: int, personality) -> Alert:
  return NormalPermanentAlert("Driving Model Lagging", f"{sm['modelV2'].frameDropPerc:.1f}% frames dropped")


def joystick_alert(CP: car.CarParams, CS: car.CarState, sm: messaging.SubMaster, metric: bool, soft_disable_time: int, personality) -> Alert:
  gb = sm['carControl'].actuators.accel / 4.
  if CP.steerControlType in (car.CarParams.SteerControlType.angle, car.CarParams.SteerControlType.curvatureDEPRECATED):
    steer = sm['carControl'].actuators.steeringAngleDeg
    vals = f"Gas: {round(gb * 100.)}%, Angle: {round(steer, 1)}°"
  else:
    steer = sm['carControl'].actuators.torque
    vals = f"Gas: {round(gb * 100.)}%, Steer: {round(steer * 100.)}%"
  return NormalPermanentAlert("Joystick Mode", vals)


def longitudinal_maneuver_alert(CP: car.CarParams, CS: car.CarState, sm: messaging.SubMaster, metric: bool, soft_disable_time: int, personality) -> Alert:
  ad = sm['alertDebug']
  audible_alert = AudibleAlert.prompt if 'Active' in ad.alertText1 else AudibleAlert.none
  alert_status = AlertStatus.userPrompt if 'Active' in ad.alertText1 else AlertStatus.normal
  alert_size = AlertSize.mid if ad.alertText2 else AlertSize.small
  return Alert(ad.alertText1, ad.alertText2,
               alert_status, alert_size,
               Priority.LOW, VisualAlert.none, audible_alert, 0.2)


def personality_changed_alert(CP: car.CarParams, CS: car.CarState, sm: messaging.SubMaster, metric: bool, soft_disable_time: int, personality) -> Alert:
  personality = str(personality).title()
  return NormalPermanentAlert(f"Driving Personality: {personality}", duration=1.5)


def invalid_lkas_setting_alert(CP: car.CarParams, CS: car.CarState, sm: messaging.SubMaster, metric: bool, soft_disable_time: int, personality) -> Alert:
  title = "Invalid LKAS setting"
  text = "Toggle stock LKAS on or off to engage"
  if CP.brand == "tesla":
    title = "Autosteer / FSD Active"
    text = "IQ.Pilot is standing by"
  elif CP.brand == "mazda":
    text = "Enable your car's LKAS to engage"
  elif CP.brand == "nissan":
    text = "Disable your car's stock LKAS to engage"
  return NormalPermanentAlert(title, text)


def invalid_lkas_setting_no_entry_alert(CP: car.CarParams, CS: car.CarState, sm: messaging.SubMaster,
                                        metric: bool, soft_disable_time: int, personality) -> Alert:
  if CP.brand == "tesla":
    return NoEntryAlert("IQ.Pilot is standing by", alert_text_1="Autosteer / FSD Active")
  return NoEntryAlert("Invalid LKAS setting")



EVENTS: dict[int, dict[str, Alert | AlertCallbackType]] = {
  EventName.stockFcw: {},
  EventName.actuatorsApiUnavailable: {},

  EventName.joystickDebug: {
    ET.WARNING: joystick_alert,
    ET.PERMANENT: NormalPermanentAlert("Joystick Mode"),
  },

  EventName.longitudinalManeuver: {
    ET.WARNING: longitudinal_maneuver_alert,
    ET.PERMANENT: NormalPermanentAlert("Longitudinal Maneuver Mode",
                                       "Ensure road ahead is clear"),
  },

  EventName.bigModelLoading: {
    ET.NO_ENTRY: NoEntryAlert("Big Model Loading"),
  },

  EventName.bigModelFailed: {
    ET.SOFT_DISABLE: soft_disable_alert("Big Model Failed"),
    ET.PERMANENT: NormalPermanentAlert("Big Model Failed ", "Restart the car to retry,\nsmall model is still available", duration=20.),
  },

  EventName.lateralManeuver: {
    ET.WARNING: longitudinal_maneuver_alert,
    ET.PERMANENT: NormalPermanentAlert("Lateral Maneuver Mode"),
  },

  EventName.selfdriveInitializing: {
    ET.NO_ENTRY: NoEntryAlert("IQ.Pilot Initializing"),
  },

  EventName.startup: {
    ET.PERMANENT: StartupAlert("Welcome to IQ.Pilot!")
  },

  EventName.startupMaster: {
    ET.PERMANENT: StartupAlert("Welcome to IQ.Pilot!"),
  },

  EventName.startupNoControl: {
    ET.PERMANENT: StartupAlert("Dashcam mode"),
    ET.NO_ENTRY: NoEntryAlert("Dashcam mode"),
  },

  EventName.startupNoCar: {
    ET.PERMANENT: StartupAlert("IQ.Pilot Dashcam mode: car unrecognized"),
  },

  EventName.startupNoSecOcKey: {
    ET.PERMANENT: NormalPermanentAlert("Dashcam Mode",
                                       "TSK Security Key Not Available",
                                       priority=Priority.HIGH),
  },

  EventName.dashcamMode: {
    ET.PERMANENT: NormalPermanentAlert("Dashcam Mode",
                                       priority=Priority.LOWEST),
  },

  EventName.invalidLkasSetting: {
    ET.PERMANENT: invalid_lkas_setting_alert,
    ET.NO_ENTRY: invalid_lkas_setting_no_entry_alert,
  },

  EventName.cruiseMismatch: {
    #ET.PERMANENT: ImmediateDisableAlert("openpilot failed to cancel cruise"),
  },

  EventName.carUnrecognized: {
    ET.PERMANENT: NormalPermanentAlert("Dashcam Mode",
                                       "Car Unrecognized",
                                       priority=Priority.LOWEST),
  },

  EventName.aeb: {
    ET.PERMANENT: Alert(
      "BRAKE!",
      "Emergency Braking: Risk of Collision",
      AlertStatus.critical, AlertSize.full,
      Priority.HIGHEST, VisualAlert.fcw, AudibleAlert.none, 2.),
    ET.NO_ENTRY: NoEntryAlert("AEB: Risk of Collision"),
  },

  EventName.stockAeb: {
    ET.PERMANENT: Alert(
      "BRAKE!",
      "Stock AEB: Risk of Collision",
      AlertStatus.critical, AlertSize.full,
      Priority.HIGHEST, VisualAlert.fcw, AudibleAlert.none, 2.),
    ET.NO_ENTRY: NoEntryAlert("Stock AEB: Risk of Collision"),
  },

  EventName.stockLkas: {
    ET.PERMANENT: Alert(
      "Stock LKAS: Lane Departure Detected",
      "",
      AlertStatus.userPrompt, AlertSize.small,
      Priority.LOW, VisualAlert.ldw, AudibleAlert.prompt, 3.),
    ET.NO_ENTRY: NoEntryAlert("Stock LKAS: Lane Departure Detected"),
  },

  EventName.fcw: {
    ET.PERMANENT: Alert(
      "BRAKE!",
      "Risk of Collision",
      AlertStatus.critical, AlertSize.full,
      Priority.HIGHEST, VisualAlert.fcw, AudibleAlert.warningSoft, 2.),
  },

  EventName.ldw: {
    ET.PERMANENT: Alert(
      "Lane Departure Detected",
      "",
      AlertStatus.userPrompt, AlertSize.small,
      Priority.LOW, VisualAlert.ldw, AudibleAlert.prompt, 3.),
  },
  EventName.steerTempUnavailableSilent: {
    ET.WARNING: Alert(
      "Steering Assist Temporarily Unavailable",
      "",
      AlertStatus.userPrompt, AlertSize.small,
      Priority.LOW, VisualAlert.steerRequired, AudibleAlert.prompt, 1.8),
  },

  EventName.preDriverDistracted: {
    ET.PERMANENT: Alert(
      "Pay Attention",
      "",
      AlertStatus.normal, AlertSize.small,
      Priority.MID, VisualAlert.none, AudibleAlert.none, .1),
  },

  EventName.promptDriverDistracted: {
    ET.PERMANENT: Alert(
      "Pay Attention",
      "Driver Distracted",
      AlertStatus.userPrompt, AlertSize.mid,
      Priority.MID, VisualAlert.steerRequired, AudibleAlert.promptDistracted, .1),
  },

  EventName.driverDistracted: {
    ET.PERMANENT: Alert(
      "DISENGAGE IMMEDIATELY",
      "Driver Distracted",
      AlertStatus.critical, AlertSize.full,
      Priority.HIGH, VisualAlert.steerRequired, AudibleAlert.warningImmediate, .1),
  },

  EventName.preDriverUnresponsive: {
    ET.PERMANENT: Alert(
      "Touch Steering Wheel: No Face Detected",
      "",
      AlertStatus.normal, AlertSize.small,
      Priority.LOW, VisualAlert.steerRequired, AudibleAlert.none, .1),
  },

  EventName.promptDriverUnresponsive: {
    ET.PERMANENT: Alert(
      "Touch Steering Wheel",
      "Driver Unresponsive",
      AlertStatus.userPrompt, AlertSize.mid,
      Priority.MID, VisualAlert.steerRequired, AudibleAlert.promptDistracted, .1),
  },

  EventName.driverUnresponsive: {
    ET.PERMANENT: Alert(
      "DISENGAGE IMMEDIATELY",
      "Driver Unresponsive",
      AlertStatus.critical, AlertSize.full,
      Priority.HIGH, VisualAlert.steerRequired, AudibleAlert.warningImmediate, .1),
  },

  EventName.manualRestart: {
    ET.WARNING: Alert(
      "TAKE CONTROL",
      "Resume Driving Manually",
      AlertStatus.userPrompt, AlertSize.mid,
      Priority.LOW, VisualAlert.none, AudibleAlert.none, .2),
  },

  EventName.resumeRequired: {
    ET.WARNING: Alert(
      "Press Resume to Exit Standstill",
      "",
      AlertStatus.userPrompt, AlertSize.small,
      Priority.LOW, VisualAlert.none, AudibleAlert.none, .2),
  },

  EventName.belowSteerSpeed: {
    ET.WARNING: below_steer_speed_alert,
  },

  EventName.preLaneChangeLeft: {
    ET.WARNING: Alert(
      "Steer Left to Start Lane Change Once Safe",
      "",
      AlertStatus.normal, AlertSize.small,
      Priority.LOW, VisualAlert.none, AudibleAlert.none, .1),
  },

  EventName.preLaneChangeRight: {
    ET.WARNING: Alert(
      "Steer Right to Start Lane Change Once Safe",
      "",
      AlertStatus.normal, AlertSize.small,
      Priority.LOW, VisualAlert.none, AudibleAlert.none, .1),
  },

  EventName.laneChangeBlocked: {
    ET.WARNING: Alert(
      "Car Detected in Blindspot",
      "",
      AlertStatus.userPrompt, AlertSize.small,
      Priority.LOW, VisualAlert.none, AudibleAlert.prompt, .1),
  },

  EventName.laneChange: {
    ET.WARNING: Alert(
      "Changing Lanes",
      "",
      AlertStatus.normal, AlertSize.small,
      Priority.LOW, VisualAlert.none, AudibleAlert.none, .1),
  },

  EventName.steerSaturated: {
    ET.WARNING: Alert(
      "Take Control",
      "Turn Exceeds Steering Limit",
      AlertStatus.userPrompt, AlertSize.mid,
      Priority.LOW, VisualAlert.steerRequired, AudibleAlert.promptRepeat, 2.),
  },

  EventName.fanMalfunction: {
    ET.PERMANENT: NormalPermanentAlert("Fan Malfunction", "Likely Hardware Issue"),
  },

  EventName.cameraMalfunction: {
    ET.PERMANENT: camera_malfunction_alert,
    ET.SOFT_DISABLE: soft_disable_alert("Camera Malfunction"),
    ET.NO_ENTRY: NoEntryAlert("Camera Malfunction: Reboot Your Device"),
  },
  EventName.cameraFrameRate: {
    ET.PERMANENT: NormalPermanentAlert("Camera Frame Rate Low", "Reboot your Device"),
    ET.SOFT_DISABLE: soft_disable_alert("Camera Frame Rate Low"),
    ET.NO_ENTRY: NoEntryAlert("Camera Frame Rate Low: Reboot Your Device"),
  },
  EventName.locationdTemporaryError: {
    ET.NO_ENTRY: NoEntryAlert("locationd Temporary Error"),
    ET.SOFT_DISABLE: soft_disable_alert("locationd Temporary Error"),
  },

  EventName.locationdPermanentError: {
    ET.NO_ENTRY: NoEntryAlert("locationd Permanent Error"),
    ET.IMMEDIATE_DISABLE: ImmediateDisableAlert("locationd Permanent Error"),
    ET.PERMANENT: NormalPermanentAlert("locationd Permanent Error"),
  },

  EventName.paramsdTemporaryError: {
    ET.NO_ENTRY: paramsd_invalid_alert,
    ET.SOFT_DISABLE: soft_disable_alert("paramsd Temporary Error"),
  },

  EventName.paramsdPermanentError: {
    ET.NO_ENTRY: NoEntryAlert("paramsd Permanent Error"),
    ET.IMMEDIATE_DISABLE: ImmediateDisableAlert("paramsd Permanent Error"),
    ET.PERMANENT: NormalPermanentAlert("paramsd Permanent Error"),
  },
  EventName.pcmEnable: {
    ET.ENABLE: EngagementAlert(AudibleAlert.engage),
  },

  EventName.buttonEnable: {
    ET.ENABLE: EngagementAlert(AudibleAlert.engage),
  },

  EventName.pcmDisable: {
    ET.USER_DISABLE: EngagementAlert(AudibleAlert.disengage),
  },

  EventName.buttonCancel: {
    ET.USER_DISABLE: EngagementAlert(AudibleAlert.disengage),
    ET.NO_ENTRY: NoEntryAlert("Cancel Pressed"),
  },

  EventName.brakeHold: {
    ET.WARNING: Alert(
      "Press Resume to Exit Brake Hold",
      "",
      AlertStatus.userPrompt, AlertSize.small,
      Priority.LOW, VisualAlert.none, AudibleAlert.none, .2),
  },

  EventName.parkBrake: {
    ET.USER_DISABLE: EngagementAlert(AudibleAlert.disengage),
    ET.NO_ENTRY: NoEntryAlert("Parking Brake Engaged"),
  },

  EventName.pedalPressed: {
    ET.USER_DISABLE: EngagementAlert(AudibleAlert.disengage),
    ET.NO_ENTRY: NoEntryAlert("Pedal Pressed",
                              visual_alert=VisualAlert.brakePressed),
  },

  EventName.steerDisengage: {
    ET.USER_DISABLE: EngagementAlert(AudibleAlert.disengage),
    ET.NO_ENTRY: NoEntryAlert("Steering Pressed"),
  },

  EventName.preEnableStandstill: {
    ET.PRE_ENABLE: Alert(
      "Release Brake to Engage",
      "",
      AlertStatus.normal, AlertSize.small,
      Priority.LOWEST, VisualAlert.none, AudibleAlert.none, .1, creation_delay=1.),
  },

  EventName.gasPressedOverride: {
    ET.OVERRIDE_LONGITUDINAL: Alert(
      "",
      "",
      AlertStatus.normal, AlertSize.none,
      Priority.LOWEST, VisualAlert.none, AudibleAlert.none, .1),
  },

  EventName.steerOverride: {
    ET.OVERRIDE_LATERAL: Alert(
      "",
      "",
      AlertStatus.normal, AlertSize.none,
      Priority.LOWEST, VisualAlert.none, AudibleAlert.none, .1),
  },

  EventName.wrongCarMode: {
    ET.USER_DISABLE: EngagementAlert(AudibleAlert.disengage),
    ET.NO_ENTRY: wrong_car_mode_alert,
  },

  EventName.resumeBlocked: {
    ET.NO_ENTRY: NoEntryAlert("Press Set to Engage"),
  },

  EventName.wrongCruiseMode: {
    ET.USER_DISABLE: EngagementAlert(AudibleAlert.disengage),
    ET.NO_ENTRY: NoEntryAlert("Adaptive Cruise Disabled"),
  },

  EventName.steerTempUnavailable: {
    ET.SOFT_DISABLE: soft_disable_alert("Steering Assist Temporarily Unavailable"),
    ET.NO_ENTRY: NoEntryAlert("Steering Temporarily Unavailable"),
  },

  EventName.steerTimeLimit: {
    ET.SOFT_DISABLE: soft_disable_alert("Vehicle Steering Time Limit"),
    ET.NO_ENTRY: NoEntryAlert("Vehicle Steering Time Limit"),
  },

  EventName.outOfSpace: {
    ET.PERMANENT: out_of_space_alert,
    ET.NO_ENTRY: NoEntryAlert("Out of Storage"),
  },

  EventName.belowEngageSpeed: {
    ET.NO_ENTRY: below_engage_speed_alert,
  },

  EventName.sensorDataInvalid: {
    ET.PERMANENT: Alert(
      "Sensor Data Invalid",
      "Possible Hardware Issue",
      AlertStatus.normal, AlertSize.mid,
      Priority.LOWER, VisualAlert.none, AudibleAlert.none, .2, creation_delay=1.),
    ET.NO_ENTRY: NoEntryAlert("Sensor Data Invalid"),
    ET.SOFT_DISABLE: soft_disable_alert("Sensor Data Invalid"),
  },

  EventName.noGps: {
  },

  EventName.tooDistracted: {
    ET.NO_ENTRY: NoEntryAlert("Distraction Level Too High"),
  },

  EventName.excessiveActuation: {
    ET.SOFT_DISABLE: soft_disable_alert("Excessive Actuation"),
    ET.NO_ENTRY: NoEntryAlert("Excessive Actuation"),
  },

  EventName.overheat: {
    ET.PERMANENT: overheat_alert,
    ET.SOFT_DISABLE: soft_disable_alert("System Overheated"),
    ET.NO_ENTRY: NoEntryAlert("System Overheated"),
  },

  EventName.wrongGear: {
    ET.SOFT_DISABLE: Alert(
      "",
      "",
      AlertStatus.normal, AlertSize.none,
      Priority.LOWEST, VisualAlert.none, AudibleAlert.none, 0.),
    ET.NO_ENTRY: Alert(
      "",
      "",
      AlertStatus.normal, AlertSize.none,
      Priority.LOWEST, VisualAlert.none, AudibleAlert.none, 0.),
  },

  EventName.calibrationInvalid: {
    ET.PERMANENT: calibration_invalid_alert,
    ET.SOFT_DISABLE: soft_disable_alert("Calibration Invalid: Remount Device & Recalibrate"),
    ET.NO_ENTRY: NoEntryAlert("Calibration Invalid: Remount Device & Recalibrate"),
  },

  EventName.calibrationIncomplete: {
    ET.PERMANENT: calibration_incomplete_alert,
    ET.SOFT_DISABLE: soft_disable_alert("Calibration Incomplete"),
    ET.NO_ENTRY: NoEntryAlert("Calibration in Progress"),
  },

  EventName.calibrationRecalibrating: {
    ET.PERMANENT: calibration_incomplete_alert,
    ET.SOFT_DISABLE: soft_disable_alert("Device Remount Detected: Recalibrating"),
    ET.NO_ENTRY: NoEntryAlert("Remount Detected: Recalibrating"),
  },

  EventName.doorOpen: {
    ET.SOFT_DISABLE: user_soft_disable_alert("Door Open"),
    ET.NO_ENTRY: NoEntryAlert("Door Open"),
  },

  EventName.seatbeltNotLatched: {
    ET.SOFT_DISABLE: user_soft_disable_alert("Seatbelt Unlatched"),
    ET.NO_ENTRY: NoEntryAlert("Seatbelt Unlatched"),
  },

  EventName.espDisabled: {
    ET.SOFT_DISABLE: soft_disable_alert("Electronic Stability Control Disabled"),
    ET.NO_ENTRY: NoEntryAlert("Electronic Stability Control Disabled"),
  },

  EventName.lowBattery: {
    ET.SOFT_DISABLE: soft_disable_alert("Low Battery"),
    ET.NO_ENTRY: NoEntryAlert("Low Battery"),
  },

  EventName.commIssue: {
  },
  EventName.commIssueAvgFreq: {
  },

  EventName.selfdrivedLagging: {
  },

  EventName.processNotRunning: {
    ET.NO_ENTRY: process_not_running_alert,
    ET.SOFT_DISABLE: process_not_running_disable_alert,
  },

  EventName.radarFault: {
    ET.SOFT_DISABLE: soft_disable_alert("Radar Error: Restart the Car"),
    ET.NO_ENTRY: NoEntryAlert("Radar Error: Restart the Car"),
  },

  EventName.radarTempUnavailable: {
    ET.SOFT_DISABLE: soft_disable_alert("Radar Temporarily Unavailable"),
    ET.NO_ENTRY: NoEntryAlert("Radar Temporarily Unavailable"),
  },

  EventName.modeldLagging: {
    ET.SOFT_DISABLE: soft_disable_alert("Driving Model Lagging"),
    ET.NO_ENTRY: NoEntryAlert("Driving Model Lagging"),
    ET.PERMANENT: modeld_lagging_alert,
  },

  EventName.posenetInvalid: {
    ET.SOFT_DISABLE: soft_disable_alert("Posenet Speed Invalid"),
    ET.NO_ENTRY: posenet_invalid_alert,
  },

  EventName.deviceFalling: {
    ET.SOFT_DISABLE: soft_disable_alert("Device Fell Off Mount"),
    ET.NO_ENTRY: NoEntryAlert("Device Fell Off Mount"),
  },

  EventName.lowMemory: {
    ET.SOFT_DISABLE: soft_disable_alert("Low Memory: Reboot Your Device"),
    ET.PERMANENT: low_memory_alert,
    ET.NO_ENTRY: NoEntryAlert("Low Memory: Reboot Your Device"),
  },

  EventName.accFaulted: {
    ET.IMMEDIATE_DISABLE: ImmediateDisableAlert("Cruise Fault: Restart the Car"),
    ET.PERMANENT: NormalPermanentAlert("Cruise Fault: Restart the car to engage"),
    ET.NO_ENTRY: NoEntryAlert("Cruise Fault: Restart the Car"),
  },

  EventName.accFaultedHardDecel: {
    ET.IMMEDIATE_DISABLE: ImmediateDisableAlert("ACC fault from hard braking"),
    ET.PERMANENT: NormalPermanentAlert("IQ.Pilot Temporarily Unavailable", "Temporary ACC fault caused by hard braking"),
    ET.NO_ENTRY: NoEntryAlert("Temporary ACC fault caused by hard braking"),
  },

  EventName.accFaultedHardDecelLateral: {
    ET.PERMANENT: NormalPermanentAlert("IQ.Pilot Temporarily Unavailable", "Temporary ACC fault caused by hard braking", duration=5.0, priority=Priority.LOW),
  },

  EventName.cruiseFaultLateralAllowed: {
    ET.PERMANENT: NormalPermanentAlert("Cruise Faulted", "Lane Assist remains available", duration=5.0, priority=Priority.LOW),
  },

  EventName.espActive: {
    ET.SOFT_DISABLE: soft_disable_alert("Electronic Stability Control Active"),
    ET.NO_ENTRY: NoEntryAlert("Electronic Stability Control Active"),
  },

  EventName.controlsMismatch: {
    ET.IMMEDIATE_DISABLE: ImmediateDisableAlert("Controls Mismatch"),
    ET.NO_ENTRY: NoEntryAlert("Controls Mismatch"),
  },

  EventName.usbError: {
    ET.SOFT_DISABLE: soft_disable_alert("USB Error: Reboot Your Device"),
    ET.PERMANENT: NormalPermanentAlert("USB Error: Reboot Your Device"),
    ET.NO_ENTRY: NoEntryAlert("USB Error: Reboot Your Device"),
  },

  EventName.canError: {
    ET.PERMANENT: Alert(
      "CAN Error",
      "",
      AlertStatus.normal, AlertSize.small,
      Priority.LOW, VisualAlert.none, AudibleAlert.none, 1., creation_delay=1.),
  },

  EventName.canBusMissing: {
    ET.IMMEDIATE_DISABLE: ImmediateDisableAlert("CAN Bus Disconnected"),
    ET.PERMANENT: Alert(
      "CAN Bus Disconnected: Likely Faulty Cable",
      "",
      AlertStatus.normal, AlertSize.small,
      Priority.LOW, VisualAlert.none, AudibleAlert.none, 1., creation_delay=1.),
    ET.NO_ENTRY: NoEntryAlert("CAN Bus Disconnected: Check Connections"),
  },

  EventName.steerUnavailable: {
    ET.IMMEDIATE_DISABLE: ImmediateDisableAlert("LKAS Fault: Restart the Car"),
    ET.PERMANENT: NormalPermanentAlert("LKAS Fault: Restart the car to engage"),
    ET.NO_ENTRY: NoEntryAlert("LKAS Fault: Restart the Car"),
  },

  EventName.reverseGear: {
    ET.PERMANENT: Alert(
      "Reverse\nGear",
      "",
      AlertStatus.normal, AlertSize.full,
      Priority.LOWEST, VisualAlert.none, AudibleAlert.none, .2, creation_delay=0.5),
    ET.USER_DISABLE: ImmediateDisableAlert("Reverse Gear"),
    ET.NO_ENTRY: NoEntryAlert("Reverse Gear"),
  },

  EventName.cruiseDisabled: {
    ET.IMMEDIATE_DISABLE: ImmediateDisableAlert("Cruise Is Off"),
  },

  EventName.relayMalfunction: {
    ET.IMMEDIATE_DISABLE: ImmediateDisableAlert("Harness Relay Malfunction"),
    ET.PERMANENT: NormalPermanentAlert("Harness Relay Malfunction", "Check Hardware"),
    ET.NO_ENTRY: NoEntryAlert("Harness Relay Malfunction"),
  },

  EventName.speedTooLow: {
    ET.IMMEDIATE_DISABLE: Alert(
      "IQ.Pilot Canceled",
      "Speed too low",
      AlertStatus.normal, AlertSize.mid,
      Priority.HIGH, VisualAlert.none, AudibleAlert.disengage, 3.),
  },

  EventName.speedTooHigh: {
    ET.WARNING: Alert(
      "Speed Too High",
      "Model uncertain at this speed",
      AlertStatus.userPrompt, AlertSize.mid,
      Priority.HIGH, VisualAlert.steerRequired, AudibleAlert.promptRepeat, 4.),
    ET.NO_ENTRY: NoEntryAlert("Slow down to engage"),
  },

  EventName.vehicleSensorsInvalid: {
    ET.IMMEDIATE_DISABLE: ImmediateDisableAlert("Vehicle Sensors Invalid"),
    ET.PERMANENT: NormalPermanentAlert("Vehicle Sensors Calibrating", "Drive to Calibrate"),
    ET.NO_ENTRY: NoEntryAlert("Vehicle Sensors Calibrating"),
  },

  EventName.personalityChanged: {
    ET.WARNING: personality_changed_alert,
  },

  EventName.userBookmark: {
    ET.PERMANENT: NormalPermanentAlert("Bookmark Saved", duration=1.5),
  },

  EventName.audioFeedback: {
    ET.PERMANENT: audio_feedback_alert,
  },
}


if HARDWARE.get_device_type() == 'mici':
  EVENTS.update({
    EventName.preDriverDistracted: {
      ET.PERMANENT: Alert(
        "Pay Attention",
        "",
        AlertStatus.normal, AlertSize.small,
        Priority.MID, VisualAlert.none, AudibleAlert.none, 2),
    },
    EventName.promptDriverDistracted: {
      ET.PERMANENT: Alert(
        "Pay Attention",
        "Driver Distracted",
        AlertStatus.userPrompt, AlertSize.mid,
        Priority.MID, VisualAlert.steerRequired, AudibleAlert.promptDistracted, 1),
    },
    EventName.resumeRequired: {
      ET.WARNING: Alert(
        "Press Resume",
        "",
        AlertStatus.userPrompt, AlertSize.small,
        Priority.LOW, VisualAlert.none, AudibleAlert.none, .2),
    },
    EventName.preLaneChangeLeft: {
      ET.WARNING: Alert(
        "Steer Left",
        "Confirm Lane Change",
        AlertStatus.normal, AlertSize.mid,
        Priority.LOW, VisualAlert.none, AudibleAlert.none, .1),
    },
    EventName.preLaneChangeRight: {
      ET.WARNING: Alert(
        "Steer Right",
        "Confirm Lane Change",
        AlertStatus.normal, AlertSize.mid,
        Priority.LOW, VisualAlert.none, AudibleAlert.none, .1),
    },
    EventName.laneChangeBlocked: {
      ET.WARNING: Alert(
        "Car in Blindspot",
        "",
        AlertStatus.userPrompt, AlertSize.small,
        Priority.LOW, VisualAlert.none, AudibleAlert.prompt, .1),
    },
    EventName.steerSaturated: {
      ET.WARNING: Alert(
        "take control",
        "turn exceeds limit",
        AlertStatus.userPrompt, AlertSize.mid,
        Priority.LOW, VisualAlert.steerRequired, AudibleAlert.promptRepeat, 2.),
    },
    EventName.calibrationIncomplete: {
      ET.PERMANENT: calibration_incomplete_alert,
      ET.SOFT_DISABLE: soft_disable_alert("Calibration Incomplete"),
      ET.NO_ENTRY: NoEntryAlert("Calibrating"),
    },
    EventName.reverseGear: {
      ET.PERMANENT: Alert(
        "Reverse",
        "",
        AlertStatus.normal, AlertSize.full,
        Priority.LOWEST, VisualAlert.none, AudibleAlert.none, .2, creation_delay=0.5),
      ET.USER_DISABLE: ImmediateDisableAlert("Reverse"),
      ET.NO_ENTRY: NoEntryAlert("Reverse"),
    },
  })


if __name__ == '__main__':
  from iqpilot.cereal.services import SERVICE_LIST
  from collections import defaultdict

  event_names = {v: k for k, v in EventName.schema.enumerants.items()}
  alerts_by_type: dict[str, dict[Priority, list[str]]] = defaultdict(lambda: defaultdict(list))

  CP = car.CarParams.new_message()
  CS = car.CarState.new_message()
  sm = messaging.SubMaster(list(SERVICE_LIST.keys()))

  for i, alerts in EVENTS.items():
    for et, alert in alerts.items():
      if callable(alert):
        alert = alert(CP, CS, sm, False, 1, log.LongitudinalPersonality.standard)
      alerts_by_type[et][alert.priority].append(event_names[i])

  all_alerts: dict[str, list[tuple[Priority, list[str]]]] = {}
  for et, priority_alerts in alerts_by_type.items():
    all_alerts[et] = sorted(priority_alerts.items(), key=lambda x: x[0], reverse=True)

  for status, evs in sorted(all_alerts.items(), key=lambda x: x[0]):
    print(f"**** {status} ****")
    for p, alert_list in evs:
      print(f"  {repr(p)}:")
      print("   ", ', '.join(alert_list), "\n")
