import os
import platform
from pathlib import Path

from iqpilot.cereal import car, custom
from iqpilot.common.params import Params
from iqpilot.system.hardware import HARDWARE, PC, TICI
from iqpilot.system.hardware.hw import Paths
from iqpilot.system.manager.process import PythonProcess, NativeProcess, BundleProcess
from iqpilot.system.wireguard import tunnel as wireguard

from iqpilot.selfdrive.iqmodeld.egpu_helpers import egpu_selected, resolve_backend, usbgpu_present
from iqpilot.selfdrive.iqmodeld.models.helpers import get_active_model_runner
from iqpilot.konn3kt.service_health import hephaestus_ready

def driverview(started: bool, params: Params, CP: car.CarParams) -> bool:
  return started or params.get_bool("IsDriverViewEnabled")

def driver_monitoring(started: bool, params: Params, CP: car.CarParams) -> bool:
  if os.path.exists('/tmp/lite_hw'):
    return False
  return driverview(started, params, CP)

def notcar(started: bool, params: Params, CP: car.CarParams) -> bool:
  return started and CP.notCar

def iscar(started: bool, params: Params, CP: car.CarParams) -> bool:
  return started and not CP.notCar

def logging(started: bool, params: Params, CP: car.CarParams) -> bool:
  run = (not CP.notCar) or not params.get_bool("DisableLogging")
  return started and run and params.get_bool("DashcamEnabled")

def ublox_available() -> bool:
  if HARDWARE.get_device_type() == "tizi" or os.path.exists('/tmp/lite_hw'):
    return False

  quectel_override = Path(Paths.persist_root()) / "comma" / "use-quectel-gps"
  return os.path.exists('/dev/ttyHS0') and not quectel_override.exists()

def ublox(started: bool, params: Params, CP: car.CarParams) -> bool:
  use_ublox = ublox_available()
  if use_ublox != params.get_bool("UbloxAvailable"):
    params.put_bool("UbloxAvailable", use_ublox)
  return started and use_ublox

def joystick(started: bool, params: Params, CP: car.CarParams) -> bool:
  return started and params.get_bool("JoystickDebugMode")

def not_joystick(started: bool, params: Params, CP: car.CarParams) -> bool:
  return started and not params.get_bool("JoystickDebugMode")

def long_maneuver(started: bool, params: Params, CP: car.CarParams) -> bool:
  return started and params.get_bool("LongitudinalManeuverMode")

def not_long_maneuver(started: bool, params: Params, CP: car.CarParams) -> bool:
  return started and not params.get_bool("LongitudinalManeuverMode")

def lat_maneuver(started: bool, params: Params, CP: car.CarParams) -> bool:
  return started and params.get_bool("LateralManeuverMode")

def not_lat_maneuver(started: bool, params: Params, CP: car.CarParams) -> bool:
  return started and not params.get_bool("LateralManeuverMode")

def qcomgps(started: bool, params: Params, CP: car.CarParams) -> bool:
  return started and not ublox_available()

def always_run(started: bool, params: Params, CP: car.CarParams) -> bool:
  return True

def nav_assist(started: bool, params: Params, CP: car.CarParams) -> bool:
  return params.get_bool("IQNavAssistDev") and params.get_bool("IQAndroidNav")

def only_onroad(started: bool, params: Params, CP: car.CarParams) -> bool:
  return started

def navigation(started: bool, params: Params, CP: car.CarParams) -> bool:
  return params.get_bool("NavigationEnabled") and (
    started or bool(params.get("NavigationDestination")) or bool(params.get("AthenaNavigationRoute"))
  )

def navrenderd_onroad(started: bool, params: Params, CP: car.CarParams) -> bool:
  return started and params.get_bool("NavigationEnabled") and params.get_bool("OnScreenNavigation")

def navincidentd_onroad(started: bool, params: Params, CP: car.CarParams) -> bool:
  return started and params.get_bool("NavigationEnabled") and bool(params.get("WazePoliceApiKey")) and (
    params.get_int("WazePoliceAlertMode") > 0 or params.get_bool("WazePoliceShadow")
  )

def iqmapd_needed(params: Params) -> bool:
  return (
    params.get_bool("IQRoadNameOverlay")
    or params.get_bool("ShowSpeedLimits")
    or params.get_bool("SpeedLimitController")
    or params.get_bool("EnableSpeedLimitControl")
    or params.get_bool("EnableSpeedLimitPredicative")
    or params.get_bool("MapCurveSpeedController")
    or params.get_bool("VisionCurveSpeedController")
  )

def iqmodeld_model_ready(started: bool, params: Params, CP: car.CarParams) -> bool:
  return (not is_tinygrad_model(started, params, CP)) or params.get_bool("IqModelLoaded")

def iqmapd_onroad(started: bool, params: Params, CP: car.CarParams) -> bool:
  return started and params.get_bool("NavigationEnabled") and iqmapd_needed(params) and iqmodeld_model_ready(started, params, CP)

def mapd_onroad(started: bool, params: Params, CP: car.CarParams) -> bool:
  return started and iqmapd_needed(params) and iqmodeld_model_ready(started, params, CP)

def constructiond_onroad(started: bool, params: Params, CP: car.CarParams) -> bool:
  return started and params.get_bool("ConstructionZoneAssist")

def iqvd_onroad(started: bool, params: Params, CP: car.CarParams) -> bool:
  # held for 1.0d: iqvd runs a detector per frame and the added load is not
  # something 1.0c needs to carry. re-enable by restoring the param check.
  return False

def only_offroad(started: bool, params: Params, CP: car.CarParams) -> bool:
  return not started

def wireguard_installed(started: bool, params: Params, CP: car.CarParams) -> bool:
  return wireguard.installed()

def sentry_enabled(started: bool, params: Params, CP: car.CarParams) -> bool:
  if not params.get_bool("IQSentry"):
    return False
  return not started or params.get_bool("IQSentryRecording") or params.get_bool("IQSentryKeepStartups")

def sentry_recording(started: bool, params: Params, CP: car.CarParams) -> bool:
  return params.get_bool("IQSentryRecording")

def livestream(started: bool, params: Params, CP: car.CarParams) -> bool:
  # Konn3kt Live View: hephaestusd sets IsLiveStreaming when a viewer connects, so the
  # manager brings up the stream encoder (and camerad/webrtcd when offroad) and tears them
  # down cleanly when the session ends — no subprocess management inside hephaestusd.
  return params.get_bool("IsLiveStreaming")

def canlive(started: bool, params: Params, CP: car.CarParams) -> bool:
  # Remote live CAN debugging via konn3kt. hephaestusd sets CanLiveStreaming when a viewer
  # connects (startCanLive) and clears it when the last one leaves (stopCanLive), so canlived
  # runs only during an active debug session — no idle connection or battery cost otherwise.
  return params.get_bool("CanLiveStreaming")

def is_tinygrad_model(started, params, CP: car.CarParams) -> bool:
  """Check if the active model runner is tinygrad."""
  return bool(get_active_model_runner(params, not started) == custom.IQModelManager.Runner.tinygrad)

def _egpu_present(params) -> bool:
  if params.get_bool("IQEgpuDisabled"):
    return False
  return usbgpu_present()


def emac_enabled(started, params, CP: car.CarParams) -> bool:
  return resolve_backend(params.get_bool("IQEmacEnabled"), egpu_selected(params), _egpu_present(params)) == "emac"

def egpu_enabled(started, params, CP: car.CarParams) -> bool:
  return (resolve_backend(params.get_bool("IQEmacEnabled"), egpu_selected(params), _egpu_present(params)) == "egpu"
          and _egpu_present(params))

def egpu_prefetch_enabled(started, params, CP: car.CarParams) -> bool:
  if params.get_bool("IQEgpuDisabled"):
    return False
  return resolve_backend(params.get_bool("IQEmacEnabled"), True, _egpu_present(params)) == "egpu"

def big_model_enabled(started, params, CP: car.CarParams) -> bool:
  return params.get_bool("IQEmacEnabled") or egpu_selected(params)

def hephaestus_ready_shim(started, params, CP: car.CarParams) -> bool:
  return hephaestus_ready(params)

_IQCORE_ARMED_ENV = {"IQCORE_ARMED": "1"}

def not_low_power(started: bool, params: Params, CP: car.CarParams) -> bool:
  # FastSleep deep standby: heavy processes are shed offroad while DevicePowerState is low_power
  return started or params.get("DevicePowerState") != "low_power"

def iquploaderd_ready(started: bool, params: Params, CP: car.CarParams) -> bool:
  if not params.get_bool("OnroadUploads"):
    return only_offroad(started, params, CP)

  return always_run(started, params, CP)

def or_(*fns):
  return lambda *args: any(fn(*args) for fn in fns)

def and_(*fns):
  return lambda *args: all(fn(*args) for fn in fns)

procs = [
  NativeProcess("loggerd", "iqpilot/system/loggerd", ["./loggerd"], logging),
  NativeProcess("encoderd", "iqpilot/system/loggerd", ["./encoderd"], only_onroad),
  NativeProcess("stream_encoderd", "iqpilot/system/loggerd", ["./encoderd", "--stream"], or_(notcar, livestream, sentry_recording),
                restart_if_crash=True),
  PythonProcess("logmessaged", "iqpilot.system.logmessaged", always_run, restart_if_crash=True),

  NativeProcess("camerad", "iqpilot/system/camerad", ["./camerad"], or_(driverview, livestream, sentry_recording), restart_if_crash=True),
  PythonProcess("proclogd", "iqpilot.system.proclogd", only_onroad, enabled=platform.system() != "Darwin"),
  PythonProcess("journald", "iqpilot.system.journald", only_onroad, platform.system() != "Darwin"),
  PythonProcess("micd", "iqpilot.system.micd", or_(iscar, livestream)),
  PythonProcess("timed", "iqpilot.system.timed", always_run, enabled=not PC),
  PythonProcess("wireguardd", "iqpilot.system.wireguard.wireguardd", wireguard_installed, enabled=TICI, restart_if_crash=True),
  BundleProcess("iqsentryd", "iqpilot_hephaestusd_private", "iqpilot_private.konn3kt.hephaestus.iqsentryd", sentry_enabled,
                enabled=TICI, restart_if_crash=True),
  BundleProcess("navassistd", "iqpilot_navassist_private", "iqpilot_private.navassist.daemon", nav_assist, enabled=TICI, restart_if_crash=True),

  PythonProcess("dmonitoringmodeld", "iqpilot.selfdrive.dmonitoringmodeld.dmonitoringmodeld", driver_monitoring, enabled=not PC, env=_IQCORE_ARMED_ENV),

  PythonProcess("sensord", "iqpilot.system.sensord.sensord", only_onroad, enabled=not PC),
  PythonProcess("ui", "iqpilot.selfdrive.ui.ui", not_low_power, restart_if_crash=True),
  PythonProcess("soundd", "iqpilot.selfdrive.ui.soundd", driverview),
  PythonProcess("locationd", "iqpilot.selfdrive.locationd.locationd", only_onroad),
  NativeProcess("_pandad", "iqpilot/selfdrive/pandad", ["./pandad"], always_run, enabled=False),
  PythonProcess("calibrationd", "iqpilot.selfdrive.locationd.calibrationd", only_onroad),
  PythonProcess("controlsd", "iqpilot.selfdrive.controls.controlsd", and_(not_joystick, iscar)),
  PythonProcess("joystickd", "iqpilot.tools.joystick.joystickd", or_(joystick, notcar)),
  PythonProcess("selfdrived", "iqpilot.selfdrive.selfdrived.selfdrived", only_onroad),
  PythonProcess("card", "iqpilot.selfdrive.car.card", only_onroad),
  PythonProcess("deleter", "iqpilot.system.loggerd.deleter", always_run),
  PythonProcess("dmonitoringd", "iqpilot.selfdrive.monitoring.dmonitoringd", driver_monitoring, enabled=not PC),
  PythonProcess("qcomgpsd", "iqpilot.system.qcomgpsd.qcomgpsd", qcomgps, enabled=TICI),
  PythonProcess("phonegpsd", "iqpilot.system.phonegps.phonegpsd", only_onroad, enabled=TICI),
  PythonProcess("pandad", "iqpilot.selfdrive.pandad.pandad", always_run),
  PythonProcess("estimatord", "iqpilot.selfdrive.locationd.estimatord", only_onroad),
  PythonProcess("ubloxd", "iqpilot.system.ubloxd.ubloxd", ublox, enabled=TICI),
  PythonProcess("pigeond", "iqpilot.system.ubloxd.pigeond", ublox, enabled=TICI),
  PythonProcess("plannerd", "iqpilot.selfdrive.controls.plannerd", not_long_maneuver),
  PythonProcess("maneuversd", "iqpilot.tools.maneuvers.longitudinal_maneuversd", long_maneuver),
  PythonProcess("lateral_maneuversd", "iqpilot.tools.maneuvers.lateral_maneuversd", lat_maneuver),
  PythonProcess("radard", "iqpilot.selfdrive.controls.radard", only_onroad),
  PythonProcess("hardwared", "iqpilot.system.hardware.hardwared", always_run, restart_if_crash=True),
  PythonProcess("tombstoned", "iqpilot.system.tombstoned", always_run, enabled=not PC),
  PythonProcess("updated", "iqpilot.system.updated.updated", and_(only_offroad, not_low_power), enabled=not PC),
  BundleProcess("iquploaderd", "iqpilot_hephaestusd_private", "iqpilot_private.konn3kt.uploaderd.iquploaderd",
                and_(iquploaderd_ready, not_low_power), restart_if_crash=True),
  BundleProcess("iqtelemetryd", "iqpilot_hephaestusd_private", "iqpilot_private.konn3kt.telemetry.iqtelemetryd",
                and_(always_run, not_low_power), enabled=TICI, restart_if_crash=True),
  PythonProcess("feedbackd", "iqpilot.selfdrive.ui.feedback.feedbackd", and_(only_onroad, not_lat_maneuver)),

  # debug procs
  NativeProcess("bridge", "iqpilot/cereal/messaging", ["./bridge"], notcar),
  PythonProcess("webrtcd", "iqpilot.system.webrtc.webrtcd", or_(iscar, livestream), restart_if_crash=True),
  PythonProcess("canlived", "iqpilot.konn3kt.canlive.canlived", canlive),
]

# iqpilot
procs += [
  # Models
  BundleProcess("models_manager", "iqpilot_model_selector_private", "iqpilot_private.models.manager", and_(only_offroad, not_low_power), restart_if_crash=True),
  NativeProcess("iqmodeld", "iqpilot/selfdrive/iqmodeld", ["./iqmodeld"], and_(only_onroad, is_tinygrad_model), restart_if_crash=True, env=_IQCORE_ARMED_ENV),
  # big-model backends: iqmodeld self-demotes to the small channel worker when
  # either backend is enabled; the selector publishes, and exactly one big
  # worker (Mac or eGPU, eMac wins) feeds the BIG channel
  PythonProcess("modeld_selector", "iqpilot.selfdrive.iqmodeld.modeld_selector",
                and_(only_onroad, and_(is_tinygrad_model, big_model_enabled)), restart_if_crash=True),
  BundleProcess("maciqmodeld", "iqpilot_emac_private", "iqpilot_private.emac.maciqmodeld",
                and_(only_onroad, and_(is_tinygrad_model, emac_enabled)), restart_if_crash=True, env=_IQCORE_ARMED_ENV),
  PythonProcess("iqegpumodeld", "iqpilot.selfdrive.iqmodeld.iqegpumodeld",
                and_(only_onroad, and_(is_tinygrad_model, egpu_enabled)), restart_if_crash=True, env=_IQCORE_ARMED_ENV),
  PythonProcess("egpu_prefetch", "iqpilot.selfdrive.iqmodeld.egpu_prefetch",
                and_(only_offroad, and_(is_tinygrad_model, egpu_prefetch_enabled)), restart_if_crash=True),

  BundleProcess("backup_manager_k3", "iqpilot_hephaestusd_private", "iqpilot_private.konn3kt.backups.backup_orchestrator",
                and_(only_offroad, hephaestus_ready_shim, not_low_power)),
  BundleProcess("navd", "iqpilot_navd_private", "iqpilot_private.navd.navd", navigation, restart_if_crash=True),
  BundleProcess("navincidentd", "iqpilot_navd_private", "iqpilot_private.navd.navincidentd", navincidentd_onroad, restart_if_crash=True),
  BundleProcess("navrenderd", "iqpilot_navd_private", "iqpilot_private.navd.navrenderd", navrenderd_onroad, restart_if_crash=True),
  BundleProcess("iqmapd", "iqpilot_navd_private", "iqpilot_private.navd.iqmapd", iqmapd_onroad, restart_if_crash=True),

  # work-zone detector for Speed Limit Assist
  PythonProcess("constructiond", "iqpilot.selfdrive.constructiond", constructiond_onroad, restart_if_crash=True),

  # iqvd: vision vehicle detector for UI ambient track dots
  BundleProcess("iqvd", "iqpilot_iqvd_private", "iqpilot_private.iqvd.iqvd", iqvd_onroad, restart_if_crash=True),

  # mapd
  NativeProcess("mapd", "iqpilot/third_party/mapd_pfeiferj", ["./mapd"], mapd_onroad, restart_if_crash=True),
  PythonProcess("mapd_manager", "iqpilot.iq_maps.orchestrator", and_(only_offroad, not_low_power)),

  # locationd
  NativeProcess("iqlocd", "iqpilot/selfdrive/iqlocd", ["./iqlocd"], only_onroad, restart_if_crash=True),
]

managed_processes = {p.name: p for p in procs}
