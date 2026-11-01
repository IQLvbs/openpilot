#!/usr/bin/env python3
import fcntl
import os
import subprocess
import sys
import queue
import struct
import threading
import time
from collections import OrderedDict, namedtuple

import psutil

import iqpilot.cereal.messaging as messaging
from iqpilot.cereal import car
from iqpilot.cereal import log
from iqpilot.cereal.services import SERVICE_LIST
from iqpilot.common.iq_perf import PerfSample, PerfTraceEmitter
from iqpilot.common.utils import strip_deprecated_keys
from iqpilot.common.filter_simple import FirstOrderFilter
from iqpilot.common.basedir import BASEDIR
from iqpilot.common.params import Params
from iqpilot.common.realtime import DT_HW
from iqpilot.selfdrive.selfdrived.alertmanager import set_offroad_alert
from iqpilot.system.hardware import HARDWARE, TICI, AGNOS
from iqpilot.system.hardware.egpu_dock.flash import dock_needs_flash
from iqpilot.system.hardware.egpu_dock.status import EgpuDockStatus
from iqpilot.system.hardware.usb import get_link_error_count, get_usb_state, set_usb_state, usb3_lane
from iqpilot.system.loggerd.config import get_available_percent
from iqpilot.common.swaglog import cloudlog
from iqpilot.system.hardware.power_monitoring import PowerMonitoring, VBATT_LOW_POWER_EXIT
from iqpilot.system.hardware.fan_controller import FanController
from iqpilot.system.version import terms_version, training_version

ThermalStatus = log.DeviceState.ThermalStatus
NetworkType = log.DeviceState.NetworkType
NetworkStrength = log.DeviceState.NetworkStrength
CURRENT_TAU = 15.   # 15s time constant
TEMP_TAU = 5.   # 5s time constant
DISCONNECT_TIMEOUT = 5.  # wait 5 seconds before going offroad after disconnect so you get an alert
PANDA_STATES_TIMEOUT = round(1000 / SERVICE_LIST['pandaStates'].frequency * 1.5)  # 1.5x the expected pandaState frequency
ONROAD_CYCLE_TIME = 1  # seconds to wait offroad after requesting an onroad cycle
CAN_STARTUP_RECOVERY_DELAY = 3.  # require a persistent CAN timeout before cycling onroad processes
CAN_STARTUP_RECOVERY_WINDOW = 30.  # only recover shortly after ignition turns on
CAN_STARTUP_RECOVERY_COOLDOWN = 5.  # allow the restarted car stack time to initialize
CAN_STARTUP_RECOVERY_MAX_ATTEMPTS = 2

ThermalBand = namedtuple("ThermalBand", ['min_temp', 'max_temp'])
HardwareState = namedtuple("HardwareState", ['network_type', 'network_info', 'network_strength', 'network_stats',
                                             'network_metered', 'modem_temps', 'usb_state', 'usb_link_errors',
                                             'usb3_lane'])

# List of thermal bands. We will stay within this region as long as we are within the bounds.
# When exiting the bounds, we'll jump to the lower or higher band. Bands are ordered in the dict.
THERMAL_BANDS = OrderedDict({
  ThermalStatus.green: ThermalBand(None, 80.0),
  ThermalStatus.yellow: ThermalBand(75.0, 96.0),
  ThermalStatus.red: ThermalBand(88.0, 107.),
  ThermalStatus.danger: ThermalBand(94.0, None),
})

# Override to highest thermal band when offroad and above this temp
OFFROAD_DANGER_TEMP = 75

prev_offroad_states: dict[str, tuple[bool, str | None]] = {}


class CanStartupRecovery:
  """Bounded recovery for a car stack that starts without a usable CAN stream."""

  def __init__(self) -> None:
    self.ignition_on_ts: float | None = None
    self.timeout_started_ts: float | None = None
    self.last_attempt_ts: float | None = None
    self.attempts = 0

  def update(self, now: float, ignition: bool, started: bool, engaged: bool,
             car_state_alive: bool, can_timeout: bool, v_ego: float) -> bool:
    if not ignition:
      self.ignition_on_ts = None
      self.timeout_started_ts = None
      self.last_attempt_ts = None
      self.attempts = 0
      return False

    if self.ignition_on_ts is None:
      self.ignition_on_ts = now

    eligible = (
      started
      and not engaged
      and car_state_alive
      and can_timeout
      and abs(v_ego) < 0.1
      and (now - self.ignition_on_ts) <= CAN_STARTUP_RECOVERY_WINDOW
      and self.attempts < CAN_STARTUP_RECOVERY_MAX_ATTEMPTS
      and (self.last_attempt_ts is None or (now - self.last_attempt_ts) >= CAN_STARTUP_RECOVERY_COOLDOWN)
    )
    if not eligible:
      self.timeout_started_ts = None
      return False

    if self.timeout_started_ts is None:
      self.timeout_started_ts = now
      return False

    if (now - self.timeout_started_ts) < CAN_STARTUP_RECOVERY_DELAY:
      return False

    self.attempts += 1
    self.last_attempt_ts = now
    self.timeout_started_ts = None
    return True


def get_top_memory_processes(limit: int = 5) -> list[dict[str, object]]:
  procs: list[dict[str, object]] = []
  for proc in psutil.process_iter(['pid', 'name', 'memory_info', 'memory_percent']):
    try:
      info = proc.info
      rss = int(getattr(info.get('memory_info'), 'rss', 0))
      procs.append({
        "pid": int(info.get('pid', -1)),
        "name": str(info.get('name', 'unknown')),
        "rss_mb": round(rss / (1024 * 1024), 1),
        "mem_pct": round(float(info.get('memory_percent') or 0.0), 2),
      })
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, TypeError, ValueError):
      continue
  procs.sort(key=lambda p: p["rss_mb"], reverse=True)
  return procs[:limit]

def _is_meb(CP) -> bool:
  try:
    if CP.brand != "volkswagen":
      return False
    from iqdbc.car.volkswagen.values import VolkswagenFlags
    return bool(CP.flags & VolkswagenFlags.MEB)
  except Exception:
    cloudlog.exception("MEB detection failed")
    return False


class _CarParamsCache:
  def __init__(self, refresh_s: float = 5.0):
    self._refresh_s = refresh_s
    self._last_check = 0.0
    self._last_bytes: bytes | None = None
    self.no_sleep = False

  def update(self, params: Params) -> None:
    now = time.monotonic()
    if (now - self._last_check) < self._refresh_s:
      return
    self._last_check = now

    cp_bytes = params.get("CarParams") or params.get("CarParamsPersistent")
    if not cp_bytes or cp_bytes == self._last_bytes:
      return
    self._last_bytes = cp_bytes

    try:
      CP = messaging.log_from_bytes(cp_bytes, car.CarParams)
      from iqdbc.car.hyundai.values import CAR
      ev6 = CP.brand == "hyundai" and CP.carFingerprint in (CAR.KIA_EV6, CAR.KIA_EV6_PE)
      self.no_sleep = ev6 or (CP.brand == "tesla") or _is_meb(CP)
    except Exception:
      self.no_sleep = False



class EgpuDockFlasher:
  """Flash the dock's firmware offroad when it does not match what we ship.

  Same policy as stock: the model runtime ignores a dock until its product
  string matches, so a mismatched dock is unusable until this runs. Bounded
  attempts, offroad only, one flash in flight at a time.
  """
  MAX_ATTEMPTS = 3
  RETRY_INTERVAL = 20.

  def __init__(self):
    self.thread: threading.Thread | None = None
    self.attempts = 0
    self.last_attempt = 0.
    self.flashed = False
    self.mismatch = False

  @property
  def failed(self) -> bool:
    return (self.mismatch and self.attempts >= self.MAX_ATTEMPTS
            and self.thread is not None and not self.thread.is_alive() and not self.flashed)

  def flash(self) -> None:
    ret = subprocess.run(["sudo", sys.executable,
                          os.path.join(BASEDIR, "iqpilot/system/hardware/egpu_dock/flash.py")],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    cloudlog.event("egpu dock flash done", returncode=ret.returncode, output=ret.stdout[-1000:],
                   error=ret.returncode != 0)
    self.flashed = ret.returncode == 0

  def update(self, offroad: bool, usb_state: list[dict]) -> None:
    self.mismatch = dock_needs_flash(usb_state)
    if not self.mismatch:
      self.flashed = False
      return

    if not offroad or self.flashed or self.attempts >= self.MAX_ATTEMPTS:
      return
    if self.thread is not None and self.thread.is_alive():
      return
    if time.monotonic() - self.last_attempt < self.RETRY_INTERVAL:
      return

    self.attempts += 1
    self.last_attempt = time.monotonic()
    cloudlog.warning(f"egpu dock firmware out of date, flashing (attempt {self.attempts})")
    self.thread = threading.Thread(target=self.flash, daemon=True)
    self.thread.start()


def set_offroad_alert_if_changed(offroad_alert: str, show_alert: bool, extra_text: str | None=None):
  if prev_offroad_states.get(offroad_alert, None) == (show_alert, extra_text):
    return
  prev_offroad_states[offroad_alert] = (show_alert, extra_text)
  set_offroad_alert(offroad_alert, show_alert, extra_text)


def touch_thread(end_event):
  count = 0

  pm = messaging.PubMaster(["touch"])

  event_format = "llHHi"
  event_size = struct.calcsize(event_format)
  event_frame = []

  with open("/dev/input/by-path/platform-894000.i2c-event", "rb") as event_file:
    fcntl.fcntl(event_file, fcntl.F_SETFL, os.O_NONBLOCK)
    while not end_event.is_set():
      if (count % int(1. / DT_HW)) == 0:
        event = event_file.read(event_size)
        if event:
          (sec, usec, etype, code, value) = struct.unpack(event_format, event)
          if etype != 0 or code != 0 or value != 0:
            touch = log.Touch.new_message()
            touch.sec = sec
            touch.usec = usec
            touch.type = etype
            touch.code = code
            touch.value = value
            event_frame.append(touch)
          else: # end of frame, push new log
            msg = messaging.new_message('touch', len(event_frame), valid=True)
            msg.touch = event_frame
            pm.send('touch', msg)
            event_frame = []
          continue

      count += 1
      time.sleep(DT_HW)


def hw_state_thread(end_event, hw_queue):
  """Handles non critical hardware state, and sends over queue"""
  count = 0
  prev_hw_state = None
  params = Params()

  modem_version = None
  modem_configured = False
  modem_missing_count = 0
  modem_restart_count = 0
  sim_detection_recovered = False

  while not end_event.is_set():
    # these are expensive calls. update every 10s
    if (count % int(10. / DT_HW)) == 0:
      try:
        network_type = HARDWARE.get_network_type()
        modem_temps = HARDWARE.get_modem_temperatures()
        if len(modem_temps) == 0 and prev_hw_state is not None:
          modem_temps = prev_hw_state.modem_temps

        # Log modem version once
        if AGNOS and (modem_version is None):
          modem_version = HARDWARE.get_modem_version()

          if modem_version is not None:
            cloudlog.event("modem version", version=modem_version)

        if AGNOS and modem_restart_count < 3 and HARDWARE.get_modem_version() is None:
          # TODO: we may be able to remove this with a MM update
          # ModemManager's probing on startup can fail
          # rarely, restart the service to probe again.
          # Also, AT commands sometimes timeout resulting in ModemManager not
          # trying to use this modem anymore.
          modem_missing_count += 1
          if (modem_missing_count % 4) == 0:
            modem_restart_count += 1
            cloudlog.event("restarting ModemManager")
            subprocess.run(["sudo", "-n", "systemctl", "restart", "--no-block", "ModemManager"], timeout=5, capture_output=True, check=False)

        tx, rx = HARDWARE.get_modem_data_usage()

        hw_state = HardwareState(
          network_type=network_type,
          network_info=HARDWARE.get_network_info(),
          network_strength=HARDWARE.get_network_strength(network_type),
          network_stats={'wwanTx': tx, 'wwanRx': rx},
          network_metered=HARDWARE.get_network_metered(network_type),
          modem_temps=modem_temps,
          usb_state=get_usb_state(),
          usb_link_errors=get_link_error_count(),
          usb3_lane=usb3_lane(),
        )

        try:
          hw_queue.put_nowait(hw_state)
        except queue.Full:
          pass

        if params.get("NetworkMetered") != hw_state.network_metered:
          params.put_bool("NetworkMetered", hw_state.network_metered)

        if not modem_configured and HARDWARE.get_modem_version() is not None:
          cloudlog.warning("configuring modem")
          HARDWARE.configure_modem()
          modem_configured = True

        if modem_configured and not sim_detection_recovered and HARDWARE.recover_sim_detection():
          cloudlog.event("sim missing with hot-swap detect armed, rebooting modem with detect disabled", error=True)
          sim_detection_recovered = True

        prev_hw_state = hw_state
      except Exception:
        cloudlog.exception("Error getting hardware state")

    count += 1
    time.sleep(DT_HW)


def memory_snapshot_thread(end_event, memory_queue) -> None:
  while not end_event.is_set():
    try:
      memory_usage_percent = memory_queue.get(timeout=DT_HW)
    except queue.Empty:
      continue
    try:
      cloudlog.event("low_memory_snapshot", memory_usage_percent=memory_usage_percent,
                     top_processes=get_top_memory_processes(), error=True)
    except Exception:
      cloudlog.exception("Error collecting low memory snapshot")


def hardware_thread(end_event, hw_queue, memory_queue) -> None:
  pm = messaging.PubMaster(['deviceState', 'iqPerfTrace'])
  sm = messaging.SubMaster(["peripheralState", "gpsLocationExternal", "selfdriveState", "pandaStates", "carState", "egpuDockState"], poll="pandaStates")
  perf = PerfTraceEmitter("hardwared", pubmaster=pm)

  count = 0

  onroad_conditions: dict[str, bool] = {
    "ignition": False,
    "not_onroad_cycle": True,
    "device_temp_good": True,
  }
  startup_conditions: dict[str, bool] = {}
  startup_conditions_prev: dict[str, bool] = {}

  off_ts: float | None = None
  started_ts: float | None = None
  started_seen = False
  startup_blocked_ts: float | None = None
  thermal_status = ThermalStatus.yellow

  last_hw_state = HardwareState(
    network_type=NetworkType.none,
    network_info=None,
    network_metered=False,
    network_strength=NetworkStrength.unknown,
    network_stats={'wwanTx': -1, 'wwanRx': -1},
    modem_temps=[],
    usb_state=[],
    usb_link_errors=0,
    usb3_lane="unknown",
  )

  all_temp_filter = FirstOrderFilter(0., TEMP_TAU, DT_HW, initialized=False)
  offroad_temp_filter = FirstOrderFilter(0., TEMP_TAU, DT_HW, initialized=False)
  low_memory_logged = False
  should_start_prev = False
  in_car = False
  engaged_prev = False
  pwrsave = False
  low_power = False
  low_power_prev = False
  offroad_cycle_count = 0
  can_startup_recovery = CanStartupRecovery()

  params = Params()
  power_monitor = PowerMonitoring()
  cp_cache = _CarParamsCache()

  uptime_offroad: float = params.get("UptimeOffroad", return_default=True)
  uptime_onroad: float = params.get("UptimeOnroad", return_default=True)
  last_uptime_ts: float = time.monotonic()

  HARDWARE.initialize_hardware()
  thermal_config = HARDWARE.get_thermal_config()

  fan_controller = FanController()
  egpu_dock_flasher = EgpuDockFlasher()
  egpu_dock_status = EgpuDockStatus()

  while not end_event.is_set():
    sm.update(PANDA_STATES_TIMEOUT)

    pandaStates = sm['pandaStates']
    peripheralState = sm['peripheralState']

    # handle requests to cycle system started state
    if params.get_bool("OnroadCycleRequested"):
      params.put_bool("OnroadCycleRequested", False)
      offroad_cycle_count = sm.frame

    car_state = sm['carState']
    if can_startup_recovery.update(
      time.monotonic(),
      ignition=onroad_conditions["ignition"],
      started=started_ts is not None,
      engaged=sm['selfdriveState'].enabled,
      car_state_alive=sm.alive['carState'],
      can_timeout=car_state.canTimeout,
      v_ego=car_state.vEgo,
    ):
      offroad_cycle_count = sm.frame
      cloudlog.event("automatic CAN startup recovery", attempt=can_startup_recovery.attempts, error=True)
    onroad_conditions["not_onroad_cycle"] = (sm.frame - offroad_cycle_count) >= ONROAD_CYCLE_TIME * SERVICE_LIST['pandaStates'].frequency

    if sm.updated['pandaStates'] and len(pandaStates) > 0:

      # Set ignition based on any panda connected
      onroad_conditions["ignition"] = any(ps.ignitionLine or ps.ignitionCan for ps in pandaStates if ps.pandaType != log.PandaState.PandaType.unknown)

      pandaState = pandaStates[0]

      in_car = pandaState.harnessStatus != log.PandaState.HarnessStatus.notConnected

    elif (time.monotonic() - sm.recv_time['pandaStates']) > DISCONNECT_TIMEOUT:
      if onroad_conditions["ignition"]:
        onroad_conditions["ignition"] = False
        cloudlog.error("panda timed out onroad")

    if params.get_bool("IQBenchIgnition"):
      onroad_conditions["ignition"] = True

    # Run at 2Hz, plus either edge of ignition
    ign_edge = (started_ts is not None) != all(onroad_conditions.values())
    if (sm.frame % round(SERVICE_LIST['pandaStates'].frequency * DT_HW) != 0) and not ign_edge:
      continue

    msg = messaging.new_message('deviceState', valid=True)
    msg.deviceState = thermal_config.get_msg()
    msg.deviceState.deviceType = HARDWARE.get_device_type()

    try:
      last_hw_state = hw_queue.get_nowait()
    except queue.Empty:
      pass

    msg.deviceState.freeSpacePercent = get_available_percent(default=100.0)
    try:
      msg.deviceState.memoryUsagePercent = int(round(psutil.virtual_memory().percent))
    except Exception:
      msg.deviceState.memoryUsagePercent = 0
    # get_top_memory_processes() costs ~500ms: must never run in the 2Hz publish loop
    if msg.deviceState.memoryUsagePercent > 95:
      if not low_memory_logged:
        try:
          memory_queue.put_nowait(msg.deviceState.memoryUsagePercent)
        except queue.Full:
          pass
        low_memory_logged = True
    else:
      low_memory_logged = False
    msg.deviceState.gpuUsagePercent = int(round(HARDWARE.get_gpu_usage_percent()))
    online_cpu_usage = [int(round(n)) for n in psutil.cpu_percent(percpu=True)]
    offline_cpu_usage = [0., ] * (len(msg.deviceState.cpuTempC) - len(online_cpu_usage))
    msg.deviceState.cpuUsagePercent = online_cpu_usage + offline_cpu_usage
    if msg.deviceState.memoryUsagePercent > 95:
      avg_cpu_usage = int(round(sum(online_cpu_usage) / max(1, len(online_cpu_usage))))
      perf.emit(
        "hardware_low_memory",
        severity="error" if msg.deviceState.memoryUsagePercent > 95 else "warning",
        frame_id=sm.frame,
        samples=[PerfSample(
          frame_id=sm.frame,
          memory_usage_percent=int(msg.deviceState.memoryUsagePercent),
          gpu_usage_percent=int(msg.deviceState.gpuUsagePercent),
          cpu_usage_percent=avg_cpu_usage,
        )],
        detail=f"memory_usage_percent={msg.deviceState.memoryUsagePercent} gpu_usage_percent={msg.deviceState.gpuUsagePercent}",
        min_interval_s=5.0,
      )

    msg.deviceState.networkType = last_hw_state.network_type
    msg.deviceState.networkMetered = last_hw_state.network_metered
    msg.deviceState.networkStrength = last_hw_state.network_strength
    msg.deviceState.networkStats = last_hw_state.network_stats
    if last_hw_state.network_info is not None:
      msg.deviceState.networkInfo = last_hw_state.network_info

    msg.deviceState.modemTempC = last_hw_state.modem_temps
    set_usb_state(msg.deviceState, last_hw_state.usb_state, last_hw_state.usb_link_errors,
                  last_hw_state.usb3_lane)
    egpu_dock_flasher.update(started_ts is None, last_hw_state.usb_state)
    egpu_valid = sm.alive["egpuDockState"] and sm.valid["egpuDockState"]
    egpu_dock_status.update(started_ts is None, last_hw_state.usb_state, egpu_dock_flasher.failed,
                            params.get_bool("UsbGpuLoading"), params.get("UsbGpuActive"),
                            params.get_bool("UsbGpuReady"),
                            sm["egpuDockState"] if egpu_valid else None, set_offroad_alert_if_changed)

    msg.deviceState.screenBrightnessPercent = HARDWARE.get_screen_brightness()

    # this subset is only used for offroad
    temp_sources = [
      msg.deviceState.memoryTempC,
      max(msg.deviceState.cpuTempC, default=0.),
      max(msg.deviceState.gpuTempC, default=0.),
    ]
    offroad_comp_temp = offroad_temp_filter.update(max(temp_sources))

    # this drives the thermal status while onroad
    temp_sources.append(max(msg.deviceState.pmicTempC, default=0.))
    all_comp_temp = all_temp_filter.update(max(temp_sources))
    msg.deviceState.maxTempC = all_comp_temp

    is_offroad_for_5_min = (started_ts is None) and ((not started_seen) or (off_ts is None) or (time.monotonic() - off_ts > 60 * 5))
    if is_offroad_for_5_min and offroad_comp_temp > OFFROAD_DANGER_TEMP:
      # if device is offroad and already hot without the extra onroad load,
      # we want to cool down first before increasing load
      thermal_status = ThermalStatus.danger
    else:
      current_band = THERMAL_BANDS[thermal_status]
      band_idx = list(THERMAL_BANDS.keys()).index(thermal_status)
      if current_band.min_temp is not None and all_comp_temp < current_band.min_temp:
        thermal_status = list(THERMAL_BANDS.keys())[band_idx - 1]
      elif current_band.max_temp is not None and all_comp_temp > current_band.max_temp:
        thermal_status = list(THERMAL_BANDS.keys())[band_idx + 1]

    # the car is running but temperature is blocking the start, so cool as fast as we can
    max_cool = (started_ts is None) and onroad_conditions["ignition"] and thermal_status >= ThermalStatus.red
    msg.deviceState.fanSpeedPercentDesired = fan_controller.update(all_comp_temp, onroad_conditions["ignition"], max_cool)

    # **** starting logic ****

    startup_conditions["up_to_date"] = True
    startup_conditions["no_excessive_actuation"] = params.get("Offroad_ExcessiveActuation") is None
    startup_conditions["not_uninstalling"] = not params.get_bool("DoUninstall")
    startup_conditions["accepted_terms"] = params.get("HasAcceptedTerms") == terms_version

    # with 2% left, we killall, otherwise the phone will take a long time to boot
    startup_conditions["free_space"] = msg.deviceState.freeSpacePercent > 2
    startup_conditions["completed_training"] = HARDWARE.get_device_type() != "mici" or params.get("CompletedTrainingVersion") == training_version
    startup_conditions["not_driver_view"] = not params.get_bool("IsDriverViewEnabled")
    startup_conditions["not_taking_snapshot"] = not params.get_bool("IsTakingSnapshot")

    # must be at an engageable thermal band to go onroad
    startup_conditions["device_temp_engageable"] = thermal_status < ThermalStatus.red

    # ensure device is fully booted
    startup_conditions["device_booted"] = startup_conditions.get("device_booted", False) or HARDWARE.booted()

    # user-forced status
    offroad_mode = params.get_bool("IQAlwaysOffroad")
    startup_conditions["not_always_offroad"] = not offroad_mode
    onroad_conditions["not_always_offroad"] = not offroad_mode

    # if the temperature enters the danger zone, go offroad to cool down
    onroad_conditions["device_temp_good"] = thermal_status < ThermalStatus.danger
    extra_text = f"{offroad_comp_temp:.1f}C"
    show_alert = (not onroad_conditions["device_temp_good"] or not startup_conditions["device_temp_engageable"]) and onroad_conditions["ignition"]
    set_offroad_alert_if_changed("Offroad_TemperatureTooHigh", show_alert, extra_text=extra_text)

    # Handle offroad/onroad transition
    should_start = all(onroad_conditions.values())
    if started_ts is None:
      should_start = should_start and all(startup_conditions.values())

    if should_start != should_start_prev or (count == 0):
      params.put_bool("IsEngaged", False)
      engaged_prev = False

    if sm.updated['selfdriveState']:
      engaged = sm['selfdriveState'].enabled
      if engaged != engaged_prev:
        params.put_bool("IsEngaged", engaged)
        engaged_prev = engaged

      try:
        with open('/dev/kmsg', 'w') as kmsg:
          kmsg.write(f"<3>[hardware] engaged: {engaged}\n")
      except Exception:
        pass

    cp_cache.update(params)
    no_sleep = cp_cache.no_sleep

    should_pwrsave = (not no_sleep) and (not onroad_conditions["ignition"] and msg.deviceState.screenBrightnessPercent < 1e-3)
    if should_pwrsave != pwrsave or (count == 0):
      HARDWARE.set_power_save(should_pwrsave)
    pwrsave = should_pwrsave

    if should_start:
      off_ts = None
      if started_ts is None:
        started_ts = time.monotonic()
        started_seen = True
        if startup_blocked_ts is not None:
          cloudlog.event("Startup after block", block_duration=(time.monotonic() - startup_blocked_ts),
                         startup_conditions=startup_conditions, onroad_conditions=onroad_conditions,
                         startup_conditions_prev=startup_conditions_prev, error=True)
      startup_blocked_ts = None
    else:
      if onroad_conditions["ignition"] and (startup_conditions != startup_conditions_prev):
        cloudlog.event("Startup blocked", startup_conditions=startup_conditions, onroad_conditions=onroad_conditions, error=True)
        startup_conditions_prev = startup_conditions.copy()
        startup_blocked_ts = time.monotonic()

      started_ts = None
      if off_ts is None:
        off_ts = time.monotonic()

    # Offroad power monitoring
    voltage = None if peripheralState.pandaType == log.PandaState.PandaType.unknown else peripheralState.voltage

    power_monitor.calculate(voltage, onroad_conditions["ignition"])
    msg.deviceState.offroadPowerUsageUwh = power_monitor.get_power_used()
    msg.deviceState.carBatteryCapacityUwh = max(0, power_monitor.get_car_battery_capacity())
    current_power_draw = HARDWARE.get_current_power_draw()
    msg.deviceState.powerDrawW = current_power_draw

    som_power_draw = HARDWARE.get_som_power_draw()
    msg.deviceState.somPowerDrawW = som_power_draw

    # FastSleep deep standby: shed heavy processes once parked with the screen idled off
    # (or at low battery) instead of shutting down, recover on ignition or once the
    # alternator is charging
    fast_sleep = params.get_bool("FastSleep")
    if fast_sleep and not no_sleep:
      if low_power:
        if onroad_conditions["ignition"] or power_monitor.car_voltage_mV >= (VBATT_LOW_POWER_EXIT * 1e3):
          low_power = False
      else:
        screen_off = msg.deviceState.screenBrightnessPercent < 1e-3
        low_power = power_monitor.should_enter_low_power(onroad_conditions["ignition"], in_car, off_ts, screen_off)
    else:
      low_power = False

    # Blank the panel only in deep standby, where not_low_power has shed the UI (so nothing
    # relights it and there is no touch grab to fight). The parked idle screen-off and
    # tap-to-wake live in the UI, which owns the touchscreen grab and brightness.
    if low_power != low_power_prev:
      params.put("DevicePowerState", "low_power" if low_power else "normal")
      cloudlog.event("hardwared.device_power_state", low_power=low_power, voltage_mV=power_monitor.car_voltage_mV, error=False)
      if low_power:
        HARDWARE.set_screen_brightness(0)
      low_power_prev = low_power

    # Check if we need to shut down
    if (not no_sleep) and power_monitor.should_shutdown(onroad_conditions["ignition"], in_car, off_ts, started_seen):
      cloudlog.warning(f"shutting device down, offroad since {off_ts}")
      params.put_bool("DoShutdown", True)

    msg.deviceState.started = started_ts is not None and not offroad_mode
    msg.deviceState.startedMonoTime = int(1e9*(started_ts or 0))

    last_ping = params.get("LastAthenaPingTime")
    if last_ping is not None:
      msg.deviceState.lastAthenaPingTime = last_ping

    msg.deviceState.thermalStatus = thermal_status
    pm.send("deviceState", msg)


    # report to server once every 10 minutes
    rising_edge_started = should_start and not should_start_prev
    if rising_edge_started or (count % int(600. / DT_HW)) == 0:
      dat = {
        'count': count,
        'pandaStates': [strip_deprecated_keys(p.to_dict()) for p in pandaStates],
        'peripheralState': strip_deprecated_keys(peripheralState.to_dict()),
        'location': (strip_deprecated_keys(sm["gpsLocationExternal"].to_dict()) if sm.alive["gpsLocationExternal"] else None),
        'deviceState': strip_deprecated_keys(msg.to_dict())
      }
      cloudlog.event("STATUS_PACKET", **dat)

      # save last one before going onroad
      if rising_edge_started:
        try:
          params.put("LastOffroadStatusPacket", dat)
        except Exception:
          cloudlog.exception("failed to save offroad status")

    now_ts = time.monotonic()
    if off_ts:
      uptime_offroad += now_ts - max(last_uptime_ts, off_ts)
    elif started_ts:
      uptime_onroad += now_ts - max(last_uptime_ts, started_ts)
    last_uptime_ts = now_ts

    if (count % int(60. / DT_HW)) == 0:
      params.put("UptimeOffroad", uptime_offroad)
      params.put("UptimeOnroad", uptime_onroad)

    count += 1
    should_start_prev = should_start


def main():
  hw_queue = queue.Queue(maxsize=1)
  memory_queue = queue.Queue(maxsize=1)
  end_event = threading.Event()

  threads = [
    threading.Thread(target=hw_state_thread, args=(end_event, hw_queue)),
    threading.Thread(target=hardware_thread, args=(end_event, hw_queue, memory_queue)),
    threading.Thread(target=memory_snapshot_thread, args=(end_event, memory_queue)),
  ]

  if TICI:
    threads.append(threading.Thread(target=touch_thread, args=(end_event,)))

  for t in threads:
    t.start()

  try:
    while True:
      time.sleep(1)
      if not all(t.is_alive() for t in threads):
        break
  finally:
    end_event.set()

  for t in threads:
    t.join()


if __name__ == "__main__":
  main()
