#!/usr/bin/env python3
import datetime
import os
import signal
import sys
import time
import traceback

# sync $PWD before cereal/kj loads (and for spawned procs) or kj warns "PWD doesn't match"
os.environ['PWD'] = os.getcwd()

from iqpilot.cereal import log
import iqpilot.cereal.messaging as messaging
import iqpilot.system.sentry as sentry
from iqpilot.common.utils import atomic_write
from iqpilot.common.params import Params, ParamKeyFlag
from iqpilot.common.text_window import TextWindow
from iqpilot.system.hardware import HARDWARE
from iqpilot.system.loggerd.crash_recovery import recover_unclean_segments
from iqpilot.system.manager.helpers import unblock_stdout, write_onroad_params, save_bootlog, heal_param_perms
from iqpilot.system.manager.process import ensure_running
from iqpilot.system.proprietary_runtime.bundle_modes import normalize_tree
from iqpilot.common.basedir import BASEDIR
from iqpilot.system.manager.process_config import managed_processes
from iqpilot.konn3kt.registration import register, UNREGISTERED_DONGLE_ID
from iqpilot.common.swaglog import cloudlog, add_file_handler
from iqpilot.system.version import get_build_metadata
from iqpilot.system.hardware.hw import Paths


MODELD_WATCHDOG_TIMEOUT = 30.0
MODELD_FIRST_PUBLISH_TIMEOUT = 120.0


def update_modeld_watchdog(deadline: float | None, started: bool, model_updated: bool, process, now: float) -> float | None:
  running = process.proc is not None and process.proc.is_alive()
  if not started or not running:
    return None
  if model_updated:
    return now + MODELD_WATCHDOG_TIMEOUT
  if deadline is None:
    return now + MODELD_FIRST_PUBLISH_TIMEOUT
  if now >= deadline:
    cloudlog.error("iqmodeld is alive but not publishing modelV2; restarting")
    process.restart()
    return None
  return deadline


def apply_curvature_controller_default(params: Params) -> None:
  if not params.get_bool("EnableCurvatureControllerDefaultApplied"):
    params.put_bool("EnableCurvatureController", True)
    params.put_bool("EnableCurvatureControllerDefaultApplied", True)


_drm_broker_conn = None

def _acquire_drm_fd() -> None:
  global _drm_broker_conn
  if os.environ.get('DRM_FD') or not os.path.exists('/tmp/drmfd.sock'):
    return
  import socket, struct, array
  try:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect('/tmp/drmfd.sock')
    _, ancdata, _, _ = s.recvmsg(8, socket.CMSG_SPACE(4))
    for level, typ, data in ancdata:
      if level == socket.SOL_SOCKET and typ == socket.SCM_RIGHTS:
        fd = struct.unpack('i', data[:4])[0]
        os.set_inheritable(fd, True)
        os.environ['DRM_FD'] = str(fd)
        _drm_broker_conn = s
        return
    s.close()
  except Exception:
    cloudlog.exception("DRM fd acquisition failed")


def manager_init() -> None:
  _acquire_drm_fd()
  normalize_tree(BASEDIR)
  heal_param_perms()
  save_bootlog()

  # loggerd isn't running yet, so any leftover .lock marks an unclean shutdown:
  # unlock those segments and preserve them (dashcam footage from power cuts)
  try:
    recover_unclean_segments()
  except Exception:
    cloudlog.exception("recover_unclean_segments failed")

  try:
    from iqpilot.system.hardware import TICI
    if TICI:
      from iqpilot.system.hardware.tici.usb_storage import ensure_ncm_gadget, suspend_usb_input
      ensure_ncm_gadget()
      if Params().get_bool("IQEmacEnabled") or Params().get_bool("IQEgpuEnabled"):
        suspend_usb_input(True)
  except Exception:
    cloudlog.exception("emac usb setup failed")

  build_metadata = get_build_metadata()

  params = Params()
  params.clear_all(ParamKeyFlag.CLEAR_ON_MANAGER_START)
  params.clear_all(ParamKeyFlag.CLEAR_ON_ONROAD_TRANSITION)
  params.clear_all(ParamKeyFlag.CLEAR_ON_OFFROAD_TRANSITION)
  params.clear_all(ParamKeyFlag.CLEAR_ON_IGNITION_ON)
  if build_metadata.release_channel:
    params.clear_all(ParamKeyFlag.DEVELOPMENT_ONLY)

  # device boot mode
  if params.get("DeviceBootMode") == 1:  # start in Always Offroad mode
    params.put_bool("IQAlwaysOffroad", True)

  if params.get_bool("RecordFrontLock"):
    params.put_bool("RecordFront", True)

  # set unset params to their default value
  initialized_defaults = {}
  for k in params.all_keys():
    default_value = params.get_default_value(k)
    if default_value is not None and params.get(k) is None:
      params.put(k, default_value)
    if default_value is not None:
      initialized_defaults[k] = params.get(k)

  try:
    from iqpilot.selfdrive.iqmodeld.models.helpers import initialize_active_model
    initialize_active_model(params)
  except Exception:
    cloudlog.exception("failed to seed default model bundle")
  for k, value in initialized_defaults.items():
    if value is not None and params.get(k) is None:
      params.put(k, value)

  from iqpilot.common.ui_defaults import apply_ui_defaults
  apply_ui_defaults(params)

  apply_curvature_controller_default(params)

  # Create folders needed for msgq
  try:
    os.mkdir(Paths.shm_path())
  except FileExistsError:
    pass
  except PermissionError:
    print(f"WARNING: failed to make {Paths.shm_path()}")

  # set params
  serial = HARDWARE.get_serial()
  params.put("Version", build_metadata.openpilot.version)
  params.put("GitCommit", build_metadata.openpilot.git_commit)
  params.put("GitCommitDate", build_metadata.openpilot.git_commit_date)
  params.put("GitBranch", build_metadata.channel)
  params.put("GitRemote", build_metadata.openpilot.git_origin)
  params.put_bool("IsDevelopmentBranch", build_metadata.development_channel)
  params.put_bool("IsTestedBranch", build_metadata.tested_channel)
  params.put_bool("IsReleaseBranch", build_metadata.release_channel)
  params.put_bool("IsReleaseIqBranch", build_metadata.release_channel)
  params.put("HardwareSerial", serial)

  # set dongle id
  reg_res = register(show_spinner=True)
  if reg_res:
    dongle_id = reg_res
  else:
    raise Exception(f"Registration failed for device {serial}")
  os.environ['DONGLE_ID'] = dongle_id  # Needed for swaglog
  os.environ['GIT_ORIGIN'] = build_metadata.openpilot.git_normalized_origin # Needed for swaglog
  os.environ['GIT_BRANCH'] = build_metadata.channel # Needed for swaglog
  os.environ['GIT_COMMIT'] = build_metadata.openpilot.git_commit # Needed for swaglog

  if not build_metadata.openpilot.is_dirty:
    os.environ['CLEAN'] = '1'

  # init logging
  sentry.init(sentry.SentryProject.SELFDRIVE)
  cloudlog.bind_global(dongle_id=dongle_id,
                       version=build_metadata.openpilot.version,
                       origin=build_metadata.openpilot.git_normalized_origin,
                       branch=build_metadata.channel,
                       commit=build_metadata.openpilot.git_commit,
                       dirty=build_metadata.openpilot.is_dirty,
                       device=HARDWARE.get_device_type())

  # preimport all processes
  for p in managed_processes.values():
    p.prepare()


def manager_cleanup() -> None:
  # send signals to kill all procs
  for p in managed_processes.values():
    p.stop(block=False)

  # ensure all are killed
  for p in managed_processes.values():
    p.stop(block=True)

  cloudlog.info("everything is dead")


def manager_thread() -> None:
  cloudlog.bind(daemon="manager")
  cloudlog.info("manager start")
  cloudlog.info({"environ": os.environ})

  params = Params()

  ignore: list[str] = []
  if params.get("DongleId") in (None, UNREGISTERED_DONGLE_ID):
    ignore += ["manage_hephaestusd", "iquploaderd"]
  if os.getenv("NOBOARD") is not None:
    ignore.append("pandad")
  ignore += [x for x in os.getenv("BLOCK", "").split(",") if len(x) > 0]

  sm = messaging.SubMaster(['deviceState', 'carParams', 'pandaStates', 'modelV2'], poll='deviceState')
  pm = messaging.PubMaster(['managerState'])

  write_onroad_params(False, params)
  ensure_running(managed_processes.values(), False, params=params, CP=sm['carParams'], not_run=ignore)

  started_prev = False
  ignition_prev = False
  running_prev = None
  modeld_deadline = None

  while True:
    sm.update(1000)

    started = sm['deviceState'].started

    if started and not started_prev:
      try:
        HARDWARE.set_power_save(False)
      except Exception:
        cloudlog.exception("failed to leave power save on onroad transition")
      params.clear_all(ParamKeyFlag.CLEAR_ON_ONROAD_TRANSITION)
    elif not started and started_prev:
      params.clear_all(ParamKeyFlag.CLEAR_ON_OFFROAD_TRANSITION)

    ignition = any(ps.ignitionLine or ps.ignitionCan for ps in sm['pandaStates'] if ps.pandaType != log.PandaState.PandaType.unknown)
    if ignition and not ignition_prev:
      params.clear_all(ParamKeyFlag.CLEAR_ON_IGNITION_ON)

    # update onroad params, which drives pandad's safety setter thread
    if started != started_prev:
      write_onroad_params(started, params)

    started_prev = started
    ignition_prev = ignition

    ensure_running(managed_processes.values(), started, params=params, CP=sm['carParams'], not_run=ignore)
    modeld_deadline = update_modeld_watchdog(
      modeld_deadline, started, sm.updated['modelV2'], managed_processes['iqmodeld'], time.monotonic()
    )

    # print only on change (reprinting every loop floods the shared tmux); always logged
    procs = [p for p in managed_processes.values() if p.proc]
    running = ' '.join(
      ("\u001b[32m{}\u001b[0m".format(p.name) if p.proc.is_alive()
       else "\u001b[1;31m\u2717 {}\u001b[0m".format(p.name))
      for p in procs)
    cloudlog.debug(running)
    alive = tuple(p.proc.is_alive() for p in procs)
    if alive != running_prev:
      print(running)
      running_prev = alive

    # send managerState
    msg = messaging.new_message('managerState', valid=True)
    msg.managerState.processes = [p.get_process_state_msg() for p in managed_processes.values()]
    pm.send('managerState', msg)

    # kick AGNOS power monitoring watchdog
    try:
      if sm.all_checks(['deviceState']):
        with atomic_write("/var/tmp/power_watchdog", "w", overwrite=True) as f:
          f.write(str(time.monotonic()))
    except Exception:
      pass

    # Exit main loop when uninstall/shutdown/reboot is needed
    shutdown = False
    for param in ("DoUninstall", "DoShutdown", "DoReboot"):
      if params.get_bool(param):
        shutdown = True
        params.put("LastManagerExitReason", f"{param} {datetime.datetime.now()}")
        cloudlog.warning(f"Shutting down manager - {param} set")

    if shutdown:
      break


def main() -> None:
  manager_init()
  if os.getenv("PREPAREONLY") is not None:
    return

  # SystemExit on sigterm
  signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(1))

  try:
    manager_thread()
  except Exception:
    traceback.print_exc()
    sentry.capture_exception()
  finally:
    manager_cleanup()

  params = Params()
  if params.get_bool("DoUninstall"):
    cloudlog.warning("uninstalling")
    HARDWARE.uninstall()
  elif params.get_bool("DoReboot"):
    cloudlog.warning("reboot")
    HARDWARE.reboot()
  elif params.get_bool("DoShutdown"):
    cloudlog.warning("shutdown")
    HARDWARE.shutdown()


if __name__ == "__main__":
  if not (os.getenv("SIMULATION") and sys.platform == "darwin"):
    unblock_stdout()

  try:
    main()
  except KeyboardInterrupt:
    print("got CTRL-C, exiting")
  except Exception:
    add_file_handler(cloudlog)
    cloudlog.exception("Manager failed to start")

    try:
      managed_processes['ui'].stop()
    except Exception:
      pass

    # Show last 3 lines of traceback
    error = traceback.format_exc(-3)
    error = "Manager failed to start\n\n" + error
    with TextWindow(error) as t:
      t.wait_for_exit()

    raise

  # manual exit because we are forked
  sys.exit(0)
