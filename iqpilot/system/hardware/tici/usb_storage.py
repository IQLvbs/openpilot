import os
import subprocess

from iqpilot.common.params import Params

SCRIPT_NAME = "set_usb_storage.sh"


def _script_path() -> str:
  # signed bundles vendor this module without the shell script beside it
  sibling = os.path.join(os.path.dirname(os.path.abspath(__file__)), SCRIPT_NAME)
  if os.path.isfile(sibling):
    return sibling
  return os.path.join(os.environ.get("BASEDIR", "/data/iqpilot"), "iqpilot", "system", "hardware", "tici", SCRIPT_NAME)


SCRIPT_PATH = _script_path()


def apply_usb_storage_state(state: bool):
  Params().put_bool("UsbStorageEnabled", state)
  try:
    args = ["sudo", SCRIPT_PATH]
    if state:
      args.append("--rebuild")
    subprocess.Popen(args)
  except OSError:
    pass


NCM_TRIED_MARKER = "/tmp/.iqemac_ncm_provisioned"
MAX_NCM_ATTEMPTS = 3


def _ncm_attempts() -> int:
  try:
    with open(NCM_TRIED_MARKER) as f:
      return int(f.read().strip() or 0)
  except (OSError, ValueError):
    return 0


def ensure_ncm_gadget() -> bool:
  # configfs is RAM backed, so the gadget must be rebuilt every boot; binding it
  # enumerates on the host, which must not happen mid-drive
  if os.path.isdir("/sys/class/net/usb0"):
    return True
  params = Params()
  from iqpilot.selfdrive.iqmodeld.egpu_helpers import egpu_selected
  if not (params.get_bool("IQEmacEnabled") or egpu_selected(params)):
    return False
  attempts = _ncm_attempts()
  if attempts >= MAX_NCM_ATTEMPTS:
    return False
  try:
    # stamped before the run: the gadget build can wedge configfs in
    # uninterruptible D state, and retrying that forever helps nobody
    with open(NCM_TRIED_MARKER, "w") as f:
      f.write(str(attempts + 1))
    subprocess.run(["sudo", "-n", SCRIPT_PATH], timeout=120, check=False)
  except (OSError, subprocess.SubprocessError):
    return False
  return os.path.isdir("/sys/class/net/usb0")


INPUT_SUSPEND = "/sys/class/power_supply/battery/input_suspend"


def suspend_usb_input(suspend: bool = True) -> bool:
  # a host on the data port makes the PMIC sink USB-PD while OBD-C feeds the same
  # rail; the SOM browns out. This closes the charge path only, data is untouched
  if not os.path.exists(INPUT_SUSPEND):
    return False
  try:
    with open(INPUT_SUSPEND) as f:
      if f.read().strip() == ("1" if suspend else "0"):
        return True
  except OSError:
    pass
  rc = subprocess.run(["sudo", "-n", "sh", "-c", f"echo {int(suspend)} > {INPUT_SUSPEND}"],
                      check=False, capture_output=True)
  return rc.returncode == 0
