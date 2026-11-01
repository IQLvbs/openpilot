"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""
import subprocess
from pathlib import Path
EGPU_DOCK_USB_IDS = ((0xADD1, 0x0001), (0x3801, 0x0001))
EGPU_DOCK_ROM_USB_IDS = ((0x174C, 0x2464), (0x174C, 0x2463))
EGPU_DOCK_FW_PRODUCT = "custom ed4e39b7-CLEAN"

def is_egpu_usb_device(vendor_id: int, product_id: int, include_bootloader: bool = False) -> bool:
  ids = EGPU_DOCK_USB_IDS + EGPU_DOCK_ROM_USB_IDS if include_bootloader else EGPU_DOCK_USB_IDS
  return (vendor_id, product_id) in ids
USB_DEVICES_PATH = Path("/sys/bus/usb/devices")
UDC_PATH = Path("/sys/class/udc")
TYPEC_CC_ORIENTATION_PATH = Path("/sys/class/power_supply/usb/typec_cc_orientation")
USB3_LANES = {1: "a", 2: "b"}  # 0 = unattached
SOC_PLATFORM_PATH = Path("/sys/devices/platform/soc")
CONTROLLER_SUFFIX = ".ssusb"
LINK_ERRORS_FILE = "portli"


def read(path: Path) -> str | None:
  try:
    return path.read_text().strip()
  except Exception:
    return None


def read_int(path: Path, base: int = 10) -> int:
  try:
    return int(path.read_text(), base)
  except Exception:
    return 0


def read_hex_counter(path: Path) -> int:
  raw = read(path)
  if raw is None:
    return 0
  try:
    return int(raw, 0) if raw.lower().startswith("0x") else int(raw, 16)
  except ValueError:
    return 0


def get_usb_topology(root: Path = USB_DEVICES_PATH) -> set[str]:
  try:
    return {p.name for p in root.iterdir()}
  except Exception:
    return set()


def usb_devices(root: Path = USB_DEVICES_PATH) -> list[Path]:
  try:
    return sorted((d for d in root.glob("*") if (d / "idVendor").exists()), key=lambda p: p.name)
  except Exception:
    return []


def controller(device: Path) -> Path | None:
  try:
    return next((p for p in device.resolve().parents if p.name.endswith(CONTROLLER_SUFFIX)), None)
  except Exception:
    return None


def usb_controllers(soc: Path = SOC_PLATFORM_PATH) -> list[Path]:
  try:
    return sorted(soc.glob(f"*{CONTROLLER_SUFFIX}"))
  except Exception:
    return []


def link_controller(udc_root: Path = UDC_PATH) -> str:
  try:
    udc = next(iter(sorted(p.name for p in udc_root.iterdir())), "")
  except Exception:
    return ""
  return f"{udc.split('.')[0]}{CONTROLLER_SUFFIX}" if udc else ""


def usb3_lane(orientation: int | None = None) -> str:
  if orientation is None:
    orientation = read_int(TYPEC_CC_ORIENTATION_PATH)
  return USB3_LANES.get(orientation, "unknown")


def link_errors(ctrl: Path | None) -> int:
  return read_hex_counter(ctrl / LINK_ERRORS_FILE) if ctrl is not None else 0


def get_link_error_count(soc: Path = SOC_PLATFORM_PATH) -> int:
  return sum(link_errors(c) for c in usb_controllers(soc))


def host_role_controller(soc: Path = SOC_PLATFORM_PATH, udc_root: Path = UDC_PATH) -> Path | None:
  ctrl = link_controller(udc_root)
  return (soc / ctrl / "mode") if ctrl else None


def ensure_host_role(mode_path: Path | None = None) -> bool:
  path = mode_path if mode_path is not None else host_role_controller()
  if path is None:
    return False
  current = read(path)
  if current == "host":
    return True
  if current is None:
    return False
  rc = subprocess.run(["sudo", "-n", "sh", "-c", f"echo host > {path}"], check=False, capture_output=True)
  return rc.returncode == 0 and read(path) == "host"


def egpu_dock_present(root: Path = USB_DEVICES_PATH) -> bool:
  return any((read_int(d / "idVendor", 16), read_int(d / "idProduct", 16)) in EGPU_DOCK_USB_IDS
             for d in usb_devices(root))


def egpu_dock_ready(root: Path = USB_DEVICES_PATH) -> bool:
  return any((read_int(d / "idVendor", 16), read_int(d / "idProduct", 16)) in EGPU_DOCK_USB_IDS
             and (read(d / "product") or "").strip() == EGPU_DOCK_FW_PRODUCT
             for d in usb_devices(root))


def get_usb_state(root: Path = USB_DEVICES_PATH, udc_root: Path = UDC_PATH) -> list[dict]:
  devices = []
  lane, link_ctrl = usb3_lane(), link_controller(udc_root)
  for device in usb_devices(root):
    ctrl = controller(device)
    devices.append({
      "usb3Lane": lane if ctrl is not None and ctrl.name == link_ctrl else "unknown",
      "busnum": read_int(device / "busnum"),
      "devnum": read_int(device / "devnum"),
      "vendorId": read_int(device / "idVendor", 16),
      "productId": read_int(device / "idProduct", 16),
      "speedMbps": read_int(device / "speed"),
      "manufacturer": read(device / "manufacturer") or "",
      "product": read(device / "product") or "",
      "linkErrorCount": link_errors(ctrl) & 0xFFFF,
    })
  return devices


def set_usb_state(device_state, devices: list[dict], link_error_count: int = 0,
                  lane: str | None = None) -> None:
  entries = device_state.usbState.init('devices', len(devices))

  dock_present = False
  for entry, device in zip(entries, devices, strict=True):
    entry.busnum = device["busnum"]
    entry.devnum = device["devnum"]
    entry.vendorId = device["vendorId"]
    entry.productId = device["productId"]
    entry.speedMbps = device["speedMbps"]
    entry.manufacturer = device["manufacturer"]
    entry.product = device["product"]
    entry.linkErrorCount = device.get("linkErrorCount", 0) & 0xFFFF
    entry.usb3Lane = device.get("usb3Lane", "unknown")

    if (entry.vendorId, entry.productId) in EGPU_DOCK_USB_IDS:
      dock_present = True

  device_state.usbState.linkErrorCount = link_error_count
  device_state.usbState.usb3Lane = lane if lane is not None else usb3_lane()
  device_state.egpuDockPresent = dock_present
