"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/

deviceState.usbState carries a per-device USB bus snapshot plus the ssusb
link-error counter (IQ.OS 4.9.1+ `portli`) into every rlog. Drive it off a
synthetic sysfs tree so parsing, eGPU dock presence and the link-error
plumbing are pinned without hardware.
"""
from iqpilot.cereal import messaging
from iqpilot.system.hardware.usb import (
  EGPU_DOCK_FW_PRODUCT, EGPU_DOCK_ROM_USB_IDS, EGPU_DOCK_USB_IDS, controller, egpu_dock_present,
  egpu_dock_ready, get_link_error_count,
  get_usb_topology, get_usb_state, link_controller, read_hex_counter, set_usb_state, usb3_lane,
)


def _mkctrl(root, name="a800000.ssusb", portli="0x00000000"):
  """Platform controller dir, mirroring /sys/devices/platform/soc/<x>.ssusb."""
  ctrl = root / "soc" / name
  ctrl.mkdir(parents=True)
  if portli is not None:
    (ctrl / "portli").write_text(portli + "\n")
  return ctrl


def _mkdev(root, name, *, vid, pid, busnum=1, devnum=2, speed=5000,
           manufacturer="ACME", product="Widget", ctrl=None):
  """USB device under the controller, symlinked into the bus view like sysfs."""
  real = (ctrl / "usb1" / name) if ctrl is not None else (root / "bus" / name)
  real.mkdir(parents=True)
  (real / "idVendor").write_text(f"{vid:04x}\n")
  (real / "idProduct").write_text(f"{pid:04x}\n")
  (real / "busnum").write_text(f"{busnum}\n")
  (real / "devnum").write_text(f"{devnum}\n")
  (real / "speed").write_text(f"{speed}\n")
  (real / "manufacturer").write_text(manufacturer + "\n")
  (real / "product").write_text(product + "\n")

  bus = root / "bus"
  bus.mkdir(parents=True, exist_ok=True)
  link = bus / name
  if real != link:
    link.symlink_to(real)
  return real


def test_missing_sysfs_is_empty(tmp_path):
  assert get_usb_state(tmp_path / "nope") == []


def test_entries_without_idvendor_are_skipped(tmp_path):
  (tmp_path / "bus" / "usb1").mkdir(parents=True)  # a root hub dir with no idVendor
  _mkdev(tmp_path, "1-2", vid=0x1234, pid=0x5678)
  state = get_usb_state(tmp_path / "bus")
  assert len(state) == 1 and state[0]["vendorId"] == 0x1234


def test_fields_parsed_with_hex_ids(tmp_path):
  _mkdev(tmp_path, "1-2", vid=0x0BDA, pid=0x8153, busnum=3, devnum=7,
         speed=480, manufacturer="Realtek", product="USB 10/100 LAN")
  (dev,) = get_usb_state(tmp_path / "bus")
  assert dev == {
    "busnum": 3, "devnum": 7,
    "vendorId": 0x0BDA, "productId": 0x8153,
    "speedMbps": 480,
    "manufacturer": "Realtek", "product": "USB 10/100 LAN",
    "linkErrorCount": 0,  # no controller in this device's path
    "usb3Lane": "unknown",  # not on the type-C port's controller
  }


def test_unreadable_strings_default_empty(tmp_path):
  real = _mkdev(tmp_path, "1-2", vid=0x1, pid=0x2)
  (real / "manufacturer").unlink()
  (real / "product").unlink()
  (dev,) = get_usb_state(tmp_path / "bus")
  assert dev["manufacturer"] == "" and dev["product"] == ""


def test_hex_counter_parsing(tmp_path):
  f = tmp_path / "portli"
  f.write_text("0x00000000\n")
  assert read_hex_counter(f) == 0
  f.write_text("0x0000002a\n")
  assert read_hex_counter(f) == 42
  f.write_text("0000002a\n")          # bare hex, no 0x prefix
  assert read_hex_counter(f) == 42
  f.write_text("garbage\n")
  assert read_hex_counter(f) == 0
  assert read_hex_counter(tmp_path / "absent") == 0  # pre-4.9.1 IQ.OS


def test_controller_resolved_from_device(tmp_path):
  ctrl = _mkctrl(tmp_path)
  _mkdev(tmp_path, "1-2", vid=0x1, pid=0x2, ctrl=ctrl)
  assert controller(tmp_path / "bus" / "1-2") == ctrl.resolve()


def test_device_carries_its_controllers_link_errors(tmp_path):
  ctrl = _mkctrl(tmp_path, portli="0x0000000c")
  _mkdev(tmp_path, "1-2", vid=0x1234, pid=0x5678, ctrl=ctrl)
  (dev,) = get_usb_state(tmp_path / "bus")
  assert dev["linkErrorCount"] == 12


def test_controller_count_without_any_enumerated_device(tmp_path):
  # peripheral mode (eMac gadget link): the peer never enumerates on our side,
  # so the counter must still be readable off the controller
  _mkctrl(tmp_path, portli="0x00000005")
  assert get_usb_state(tmp_path / "bus") == []
  assert get_link_error_count(tmp_path / "soc") == 5


def test_link_errors_summed_across_controllers(tmp_path):
  _mkctrl(tmp_path, name="a800000.ssusb", portli="0x00000002")
  _mkctrl(tmp_path, name="a600000.ssusb", portli="0x00000003")
  assert get_link_error_count(tmp_path / "soc") == 5


def test_missing_portli_reads_zero(tmp_path):
  _mkctrl(tmp_path, portli=None)  # pre-4.9.1 kernel: file absent
  assert get_link_error_count(tmp_path / "soc") == 0


def test_set_usb_state_populates_message_and_flags_dock(tmp_path):
  ctrl = _mkctrl(tmp_path, portli="0x00000007")
  _mkdev(tmp_path, "1-1", vid=0x1234, pid=0x5678, speed=480, ctrl=ctrl)
  _mkdev(tmp_path, "1-2", vid=EGPU_DOCK_USB_IDS[0][0], pid=EGPU_DOCK_USB_IDS[0][1], speed=5000, ctrl=ctrl)

  msg = messaging.new_message('deviceState')
  set_usb_state(msg.deviceState, get_usb_state(tmp_path / "bus"), get_link_error_count(tmp_path / "soc"))

  devices = list(msg.deviceState.usbState.devices)
  assert len(devices) == 2
  assert {d.speedMbps for d in devices} == {480, 5000}
  assert all(d.linkErrorCount == 7 for d in devices)
  assert msg.deviceState.usbState.linkErrorCount == 7
  assert msg.deviceState.egpuDockPresent


def test_dock_absent_when_not_plugged(tmp_path):
  _mkdev(tmp_path, "1-1", vid=0x1234, pid=0x5678)
  msg = messaging.new_message('deviceState')
  set_usb_state(msg.deviceState, get_usb_state(tmp_path / "bus"))
  assert not msg.deviceState.egpuDockPresent


def test_both_shipped_dock_usb_ids_detected(tmp_path):
  # comma ships the dock under two VID/PIDs; only the first was known before
  for i, (vid, pid) in enumerate(EGPU_DOCK_USB_IDS):
    root = tmp_path / f"v{i}"
    _mkdev(root, "1-1", vid=vid, pid=pid)
    assert egpu_dock_present(root / "bus"), f"{vid:#06x}:{pid:#06x} not detected"


def test_dock_in_rom_mode_is_not_present(tmp_path):
  # bootloader/ROM state enumerates but cannot serve a GPU until flashed
  vid, pid = EGPU_DOCK_ROM_USB_IDS[0]
  _mkdev(tmp_path, "1-1", vid=vid, pid=pid)
  assert not egpu_dock_present(tmp_path / "bus")


def test_empty_bus_clears_flag():
  msg = messaging.new_message('deviceState')
  set_usb_state(msg.deviceState, [])
  assert len(msg.deviceState.usbState.devices) == 0
  assert msg.deviceState.usbState.linkErrorCount == 0
  assert not msg.deviceState.egpuDockPresent


def test_link_error_count_masked_to_16_bits(tmp_path):
  # the per-device field is UInt16 upstream; a wrapped counter must not overflow it
  ctrl = _mkctrl(tmp_path, portli="0x0001ffff")
  _mkdev(tmp_path, "1-2", vid=0x1, pid=0x2, ctrl=ctrl)
  (dev,) = get_usb_state(tmp_path / "bus")
  assert dev["linkErrorCount"] == 0xFFFF

  msg = messaging.new_message('deviceState')
  set_usb_state(msg.deviceState, get_usb_state(tmp_path / "bus"))
  assert list(msg.deviceState.usbState.devices)[0].linkErrorCount == 0xFFFF


def test_usb_topology_lists_bus_entries(tmp_path):
  _mkdev(tmp_path, "1-1", vid=0x1, pid=0x2)
  _mkdev(tmp_path, "1-2", vid=0x3, pid=0x4)
  assert {"1-1", "1-2"} <= get_usb_topology(tmp_path / "bus")
  assert get_usb_topology(tmp_path / "nope") == set()


def _mkudc(root, name="a600000.dwc3"):
  udc = root / "udc" / name
  udc.mkdir(parents=True, exist_ok=True)
  (udc / "state").write_text("not attached\n")
  return root / "udc"


def test_link_controller_derived_from_udc_not_hardcoded(tmp_path):
  # comma pins "a600000.ssusb"; we derive it, so another board still resolves
  assert link_controller(_mkudc(tmp_path)) == "a600000.ssusb"
  assert link_controller(_mkudc(tmp_path / "other", "a800000.dwc3")) == "a800000.ssusb"


def test_link_controller_absent_udc_is_empty(tmp_path):
  assert link_controller(tmp_path / "nope") == ""


def test_usb3_lane_mapping():
  assert usb3_lane(1) == "a"
  assert usb3_lane(2) == "b"
  assert usb3_lane(0) == "unknown"    # unattached
  assert usb3_lane(None if False else 7) == "unknown"


def test_port_lane_survives_gadget_mode(tmp_path):
  """The case upstream cannot report: in peripheral mode nothing enumerates on
  the link controller, so every Device row is 'unknown' while the eMac link is
  up. The port-level field still carries it."""
  ctrl = _mkctrl(tmp_path, name="a800000.ssusb")
  _mkdev(tmp_path, "1-1", vid=0x1234, pid=0x5678, ctrl=ctrl)   # panda, host controller
  _mkudc(tmp_path)                                             # gadget is a600000

  devices = get_usb_state(tmp_path / "bus", tmp_path / "udc")
  assert all(d["usb3Lane"] == "unknown" for d in devices), "no device sits on the gadget controller"

  msg = messaging.new_message('deviceState')
  set_usb_state(msg.deviceState, devices, 0, lane="b")
  assert msg.deviceState.usbState.usb3Lane == "b"
  assert all(d.usb3Lane == "unknown" for d in msg.deviceState.usbState.devices)


def test_device_on_link_controller_gets_the_lane(tmp_path):
  # host mode on the type-C port (eGPU dock): upstream's per-device field populates
  ctrl = _mkctrl(tmp_path, name="a600000.ssusb")
  _mkdev(tmp_path, "1-1", vid=EGPU_DOCK_USB_IDS[0][0], pid=EGPU_DOCK_USB_IDS[0][1], ctrl=ctrl)
  _mkudc(tmp_path)
  import iqpilot.system.hardware.usb as usbmod
  orig = usbmod.usb3_lane
  usbmod.usb3_lane = lambda orientation=None: "a"
  try:
    devices = get_usb_state(tmp_path / "bus", tmp_path / "udc")
  finally:
    usbmod.usb3_lane = orig
  assert devices[0]["usb3Lane"] == "a"


def test_dock_ready_requires_the_bundled_firmware_product(tmp_path):
  root = tmp_path
  vid, pid = EGPU_DOCK_USB_IDS[0]
  _mkdev(root, "1-1", vid=vid, pid=pid, product=EGPU_DOCK_FW_PRODUCT)
  assert egpu_dock_present(root / "bus")
  assert egpu_dock_ready(root / "bus")


def test_dock_on_foreign_firmware_is_present_but_not_ready(tmp_path):
  root = tmp_path
  vid, pid = EGPU_DOCK_USB_IDS[0]
  _mkdev(root, "1-1", vid=vid, pid=pid, product="custom deadbeef-CLEAN")
  assert egpu_dock_present(root / "bus")
  assert not egpu_dock_ready(root / "bus")


def test_rom_mode_dock_is_neither_present_nor_ready(tmp_path):
  root = tmp_path
  vid, pid = EGPU_DOCK_ROM_USB_IDS[0]
  _mkdev(root, "1-1", vid=vid, pid=pid, product="USB 3.2 PCIe TinyEnclosure")
  assert not egpu_dock_present(root / "bus")
  assert not egpu_dock_ready(root / "bus")


class TestEnsureHostRole:
  def test_already_host_needs_no_write(self, tmp_path):
    from iqpilot.system.hardware.usb import ensure_host_role
    mode = tmp_path / "mode"
    mode.write_text("host\n")
    assert ensure_host_role(mode)

  def test_missing_controller_is_false(self, tmp_path):
    from iqpilot.system.hardware.usb import ensure_host_role
    assert not ensure_host_role(tmp_path / "absent" / "mode")
