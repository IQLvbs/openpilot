"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/

The eGPU dock flasher writes SPI flash, so the parts that decide WHETHER and
WHAT to write are pinned here. The transfer path itself needs the hardware; the
image validator, product parsing, config preservation and the needs-flash
decision do not, and those are what stop a bad write.
"""
import os
import zlib

import pytest

from iqpilot.system.hardware.egpu_dock import flash as f


def _wrap(body: bytes, *, magic=0xA5, checksum=None, crc=None, body_len=None) -> bytes:
  n = len(body) if body_len is None else body_len
  cs = (sum(body) & 0xFF) if checksum is None else checksum
  c = zlib.crc32(body).to_bytes(4, "little") if crc is None else crc
  return n.to_bytes(4, "little") + body + bytes([magic, cs]) + c


def test_bundled_firmware_is_valid_and_named():
  image = f.FIRMWARE_PATH.read_bytes()
  f.validate_image(image)                       # raises if the shipped blob is corrupt
  assert f.image_product(image).startswith("custom ")
  assert f.image_product(image).endswith("-CLEAN")
  assert f.bundled_version() == f.image_product(image)


def test_validate_rejects_corruption():
  body = b"custom deadbeef-CLEAN" + bytes(64)
  f.validate_image(_wrap(body))                 # the good case
  with pytest.raises(ValueError):
    f.validate_image(b"\x00" * 4)               # too short
  with pytest.raises(ValueError):
    f.validate_image(_wrap(body, magic=0x00))   # bad magic
  with pytest.raises(ValueError):
    f.validate_image(_wrap(body, checksum=0x00))
  with pytest.raises(ValueError):
    f.validate_image(_wrap(body, crc=b"\x00\x00\x00\x00"))
  with pytest.raises(ValueError):
    f.validate_image(_wrap(body, body_len=len(body) + 1))   # length disagrees
  with pytest.raises(ValueError):
    f.validate_image((f.MAX_CODE_SIZE + 1).to_bytes(4, "little") + bytes(16))


def test_image_product_requires_a_version_string():
  with pytest.raises(ValueError):
    f.image_product(b"no version here")


def test_saved_config_preserves_the_first_backup(tmp_path):
  # the config page is per-unit; a reflash must rewrite the ORIGINAL, never the
  # bytes read back from a half-written dock
  p = str(tmp_path / "dock.bin")
  original = bytes(range(256))
  assert f.saved_config(p, original) == original
  # later flash reads something different -> the stored original wins
  assert f.saved_config(p, bytes(256)) == original
  with open(p, "rb") as fh:
    assert fh.read() == original


def test_saved_config_rejects_wrong_size_backup(tmp_path):
  p = str(tmp_path / "dock.bin")
  with open(p, "wb") as fh:
    fh.write(b"\x00" * 8)
  with pytest.raises(RuntimeError):
    f.saved_config(p, bytes(256))


def test_needs_flash_only_for_a_dock_on_wrong_firmware():
  expected = f.bundled_version()
  vid, pid = (int(x, 16) for x in f.VID_PIDS[0])
  rom_vid, rom_pid = (int(x, 16) for x in f.ROM_VID_PIDS[0])

  assert not f.dock_needs_flash([])
  assert not f.dock_needs_flash([{"vendorId": 0x1234, "productId": 0x5678, "product": "something else"}])
  assert not f.dock_needs_flash([{"vendorId": vid, "productId": pid, "product": expected}])
  assert f.dock_needs_flash([{"vendorId": vid, "productId": pid, "product": "custom 00000000-CLEAN"}])
  # a ROM-mode board always needs flashing
  assert f.dock_needs_flash([{"vendorId": rom_vid, "productId": rom_pid, "product": f.ROM_PRODUCT}])


def test_both_shipped_ids_trigger_the_check():
  for pair in f.VID_PIDS:
    vid, pid = (int(x, 16) for x in pair)
    assert f.dock_needs_flash([{"vendorId": vid, "productId": pid, "product": "custom 00000000-CLEAN"}])


def test_rom_detection():
  assert f.in_rom_bootloader(f.ROM_VID_PIDS[0], "anything")
  assert f.in_rom_bootloader(("add1", "0001"), f.ROM_PRODUCT)
  assert f.in_rom_bootloader(("add1", "0001"), "AS2462something")
  assert not f.in_rom_bootloader(("add1", "0001"), f.bundled_version())
  assert not f.in_rom_bootloader(("add1", "0001"), None)


def test_config_paths_are_per_host_and_have_a_legacy_fallback():
  host = os.uname().nodename
  assert f.config_path().endswith(f"{host}.bin")
  assert f.config_path().startswith(f.CONFIG_DIR)
  # a dock flashed on this device by stock openpilot left its backup elsewhere
  assert f.legacy_config_path().startswith(f.LEGACY_CONFIG_DIR)
  assert f.legacy_config_path() != f.config_path()


def test_we_do_not_autoflash():
  # upstream flashes from hardwared automatically; ours must stay deliberate
  # until it has been validated against a real dock
  import subprocess
  root = os.path.join(os.path.dirname(f.__file__), "..", "..", "..")
  hits = subprocess.run(["grep", "-rnI", "--exclude-dir=__pycache__", "flash_dock", os.path.join(root, "system"),
                         os.path.join(root, "iqpilot")], capture_output=True, text=True).stdout
  callers = [ln for ln in hits.splitlines() if "egpu_dock/flash.py" not in ln and "test_" not in ln]
  assert callers == [], f"unexpected automatic flash caller: {callers}"


def test_runtime_fw_gate_is_pinned_to_the_bundled_firmware():
  from iqpilot.system.hardware.usb import EGPU_DOCK_FW_PRODUCT
  assert EGPU_DOCK_FW_PRODUCT == f.bundled_version()


def test_register_reads_default_to_superspeed_size():
  assert f.MAX_REGISTER_READ_SIZE == 255
  assert f.Flash().max_register_read_size == f.MAX_REGISTER_READ_SIZE


def test_link_up_is_false_without_a_dock():
  assert f.link_up() is False
