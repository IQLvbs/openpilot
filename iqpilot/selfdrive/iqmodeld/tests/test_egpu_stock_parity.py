"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""
import ctypes
import os
import struct
import subprocess
import sys
import types

import pytest

from iqpilot.cereal import log, messaging
from iqpilot.cereal.services import SERVICE_LIST


class TestTelemetryContract:
  def test_service_is_published_at_stock_cadence(self):
    assert "egpuDockState" in SERVICE_LIST
    assert SERVICE_LIST["egpuDockState"].frequency == 10.

  def test_message_carries_every_stock_field(self):
    msg = messaging.new_message("egpuDockState")
    state = msg.egpuDockState
    for field in ("tempC", "memoryTempC", "powerDrawW", "powerLimitW", "gpuUsagePercent",
                  "gpuClockMhz", "fanSpeedRpm", "pcieLtssm", "supplyVoltage", "supplyCurrent"):
      setattr(state, field, 1)
      assert getattr(state, field) == 1

  def test_metrics_refresh_matches_stock(self):
    from iqpilot.selfdrive.iqmodeld.egpu_telemetry import METRICS_REFRESH_EVERY
    assert METRICS_REFRESH_EVERY == 100

  def test_send_without_a_gpu_publishes_an_invalid_message(self):
    from iqpilot.selfdrive.iqmodeld import egpu_telemetry
    sent = []
    telemetry = egpu_telemetry.EgpuDockTelemetry(types.SimpleNamespace(send=lambda n, m: sent.append((n, m))), big=True)
    telemetry._device = lambda: types.SimpleNamespace(_opened_devices=set())
    telemetry.send()
    assert sent and sent[0][0] == "egpuDockState"
    assert sent[0][1].valid is False


class _Metrics(ctypes.Structure):
  _fields_ = [("AvgTemperature", ctypes.c_uint16 * 4), ("AverageSocketPower", ctypes.c_uint16),
              ("AverageGfxActivity", ctypes.c_uint16), ("AverageGfxclkFrequencyPostDs", ctypes.c_uint16), ("AvgFanRpm", ctypes.c_uint16)]


class _MetricsExternal(ctypes.Structure):
  _fields_ = [("SmuMetrics", _Metrics)]


def _fake_dock(table: bytes, calls: list):
  smu_mod = types.SimpleNamespace(SmuMetricsExternal_t=_MetricsExternal, TABLE_SMU_METRICS=7, PPSMC_MSG_TransferTableSmu2Dram=0x13,
                                  PPSMC_MSG_GetPptLimit=0x5d, TEMP_HOTSPOT=1, TEMP_MEM=2)

  def send_msg(msg, arg, read_back_arg=False, timeout=None):
    calls.append(msg)
    return 100 if read_back_arg else None

  def vram_view(paddr, size):
    calls.append(("vram", paddr, size))
    return memoryview(table[:size])

  def read_table(*args):
    raise AssertionError("read_table repeats the SMU table transfer")

  adev = types.SimpleNamespace(vram=types.SimpleNamespace(view=vram_view))
  smu = types.SimpleNamespace(smu_mod=smu_mod, adev=adev, driver_table_paddr=0x4000, _send_msg=send_msg, read_table=read_table)
  usb = types.SimpleNamespace(usb=types.SimpleNamespace(control_read=lambda req, n: struct.pack("<Hh?", 5100, 900, False)),
                              read=lambda addr, n: bytes([0x78]))
  return types.SimpleNamespace(iface=types.SimpleNamespace(dev_impl=types.SimpleNamespace(smu=smu), pci_dev=types.SimpleNamespace(usb=usb)))


class TestTelemetrySmuRead:
  def test_metrics_refresh_transfers_the_table_once(self):
    from iqpilot.selfdrive.iqmodeld import egpu_telemetry
    metrics = _MetricsExternal()
    metrics.SmuMetrics.AvgTemperature[1], metrics.SmuMetrics.AvgTemperature[2] = 61, 55
    metrics.SmuMetrics.AverageSocketPower, metrics.SmuMetrics.AverageGfxActivity = 42, 7
    calls: list = []
    amd = _fake_dock(bytes(metrics) + bytes(64), calls)

    class Device:
      _opened_devices = {"AMD"}

      def __class_getitem__(cls, key):
        return amd

    sent = []
    telemetry = egpu_telemetry.EgpuDockTelemetry(types.SimpleNamespace(send=lambda n, m: sent.append((n, m))), big=True)
    telemetry._device = lambda: Device
    telemetry.send()

    assert calls.count(0x13) == 1
    assert ("vram", 0x4000, ctypes.sizeof(_MetricsExternal)) in calls
    state = sent[0][1].egpuDockState
    assert sent[0][1].valid
    assert (state.tempC, state.memoryTempC, state.powerDrawW, state.gpuUsagePercent) == (61, 55, 42, 7)
    assert state.supplyVoltage == 5100 and state.pcieLtssm == 0x78


class TestDockPowerLimit:
  def test_limit_matches_stock(self):
    from iqpilot.selfdrive.iqmodeld.egpu_helpers import EGPU_POWER_LIMIT_W
    assert EGPU_POWER_LIMIT_W == 100

  def test_default_and_override(self, monkeypatch):
    from iqpilot.selfdrive.iqmodeld import egpu_helpers
    monkeypatch.setenv("AM_POWER_LIMIT", "")
    monkeypatch.delenv("AM_POWER_LIMIT")
    egpu_helpers.apply_dock_power_limit()
    assert os.environ["AM_POWER_LIMIT"] == str(egpu_helpers.EGPU_POWER_LIMIT_W)
    monkeypatch.setenv("AM_POWER_LIMIT", "150")
    egpu_helpers.apply_dock_power_limit()
    assert os.environ["AM_POWER_LIMIT"] == "150"

  def test_worker_sets_it_before_opening_the_dock(self):
    env = {k: v for k, v in os.environ.items() if k != "AM_POWER_LIMIT"}
    code = "import os, sys, iqpilot.selfdrive.iqmodeld.iqegpumodeld; assert 'tinygrad.runtime.ops_amd' not in sys.modules; print(os.environ['AM_POWER_LIMIT'])"
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().splitlines()[-1] == "100"

  def test_prefetch_sets_it_before_opening_the_dock(self, monkeypatch):
    from iqpilot.selfdrive.iqmodeld import egpu_prefetch
    monkeypatch.setenv("AM_POWER_LIMIT", "")
    monkeypatch.delenv("AM_POWER_LIMIT")
    seen = {}

    class FakeDevice:
      def __class_getitem__(cls, key):
        seen[key] = os.environ.get("AM_POWER_LIMIT")

    monkeypatch.setenv("DEV", "USB+AMD:LLVM")
    monkeypatch.setenv("GMMU", "0")
    monkeypatch.setattr(egpu_prefetch, "patch_tinygrad_fetch_fw", lambda: None)
    monkeypatch.setitem(sys.modules, "tinygrad.device", types.SimpleNamespace(Device=FakeDevice))
    egpu_prefetch._open_dock_once()
    assert seen == {"AMD": "100"}


class TestBigFrameFlag:
  def test_model_message_carries_the_big_flag(self):
    msg = messaging.new_message("modelV2")
    msg.modelV2.big = True
    assert msg.modelV2.big


class TestStatusParams:
  def test_loading_param_exists_and_is_cleared_like_stock(self):
    from pathlib import Path
    root = Path(__file__).resolve().parents[3]
    keys = (root / "common" / "params_keys.h").read_text()
    assert '{"UsbGpuLoading"' in keys
    line = next(ln for ln in keys.splitlines() if '"UsbGpuLoading"' in ln)
    for flag in ("CLEAR_ON_MANAGER_START", "CLEAR_ON_OFFROAD_TRANSITION", "CLEAR_ON_IGNITION_ON"):
      assert flag in line


class TestAlerts:
  def test_both_stock_big_model_events_exist(self):
    assert hasattr(log.OnroadEvent.EventName, "bigModelLoading")
    assert hasattr(log.OnroadEvent.EventName, "bigModelFailed")

  def test_alerts_are_wired_with_stock_severities(self):
    from iqpilot.selfdrive.selfdrived.events import EVENTS, ET
    EventName = log.OnroadEvent.EventName
    loading = EVENTS[EventName.bigModelLoading]
    failed = EVENTS[EventName.bigModelFailed]
    assert ET.NO_ENTRY in loading
    assert ET.SOFT_DISABLE in failed and ET.PERMANENT in failed


class TestFirmwareGate:
  def test_runtime_refuses_a_dock_on_other_firmware(self, tmp_path):
    from iqpilot.selfdrive.iqmodeld.egpu_helpers import usbgpu_present
    from iqpilot.system.hardware.usb import EGPU_DOCK_FW_PRODUCT, EGPU_DOCK_USB_IDS
    vid, pid = EGPU_DOCK_USB_IDS[0]
    d = tmp_path / "1-1"
    d.mkdir()
    (d / "idVendor").write_text(f"{vid:04x}\n")
    (d / "idProduct").write_text(f"{pid:04x}\n")
    (d / "product").write_text("custom deadbeef-CLEAN\n")
    assert not usbgpu_present(str(tmp_path))
    (d / "product").write_text(EGPU_DOCK_FW_PRODUCT + "\n")
    assert usbgpu_present(str(tmp_path))


class TestAutoFlash:
  def test_hardwared_drives_the_flasher_offroad_only(self):
    from iqpilot.system.hardware.hardwared import EgpuDockFlasher
    f = EgpuDockFlasher()
    calls = []
    f.flash = lambda: calls.append(1)
    stale = [{"vendorId": 0xADD1, "productId": 0x0001, "product": "custom deadbeef-CLEAN"}]
    f.update(False, stale)
    assert f.attempts == 0, "must not flash onroad"
    f.update(True, stale)
    assert f.attempts == 1
    if f.thread is not None:
      f.thread.join(timeout=5)

  def test_matching_firmware_is_never_flashed(self):
    from iqpilot.system.hardware.egpu_dock.flash import bundled_version
    from iqpilot.system.hardware.hardwared import EgpuDockFlasher
    f = EgpuDockFlasher()
    f.flash = lambda: pytest.fail("flashed a dock that already matches")
    f.update(True, [{"vendorId": 0xADD1, "productId": 0x0001, "product": bundled_version()}])
    assert f.attempts == 0

  def test_attempts_are_bounded_like_stock(self):
    from iqpilot.system.hardware.hardwared import EgpuDockFlasher
    assert EgpuDockFlasher.MAX_ATTEMPTS == 3
    assert EgpuDockFlasher.RETRY_INTERVAL == 20.


class TestDockIsItsOwnConsent:

  def _params(self, **flags):
    class P:
      def get_bool(self, k):
        return bool(flags.get(k, False))
      def get(self, k, *a, **kw):
        return None
    return P()

  def _sysfs_with_dock(self, tmp_path, product=None):
    from iqpilot.system.hardware.usb import EGPU_DOCK_FW_PRODUCT, EGPU_DOCK_USB_IDS
    vid, pid = EGPU_DOCK_USB_IDS[0]
    d = tmp_path / "1-1"
    d.mkdir()
    (d / "idVendor").write_text(f"{vid:04x}\n")
    (d / "idProduct").write_text(f"{pid:04x}\n")
    (d / "product").write_text((product or EGPU_DOCK_FW_PRODUCT) + "\n")
    return str(tmp_path)

  def test_a_plugged_in_dock_selects_itself(self, tmp_path):
    from iqpilot.selfdrive.iqmodeld.egpu_helpers import egpu_selected
    assert egpu_selected(self._params(), self._sysfs_with_dock(tmp_path))

  def test_nothing_plugged_in_selects_nothing(self, tmp_path):
    from iqpilot.selfdrive.iqmodeld.egpu_helpers import egpu_selected
    assert not egpu_selected(self._params(), str(tmp_path))

  def test_a_dock_on_foreign_firmware_does_not_select_itself(self, tmp_path):
    from iqpilot.selfdrive.iqmodeld.egpu_helpers import egpu_selected
    assert not egpu_selected(self._params(), self._sysfs_with_dock(tmp_path, "custom deadbeef-CLEAN"))

  def test_the_user_can_force_it_off(self, tmp_path):
    from iqpilot.selfdrive.iqmodeld.egpu_helpers import egpu_selected
    root = self._sysfs_with_dock(tmp_path)
    assert not egpu_selected(self._params(IQEgpuDisabled=True), root)

  def test_the_param_can_force_it_on_without_hardware(self, tmp_path):
    from iqpilot.selfdrive.iqmodeld.egpu_helpers import egpu_selected
    assert egpu_selected(self._params(IQEgpuEnabled=True), str(tmp_path))

  def test_present_dock_wins_even_with_emac_enabled(self, tmp_path):
    from iqpilot.selfdrive.iqmodeld.egpu_helpers import egpu_selected, resolve_backend, usbgpu_present
    root = self._sysfs_with_dock(tmp_path)
    assert resolve_backend(True, egpu_selected(self._params(), root), usbgpu_present(root)) == "egpu"

  def test_force_param_without_hardware_yields_to_emac(self, tmp_path):
    from iqpilot.selfdrive.iqmodeld.egpu_helpers import egpu_selected, resolve_backend, usbgpu_present
    assert resolve_backend(True, egpu_selected(self._params(IQEgpuEnabled=True), str(tmp_path)),
                           usbgpu_present(str(tmp_path))) == "emac"
