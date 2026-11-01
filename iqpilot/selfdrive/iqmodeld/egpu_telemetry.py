"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""
from __future__ import annotations

import ctypes
import struct

from iqpilot.cereal import messaging
from iqpilot.common.swaglog import cloudlog

METRICS_REFRESH_EVERY = 100


class EgpuDockTelemetry:

  def __init__(self, pm, big: bool):
    self.pm = pm
    self.big = big
    self.valid = True
    self.sends = 0
    self.metrics: dict[str, float] = {}
    self._power_limit: int | None = None
    self._asm_usb = None
    self._asic_logged = False

  def _device(self):
    from tinygrad.device import Device
    return Device

  def _open_asm_usb(self):
    import usb1
    from iqpilot.system.hardware.usb import EGPU_DOCK_USB_IDS
    context = usb1.USBContext()
    for vendor_id, product_id in EGPU_DOCK_USB_IDS:
      handle = context.openByVendorIDAndProductID(vendor_id, product_id, skip_on_error=True)
      if handle is not None:
        return handle
    context.close()
    return None

  def _read_ina(self):
    Device = self._device()
    if "AMD" in Device._opened_devices and self._asm_usb is None:
      try:
        raw = Device["AMD"].iface.pci_dev.usb.usb.control_read(0xC0, 5)
        return struct.unpack("<Hh?", bytes(raw))
      except Exception:
        pass
    if self._asm_usb is None:
      self._asm_usb = self._open_asm_usb()
    if self._asm_usb is None:
      raise RuntimeError("no egpu ASM usb handle")
    try:
      raw = self._asm_usb.controlRead(0xC0, 0xC0, 0, 0, 5, timeout=100)
    except Exception:
      self._asm_usb = None
      raise
    return struct.unpack("<Hh?", bytes(raw))

  def power_limit(self, smu) -> int:
    if self._power_limit is None:
      self._power_limit = smu._send_msg(smu.smu_mod.PPSMC_MSG_GetPptLimit, 0, read_back_arg=True, timeout=100)
    return self._power_limit

  def _log_detail(self, metrics, smu) -> None:
    def g(name):
      return getattr(metrics, name, None)
    try:
      if not self._asic_logged:
        cloudlog.warning(f"iqegpumodeld dock smu={getattr(smu.smu_mod, '__name__', '?')}")
        self._asic_logged = True
      throttle = g("ThrottlingPercentage")
      cloudlog.event("egpu_dock_detail",
                     gfxclk_running=g("AverageGfxclkFrequencyPreDs"), gfxclk_idle=g("AverageGfxclkFrequencyPostDs"),
                     gfxclk_target=g("AverageGfxclkFrequencyTarget"), gfx_activity=g("AverageGfxActivity"),
                     uclk_activity=g("AverageUclkActivity"), pcie_busy=g("PCIeBusy"),
                     pcie_rate=g("PcieRate"), pcie_width=g("PcieWidth"),
                     throttle_max=(max(throttle) if throttle is not None else None),
                     board_w=g("AverageTotalBoardPower"))
    except Exception:
      pass

  def send(self) -> None:
    Device = self._device()
    msg = messaging.new_message("egpuDockState")
    state = msg.egpuDockState
    self.sends += 1

    if self.big and "AMD" in Device._opened_devices and self.sends % METRICS_REFRESH_EVERY == 1:
      try:
        smu = Device["AMD"].iface.dev_impl.smu
        metrics_t = smu.smu_mod.SmuMetricsExternal_t
        smu._send_msg(smu.smu_mod.PPSMC_MSG_TransferTableSmu2Dram, smu.smu_mod.TABLE_SMU_METRICS, timeout=100)
        metrics_buf = bytearray(smu.adev.vram.view(smu.driver_table_paddr, ctypes.sizeof(metrics_t))[:])
        metrics = metrics_t.from_buffer(metrics_buf).SmuMetrics
        self.metrics = {"tempC": metrics.AvgTemperature[smu.smu_mod.TEMP_HOTSPOT],
                        "memoryTempC": metrics.AvgTemperature[smu.smu_mod.TEMP_MEM],
                        "powerDrawW": metrics.AverageSocketPower,
                        "powerLimitW": self.power_limit(smu),
                        "gpuUsagePercent": metrics.AverageGfxActivity,
                        "gpuClockMhz": metrics.AverageGfxclkFrequencyPostDs,
                        "fanSpeedRpm": metrics.AvgFanRpm}
        self._log_detail(metrics, smu)
        self.valid = True
      except Exception:
        if self.valid:
          cloudlog.exception("egpu dock state read failed")
        self.valid = False
        self.metrics.clear()

    if self.big:
      for k, v in self.metrics.items():
        setattr(state, k, v)

    asm_valid = False
    try:
      state.supplyVoltage, state.supplyCurrent, state.supplyFault = self._read_ina()
      asm_valid = True
    except Exception:
      pass
    if "AMD" in Device._opened_devices:
      try:
        state.pcieLtssm = Device["AMD"].iface.pci_dev.usb.read(0xB450, 1)[0]
      except Exception:
        pass

    msg.valid = asm_valid and (not self.big or self.valid)
    self.pm.send("egpuDockState", msg)
