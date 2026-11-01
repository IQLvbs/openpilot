"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""

from iqpilot.system.hardware.usb import EGPU_DOCK_FW_PRODUCT, is_egpu_usb_device

EGPU_POWERED_VOLTAGE = 5000
GPU_TEMP_LIMIT = 110.
MEMORY_TEMP_LIMIT = 108.
TEMP_HYSTERESIS = 5.
FAN_START_GPU_TEMP = 60.
FAN_STOP_GPU_TEMP = 50.
FAN_START_MEMORY_TEMP = 70.
FAN_STOP_MEMORY_TEMP = 60.
FAN_STALLED_RPM = 250


class EgpuDockStatus:
  def __init__(self):
    self.offroad = True
    self.pcie_failed = False
    self.power_lost = False
    self.power_restored = False
    self.link_failures = 0
    self.model_loading_seen = False
    self.model_attempted = False
    self.overheated = False
    self.fans_obstructed = False
    self.usb_seen = False
    self.usb_failed = False

  def update(self, offroad, usb_state, firmware_failed, model_loading, model_active, compiled, state, set_alert):
    detected = [d for d in usb_state if is_egpu_usb_device(d["vendorId"], d["productId"], include_bootloader=True)]
    devices = [d for d in detected if is_egpu_usb_device(d["vendorId"], d["productId"])]
    firmware_ok = len(devices) == 1 and devices[0]["product"] == EGPU_DOCK_FW_PRODUCT

    if self.offroad and not offroad:
      self.pcie_failed = False
      self.power_lost = False
      self.power_restored = False
      self.link_failures = 0
      self.model_loading_seen = False
      self.model_attempted = False
      self.usb_seen = firmware_ok
      self.usb_failed = False

    self.model_loading_seen |= model_loading
    self.model_attempted |= self.model_loading_seen and not model_loading and model_active is not None

    if not offroad and self.usb_seen and not firmware_ok:
      self.usb_failed = True

    if not offroad and self.model_attempted and state is not None:
      power_lost = state.supplyFault or state.supplyVoltage < EGPU_POWERED_VOLTAGE
      self.link_failures = self.link_failures + 1 if state.pcieLtssm != 0x78 else 0
      self.pcie_failed |= self.link_failures >= 2 or power_lost
      self.power_lost |= power_lost

    if self.pcie_failed and self.power_lost and state is not None:
      self.power_restored |= not state.supplyFault and state.supplyVoltage >= EGPU_POWERED_VOLTAGE
    if self.usb_failed:
      self.pcie_failed = False
      self.power_lost = False
      self.power_restored = False

    if state is not None:
      gpu_limit = GPU_TEMP_LIMIT - (TEMP_HYSTERESIS if self.overheated else 0.)
      memory_limit = MEMORY_TEMP_LIMIT - (TEMP_HYSTERESIS if self.overheated else 0.)
      self.overheated = state.tempC >= gpu_limit or state.memoryTempC >= memory_limit
      fan_hot = (state.tempC >= (FAN_STOP_GPU_TEMP if self.fans_obstructed else FAN_START_GPU_TEMP) or
                 state.memoryTempC >= (FAN_STOP_MEMORY_TEMP if self.fans_obstructed else FAN_START_MEMORY_TEMP))
      self.fans_obstructed = fan_hot and state.fanSpeedRpm < FAN_STALLED_RPM

    slow_usb = offroad and len(devices) == 1 and devices[0]["speedMbps"] < 5000
    set_alert("Offroad_EgpuNotDetected", self.usb_failed)
    set_alert("Offroad_EgpuFansObstructed", self.fans_obstructed)
    set_alert("Offroad_EgpuOverheated", self.overheated)
    set_alert("Offroad_EgpuUsbSlow", slow_usb, f"{devices[0]['speedMbps']} Mbps" if slow_usb else None)
    if self.power_lost:
      pcie_action = "12V power was interrupted, possibly by engine start-stop. "
      pcie_action += ("Cycle ignition to reload the model." if self.power_restored else
                      "Check 12V, then cycle ignition to reload the model.")
    else:
      pcie_action = "Check 12V connection."
    set_alert("Offroad_EgpuPcieUnavailable", self.pcie_failed, pcie_action)
    set_alert("Offroad_EgpuUncompiled", offroad and firmware_ok and not compiled)
    set_alert("Offroad_EgpuUpdateFailed", offroad and firmware_failed)
    self.offroad = offroad
