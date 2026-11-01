from iqpilot.cereal import log
from iqpilot.system.hardware.base import HardwareBase

NetworkType = log.DeviceState.NetworkType


class Pc(HardwareBase):
  def get_device_type(self):
    return "pc"

  def get_network_type(self):
    return NetworkType.wifi
