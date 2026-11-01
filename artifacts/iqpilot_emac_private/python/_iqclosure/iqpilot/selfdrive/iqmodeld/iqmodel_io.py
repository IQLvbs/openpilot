# Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
from iqpilot._proprietary_loader import ProprietaryModuleMissing, load_private_module

try:
  load_private_module(__name__, "iqpilot_private.context.iqmodel_io")
except ProprietaryModuleMissing:
  def publish_outputs(model_v2, iq_data, outputs) -> None:
    pass

  def fill_reasoning(iq_data, outputs, reasoning_wire, frame_id, mono_ns) -> bool:
    return False

  class IQModelIO:
    services: list = []
    carry: dict = {}

    def __init__(self, runner):
      pass

    def compose(self, numpy_inputs, sub, params, frame_timestamp_eof_ns, vehicle_speed) -> dict:
      return {}

    def memory_reset(self, sub, frame_id, timestamp_ns) -> bool:
      return False

    def publish(self, model_v2, iq_data, outputs, frame_id, mono_ns) -> None:
      pass
