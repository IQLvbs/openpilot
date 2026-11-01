# Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos
from iqpilot._proprietary_loader import ProprietaryModuleMissing, load_private_module

try:
  load_private_module(__name__, "iqpilot_private.context.cot_decode")
except ProprietaryModuleMissing:
  def validate_reasoning_wire(metadata):
    return False

  def decode_reasoning(*args, **kwargs):
    return None

  class ReasoningWire:
    def __init__(self, metadata):
      self.metadata = metadata
      self.enabled = False
      self.last_frame_id = None
      self.debug_path = ''

    def update(self, output, frame_id, mono_ns):
      return None
