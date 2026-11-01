# Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
from iqpilot._proprietary_loader import ProprietaryModuleMissing, load_private_module

try:
  load_private_module(__name__, "iqpilot_private.context.model_context")
except ProprietaryModuleMissing:
  def context_model(input_shapes):
    return False

  def device_context_inputs(numpy_inputs, sub, frame_timestamp_eof_ns):
    return {}
