"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""
from iqpilot._proprietary_loader import ProprietaryModuleMissing, load_private_module

try:
  load_private_module(__name__, "iqpilot_private.context.aux_inputs")
except ProprietaryModuleMissing:
  class AuxModelInputs:
    def __init__(self, input_spec: dict):
      self.active = False

    def poll(self) -> None:
      pass

    def extras(self) -> dict:
      return {}
