"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""
from iqpilot._proprietary_loader import ProprietaryModuleMissing, load_private_module

try:
  load_private_module(__name__, "iqpilot_private.models.emac_model_meta")
except ProprietaryModuleMissing:
  from iqpilot.models_private_src.emac_model_meta import *
