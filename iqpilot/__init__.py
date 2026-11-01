"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos
"""
import sys
from pathlib import Path

_LOADER_SHIMS = Path(__file__).resolve().parents[1] / "scripts" / "iqpilot" / "ci_shims"
if sys.platform != "linux" and _LOADER_SHIMS.is_dir():
  __path__.append(str(_LOADER_SHIMS))
