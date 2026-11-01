#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

from iqpilot.common.prefix import OpenpilotPrefix

command = [str(Path(sys.argv[1]).resolve()), *sys.argv[2:]]
with OpenpilotPrefix():
  ret = subprocess.call(command)

sys.exit(ret)
