# Copyright (c) 2026 IQ.Lvbs. All rights reserved.
import subprocess
import sys


def test_compiled_mpc_runtime_without_symbolic_dependencies():
  subprocess.run([sys.executable, '-c', '''
import sys
sys.modules['casadi'] = None
sys.modules['sympy'] = None
from iqpilot.selfdrive.controls.lib.lateral_mpc_lib.lat_mpc import LateralMpc
from iqpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LongitudinalMpc
assert LateralMpc().solver is not None
assert LongitudinalMpc().solver is not None
'''], check=True)
