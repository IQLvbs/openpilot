"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""
import os
import subprocess
import sys

import pytest

from iqpilot.selfdrive.iqmodeld.tools.egpu_host_mock import tinygrad_tree

PROBE = """
import os
os.environ["JIT_BATCH_SIZE"] = "0"
from iqpilot.selfdrive.iqmodeld.tools.egpu_host_mock import activate
activate("gfx1200")
from tinygrad import Tensor
from tinygrad.device import Device
from tinygrad.engine.jit import TinyJit
dev = Device["AMD"]
assert dev.arch == "gfx1200", dev.arch
assert type(dev.iface).__name__ == "MOCKUSBIface", type(dev.iface).__name__
run = TinyJit(lambda x: (x * 2 + 1).sum(axis=1).realize())
for i in range(3):
  run(Tensor.ones(64, 64, device="AMD") * i)
import numpy as np
w = np.random.default_rng(0).integers(0, 256, (3 << 20) + 5, dtype=np.uint8)
assert (Tensor(w, device="AMD").realize().numpy() == w).all()
print("MOCK_OK")
"""


@pytest.mark.skipif(not os.path.isdir(os.path.join(tinygrad_tree(), "test", "mockgpu")), reason="tinygrad mockgpu tree not checked out")
def test_mock_dock_captures_a_jit_without_hardware():
  out = subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True, timeout=600)
  assert out.returncode == 0, out.stderr[-2000:]
  assert "MOCK_OK" in out.stdout
