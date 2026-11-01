"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""
from __future__ import annotations

import os
import sys
import tempfile

DEFAULT_ARCH = "gfx1200"
MOCK_DEV = "MOCKUSB+AMD:LLVM"


def tinygrad_tree() -> str:
  override = os.environ.get("IQ_TINYGRAD_TREE")
  if override:
    return override
  here = os.path.dirname(os.path.abspath(__file__))
  root = os.path.abspath(os.path.join(here, "..", "..", "..", ".."))
  return os.path.join(root, "components", "tinygrad")


def activate(arch: str = DEFAULT_ARCH, execute: bool = False) -> None:
  assert "tinygrad" not in sys.modules, "egpu_host_mock.activate must run before tinygrad is imported"
  os.environ["DEV"] = f"{MOCK_DEV}:{arch}"
  if not execute:
    # the wait passthrough below lets a pipelined copyin's SDMA copy run before its USB data lands, so weights read back wrong
    os.environ["AMD_USB_PIPELINED_COPYIN"] = "0"
  tree = tinygrad_tree()
  if tree not in sys.path:
    sys.path.insert(0, tree)
  from test.mockgpu.am import amgpu

  # The mock dock models 512MB VRAM; big-model weights alone exceed that. Must be set before amdriver binds it.
  amgpu.VRAM_SIZE = int(os.environ.get("IQ_MOCK_VRAM_GB", "4")) << 30
  from tinygrad.runtime.autogen import libc
  if sys.platform == "darwin":
    # A Homebrew-LLVM gfx1200 kernel (no s_code_end padding) hung a real dock; ship only container-built artifacts.
    print("egpu_host_mock: native macOS LLVM output is for tests only; use scripts/iqpilot/host_egpu_compile_docker.sh for artifacts",
          file=sys.stderr)

    def memfd_create(name, flags):
      fd, path = tempfile.mkstemp(prefix=b"iq_mock_" + bytes(name) + b"_")
      os.unlink(path)
      return fd
    libc.memfd_create = memfd_create
    if not hasattr(libc, "MFD_CLOEXEC"):
      libc.MFD_CLOEXEC = 1
  if not execute:
    import ctypes
    from test.mockgpu.amd import amdgpu
    amdgpu.remu.run_asm = lambda *args, **kwargs: 0
    pm4_wait = amdgpu.PM4Executor._exec_wait_reg_mem
    sdma_poll = amdgpu.SDMAExecutor._execute_poll_regmem

    # Without kernel execution no memory wait carries information; a blocked wait would need a host write to re-poll it.
    def pm4_wait_passthrough(self, n):
      if not pm4_wait(self, n):
        self.rptr[0] += 7
      return True

    def sdma_poll_passthrough(self):
      if not sdma_poll(self):
        self.rptr[0] += ctypes.sizeof(amdgpu.sdma_pkts.poll_regmem)
      return True
    amdgpu.PM4Executor._exec_wait_reg_mem = pm4_wait_passthrough
    amdgpu.SDMAExecutor._execute_poll_regmem = sdma_poll_passthrough
