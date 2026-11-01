"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""
import hashlib
import os

from iqpilot.selfdrive.iqmodeld import egpu_helpers as eh


def test_fetch_fw_mirrors_and_serves_offline(tmp_path, monkeypatch):
  from tinygrad import helpers
  blob = os.urandom(4096)
  sha = hashlib.sha256(blob).hexdigest()
  calls = []

  def orig(path, name, sha256):
    calls.append((path, name))
    return blob

  monkeypatch.setattr(helpers, "fetch_fw", orig, raising=False)
  helpers.fetch_fw._iq_patched = False
  monkeypatch.setattr(eh, "FIRMWARE_MIRROR", str(tmp_path / "mirror"))
  eh.patch_tinygrad_fetch_fw()
  assert helpers.fetch_fw("amdgpu", "gc.bin", sha) == blob and calls == [("amdgpu", "gc.bin")]
  mirrored = tmp_path / "mirror" / "amdgpu" / "gc.bin"
  assert mirrored.read_bytes() == blob
  assert helpers.fetch_fw("amdgpu", "gc.bin", sha) == blob and len(calls) == 1
  mirrored.write_bytes(b"corrupt")
  assert helpers.fetch_fw("amdgpu", "gc.bin", sha) == blob and len(calls) == 2
  assert mirrored.read_bytes() == blob
