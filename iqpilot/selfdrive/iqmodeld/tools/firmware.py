"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos
"""

def patch_firmware_fetch() -> None:
  import hashlib
  import pathlib

  import zstandard
  from tinygrad import helpers

  if not hasattr(helpers, "fetch_fw"):
    return

  original_fetch = helpers.fetch_fw

  def fetch_fw(path, name, sha256):
    archive_path = pathlib.Path(f"/lib/firmware/{path}/{name}.zst")
    if archive_path.is_file():
      blob = zstandard.ZstdDecompressor().stream_reader(archive_path.read_bytes()).read()
      if hashlib.sha256(blob).hexdigest() == sha256:
        return blob
    return original_fetch(path, name, sha256)

  helpers.fetch_fw = fetch_fw
