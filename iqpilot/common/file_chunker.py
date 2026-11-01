#!/usr/bin/env python3
import sys
import atexit
import shutil
import tempfile
import math
import os
from pathlib import Path

CHUNK_SIZE = 45 * 1024 * 1024  # 45MB, under GitHub's 50MB limit

def get_chunk_name(name, idx, num_chunks):
  return f"{name}.chunk{idx+1:02d}of{num_chunks:02d}"

def get_manifest_path(name):
  return f"{name}.chunkmanifest"

def _chunk_paths(path, num_chunks):
  return [get_manifest_path(path)] + [get_chunk_name(path, i, num_chunks) for i in range(num_chunks)]

def get_chunk_targets(path, file_size):
  num_chunks = math.ceil(file_size / CHUNK_SIZE)
  return _chunk_paths(path, num_chunks)

def chunk_file(path, targets):
  manifest_path, *chunk_paths = targets
  with open(path, 'rb') as f:
    data = f.read()
  actual_num_chunks = max(1, math.ceil(len(data) / CHUNK_SIZE))
  assert len(chunk_paths) >= actual_num_chunks, f"Allowed {len(chunk_paths)} chunks but needs at least {actual_num_chunks}, for path {path}"
  for i, chunk_path in enumerate(chunk_paths):
    with open(chunk_path, 'wb') as f:
      f.write(data[i * CHUNK_SIZE:(i + 1) * CHUNK_SIZE])
  Path(manifest_path).write_text(str(len(chunk_paths)))
  os.remove(path)

def _file_paths(path):
  manifest_path = get_manifest_path(path)
  if os.path.isfile(manifest_path):
    num_chunks = int(Path(manifest_path).read_text().strip())
    yield from (get_chunk_name(path, i, num_chunks) for i in range(num_chunks))
  elif os.path.isfile(path):
    yield path
  else:
    raise FileNotFoundError(path)


def read_file_chunked(path):
  return b''.join(Path(filename).read_bytes() for filename in _file_paths(path))


def stage_file_chunked(path, directory, prefix=None):
  target = (tempfile.NamedTemporaryFile(prefix=prefix, dir=directory, delete=False) if prefix is not None
            else open(os.path.join(directory, os.path.basename(path)), "wb"))
  try:
    with target:
      for filename in _file_paths(path):
        with open(filename, "rb") as source:
          shutil.copyfileobj(source, target, length=1024 * 1024)
  except BaseException:
    Path(target.name).unlink(missing_ok=True)
    raise
  staged_path = target.name
  atexit.register(lambda: os.path.exists(staged_path) and os.remove(staged_path))
  return staged_path


if __name__ == "__main__":
  path = sys.argv[1]
  chunk_paths = get_chunk_targets(path, os.path.getsize(path))
  chunk_file(path, chunk_paths)
