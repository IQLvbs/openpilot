# Copyright (c) 2026 IQ.Lvbs. All rights reserved.
from pathlib import Path

import pytest

from iqpilot.common import file_chunker as chunker


@pytest.mark.parametrize('chunks', [None, [], [b''], [b'abc', b'', b'\x00\xfftail']])
@pytest.mark.parametrize('prefix', [None, 'compile_modeld_'])
def test_stage_bytes_and_cleanup(tmp_path, mocker, chunks, prefix):
  source = tmp_path / 'model.onnx'
  expected = b'plain\x00\xff' if chunks is None else b''.join(chunks)
  if chunks is None:
    source.write_bytes(expected)
  else:
    Path(chunker.get_manifest_path(source)).write_text(str(len(chunks)))
    for index, data in enumerate(chunks):
      Path(chunker.get_chunk_name(source, index, len(chunks))).write_bytes(data)
  destination = tmp_path / 'shared'
  destination.mkdir()
  register = mocker.patch.object(chunker.atexit, 'register')
  staged = Path(chunker.stage_file_chunked(source, destination, prefix))
  assert staged.read_bytes() == expected == chunker.read_file_chunked(source)
  assert staged.parent == destination
  assert staged.name == source.name if prefix is None else staged.name.startswith(prefix)
  register.call_args.args[0]()
  assert not staged.exists()


@pytest.mark.parametrize('prefix', [None, 'compile_modeld_'])
@pytest.mark.parametrize('failure', ['missing_chunk', 'read', 'write'])
def test_failed_staging_leaves_no_output(tmp_path, mocker, prefix, failure):
  source = tmp_path / 'model.onnx'
  Path(chunker.get_manifest_path(source)).write_text('2')
  Path(chunker.get_chunk_name(source, 0, 2)).write_bytes(b'first chunk')
  if failure != 'missing_chunk':
    Path(chunker.get_chunk_name(source, 1, 2)).write_bytes(b'second chunk')
    original_copy = chunker.shutil.copyfileobj

    def fail_copy(src, dst, length):
      if src.name.endswith('chunk02of02'):
        if failure == 'write':
          dst.write(b'partial')
        raise OSError(failure)
      return original_copy(src, dst, length)

    mocker.patch.object(chunker.shutil, 'copyfileobj', side_effect=fail_copy)
  register = mocker.patch.object(chunker.atexit, 'register')
  destination = tmp_path / 'shared'
  destination.mkdir()
  with pytest.raises(OSError):
    chunker.stage_file_chunked(source, destination, prefix)
  assert list(destination.iterdir()) == []
  register.assert_not_called()
