# Copyright (c) 2026 IQ.Lvbs. All rights reserved.
import bz2
from types import SimpleNamespace

import numpy as np
import pytest

from iqpilot.cereal import messaging
from iqpilot.tools.clip import run


@pytest.fixture
def cameras(tmp_path, mocker):
  opened = []

  class Container:
    def __init__(self, source):
      self.source, self.closed, self.decoded = source, False, 0
      opened.append(self)

    def __enter__(self):
      return self

    def __exit__(self, *args):
      self.closed = True

    def decode(self, video):
      for index in range(60):
        self.decoded += 1
        data = np.full((3, 2), index, dtype=np.uint8)
        yield SimpleNamespace(width=2, height=2, to_ndarray=lambda data=data, **kwargs: data)

  mocker.patch.object(run.av, 'open', side_effect=Container)
  paths = [tmp_path / f'{i}.ts' for i in range(2)]
  for path in paths:
    path.write_bytes(b'fixture')
  return [str(path) for path in paths], opened


def test_qcamera_segment_boundary(cameras):
  paths, opened = cameras
  callbacks = []
  frames = list(run.iter_segment_frames(paths, 59, 62, fps=1, use_qcam=True, frame_size=(2, 2),
                                        on_segment_open=lambda *args: callbacks.append(args)))
  assert [index for index, _ in frames] == [59, 60, 61]
  assert [data.tolist() for _, data in frames] == [[59]*6, [0]*6, [1]*6]
  assert callbacks == list(enumerate(paths))
  assert [container.decoded for container in opened] == [60, 2]
  assert all(container.closed and container.source.closed for container in opened)


def test_qcamera_early_close(cameras):
  paths, opened = cameras
  frames = run.iter_segment_frames(paths, 0, 60, fps=1, use_qcam=True)
  next(frames)
  frames.close()
  assert opened[0].decoded == 1
  assert opened[0].closed and opened[0].source.closed


def test_qcamera_short_segment(cameras):
  paths, opened = cameras
  with pytest.raises(IndexError, match='Missing camera frame 60'):
    list(run.iter_segment_frames(paths, 29, 31, fps=2, use_qcam=True))
  assert all(container.closed and container.source.closed for container in opened)


def test_log_pipeline_order(tmp_path):
  paths = []
  for index in range(3):
    path = tmp_path / f'{index}.bz2'
    message = messaging.new_message('carState')
    message.logMonoTime = 1_000_000_000 * (index + 1)
    message.carState.vEgo = float(index)
    path.write_bytes(bz2.compress(b'' if index == 1 else message.to_bytes()))
    paths.append(str(path))
  chunks = run.load_logs_parallel(paths)
  assert [chunk['carState'].carState.vEgo for chunk in chunks] == [0., 2.]
  with pytest.raises(FileNotFoundError):
    run.load_logs_parallel([str(tmp_path / 'missing.bz2')])
