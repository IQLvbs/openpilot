# Copyright (c) 2026 IQ.Lvbs. All rights reserved.
import numpy as np
import pytest

from iqpilot.tools.lib.framereader import DataUnreadableError, FrameReader


@pytest.fixture
def reader(tmp_path):
  video = tmp_path / 'video.hevc'
  video.write_bytes(b'abc')
  return FrameReader(str(video), index_data={
    'index': np.array([(2, 0), (1, 1), (1, 2), (0xFFFFFFFF, 3)], dtype=np.uint32),
    'global_prefix': b'', 'probe': {'streams': [{'width': 2, 'height': 2}]},
  })


@pytest.mark.parametrize('count', [0, 1, 2, 4])
def test_incomplete_or_excess_gop_fails(reader, mocker, count):
  mocker.patch('iqpilot.tools.lib.framereader.decompress_video_data', side_effect=[np.zeros((count, 2, 2, 3))])
  with pytest.raises(DataUnreadableError, match=f'decoded {count} frames, expected 3'):
    reader.get(2)


def test_indexed_frames_and_slice(reader, mocker):
  frames = np.arange(36, dtype=np.uint8).reshape(3, 2, 2, 3)
  mocker.patch('iqpilot.tools.lib.framereader.decompress_video_data', return_value=frames)
  for idx in (2, 0, 1):
    np.testing.assert_array_equal(reader.get(idx), frames[idx])
  selected = list(reader.decoder.get_iterator(0, 2, frame_skip=2))
  assert [idx for idx, _ in selected] == [0]
  np.testing.assert_array_equal(selected[0][1], frames[0])
