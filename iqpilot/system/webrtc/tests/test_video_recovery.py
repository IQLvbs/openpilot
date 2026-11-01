# Copyright (c) 2026 IQ.Lvbs. All rights reserved.

import asyncio
import time

import pytest

from iqpilot.cereal import messaging
from iqpilot.system.webrtc.device import native_video


def frame(sequence, keyframe=False, age_ns=0, codec="livestreamH264"):
  event = messaging.new_message("livestreamRoadEncodeData")
  event.logMonoTime = time.monotonic_ns() - age_ns
  event.livestreamRoadEncodeData.idx.type = codec
  event.livestreamRoadEncodeData.idx.frameId = sequence
  event.livestreamRoadEncodeData.header = b"header" if keyframe else b""
  event.livestreamRoadEncodeData.data = b"\x00\x00\x00\x01" + (b"\x65" if keyframe else b"\x41") + bytes([sequence])
  return event


@pytest.fixture
def track(mocker):
  mocker.patch.object(native_video, "Params", return_value=mocker.Mock())
  mocker.patch.object(native_video.messaging, "sub_sock", return_value=mocker.Mock())
  native_video.LiveStreamVideoStreamTrack._kf_pending_count = 0
  instance = native_video.LiveStreamVideoStreamTrack("road")
  yield instance
  instance.stop()
  native_video.LiveStreamVideoStreamTrack._kf_pending_count = 0


def test_only_ordered_h264_topics_are_subscribed(track):
  native_video.messaging.sub_sock.assert_called_once_with("livestreamRoadEncodeData", conflate=False)
  track.switch_camera("wideRoad")
  assert track._candidate_topics == ["livestreamWideRoadEncodeData"]
  assert track._previous_frame_id is None


@pytest.mark.asyncio
async def test_gap_waits_for_new_keyframe(track, mocker):
  mocker.patch.object(native_video.messaging, "recv_one_or_none", side_effect=[frame(1, True), frame(2), frame(4), frame(5, True)])
  first = await track.recv()
  second = await track.recv()
  recovered = await track.recv()
  assert bytes(first).endswith(b"\x65\x01")
  assert bytes(second).endswith(b"\x41\x02")
  assert bytes(recovered).endswith(b"\x65\x05")
  assert first.pts < second.pts < recovered.pts


@pytest.mark.asyncio
async def test_stale_gop_is_skipped_until_fresh_idr(track, mocker):
  mocker.patch.object(native_video.messaging, "recv_one_or_none", side_effect=[frame(1, True, 1_000_000_000), frame(2), frame(3, True)])
  packet = await track.recv()
  assert bytes(packet).endswith(b"\x65\x03")


@pytest.mark.asyncio
async def test_hevc_cannot_enter_h264_connection(track, mocker):
  mocker.patch.object(native_video.messaging, "recv_one_or_none", side_effect=[frame(1, True, codec="fullHEVC"), frame(2, True)])
  packet = await track.recv()
  assert bytes(packet).endswith(b"\x65\x02")


@pytest.mark.asyncio
async def test_silent_camera_does_not_fall_back_to_recording(track, mocker):
  mocker.patch.object(native_video.messaging, "recv_one_or_none", return_value=None)
  with pytest.raises(TimeoutError):
    await asyncio.wait_for(track.recv(), .03)
  assert set(track._socks) == {"livestreamRoadEncodeData"}
