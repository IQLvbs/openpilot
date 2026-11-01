import asyncio
import os

import pytest

from iqpilot.system.webrtc.rtc import WebRTCOfferBuilder
from iqpilot.system.webrtc.rtc.stream import RTCSessionDescription
from iqpilot.system.webrtc.session import StreamSession


@pytest.mark.asyncio
@pytest.mark.parametrize("cameras", [["road"], ["road", "wideRoad"], ["road", "driver", "wideRoad"]])
async def test_each_negotiated_camera_receives_video(cameras):
  if not os.environ.get("CI"):
    return
  answer_session = None
  loop = asyncio.get_running_loop()
  received = {camera: asyncio.Event() for camera in cameras}

  async def connect(offer):
    nonlocal answer_session
    answer_session = StreamSession(offer.sdp, offer.video, [], [], [], debug_mode=True)
    answer = await answer_session.get_answer()
    mids = [track.mid() for track, _, _ in answer_session.stream._track_state]
    assert len(set(mids)) == len(cameras)
    answer_session.start()
    return RTCSessionDescription(answer.sdp, answer.type)

  builder = WebRTCOfferBuilder(connect, ice_servers=[])
  for camera in cameras:
    builder.offer_to_receive_video_stream(camera)
  stream = builder.stream()
  try:
    await asyncio.wait_for(stream.start(), 10)
    for camera, event in received.items():
      stream.get_incoming_video_track(camera).on_frame(lambda *_, event=event: loop.call_soon_threadsafe(event.set))
    await asyncio.wait_for(stream.wait_for_connection(), 10)
    await asyncio.wait_for(asyncio.gather(*(event.wait() for event in received.values())), 10)
  finally:
    await stream.stop()
    if answer_session is not None:
      await answer_session.stop_async()


@pytest.mark.asyncio
async def test_native_video_audio_and_data_channel():
  if not os.environ.get("CI"):
    return

  answer_session = None
  video_received = asyncio.Event()
  audio_received = asyncio.Event()

  async def connect(offer):
    nonlocal answer_session
    browser_offer = offer.sdp.replace("profile-level-id=42e01f", "profile-level-id=640c1f", 1)
    audio_offset = browser_offer.index("m=audio")
    browser_offer = browser_offer[:audio_offset] + browser_offer[audio_offset:].replace(
      "a=recvonly",
      "a=sendrecv\r\na=msid:ios-microphone ios-audio-track\r\na=ssrc:123456 cname:ios-audio\r\na=ssrc:123456 msid:ios-microphone ios-audio-track",
      1,
    )
    answer_session = StreamSession(browser_offer, offer.video, [], [], [], debug_mode=True)
    answer = await answer_session.get_answer()
    assert not any(line.startswith("m=video 0 ") for line in answer.sdp.splitlines())
    assert answer_session.stream._track_state[0][0] is answer_session.stream._negotiated_tracks["road"]
    answer_session.start()
    return RTCSessionDescription(answer.sdp, answer.type)

  builder = WebRTCOfferBuilder(connect, ice_servers=[])
  builder.offer_to_receive_video_stream("road")
  builder.offer_to_receive_audio_stream()
  builder.add_messaging()
  stream = builder.stream()

  try:
    await asyncio.wait_for(stream.start(), 10)
    stream.get_incoming_video_track("road").on_frame(lambda *_: video_received.set())
    stream.get_incoming_audio_track().on_frame(lambda *_: audio_received.set())
    await asyncio.wait_for(stream.wait_for_connection(), 10)
    await asyncio.wait_for(video_received.wait(), 10)
    await asyncio.wait_for(audio_received.wait(), 10)
    stream.get_messaging_channel().send('{"type":"timingSei","enabled":true}')
    await asyncio.sleep(0.1)
    assert answer_session is not None
    assert answer_session.video_tracks[0].timing_sei_enabled
  finally:
    await stream.stop()
    if answer_session is not None:
      await answer_session.stop_async()


@pytest.mark.asyncio
async def test_native_duplex_audio_negotiation():
  if not os.environ.get("CI"):
    return

  answer_session = None

  async def connect(offer):
    nonlocal answer_session
    sendrecv_offer = offer.sdp.replace(
      "a=recvonly",
      "a=sendrecv\r\na=msid:ios-microphone ios-audio-track\r\na=ssrc:123456 cname:ios-audio\r\na=ssrc:123456 msid:ios-microphone ios-audio-track",
      1,
    )
    answer_session = StreamSession(sendrecv_offer, offer.video, [], [], [], debug_mode=True)
    answer = await answer_session.get_answer()
    assert "ios-microphone" not in answer.sdp
    assert "ios-audio-track" not in answer.sdp
    assert answer.sdp.count("a=msid:audio audio") == 1
    answer_session.start()
    return RTCSessionDescription(answer.sdp, answer.type)

  builder = WebRTCOfferBuilder(connect, ice_servers=[])
  builder.offer_to_receive_audio_stream()
  stream = builder.stream()

  try:
    await asyncio.wait_for(stream.start(), 10)
    await asyncio.wait_for(stream.wait_for_connection(), 10)
    assert stream.has_incoming_audio_track()
    assert answer_session is not None
    assert answer_session.audio_recv_requested
    assert answer_session.audio_output is not None
  finally:
    await stream.stop()
    if answer_session is not None:
      await answer_session.stop_async()
