import asyncio
import gc
import json

import capnp
from iqpilot.cereal import messaging, log

from iqpilot.system.webrtc.webrtcd import CerealOutgoingMessageProxy, CerealIncomingMessageProxy
from iqpilot.system.webrtc.device.native_audio import AudioInputOpusProducer, DebugAudioOpusProducer
from iqpilot.system.webrtc.device.native_video import DebugVideoStreamTrack, LiveStreamVideoStreamTrack
from iqpilot.system.webrtc.rtc.tracks import VIDEO_TIME_BASE


class TestStreamSession:
  def setup_method(self):
    self.loop = asyncio.new_event_loop()

  def teardown_method(self):
    self.loop.stop()
    self.loop.close()
    gc.collect()

  def test_outgoing_proxy(self, mocker):
    test_msg = log.Event.new_message()
    test_msg.logMonoTime = 123
    test_msg.valid = True
    test_msg.customReservedRawData0 = b"test"
    expected_dict = {"type": "customReservedRawData0", "logMonoTime": 123, "valid": True, "data": "test"}
    expected_json = json.dumps(expected_dict).encode()

    channel = mocker.Mock()
    mocked_submaster = mocker.MagicMock()
    mocked_submaster.updated = {"customReservedRawData0": True}
    mocked_submaster.logMonoTime = {"customReservedRawData0": 123}
    mocked_submaster.valid = {"customReservedRawData0": True}
    mocked_submaster.__getitem__.return_value = test_msg.customReservedRawData0
    proxy = CerealOutgoingMessageProxy(mocked_submaster)
    proxy.add_channel(channel)

    proxy.update()

    channel.send.assert_called_once_with(expected_json)

  def test_incoming_proxy(self, mocker):
    tested_msgs = [
      {"type": "customReservedRawData0", "data": "test"},
      {"type": "can", "data": [{"address": 0, "dat": "", "src": 0}]},
      {"type": "testJoystick", "data": {"axes": [0, 0], "buttons": [False]}},
    ]

    mocked_pubmaster = mocker.MagicMock(spec=messaging.PubMaster)

    proxy = CerealIncomingMessageProxy(mocked_pubmaster)

    for msg in tested_msgs:
      proxy.send(json.dumps(msg).encode())

      mocked_pubmaster.send.assert_called_once()
      mt, md = mocked_pubmaster.send.call_args.args
      assert mt == msg["type"]
      assert isinstance(md, capnp._DynamicStructBuilder)
      assert hasattr(md, msg["type"])

      mocked_pubmaster.reset_mock()

  def test_livestream_track(self, mocker):
    fake_msg = messaging.new_message("livestreamDriverEncodeData")
    fake_msg.livestreamDriverEncodeData.header = b"header"
    fake_msg.livestreamDriverEncodeData.data = b"\x00\x00\x00\x01\x65"
    fake_msg.livestreamDriverEncodeData.idx.type = "livestreamH264"

    mocker.patch("iqpilot.system.webrtc.device.native_video.messaging.sub_sock", return_value=mocker.Mock())
    mocker.patch("iqpilot.system.webrtc.device.native_video.messaging.recv_one_or_none", return_value=fake_msg)
    track = LiveStreamVideoStreamTrack("driver")

    assert track.id.startswith("driver")
    packet = self.loop.run_until_complete(track.recv())
    assert packet.time_base == VIDEO_TIME_BASE
    assert packet.pts is not None
    assert packet.size == len(b"header\x00\x00\x00\x01\x65")

  def test_input_audio_track(self, mocker):
    packet_time, rate = 0.02, 16000
    sample_count = int(packet_time * rate)
    fake_msg = messaging.new_message("rawAudioData")
    fake_msg.rawAudioData.data = b"\x00" * 2 * sample_count
    fake_msg.rawAudioData.sampleRate = rate
    mocker.patch("iqpilot.system.webrtc.device.native_audio.messaging.sub_sock", return_value=mocker.Mock())
    track = AudioInputOpusProducer()
    track._source_rate = rate
    mocker.patch("iqpilot.system.webrtc.device.native_audio.messaging.recv_one_or_none", return_value=fake_msg)

    packet = self.loop.run_until_complete(track.recv())
    assert packet is not None
    payload, pts = packet
    assert payload
    assert pts >= 0

  def test_debug_video_track(self):
    track = DebugVideoStreamTrack("road")
    packet = self.loop.run_until_complete(track.recv())
    assert packet.size > 0
    assert packet.pts == 0

  def test_debug_audio_track(self):
    track = DebugAudioOpusProducer()
    packet = self.loop.run_until_complete(track.recv())
    assert packet is not None
    payload, pts = packet
    assert payload
    assert pts == 0
