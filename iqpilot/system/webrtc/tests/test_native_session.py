from types import SimpleNamespace

import pytest

from iqpilot.system.webrtc import session as session_module


class FakeStream:
  def get_receiver_report_stats(self):
    return {}


class FakeBuilder:
  instance = None

  def __init__(self, sdp, bind_address=None, ice_servers=None):
    self.sdp = sdp
    self.bind_address = bind_address
    self.ice_servers = ice_servers
    self.video = []
    self.audio = []
    self.receive_audio = False
    self.result = FakeStream()
    FakeBuilder.instance = self

  def add_video_stream(self, camera, track):
    self.video.append((camera, track))

  def add_audio_stream(self, track):
    self.audio.append(track)

  def offer_to_receive_audio_stream(self):
    self.receive_audio = True

  def stream(self):
    return self.result


class FakeVideoTrack:
  def __init__(self, camera):
    self.camera = camera
    self.timing_sei_enabled = False
    self.switched = None

  def switch_camera(self, camera):
    self.switched = camera


class FakeAudioProducer:
  def __init__(self):
    self.enabled = True

  def enable(self, enabled):
    self.enabled = enabled


@pytest.fixture
def native_session(mocker):
  config = SimpleNamespace(
    n_expected_camera_tracks=1,
    expected_audio_track=True,
    incoming_audio_track=True,
    incoming_datachannel=True,
  )
  mocker.patch("iqpilot.system.webrtc.rtc.info.parse_info_from_offer", return_value=config)
  mocker.patch("iqpilot.system.webrtc.rtc.builder.WebRTCAnswerBuilder", FakeBuilder)
  mocker.patch("iqpilot.system.webrtc.device.native_video.LiveStreamVideoStreamTrack", FakeVideoTrack)
  mocker.patch("iqpilot.system.webrtc.device.native_audio.AudioInputOpusProducer", FakeAudioProducer)
  mocker.patch.object(session_module, "_default_route_ip", return_value="192.0.2.1")
  mocker.patch.object(session_module, "Params", return_value=mocker.Mock())
  return session_module.StreamSession(
    "offer", ["road"], [], [], [{"urls": "turn:example.com"}], ui_stream=False,
  )


def test_native_session_builds_duplex_audio(native_session):
  builder = FakeBuilder.instance
  assert builder is not None
  assert builder.bind_address == "192.0.2.1"
  assert builder.ice_servers == [{"urls": "turn:example.com"}]
  assert [camera for camera, _ in builder.video] == ["road"]
  assert builder.audio == [native_session.audio_output]
  assert builder.receive_audio
  assert native_session.audio_recv_requested


def test_native_session_controls(native_session, mocker):
  native_session.bitrate_controller = mocker.Mock()
  native_session.message_handler('{"type":"timingSei","enabled":true}')
  assert native_session.video_tracks[0].timing_sei_enabled
  native_session.message_handler('{"type":"switchCamera","camera":"driver"}')
  assert native_session.video_tracks[0].switched == "driver"
  native_session.message_handler('{"type":"setAudioEnabled","enabled":false}')
  assert not native_session.audio_output.enabled
  native_session.message_handler('{"type":"setQuality","quality":"low"}')
  native_session.bitrate_controller.set_quality.assert_called_once_with("low")
