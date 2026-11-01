import pytest

from iqpilot.system.webrtc.rtc.info import parse_info_from_offer
from iqpilot.system.webrtc.rtc.stream import WebRTCBaseStream
from iqpilot.system.webrtc.rtc.tracks import TiciTrackWrapper, TiciVideoStreamTrack, parse_video_track_id, video_track_id


def sdp_with_media(media):
  mids = " ".join(str(i) for i in range(len(media)))
  sections = []
  for index, (kind, direction) in enumerate(media):
    if kind == "video":
      payload, mapping = "96", "H264/90000"
      protocol = "UDP/TLS/RTP/SAVPF"
    elif kind == "audio":
      payload, mapping = "111", "opus/48000/2"
      protocol = "UDP/TLS/RTP/SAVPF"
    else:
      sections.extend([
        "m=application 9 UDP/DTLS/SCTP webrtc-datachannel",
        "c=IN IP4 0.0.0.0",
        f"a=mid:{index}",
        "a=sctp-port:5000",
      ])
      continue
    sections.extend([
      f"m={kind} 9 {protocol} {payload}",
      "c=IN IP4 0.0.0.0",
      f"a=mid:{index}",
      f"a={direction}",
      f"a=rtpmap:{payload} {mapping}",
      "a=rtcp-mux",
    ])
  lines = [
    "v=0",
    "o=- 1 1 IN IP4 0.0.0.0",
    "s=-",
    "t=0 0",
    f"a=group:BUNDLE {mids}",
    *sections,
  ]
  return "\r\n".join(lines) + "\r\n"


@pytest.mark.parametrize("direction,expected_outgoing,expected_incoming", [
  ("recvonly", True, False),
  ("sendonly", False, True),
  ("sendrecv", True, True),
  ("inactive", False, False),
])
def test_audio_directions(direction, expected_outgoing, expected_incoming):
  info = parse_info_from_offer(sdp_with_media([("audio", direction)]))
  assert info.expected_audio_track == expected_outgoing
  assert info.incoming_audio_track == expected_incoming


def test_video_and_data_channel_metadata():
  info = parse_info_from_offer(sdp_with_media([
    ("video", "recvonly"),
    ("video", "sendrecv"),
    ("application", "sendrecv"),
  ]))
  assert info.n_expected_camera_tracks == 2
  assert info.incoming_datachannel


def test_each_camera_uses_a_distinct_receiving_video_slot():
  from types import SimpleNamespace
  from iqpilot.system.webrtc.rtc.stream import WebRTCAnswerStream
  stream = WebRTCAnswerStream.__new__(WebRTCAnswerStream)
  offer = sdp_with_media([("audio", "sendrecv"), ("video", "inactive"), ("video", "recvonly"), ("video", "sendrecv"), ("video", "recvonly")])
  mids = [stream._make_video_media(SimpleNamespace(id=f"{camera}:test"), offer, index)[0].mid()
          for index, camera in enumerate(("road", "driver", "wideRoad"))]
  assert mids == ["2", "3", "4"]
  with pytest.raises(ValueError, match="track 3"):
    stream._find_offer_video(offer, 3)


def test_explicit_empty_ice_servers_disable_defaults():
  assert WebRTCBaseStream._make_ice_servers([]) == []


def test_default_ice_server():
  servers = WebRTCBaseStream._make_ice_servers(None)
  assert len(servers) == 1
  assert servers[0].hostname == "stun.l.google.com"
  assert servers[0].port == 19302


def test_authenticated_ice_servers():
  servers = WebRTCBaseStream._make_ice_servers([{
    "urls": ["turn:relay.example.com:3478", "stun:stun.example.com:3478"],
    "username": "user",
    "credential": "secret",
  }])
  assert [(server.hostname, server.port) for server in servers] == [
    ("relay.example.com", 3478),
    ("stun.example.com", 3478),
  ]
  assert all(server.username == "user" and server.password == "secret" for server in servers)


def test_track_id_roundtrip():
  assert parse_video_track_id(video_track_id("driver", "track")) == ("driver", "track")


def test_invalid_track_id():
  with pytest.raises(ValueError):
    parse_video_track_id("driver")


def test_track_wrapper_preserves_camera():
  class Track:
    kind = "video"
    id = "source"

    async def recv(self):
      return b"frame"

  wrapper = TiciTrackWrapper("road", Track())
  assert parse_video_track_id(wrapper.id)[0] == "road"
  wrapper.stop()
  assert wrapper.readyState == "ended"


def test_track_stores_frame_period():
  track = TiciVideoStreamTrack("wideRoad", 0.05)
  assert track._dt == 0.05
