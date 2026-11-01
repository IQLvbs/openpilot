import asyncio
import logging
import struct
import time

import av
from iqpilot.system.webrtc.rtc.tracks import TiciVideoStreamTrack

from iqpilot.cereal import messaging
from iqpilot.common.params import Params
from iqpilot.common.realtime import DT_MDL, DT_DMON

# Arbitrary 16-byte UUID identifying konn3kt frame-timing SEI messages. When timing
# telemetry is enabled, each frame carries a user_data_unregistered SEI NAL with four
# big-endian doubles (ms): encode duration, IPC/queue delay, host transit, and the
# device wall clock. The client decodes these to compute true glass-to-glass latency.
TIMING_SEI_UUID = bytes([
  0xa5, 0xe0, 0xc4, 0xa4, 0x5b, 0x6e, 0x4e, 0x1e,
  0x9c, 0x7e, 0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc,
])
# Annex-B start code + SEI NAL (type 6) + user_data_unregistered (type 5) + payload size
# (0x30 = 48 bytes = 16 UUID + 32 data). Trailing 0x80 is the RBSP stop bit.
_SEI_PREFIX = b'\x00\x00\x00\x01\x06\x05\x30' + TIMING_SEI_UUID


class LiveStreamVideoStreamTrack(TiciVideoStreamTrack):
  livestream_camera_to_sock_mapping = {
    "driver": "livestreamDriverEncodeData",
    "wideRoad": "livestreamWideRoadEncodeData",
    "road": "livestreamRoadEncodeData",
  }
  main_camera_to_sock_mapping = {
    "driver": "driverEncodeData",
    "wideRoad": "wideRoadEncodeData",
    "road": "roadEncodeData",
  }

  # Number of live tracks still waiting for their first keyframe. The on-demand
  # keyframe request (LivestreamRequestKeyframe) is a single global param honored by
  # every encoder, so with multiple concurrent tracks (dual-camera PiP) we must not
  # clear it until *all* tracks have received an IDR — otherwise the first track to
  # get its keyframe clears the request and starves the others (black feed).
  _kf_pending_count = 0

  def __init__(self, camera_type: str):
    dt = DT_DMON if camera_type == "driver" else DT_MDL
    super().__init__(camera_type, dt)

    self._params = Params()
    self._camera_type = camera_type
    self._candidate_topics = [self.livestream_camera_to_sock_mapping[camera_type]]
    self._socks = {topic: messaging.sub_sock(topic, conflate=False) for topic in self._candidate_topics}
    self._active_topic = self._preferred_topics()[0]
    self._pts = 0
    self._t0_ns = time.monotonic_ns()
    self._cached_header: bytes = b""
    self._sent_keyframe = False
    self._kf_requested = False  # whether this track counts toward _kf_pending_count
    self._frame_count = 0
    self._last_frame_time = 0.0
    self._previous_frame_id: int | None = None
    # Opt-in glass-to-glass latency telemetry (toggled by the client over the data channel).
    self.timing_sei_enabled = False
    self._logger = logging.getLogger("LiveStreamVideoStreamTrack")

    # Ask the encoder for an immediate IDR so the stream starts fast instead of waiting up to a full
    # GOP for the next periodic keyframe (encoderd honors LivestreamRequestKeyframe per-frame).
    self._mark_keyframe_needed()

  def _request_keyframe(self, enabled: bool) -> None:
    try:
      self._params.put_bool("LivestreamRequestKeyframe", enabled)
    except Exception:
      self._logger.exception("failed to set LivestreamRequestKeyframe")

  def _mark_keyframe_needed(self) -> None:
    """This track needs (another) keyframe: keep the global request asserted."""
    if not self._kf_requested:
      LiveStreamVideoStreamTrack._kf_pending_count += 1
      self._kf_requested = True
    self._request_keyframe(True)

  def request_keyframe(self) -> None:
    """RTCP PLI hook used by libdatachannel's native H.264 packetizer."""
    self._mark_keyframe_needed()

  def _mark_keyframe_received(self) -> None:
    """This track got its keyframe; only clear the global request once no track needs one."""
    if self._kf_requested:
      self._kf_requested = False
      LiveStreamVideoStreamTrack._kf_pending_count = max(0, LiveStreamVideoStreamTrack._kf_pending_count - 1)
    if LiveStreamVideoStreamTrack._kf_pending_count == 0:
      self._request_keyframe(False)

  def stop(self):
    # Release our pending-keyframe hold so a torn-down track that never received an
    # IDR doesn't pin LivestreamRequestKeyframe True forever (continuous keyframes).
    if getattr(self, "_kf_requested", False):
      self._kf_requested = False
      LiveStreamVideoStreamTrack._kf_pending_count = max(0, LiveStreamVideoStreamTrack._kf_pending_count - 1)
    try:
      super().stop()
    except Exception:
      pass

  def switch_camera(self, camera_type: str) -> None:
    """Repoint this track at a different camera without renegotiating the peer connection.

    Lets a single video track back the whole Live View — the client flips cameras over the
    data channel and we swap the source here, instead of uplinking every camera at once."""
    if camera_type not in self.livestream_camera_to_sock_mapping:
      self._logger.warning("[%s] ignoring switch to unknown camera %s", self._id, camera_type)
      return
    if camera_type == self._camera_type:
      return
    self._logger.info("[%s] switching camera %s -> %s", self._id, self._camera_type, camera_type)
    self._camera_type = camera_type
    self._candidate_topics = [self.livestream_camera_to_sock_mapping[camera_type]]
    self._socks = {topic: messaging.sub_sock(topic, conflate=False) for topic in self._candidate_topics}
    self._active_topic = self._preferred_topics()[0]
    # Force a fresh keyframe/header before emitting frames from the new source, and ask the encoder
    # for an immediate IDR so the camera switch isn't stalled waiting for the next periodic keyframe.
    self._cached_header = b""
    self._sent_keyframe = False
    self._previous_frame_id = None
    self._mark_keyframe_needed()

  def _preferred_topics(self) -> list[str]:
    return [self.livestream_camera_to_sock_mapping[self._camera_type]]

  def _reset_decoder_state(self, topic: str) -> None:
    if topic == self._active_topic:
      return
    self._logger.info("[%s] switching video source from %s to %s", self._id, self._active_topic, topic)
    self._active_topic = topic
    self._cached_header = b""
    self._sent_keyframe = False

  def _timing_sei(self, evta, log_mono_time: int) -> bytes:
    """Build a timing SEI NAL from encode metadata, or empty bytes when disabled."""
    if not self.timing_sei_enabled:
      return b""
    idx = evta.idx
    return _SEI_PREFIX + struct.pack(
      '>4d',
      (idx.timestampEof - idx.timestampSof) / 1e6,   # encode duration (ms)
      (log_mono_time - idx.timestampEof) / 1e6,       # IPC/queue delay (ms)
      (time.monotonic_ns() - log_mono_time) / 1e6,    # host transit so far (ms)
      time.time() * 1000,                             # device wall clock (ms)  # noqa: TID251
    ) + b'\x80'

  def _is_keyframe(self, data: bytes) -> bool:
    """Check if H.264 NAL unit contains an IDR keyframe (NAL type 5)."""
    i = 0
    while i < len(data) - 4:
      # Look for Annex B start codes: 0x000001 or 0x00000001
      if data[i:i+3] == b'\x00\x00\x01':
        nal_type = data[i+3] & 0x1f
        if nal_type == 5:  # IDR slice
          return True
        i += 3
      elif data[i:i+4] == b'\x00\x00\x00\x01':
        nal_type = data[i+4] & 0x1f
        if nal_type == 5:  # IDR slice
          return True
        i += 4
      else:
        i += 1
    return False

  async def recv(self):
    while True:
      await asyncio.sleep(0)
      live_topic = self.livestream_camera_to_sock_mapping[self._camera_type]
      msg = messaging.recv_one_or_none(self._socks[live_topic])
      if msg is None:
        await asyncio.sleep(0.005)
        continue
      self._last_frame_time = time.monotonic()
      evta = getattr(msg, msg.which())
      if str(evta.idx.type) not in ("qcameraH264", "livestreamH264"):
        continue
      header = bytes(evta.header)
      data = bytes(evta.data)
      self._frame_count += 1
      frame_id = evta.idx.frameId
      discontinuity = self._previous_frame_id is not None and frame_id != self._previous_frame_id + 1
      self._previous_frame_id = frame_id
      stale = msg.logMonoTime > 0 and time.monotonic_ns() - msg.logMonoTime > 250_000_000
      if discontinuity or stale:
        self._sent_keyframe = False
        if not self._kf_requested:
          self._mark_keyframe_needed()
      if header:
        self._cached_header = header
      if stale or not self._cached_header:
        continue
      is_keyframe = self._is_keyframe(data)
      if not self._sent_keyframe:
        if not is_keyframe:
          continue
        self._sent_keyframe = True
        self._mark_keyframe_received()
        self._logger.info("[%s] first keyframe received, starting stream", self._id)
      break

    # Optional timing SEI NAL, inserted before the slice data (and after SPS/PPS on keyframes).
    sei_nal = self._timing_sei(evta, msg.logMonoTime)

    # Prepend SPS/PPS header to keyframes (required by some decoders)
    # For non-keyframes, header is optional but safe to include
    if is_keyframe:
      payload = self._cached_header + sei_nal + data
    else:
      payload = sei_nal + data

    self._pts = ((time.monotonic_ns() - self._t0_ns) * self._clock_rate) // 1_000_000_000

    packet = av.Packet(payload)
    packet.time_base = self._time_base
    packet.pts = int(self._pts)
    packet.dts = int(self._pts)
    packet.duration = int(self._dt * self._clock_rate)

    if is_keyframe:
      packet.is_keyframe = True

    self.log_debug("track sending frame %s (keyframe=%s, size=%d)", self._pts, is_keyframe, len(payload))

    return packet

  def codec_preference(self) -> str | None:
    return "H264"


class DebugVideoStreamTrack(TiciVideoStreamTrack):
  def __init__(self, camera_type: str):
    super().__init__(camera_type, 0.05)
    self._codec = av.CodecContext.create("libx264", "w")
    self._codec.width = 640
    self._codec.height = 480
    self._codec.pix_fmt = "yuv420p"
    self._codec.time_base = self._time_base
    self._codec.framerate = 20
    self._codec.options = {"preset": "ultrafast", "tune": "zerolatency"}
    self._codec.open()
    self._pts = 0
    self.timing_sei_enabled = False

  async def recv(self):
    await asyncio.sleep(self._dt)
    frame = av.VideoFrame(self._codec.width, self._codec.height, "yuv420p")
    frame.planes[0].update(bytes(frame.planes[0].buffer_size))
    for plane in frame.planes[1:]:
      plane.update(bytes([128]) * plane.buffer_size)
    frame.pts = self._pts
    self._pts += int(self._dt * self._clock_rate)
    packets = self._codec.encode(frame)
    if not packets:
      return await self.recv()
    packet = av.Packet(b"".join(bytes(encoded) for encoded in packets))
    packet.pts = frame.pts
    packet.dts = frame.pts
    packet.time_base = self._time_base
    packet.duration = int(self._dt * self._clock_rate)
    return packet

  def switch_camera(self, camera_type: str) -> None:
    if camera_type not in LiveStreamVideoStreamTrack.livestream_camera_to_sock_mapping:
      raise ValueError(f"Unknown camera {camera_type}")
