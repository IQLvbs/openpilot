#!/usr/bin/env python3

import argparse
import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import capnp
from aiohttp import web

from iqpilot.system.webrtc.schema import generate_field
from iqpilot.cereal import messaging, log


class CerealOutgoingMessageProxy:
  def __init__(self, sm: messaging.SubMaster):
    self.sm = sm
    self.channels: list[Any] = []

  def add_channel(self, channel: Any):
    self.channels.append(channel)

  def to_json(self, msg_content: Any):
    if isinstance(msg_content, capnp._DynamicStructReader):
      msg_dict = msg_content.to_dict()
    elif isinstance(msg_content, capnp._DynamicListReader):
      msg_dict = [self.to_json(msg) for msg in msg_content]
    elif isinstance(msg_content, bytes):
      msg_dict = msg_content.decode()
    else:
      msg_dict = msg_content

    return msg_dict

  def update(self):
    # this is blocking in async context...
    self.sm.update(0)
    for service, updated in self.sm.updated.items():
      if not updated:
        continue
      msg_dict = self.to_json(self.sm[service])
      mono_time, valid = self.sm.logMonoTime[service], self.sm.valid[service]
      outgoing_msg = {"type": service, "logMonoTime": mono_time, "valid": valid, "data": msg_dict}
      encoded_msg = json.dumps(outgoing_msg).encode()
      for channel in self.channels:
        channel.send(encoded_msg)


class CerealIncomingMessageProxy:
  def __init__(self, pm: messaging.PubMaster):
    self.pm = pm

  def send(self, message: bytes):
    msg_json = json.loads(message)
    msg_type, msg_data = msg_json["type"], msg_json["data"]
    size = None
    if not isinstance(msg_data, dict):
      size = len(msg_data)

    msg = messaging.new_message(msg_type, size=size)
    setattr(msg, msg_type, msg_data)
    self.pm.send(msg_type, msg)


class CerealProxyRunner:
  def __init__(self, proxy: CerealOutgoingMessageProxy):
    self.proxy = proxy
    self.is_running = False
    self.task = None
    self.logger = logging.getLogger("webrtcd")

  def start(self):
    assert self.task is None
    self.task = asyncio.create_task(self.run())

  def stop(self):
    if self.task is None or self.task.done():
      return
    self.task.cancel()
    self.task = None

  async def run(self):
    while True:
      try:
        self.proxy.update()
      except Exception:
        self.logger.exception("Cereal outgoing proxy failure")
      await asyncio.sleep(0.01)


class DynamicPubMaster(messaging.PubMaster):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.lock = asyncio.Lock()

  async def add_services_if_needed(self, services):
    async with self.lock:
      for service in services:
        if service not in self.sock:
          self.sock[service] = messaging.pub_sock(service)


def _is_retryable_stream_error(e: Exception) -> bool:
  return isinstance(e, (ValueError, OSError))


async def _cleanup_failed_session(session: Any | None, logger: logging.Logger) -> None:
  if session is None:
    return
  try:
    await session.stop_async()
  except Exception:
    logger.exception("Failed to clean up failed stream session")


def _strip_mdns_host_candidates(sdp: str) -> tuple[str, int]:
  lines = sdp.split("\r\n")
  kept = [line for line in lines if not (line.startswith("a=candidate:") and ".local" in line)]
  return "\r\n".join(kept), len(lines) - len(kept)


@dataclass
class StreamRequestBody:
  sdp: str
  cameras: list[str]
  bridge_services_in: list[str] = field(default_factory=list)
  bridge_services_out: list[str] = field(default_factory=list)
  iceServers: list[dict[str, Any]] = field(default_factory=list)
  ui_stream: bool = False


def _new_stream_session(offer_sdp: str, body: StreamRequestBody, debug_mode: bool):
  from iqpilot.system.webrtc.session import StreamSession
  return StreamSession(
    offer_sdp, body.cameras, body.bridge_services_in, body.bridge_services_out, body.iceServers, debug_mode,
    ui_stream=body.ui_stream,
  )


async def get_stream(request: 'web.Request'):
  lock = request.app.setdefault('stream_lock', asyncio.Lock())
  async with lock:
    return await _get_stream(request)


async def _get_stream(request: 'web.Request'):
  stream_dict, debug_mode = request.app['streams'], request.app['debug']
  logger = logging.getLogger("webrtcd")
  session: Any | None = None
  try:
    raw_body = await request.json()
    body = StreamRequestBody(**raw_body)
    offer_sdp = body.sdp

    # Single active session on the device: tear down any prior session before starting a new
    # one. webrtcd is long-lived (manager-owned), so without this, repeated offers would leak
    # sessions and contend for the same livestream topics.
    for prev in list(stream_dict.values()):
      try:
        await prev.stop_async()
      except Exception:
        logger.exception("Failed to stop previous stream session")
    stream_dict.clear()

    session = _new_stream_session(offer_sdp, body, debug_mode)
    # Creating an answer can occasionally stall (ICE gathering, codec negotiation, etc).
    # Bound it so the HTTP request doesn't hang forever and athena can surface a useful error.
    try:
      answer = await asyncio.wait_for(session.get_answer(), timeout=15.0)
    except Exception as e:
      if not _is_retryable_stream_error(e):
        raise

      logger.warning("Transient stream creation error (%s); retrying once with a fresh session", e)
      await _cleanup_failed_session(session, logger)
      retry_offer_sdp, removed_mdns = _strip_mdns_host_candidates(offer_sdp)
      if removed_mdns > 0:
        logger.info("Retrying with SDP sanitized; removed %d mDNS host ICE candidate(s)", removed_mdns)
      else:
        logger.info("Retrying with fresh session and original SDP (no mDNS host candidates removed)")

      session = _new_stream_session(retry_offer_sdp, body, debug_mode)
      answer = await asyncio.wait_for(session.get_answer(), timeout=15.0)

    session.start()

    stream_dict[session.identifier] = session

    return web.json_response({"sdp": answer.sdp, "type": answer.type})
  except asyncio.CancelledError:
    await _cleanup_failed_session(session, logger)
    raise
  except TimeoutError:
    await _cleanup_failed_session(session, logger)
    logger.exception("Timed out generating WebRTC answer")
    return web.json_response({"error": "answer_timeout", "message": "Timed out generating WebRTC answer"}, status=504)
  except Exception as e:
    await _cleanup_failed_session(session, logger)
    logger.exception("Failed to create WebRTC stream session")
    return web.json_response({"error": "stream_create_failed", "message": str(e)}, status=500)


async def add_ice(request: 'web.Request'):
  stream_dict = request.app['streams']
  try:
    body = await request.json()
  except Exception:
    return web.json_response({"error": "bad_request"}, status=400)
  cand = body.get("candidate")
  # Single active session on the device; apply to whatever is live.
  for session in list(stream_dict.values()):
    await session.add_ice_candidate(cand)
  return web.json_response({"ok": True})


async def get_schema(request: 'web.Request'):
  services = request.query["services"].split(",")
  services = [s for s in services if s]
  assert all(s in log.Event.schema.fields and not s.endswith("DEPRECATED") for s in services), "Invalid service name"
  schema_dict = {s: generate_field(log.Event.schema.fields[s]) for s in services}
  return web.json_response(schema_dict)


async def on_shutdown(app: 'web.Application'):
  for session in app['streams'].values():
    await session.stop_async()
  del app['streams']


def webrtcd_thread(host: str, port: int, debug: bool):
  logging.basicConfig(level=logging.CRITICAL, handlers=[logging.StreamHandler()])
  logging_level = logging.DEBUG if debug else logging.INFO
  logging.getLogger("WebRTCStream").setLevel(logging_level)
  logging.getLogger("webrtcd").setLevel(logging_level)
  logging.getLogger("LiveStreamVideoStreamTrack").setLevel(logging_level)

  app = web.Application()

  app['streams'] = dict()
  app['stream_lock'] = asyncio.Lock()
  app['debug'] = debug
  app.on_shutdown.append(on_shutdown)
  app.router.add_post("/stream", get_stream)
  app.router.add_post("/ice", add_ice)
  app.router.add_get("/schema", get_schema)

  web.run_app(app, host=host, port=port)


def main():
  parser = argparse.ArgumentParser(description="WebRTC daemon")
  parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to listen on")
  parser.add_argument("--port", type=int, default=5001, help="Port to listen on")
  parser.add_argument("--debug", action="store_true", help="Enable debug mode")
  args = parser.parse_args()

  webrtcd_thread(args.host, args.port, args.debug)


if __name__=="__main__":
  main()
