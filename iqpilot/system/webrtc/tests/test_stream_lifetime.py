# Copyright (c) 2026 IQ.Lvbs. All rights reserved.
import asyncio
from types import SimpleNamespace

import pytest

from iqpilot.system.webrtc import webrtcd


@pytest.fixture
def stream_request(monkeypatch):
  sessions = []

  class Session:
    def __init__(self):
      self.identifier = str(len(sessions))
      self.stopped = False
      self.answer_started = asyncio.Event()
      sessions.append(self)

    async def get_answer(self):
      self.answer_started.set()
      await asyncio.sleep(0)
      return SimpleNamespace(sdp='answer', type='answer')

    def start(self):
      pass

    async def stop_async(self):
      self.stopped = True

  async def body():
    return {'sdp': 'offer', 'cameras': ['road']}

  monkeypatch.setattr(webrtcd, '_new_stream_session', lambda *args: Session())
  return SimpleNamespace(app={'streams': {}, 'debug': False}, json=body), sessions, Session


@pytest.mark.asyncio
async def test_overlapping_offers_leave_only_one_live_session(stream_request):
  request, sessions, _ = stream_request
  responses = await asyncio.gather(*(webrtcd.get_stream(request) for _ in range(12)))
  assert all(response.status == 200 for response in responses)
  assert len(request.app['streams']) == 1
  assert sum(not session.stopped for session in sessions) == 1
  assert not sessions[-1].stopped


@pytest.mark.asyncio
async def test_cancelled_offer_releases_partial_session(stream_request, monkeypatch):
  request, sessions, session_class = stream_request
  started = asyncio.Event()

  async def stalled_answer(self):
    started.set()
    await asyncio.Event().wait()

  monkeypatch.setattr(session_class, 'get_answer', stalled_answer)
  task = asyncio.create_task(webrtcd.get_stream(request))
  await asyncio.wait_for(started.wait(), 2)
  task.cancel()
  with pytest.raises(asyncio.CancelledError):
    await task
  assert len(sessions) == 1
  assert sessions[0].stopped
  assert not request.app['streams']
