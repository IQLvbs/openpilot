import json
from types import SimpleNamespace

from parameterized import parameterized_class
import pytest

from iqpilot.system.webrtc.webrtcd import add_ice, get_stream


class FakeSession:
  instances = []

  def __init__(self, identifier="session"):
    self.identifier = identifier
    self.started = False
    self.stopped = False
    self.candidates = []
    self.instances.append(self)

  async def get_answer(self):
    return SimpleNamespace(sdp="answer", type="answer")

  def start(self):
    self.started = True

  async def stop_async(self):
    self.stopped = True

  async def add_ice_candidate(self, candidate):
    self.candidates.append(candidate)


@parameterized_class(("in_services", "out_services"), [
  (["testJoystick"], ["carState"]),
  ([], ["carState"]),
  (["testJoystick"], []),
  ([], []),
])
@pytest.mark.asyncio
class TestWebrtcdProc:
  async def test_webrtcd(self, mocker):
    session = FakeSession()
    mocker.patch("iqpilot.system.webrtc.webrtcd._new_stream_session", return_value=session)
    request = mocker.MagicMock()
    request.app = {"streams": {}, "debug": False}
    request.json = mocker.AsyncMock(return_value={
      "sdp": "offer",
      "cameras": ["road"],
      "bridge_services_in": self.in_services,
      "bridge_services_out": self.out_services,
    })

    response = await get_stream(request)

    assert response.status == 200
    assert json.loads(response.text) == {"sdp": "answer", "type": "answer"}
    assert request.app["streams"] == {session.identifier: session}
    assert session.started

  async def test_replaces_session_and_routes_ice(self, mocker):
    previous = FakeSession("previous")
    session = FakeSession("current")
    mocker.patch("iqpilot.system.webrtc.webrtcd._new_stream_session", return_value=session)
    request = mocker.MagicMock()
    request.app = {"streams": {previous.identifier: previous}, "debug": False}
    request.json = mocker.AsyncMock(return_value={"sdp": "offer", "cameras": ["road"]})

    response = await get_stream(request)

    assert response.status == 200
    assert previous.stopped
    request.json = mocker.AsyncMock(return_value={"candidate": {"candidate": "candidate:1"}})
    ice_response = await add_ice(request)
    assert ice_response.status == 200
    assert session.candidates == [{"candidate": "candidate:1"}]
