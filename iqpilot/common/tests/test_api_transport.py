# Copyright (c) 2026 IQ.Lvbs. All rights reserved.
from types import SimpleNamespace

import pytest

from iqpilot.common.api import base


@pytest.fixture
def client(monkeypatch):
  calls = []
  response = object()

  def send(**options):
    calls.append(options)
    return response

  monkeypatch.setattr(base.BaseApi, 'get_key_pair', staticmethod(lambda: (None, None, None)))
  monkeypatch.setattr(base, 'get_version', lambda: '1.2-beta')
  api = base.BaseApi('device', 'https://example.invalid/root/', 'IQ-', transport=SimpleNamespace(request=send))
  return api, calls, response


def test_request_options_are_reviewable_without_transport(client):
  api, calls, _ = client
  options = api.request_options('PUT', '/v2/backup', timeout=(2, 30), access_token='test-token', json={'version': 1}, limit=3)
  assert options == {'method': 'PUT', 'url': 'https://example.invalid/root/v2/backup', 'timeout': (2, 30),
                     'headers': {'User-Agent': 'IQ-1.2-beta', 'Authorization': 'JWT test-token'},
                     'json': {'version': 1}, 'params': {'limit': 3}}
  assert not calls


@pytest.mark.parametrize('entry,method', [('get', 'GET'), ('post', 'POST'), ('api_get', 'GET')])
def test_compatibility_methods_use_the_same_transport(client, entry, method):
  api, calls, response = client
  assert getattr(api, entry)('route', timeout=5, item='one') is response
  assert calls == [api.request_options(method, 'route', timeout=5, item='one')]


def test_legacy_positional_session_is_request_local(client):
  api, calls, response = client
  override = []
  session = SimpleNamespace(request=lambda **options: override.append(options))
  api.api_get('first', 'PATCH', 8, 'test-token', session, {'name': 'route'})
  assert override == [api.request_options('PATCH', 'first', timeout=8, access_token='test-token', json={'name': 'route'})]
  assert not calls
  assert api.get('second') is response
  assert len(calls) == 1
  assert 'Authorization' not in calls[0]['headers']


def test_headers_do_not_share_authentication_between_requests(client):
  api, calls, _ = client
  api.get('authenticated', access_token='test-token')
  api.get('public')
  assert calls[0]['headers']['Authorization'] == 'JWT test-token'
  assert 'Authorization' not in calls[1]['headers']


def test_user_agent_handles_unicode_and_control_characters(client, monkeypatch):
  api, _, _ = client
  api.user_agent = 'IQ-é-'
  monkeypatch.setattr(base, 'get_version', lambda: 'βeta\r\n1.2')
  assert api.request_options('GET', 'route')['headers']['User-Agent'] == 'IQ-e-eta1.2'


def test_absolute_looking_endpoint_does_not_replace_configured_host(client):
  api, _, _ = client
  assert api.request_options('GET', 'https://other.invalid/path')['url'] == 'https://example.invalid/root/https://other.invalid/path'


def test_transport_failures_remain_visible_to_callers(client):
  api, _, _ = client

  def fail(**_):
    raise TimeoutError('timed out')

  with pytest.raises(TimeoutError, match='timed out'):
    api.get('route', session=SimpleNamespace(request=fail))


@pytest.mark.parametrize('client_type', ['common', 'device', 'facade'])
def test_existing_clients_keep_their_wire_identity(client, monkeypatch, client_type):
  from iqpilot.common.api import Api
  from iqpilot.common.api.konn3kt import Konn3ktApi
  from iqpilot.konn3kt.cloud_client import Konn3ktApi as DeviceApi

  _, calls, response = client
  monkeypatch.setattr(base.requests, 'request', lambda **options: calls.append(options) or response)
  api = {'common': Konn3ktApi, 'device': DeviceApi, 'facade': Api}[client_type]('device')
  assert api.get('status', access_token='test-token', include='routes') is response
  assert calls[0]['headers']['User-Agent'] == ('konn3kt-device-' if client_type == 'device' else 'openpilot-') + '1.2-beta'
  assert calls[0]['headers']['Authorization'] == 'JWT test-token'
  assert calls[0]['params'] == {'include': 'routes'}
