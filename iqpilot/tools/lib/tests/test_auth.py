# Copyright (c) 2026 IQ.Lvbs LLC. All Rights Reserved.
import json
import socket
import threading
from urllib.parse import parse_qs, urlsplit

import pytest
import requests

from iqpilot.tools.lib import api, auth


@pytest.fixture
def token_home(tmp_path, monkeypatch):
  monkeypatch.setenv('HOME', str(tmp_path))
  monkeypatch.setenv('OPENPILOT_PREFIX', 'test-cabana')
  legacy = tmp_path / '.comma' / 'auth.json'
  legacy.parent.mkdir()
  legacy.write_text(json.dumps({'access_token': 'legacy-token'}))
  return tmp_path


def test_token_storage_is_iq(token_home):
  assert api.get_token() is None
  api.set_token('konn3kt-test-token')
  assert api.get_token() == 'konn3kt-test-token'
  assert json.loads((token_home / '.iq/auth.json').read_text()) == {'access_token': 'konn3kt-test-token'}
  assert (token_home / '.iq/auth.json').stat().st_mode & 0o777 == 0o600
  api.clear_token()
  api.clear_token()
  assert api.get_token() is None
  assert json.loads((token_home / '.comma/auth.json').read_text()) == {'access_token': 'legacy-token'}


@pytest.mark.parametrize('provider', auth.PROVIDERS)
def test_provider_redirect(provider):
  url = urlsplit(auth.auth_redirect_link(provider, 4567, 'test-state'))
  assert f'{url.scheme}://{url.netloc}' == api.API_HOST
  assert url.path == f'/auth/{provider}'
  assert parse_qs(url.query)['redirect'] == ['http://localhost:4567/auth?state=test-state']


@pytest.mark.parametrize('provider', auth.PROVIDERS)
def test_browser_callback(token_home, monkeypatch, mocker, provider):
  clients = []
  errors = []

  def open_browser(args, **kwargs):
    callback = parse_qs(urlsplit(args[1]).query)['redirect'][0]
    port = urlsplit(callback).port
    state = parse_qs(urlsplit(callback).query)['state'][0]

    def send_callback():
      try:
        with socket.create_connection(('localhost', port), timeout=2):
          assert requests.get(f'http://localhost:{port}/favicon.ico', timeout=2).status_code == 204
          response = requests.get(callback, timeout=2)
          assert response.status_code == 200
          assert 'location.hash' in response.text
          assert requests.post(f'http://localhost:{port}/auth', data={'state': 'wrong', 'token': 'wrong-token'}, timeout=2).status_code == 400
          assert requests.post(f'http://localhost:{port}/auth', data={'state': state, 'token': ''}, timeout=2).status_code == 400
          response = requests.post(f'http://localhost:{port}/auth', data={'state': state, 'token': 'test-token'}, timeout=2)
          assert response.status_code == 200
      except Exception as error:
        errors.append(error)

    thread = threading.Thread(target=send_callback)
    clients.append(thread)
    thread.start()
    return mocker.Mock(poll=lambda: 0)

  client = mocker.Mock()
  monkeypatch.setattr(auth.subprocess, 'Popen', open_browser)
  factory = mocker.patch.object(auth, 'CommaApi', return_value=client)
  result = auth.login(provider, timeout=3)
  for thread in clients:
    thread.join()
  assert not errors
  assert result == {'success': True}
  assert api.get_token() == 'test-token'
  factory.assert_called_once_with('test-token')
  client.get.assert_called_once_with('v1/me', timeout=30)


def test_timeout_preserves_existing_token(token_home, mocker):
  api.set_token('existing-token')
  mocker.patch.object(auth.subprocess, 'Popen', return_value=mocker.Mock(poll=lambda: 0))
  assert 'timed out' in auth.login('github', timeout=0.01)['error']
  assert api.get_token() == 'existing-token'


def test_rejected_token_is_not_saved(token_home, mocker):
  api.set_token('existing-token')
  server = mocker.MagicMock()
  server.__enter__.return_value = server
  server.server_port = 4567
  server.state = 'test-state'
  server.query_params = {'token': ['invalid-token']}
  mocker.patch.object(auth, 'ClientRedirectServer', return_value=server)
  mocker.patch.object(auth.subprocess, 'Popen', return_value=mocker.Mock(poll=lambda: 0))
  mocker.patch.object(auth, 'CommaApi').return_value.get.side_effect = api.UnauthorizedError()
  assert 'error' in auth.login('google')
  assert api.get_token() == 'existing-token'
