# Copyright (c) 2026 IQ.Lvbs LLC. All Rights Reserved.
import json
import os
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from iqpilot.tools.lib import api

ROOT = Path(__file__).resolve().parents[4]


@pytest.mark.parametrize('status,body', [(401, b''), (401, b'{"detail":"expired"}'), (403, b'Forbidden')])
def test_python_rejects_unauthorized_regardless_of_body(status, body, mocker):
  response = api.requests.Response()
  response.status_code = status
  response._content = body
  response._content_consumed = True
  client = api.CommaApi('stale-token')
  mocker.patch.object(client.session, 'request', return_value=response)
  with pytest.raises(api.UnauthorizedError, match='sign in again'):
    client.get('v1/me')


@pytest.mark.parametrize('mode', ['missing', 'malformed', 'expired', 'revoked', 'valid', 'offline'])
def test_native_route_browsing_requires_a_valid_session(tmp_path, mode):
  paths = []
  devices = '[{"dongle_id":"test-device"}]'
  routes = '[]'

  class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
      paths.append(self.path)
      assert self.headers.get('Authorization') == 'JWT test-token'
      if self.path == '/v1/me':
        status = {'expired': 401, 'revoked': 403, 'offline': 503}.get(mode, 200)
        body = '{"id":"test-user"}' if status == 200 else '{}'
      else:
        status = 200
        body = devices if self.path == '/v1/me/devices/' else routes
      self.send_response(status)
      self.send_header('Content-Type', 'application/json')
      self.end_headers()
      self.wfile.write(body.encode())

    def log_message(self, *args):
      pass

  server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
  worker = threading.Thread(target=server.serve_forever)
  worker.start()
  (tmp_path / '.iq').mkdir()
  if mode == 'malformed':
    (tmp_path / '.iq/auth.json').write_text('broken')
  elif mode != 'missing':
    (tmp_path / '.iq/auth.json').write_text(json.dumps({'access_token': 'test-token'}))
  error = '{"error": "network"}' if mode == 'offline' else '{"error": "unauthorized"}'
  env = {**os.environ, 'HOME': str(tmp_path), 'API_HOST': f'http://127.0.0.1:{server.server_port}',
         'TEST_DEVICES_RESPONSE': devices if mode == 'valid' else error,
         'TEST_ROUTES_RESPONSE': routes if mode == 'valid' else error}
  try:
    result = subprocess.run([ROOT / 'iqpilot/tools/replay/tests/test_api', '[.tools_session]'], env=env,
                            capture_output=True, text=True, timeout=10)
  finally:
    server.shutdown()
    worker.join()
    server.server_close()
  assert result.returncode == 0, result.stdout + result.stderr
  assert 'Error parsing auth.json' not in result.stderr
  assert paths == ([] if mode in ('missing', 'malformed') else
                   ['/v1/me', '/v1/me/devices/', '/v1/me', '/v1/devices/test-device/routes_segments?start=1000&end=2000']
                   if mode == 'valid' else ['/v1/me', '/v1/me'])
