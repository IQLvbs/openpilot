#!/usr/bin/env python3
import argparse
import json
import sys
import subprocess
import pprint
import time
import threading
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlencode, urlsplit

from iqpilot.tools.lib.api import API_HOST, CommaApi, UnauthorizedError, set_token, get_token

class ClientRedirectServer(ThreadingHTTPServer):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.query_params: dict[str, Any] = {}
    self.result_lock = threading.Lock()
    self.state = secrets.token_urlsafe(32)

  def get_request(self):
    request, address = super().get_request()
    request.settimeout(1)
    return request, address


class ClientRedirectHandler(BaseHTTPRequestHandler):
  def do_GET(self):
    if urlsplit(self.path).path != '/auth':
      self.send_response(204)
      self.end_headers()
      return
    params = parse_qs(urlsplit(self.path).query)
    if params.get('state') != [self.server.state]:
      self.send_error(400)
      return
    self.send_response(200)
    self.send_header('Content-Type', 'text/html; charset=utf-8')
    self.send_header('Cache-Control', 'no-store')
    self.send_header('Referrer-Policy', 'no-referrer')
    self.end_headers()
    try:
      self.wfile.write(CALLBACK_HTML.encode())
    except ConnectionError:
      pass

  def do_POST(self):
    if urlsplit(self.path).path != '/auth':
      self.send_error(404)
      return
    try:
      length = int(self.headers.get('Content-Length', '0'))
      if not 0 < length <= 16384:
        raise ValueError
      params = parse_qs(self.rfile.read(length).decode(), keep_blank_values=True)
    except (ValueError, UnicodeError):
      self.send_error(400)
      return
    if params.get('state') != [self.server.state] or len(params.get('token', [])) != 1 or not params['token'][0].strip():
      self.send_error(400)
      return
    with self.server.result_lock:
      if not self.server.query_params:
        self.server.query_params = params
    self.send_response(200)
    self.send_header('Cache-Control', 'no-store')
    self.end_headers()

  def log_message(self, fmt: str, *args: object) -> None:
    pass


PROVIDERS = ('google', 'github', 'apple', 'microsoft')
CALLBACK_HTML = """<!doctype html><meta charset="utf-8"><title>Konn3kt sign-in</title>
<p id="status">Completing Konn3kt sign-in...</p><script>
const token = new URLSearchParams(location.hash.slice(1)).get('token');
const state = new URLSearchParams(location.search).get('state');
history.replaceState(null, '', location.pathname);
if (token && state) {
  fetch('/auth', {method: 'POST', body: new URLSearchParams({token, state})})
    .then(response => {document.getElementById('status').textContent = response.ok
      ? 'Sign-in received. Return to Cabana or your terminal.' : 'Sign-in failed. Please try again.';})
    .catch(() => {document.getElementById('status').textContent = 'Sign-in failed. Please try again.';});
} else {
  document.getElementById('status').textContent = 'Sign-in did not return a token. Please try again.';
}
</script>"""


def auth_redirect_link(method, port, state):
  if method not in PROVIDERS:
    raise NotImplementedError(f"no redirect implemented for method {method}")
  callback = f'http://localhost:{port}/auth?' + urlencode({'state': state})
  return API_HOST.rstrip('/') + f'/auth/{method}?' + urlencode({'redirect': callback})


def login(method, timeout=180):
  try:
    with ClientRedirectServer(('localhost', 0), ClientRedirectHandler) as server:
      url = auth_redirect_link(method, server.server_port, server.state)
      print(f'To sign in, use your browser and navigate to {url}', file=sys.stderr)
      browser = subprocess.Popen(['open' if sys.platform == 'darwin' else 'xdg-open', url],
                                 stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
      deadline = time.monotonic() + timeout
      while time.monotonic() < deadline:
        server.timeout = min(0.1, max(0, deadline - time.monotonic()))
        server.handle_request()
        params = server.query_params
        if 'error' in params:
          return {"error": "Sign-in was declined. Please try again."}
        if 'token' in params:
          token = params['token'][0]
          CommaApi(token).get('v1/me', timeout=30)
          set_token(token)
          return {"success": True}
        if browser.poll() not in (None, 0):
          return {"error": "Could not open your browser. Check your default browser and try again."}
      return {"error": "Sign-in timed out. Please try again."}
  except Exception:
    return {"error": "Could not complete sign-in. Check your connection and try again."}


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Login to your Konn3kt account')
  parser.add_argument('method', default='github', const='github', nargs='?', choices=[*PROVIDERS, 'jwt'])
  parser.add_argument('jwt', nargs='?')
  parser.add_argument('--json', action='store_true', help='Return browser sign-in status as JSON')

  args = parser.parse_args()
  if args.json:
    if args.method == 'jwt':
      parser.error('--json requires a browser sign-in provider')
    print(json.dumps(login(args.method)))
    sys.exit(0)

  if args.method == 'jwt':
    if args.jwt is None:
      print("method JWT selected, but no JWT was provided")
      exit(1)

    set_token(args.jwt)
  else:
    result = login(args.method)
    if "error" in result:
      print(result["error"], file=sys.stderr)
      sys.exit(1)

  try:
    me = CommaApi(token=get_token()).get('/v1/me')
    print("Authenticated!")
    pprint.pprint(me)
  except UnauthorizedError:
    print("Got invalid JWT")
    exit(1)
