import os
import json
import requests
import tempfile
from requests.adapters import HTTPAdapter, Retry

from pathlib import Path

API_HOST = os.getenv('API_HOST', 'https://api-iqlabs.konn3kt.com')

# TODO: this should be merged into common.api

class CommaApi:
  def __init__(self, token=None):
    self.session = requests.Session()
    self.session.headers['User-agent'] = 'OpenpilotTools'
    if token:
      self.session.headers['Authorization'] = 'JWT ' + token

    retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    self.session.mount('https://', HTTPAdapter(max_retries=retries))

  def request(self, method, endpoint, **kwargs):
    with self.session.request(method, API_HOST + '/' + endpoint, **kwargs) as resp:
      if resp.status_code in (401, 403):
        raise UnauthorizedError('Your Konn3kt login is missing, invalid, or expired. Please sign in again.')
      resp_json = resp.json()
      if isinstance(resp_json, dict) and resp_json.get('error'):
        e = APIError(str(resp.status_code) + ":" + resp_json.get('description', str(resp_json['error'])))
        e.status_code = resp.status_code
        raise e
      resp.raise_for_status()
      return resp_json

  def get(self, endpoint, **kwargs):
    return self.request('GET', endpoint, **kwargs)

  def post(self, endpoint, **kwargs):
    return self.request('POST', endpoint, **kwargs)

class APIError(Exception):
  pass

class UnauthorizedError(Exception):
  pass

def get_token():
  try:
    with open(os.path.join((Path.home() / ".iq"), 'auth.json')) as f:
      return json.load(f)['access_token']
  except Exception:
    return None

def set_token(token):
  root = Path.home() / '.iq'
  root.mkdir(mode=0o700, parents=True, exist_ok=True)
  with tempfile.NamedTemporaryFile(mode='w', dir=root, delete=False) as f:
    try:
      json.dump({'access_token': token}, f)
      f.close()
      os.replace(f.name, root / 'auth.json')
    finally:
      if os.path.exists(f.name):
        os.unlink(f.name)


def clear_token():
  try:
    os.unlink(os.path.join((Path.home() / ".iq"), 'auth.json'))
  except FileNotFoundError:
    pass
