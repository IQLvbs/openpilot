"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos
"""
import threading


def _create_endpoint():
  from iqpilot.ui.reasoning_relay import RasterEndpoint

  return RasterEndpoint()


class OptionalPrivateUI:
  def __init__(self):
    self._endpoint = None
    self._started = False

  def _load(self):
    try:
      self._endpoint = _create_endpoint()
    except BaseException:
      self._endpoint = None

  def get(self):
    if not self._started:
      self._started = True
      try:
        threading.Thread(target=self._load, name="optional-ui", daemon=True).start()
      except Exception:
        self._endpoint = None
    return self._endpoint

  def render_onroad(self, rect):
    endpoint = self.get()
    if endpoint is not None:
      try:
        endpoint.render_onroad(rect)
      except Exception:
        self._endpoint = None
        if hasattr(endpoint, 'close'):
          endpoint.close()

  def attach(self, scroller):
    endpoint = self.get()
    if endpoint is None:
      return False
    try:
      return endpoint.attach(scroller) is not False
    except Exception:
      self._endpoint = None
      return False


optional_private_ui = OptionalPrivateUI()
