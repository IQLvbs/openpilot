# Copyright (c) 2026 IQ.Lvbs. All rights reserved.
import os

from iqpilot.common.api.base import BaseApi

API_HOST = os.getenv('API_HOST', 'https://api-iqlabs.konn3kt.com')


class Konn3ktApi(BaseApi):
  def __init__(self, dongle_id):
    super().__init__(dongle_id, API_HOST)
    self.user_agent = "openpilot-"
