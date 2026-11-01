import time

from iqpilot.common.params import Params
from iqpilot.common.swaglog import cloudlog
from iqpilot.common.geo_regions import UNKNOWN_REGION, region_for_position, region_is_metric

CHECK_INTERVAL = 10.0
CONFIRMATIONS = 3


class AutoUnits:
  def __init__(self, params: Params | None = None):
    self.params = params or Params()
    self._next_check = 0.0
    self._candidate = UNKNOWN_REGION
    self._confirmations = 0

  def _position(self) -> tuple[float, float, bool]:
    from iqpilot.selfdrive.ui.lib.nav_helpers import current_or_last_gps_position

    lat, lon, _, valid = current_or_last_gps_position(self.params)
    return lat, lon, valid

  def update(self, now: float | None = None) -> None:
    if not self.params.get_bool("IQAutoUnits"):
      self._candidate = UNKNOWN_REGION
      self._confirmations = 0
      return

    now = time.monotonic() if now is None else now
    if now < self._next_check:
      return
    self._next_check = now + CHECK_INTERVAL

    lat, lon, valid = self._position()
    region = region_for_position(lat, lon) if valid else UNKNOWN_REGION
    if region == UNKNOWN_REGION:
      self._confirmations = 0
      return

    if region != self._candidate:
      self._candidate = region
      self._confirmations = 1
      return

    self._confirmations += 1
    if self._confirmations < CONFIRMATIONS:
      return

    if region == self.params.get("IQAutoUnitsRegion"):
      return

    self.params.put("IQAutoUnitsRegion", region)

    metric = region_is_metric(region)
    if metric != self.params.get_bool("IsMetric"):
      self.params.put_bool("IsMetric", metric)
      cloudlog.warning(f"auto units: {region} detected, switching to {'km/h' if metric else 'mph'}")
