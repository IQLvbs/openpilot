import math

try:
  import requests
except ImportError:
  requests = None

from iqpilot.common.slc_variables import EARTH_RADIUS


def calculate_bearing_offset(latitude, longitude, current_bearing, distance):
  """
  Calculate new GPS coordinates given a starting point, bearing, and distance.
  Used for Mapbox API lookahead calculations.

  Args:
    latitude: Starting latitude in degrees
    longitude: Starting longitude in degrees
    current_bearing: Bearing in degrees (0-360)
    distance: Distance to project in meters

  Returns:
    Tuple of (new_latitude, new_longitude) in degrees
  """
  bearing = math.radians(current_bearing)
  lat_rad = math.radians(latitude)
  lon_rad = math.radians(longitude)

  delta = distance / EARTH_RADIUS

  new_lat = math.asin(math.sin(lat_rad) * math.cos(delta) + math.cos(lat_rad) * math.sin(delta) * math.cos(bearing))
  new_lon = lon_rad + math.atan2(math.sin(bearing) * math.sin(delta) * math.cos(lat_rad), math.cos(delta) - math.sin(lat_rad) * math.sin(new_lat))
  return math.degrees(new_lat), math.degrees(new_lon)


def is_url_pingable(url):
  """
  Check if a URL is accessible and responding.
  Used to verify Mapbox/Overpass API availability before making requests.

  Args:
    url: URL to ping

  Returns:
    Boolean indicating if URL is accessible
  """
  if not url:
    return False

  if requests is None:
    return False

  if not hasattr(is_url_pingable, "session"):
    is_url_pingable.session = requests.Session()
    is_url_pingable.session.headers.update({"User-Agent": "iqpilot-ping-test/1.0"})

  try:
    response = is_url_pingable.session.head(url, timeout=10, allow_redirects=True)
    if response.status_code in (405, 501):
      response = is_url_pingable.session.get(url, timeout=10, allow_redirects=True, stream=True)

    is_accessible = response.ok
    response.close()
    return is_accessible
  except Exception:
    return False
