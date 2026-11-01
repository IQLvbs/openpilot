# Earth radius in meters (for GPS calculations)
EARTH_RADIUS = 6378137

# Mapbox API limits
FREE_MAPBOX_REQUESTS = 100_000

# Speed limit offset zones for different unit systems
# Each entry is (min_speed_ms, max_speed_ms, param_name); the param value is a
# percent offset applied to the resolved limit (e.g. 10 -> +10%), lower bound inclusive

OFFSET_PERCENT_MAX = 50.0

OFFSET_MAP_IMPERIAL = [
  (0, 8.94, "speed_limit_offset1"),               # 0-20 mph
  (8.94, 17.88, "speed_limit_offset2"),           # 20-40 mph
  (17.88, float("inf"), "speed_limit_offset3"),   # 40+ mph
]

OFFSET_MAP_METRIC = [
  (0, 8.33, "speed_limit_offset1"),               # 0-30 km/h
  (8.33, 16.67, "speed_limit_offset2"),           # 30-60 km/h
  (16.67, float("inf"), "speed_limit_offset3"),   # 60+ km/h
]

# Speed limit filler constants
BOUNDING_BOX_RADIUS_DEGREE = 0.1
MAX_ENTRIES = 1_000_000
MAX_OVERPASS_DATA_BYTES = 1_073_741_824
MAX_OVERPASS_REQUESTS = 10_000
METERS_PER_DEG_LAT = 111_320
VETTING_INTERVAL_DAYS = 7

# Overpass API URLs
OVERPASS_API_URL = "https://overpass-api.de/api/interpreter"
OVERPASS_STATUS_URL = "https://overpass-api.de/api/status"
