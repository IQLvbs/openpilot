import math

EARTH_RADIUS_M = 6378137.0
TILE_SIZE = 256.0
NAV_DRIVE_ZOOM_BOOST = 1.0


class MapMarkerMotion:
  def __init__(self):
    self.pose = None
    self.predicted = None
    self.fix = None
    self.updated_at = 0.0

  def update(self, now, fix_time, latitude, longitude, heading, speed, yaw_rate, motion_valid):
    target = (latitude, longitude, heading)
    if not all(math.isfinite(value) for value in target):
      return self.pose
    dt = max(0.0, now - self.updated_at)
    fix = (fix_time, *target)
    reset = self.pose is None or dt > 0.5 or now < self.updated_at
    if self.pose is not None and fix != self.fix:
      north = math.radians(latitude - self.pose[0]) * EARTH_RADIUS_M
      east = math.radians((longitude - self.pose[1] + 180) % 360 - 180) * EARTH_RADIUS_M * math.cos(math.radians(latitude))
      reset |= math.hypot(north, east) > 50.0
    if reset or fix != self.fix:
      self.predicted = target
    elif motion_valid and math.isfinite(speed) and math.isfinite(yaw_rate) and speed > 0.1:
      elapsed = min(dt, max(0.0, fix_time + 1.25 - self.updated_at))
      lat, lon, bearing = self.predicted
      turn = -math.degrees(yaw_rate) * elapsed
      lat, lon = destination_point(lat, lon, bearing + turn * 0.5, speed * elapsed)
      self.predicted = (lat, (lon + 180) % 360 - 180, (bearing + turn) % 360)
    if reset:
      self.pose = target
    else:
      alpha = -math.expm1(-dt / 0.06)
      lat, lon, bearing = self.pose
      target_lat, target_lon, target_bearing = self.predicted
      self.pose = (
        lat + (target_lat - lat) * alpha,
        (lon + ((target_lon - lon + 180) % 360 - 180) * alpha + 180) % 360 - 180,
        (bearing + ((target_bearing - bearing + 180) % 360 - 180) * alpha) % 360,
      )
    self.fix = fix
    self.updated_at = now
    return self.pose


def closest_polyline_segment(points, x, y):
  best_index, best_fraction, best_distance = 0, 0.0, math.inf
  for index in range(len(points) - 1):
    ax, ay = points[index]
    bx, by = points[index + 1]
    dx, dy = bx - ax, by - ay
    length_squared = dx * dx + dy * dy
    fraction = min(1.0, max(0.0, ((x - ax) * dx + (y - ay) * dy) / length_squared)) if length_squared else 0.0
    distance = (x - ax - fraction * dx) ** 2 + (y - ay - fraction * dy) ** 2
    if distance < best_distance:
      best_index, best_fraction, best_distance = index, fraction, distance
  return best_index, best_fraction

def _mercator_normalized(latitude: float, longitude: float) -> tuple[float, float]:
  x = (longitude + 180.0) / 360.0
  siny = min(max(math.sin(math.radians(latitude)), -0.9999), 0.9999)
  y = 0.5 - math.log((1.0 + siny) / (1.0 - siny)) / (4.0 * math.pi)
  return x, y

def mercator_world_px(latitude: float, longitude: float, zoom: float) -> tuple[float, float]:
  world_size = TILE_SIZE * (2.0 ** zoom)
  nx, ny = _mercator_normalized(latitude, longitude)
  x = nx * world_size
  y = ny * world_size
  return x, y

def destination_point(latitude: float, longitude: float, bearing_deg: float, distance_m: float) -> tuple[float, float]:
  if abs(distance_m) < 1e-3:
    return latitude, longitude

  angular_distance = distance_m / EARTH_RADIUS_M
  bearing = math.radians(bearing_deg)
  lat1 = math.radians(latitude)
  lon1 = math.radians(longitude)

  sin_lat1 = math.sin(lat1)
  cos_lat1 = math.cos(lat1)
  sin_ad = math.sin(angular_distance)
  cos_ad = math.cos(angular_distance)

  lat2 = math.asin(sin_lat1 * cos_ad + cos_lat1 * sin_ad * math.cos(bearing))
  lon2 = lon1 + math.atan2(
    math.sin(bearing) * sin_ad * cos_lat1,
    cos_ad - sin_lat1 * math.sin(lat2),
  )
  return math.degrees(lat2), math.degrees(lon2)

def fit_zoom_for_points(points, width: float, height: float, max_zoom: float = 17.6,
                        min_zoom: float = 12.8, padding: float = 56.0) -> float:
  coords = [(float(point.latitude), float(point.longitude)) for point in points if point is not None]
  if len(coords) < 2:
    return max_zoom

  xs, ys = zip(*[_mercator_normalized(lat, lon) for lat, lon in coords], strict=True)
  span_x = max(max(xs) - min(xs), 1e-6)
  span_y = max(max(ys) - min(ys), 1e-6)

  usable_width = max(width - 2.0 * padding, 32.0)
  usable_height = max(height - 2.0 * padding, 32.0)
  zoom_x = math.log2(usable_width / (TILE_SIZE * span_x))
  zoom_y = math.log2(usable_height / (TILE_SIZE * span_y))
  return max(min(min(zoom_x, zoom_y), max_zoom), min_zoom)

def choose_nav_camera(current_latitude: float, current_longitude: float, bearing_deg: float, points,
                      width: float, height: float, preferred_zoom: float) -> tuple[float, float, float]:
  preferred_zoom += NAV_DRIVE_ZOOM_BOOST
  zoom = preferred_zoom
  if points:
    zoom = fit_zoom_for_points(points, width, height * 0.78, max_zoom=preferred_zoom + 0.6)
    zoom = min(max(zoom, preferred_zoom - 1.2), preferred_zoom + 0.6)

  meters_per_pixel = 156543.03392 * math.cos(math.radians(current_latitude)) / (2.0 ** zoom)
  lookahead_pixels = height * 0.16
  lookahead_m = max(lookahead_pixels * meters_per_pixel, 12.0)
  center_latitude, center_longitude = destination_point(current_latitude, current_longitude, bearing_deg, lookahead_m)
  return center_latitude, center_longitude, zoom

def build_mapbox_tile_url(z: int, x: int, y: int, tile_size: int = 256, scale: int = 2,
                          style: str = "navigation-night-v1") -> str:
  suffix = f"@{scale}x" if scale > 1 else ""
  return (
    f"https://api.mapbox.com/styles/v1/mapbox/{style}/tiles/" +
    f"{tile_size}/{z}/{x}/{y}{suffix}"
  )

def tile_world_size(z: int, tile_size: int = 256) -> int:
  return tile_size * (2 ** z)

def mercator_world_px_at_zoom(latitude: float, longitude: float, z: int, tile_size: int = 256) -> tuple[float, float]:
  world_size = tile_world_size(z, tile_size)
  nx, ny = _mercator_normalized(latitude, longitude)
  return nx * world_size, ny * world_size

def project_nav_point(latitude: float, longitude: float, center_latitude: float, center_longitude: float,
                      zoom: float, bearing_deg: float, width: float, height: float,
                      anchor_x: float = 0.5, anchor_y: float = 0.5) -> tuple[float, float]:
  px, py = mercator_world_px(latitude, longitude, zoom)
  cx, cy = mercator_world_px(center_latitude, center_longitude, zoom)
  dx = px - cx
  dy = py - cy

  theta = math.radians(bearing_deg)
  cos_theta = math.cos(theta)
  sin_theta = math.sin(theta)
  rx = dx * cos_theta + dy * sin_theta
  ry = -dx * sin_theta + dy * cos_theta
  return width * anchor_x + rx, height * anchor_y + ry

def project_nav_polyline(points, center_latitude: float, center_longitude: float, zoom: float, bearing_deg: float,
                         width: float, height: float, anchor_x: float = 0.5, anchor_y: float = 0.5) -> list[tuple[float, float]]:
  center_x, center_y = mercator_world_px(center_latitude, center_longitude, zoom)
  theta = math.radians(bearing_deg)
  cos_theta, sin_theta = math.cos(theta), math.sin(theta)
  origin_x, origin_y = width * anchor_x, height * anchor_y
  projected = []
  for point in points:
    x, y = mercator_world_px(float(point.latitude), float(point.longitude), zoom)
    dx, dy = x - center_x, y - center_y
    projected.append((origin_x + dx * cos_theta + dy * sin_theta,
                      origin_y - dx * sin_theta + dy * cos_theta))
  return projected
