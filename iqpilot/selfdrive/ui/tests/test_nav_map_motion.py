# Copyright (c) 2026 IQ.Lvbs. All rights reserved.
import math
from types import SimpleNamespace

import pytest

from iqpilot.ui.onroad.nav_map_utils import MapMarkerMotion, closest_polyline_segment, destination_point, EARTH_RADIUS_M


def east_meters(pose):
  return math.radians(pose[1]) * EARTH_RADIUS_M


def test_marker_moves_between_one_hertz_fixes():
  motion = MapMarkerMotion()
  positions = []
  for frame in range(181):
    now = frame / 60
    fix_time = float(int(now))
    lat, lon = destination_point(0, 0, 90, 30 * fix_time)
    positions.append(east_meters(motion.update(now, fix_time, lat, lon, 90, 30, 0, True)))
  steps = [b - a for a, b in zip(positions[30:-1], positions[31:], strict=True)]
  assert min(steps) > 0.3
  assert max(steps) < 0.6
  assert positions[-1] == pytest.approx(90, abs=2.5)


@pytest.mark.parametrize('speed,valid', [(0, True), (30, False), (float('nan'), True)])
def test_no_extrapolation_without_valid_motion(speed, valid):
  motion = MapMarkerMotion()
  for frame in range(61):
    pose = motion.update(frame / 60, 0, 0, 0, 90, speed, 0, valid)
    assert pose == (0, 0, 90)


def test_prediction_freezes_after_fix_timeout_without_snapping_back():
  motion = MapMarkerMotion()
  for frame in range(301):
    pose = motion.update(frame / 60, 0, 0, 0, 90, 60, 0, True)
  assert east_meters(pose) == pytest.approx(75, abs=0.1)
  assert motion.update(5.01, 0, 0, 0, 90, 60, 0, True) == pytest.approx(pose)


def test_stop_does_not_continue_dead_reckoning():
  motion = MapMarkerMotion()
  for frame in range(121):
    pose = motion.update(frame / 60, 0, 0, 0, 90, 20 if frame <= 30 else 0, 0, True)
  assert east_meters(pose) == pytest.approx(10, abs=0.1)


def test_yaw_and_heading_wraparound():
  motion = MapMarkerMotion()
  for frame in range(61):
    pose = motion.update(frame / 60, 0, 0, 0, 359, 10, -math.radians(10), True)
  assert 7 < pose[2] < 10
  assert all(math.isfinite(value) for value in pose)


def test_discontinuous_fix_resets_and_invalid_fix_preserves_pose():
  motion = MapMarkerMotion()
  motion.update(0, 0, 0, 0, 90, 30, 0, True)
  pose = motion.update(0.1, 0.1, 1, 1, 180, 30, 0, True)
  assert pose == (1, 1, 180)
  assert motion.update(0.2, 0.2, float('nan'), 1, 180, 30, 0, True) == pose
  assert motion.update(10, 10, 2, 2, 180, 30, 0, True) == (2, 2, 180)


@pytest.mark.parametrize('x,index,fraction', [(25, 0, 0.25), (75, 0, 0.75), (99, 0, 0.99), (101, 1, 0.01), (175, 1, 0.75)])
def test_route_trimming_is_continuous_inside_segments(x, index, fraction):
  points = [(0, 0), (100, 0), (200, 0)]
  result = closest_polyline_segment(points, x, 2)
  assert result == pytest.approx((index, fraction))


def test_segment_projection_handles_corners_and_duplicate_points():
  assert closest_polyline_segment([(0, 0), (0, 0), (100, 0), (100, 100)], 100, 40) == pytest.approx((2, 0.4))


def test_live_route_starts_under_marker_using_cached_camera(monkeypatch):
  from iqpilot.ui.onroad import nav_map_panel as nav
  panel = nav.NavMapPanel.__new__(nav.NavMapPanel)
  panel.nav_active = True
  panel._route_camera = (0, 0, 16, 200, 200)
  panel._projected_route_points = [(0, 100), (100, 100), (200, 100)]
  panel.destination_latitude = panel.destination_longitude = 0
  lines = []
  monkeypatch.setattr(nav.rl, 'draw_line_ex', lambda a, b, *args: lines.append((a.x, a.y, b.x, b.y)))
  rect = nav.rl.Rectangle(0, 0, 200, 200)
  panel._draw_route_overlay(rect, (0, 0, 90), (0, 0, 16, 0))
  assert all(line == pytest.approx((100, 100, 200, 100)) for line in lines)
  assert len(lines) == 3


def test_route_ahead_retains_segment_containing_vehicle():
  from iqpilot.ui.onroad.nav_map_panel import NavMapPanel
  panel = NavMapPanel.__new__(NavMapPanel)
  panel.current_latitude, panel.current_longitude = 0, 0.00075
  panel.route_points = [SimpleNamespace(latitude=0, longitude=x) for x in (0, 0.001, 0.002)]
  assert panel._route_ahead_points()[0] is panel.route_points[0]
