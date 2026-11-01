from pathlib import Path

from iqpilot.selfdrive.ui.lib.local_routes import list_local_routes


def _touch(path: Path) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_bytes(b"")


def test_list_local_routes_from_segment_directories(tmp_path):
  route_name = "00000051--3141cf1d76"
  _touch(tmp_path / f"{route_name}--0" / "qcamera.ts")
  _touch(tmp_path / f"{route_name}--1" / "qlog.zst")

  routes = list_local_routes(tmp_path)

  assert len(routes) == 1
  assert routes[0].name == route_name
  assert routes[0].segment_count == 2
  assert routes[0].cameras == ("road",)


def test_list_local_routes_from_single_segment_directory(tmp_path):
  route_name = "00000052--3141cf1d77"
  _touch(tmp_path / f"{route_name}--0" / "qcamera.ts")

  routes = list_local_routes(tmp_path)

  assert len(routes) == 1
  assert routes[0].name == route_name
  assert routes[0].subtitle == "1:00  ·  Road Cam"


def test_list_local_routes_ignores_invalid_entries(tmp_path):
  _touch(tmp_path / "not-a-route" / "fcamera.hevc")

  assert list_local_routes(tmp_path) == []
