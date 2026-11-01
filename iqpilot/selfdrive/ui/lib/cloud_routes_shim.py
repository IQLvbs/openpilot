"""Public, konn3kt-agnostic shim to the private cloud route client.
"""
from __future__ import annotations

from functools import cache

UPLOAD_NONE = "none"
UPLOAD_UPLOADING = "uploading"
UPLOAD_UPLOADED = "uploaded"

@cache
def _load_cloud():
  try:
    from iqpilot.system.proprietary_runtime._verified_import import import_verified_module
    return import_verified_module("iqpilot_hephaestusd_private",
                                  "iqpilot_private.konn3kt.hephaestus.cloud_routes")
  except Exception:
    return None

def cloud_available() -> bool:
  return _load_cloud() is not None


def get_dongle_id() -> str | None:
  cloud = _load_cloud()
  if cloud is None:
    return None
  try:
    return cloud.get_dongle_id()
  except Exception:
    return None


def list_cloud_routes(dongle_id: str) -> list:
  cloud = _load_cloud()
  if cloud is None:
    return []
  try:
    return cloud.list_cloud_routes(dongle_id)
  except Exception:
    return []


def cloud_route_road_segments(dongle_id: str, fullname: str) -> list:
  cloud = _load_cloud()
  if cloud is None:
    return []
  try:
    return cloud.cloud_route_road_segments(dongle_id, fullname)
  except Exception:
    return []


def cloud_route_camera_urls(dongle_id: str, fullname: str, camera: str = "road") -> list:
  cloud = _load_cloud()
  if cloud is None:
    return []
  try:
    return cloud.cloud_route_camera_urls(dongle_id, fullname, camera)
  except Exception:
    return []


def request_mp4_conversion(dongle_id: str, segment_canonical_name: str, camera: str):
  cloud = _load_cloud()
  if cloud is None:
    return None
  try:
    return cloud.request_mp4_conversion(dongle_id, segment_canonical_name, camera)
  except Exception:
    return None


def get_mp4_conversion(dongle_id: str, segment_canonical_name: str, camera: str):
  cloud = _load_cloud()
  if cloud is None:
    return None
  try:
    return cloud.get_mp4_conversion(dongle_id, segment_canonical_name, camera)
  except Exception:
    return None


def merge_routes(local_routes: list, cloud_routes: list) -> list:
  cloud = _load_cloud()
  if cloud is not None:
    try:
      return cloud.merge_routes(local_routes, cloud_routes)
    except Exception:
      pass
  return [_LocalOnly(local) for local in local_routes]


class _LocalOnly:
  def __init__(self, local):
    self.name = local.name
    self.local = local
    self.cloud = None
    self.is_local = True
    self.is_cloud = False
    self.upload_state = UPLOAD_NONE
