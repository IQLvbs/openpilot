"""Destination search + persistence for the offroad Navigate screen.

Uses Mapbox's Search Box API (the same public Mapbox service the nav map preview already calls) for
POI-and-address autocomplete biased to the device's location — so "Walmart" returns the nearest
Walmart store, not a street named Walmart, and partial addresses complete as you type. Selecting a
result writes NavigationDestination, which navd picks up to build the route.

Home/Work/Recents live in a small JSON file on /data (persistent, no new param key needed).
"""
from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path
from dataclasses import dataclass
from types import SimpleNamespace

import requests

from iqpilot.common.params import Params
from iqpilot.common.swaglog import cloudlog
from iqpilot.system.hardware.hw import Paths
from iqpilot.selfdrive.ui.lib.nav_helpers import resolve_mapbox_token, current_or_last_gps_position

SEARCHBOX = "https://api.mapbox.com/search/searchbox/v1"
FAVORITES_PATH = str(Path(Paths.params()).parent / "nav_favorites.json")
MAX_RESULTS = 6
MAX_RECENTS = 8


def _load_amap_client():
  try:
    from iqpilot.system.proprietary_runtime._verified_import import import_verified_module
    return import_verified_module("iqpilot_navd_private", "iqpilot_private.navd.amap_client")
  except Exception:
    return None


_amap_client = _load_amap_client()


@dataclass
class SearchResult:
  name: str
  address: str
  mapbox_id: str = ""
  distance_m: float | None = None
  lat: float | None = None
  lon: float | None = None
  provider: str = "mapbox"

  @property
  def has_coords(self) -> bool:
    return self.lat is not None and self.lon is not None


class NavSearch:
  """Threaded, debounced Search Box autocomplete. The UI calls search() as the user types and reads
  results()/searching each frame; a stale query's results are dropped so only the latest shows."""

  def __init__(self):
    self._params = Params()
    self._session = str(uuid.uuid4())
    self._lock = threading.Lock()
    self._results: list[SearchResult] = []
    self._seq = 0
    self._last_query = ""
    self._searching = False
    self._amap_adcode = ""
    self.error = ""

  def new_session(self) -> None:
    # A Search Box "session" groups suggest+retrieve for billing; start one per search visit.
    self._session = str(uuid.uuid4())
    with self._lock:
      self._seq += 1
      self._results = []
      self._searching = False
    self._last_query = ""
    self._amap_adcode = ""
    self.error = ""

  def results(self) -> list[SearchResult]:
    with self._lock:
      return list(self._results)

  @property
  def searching(self) -> bool:
    return self._searching

  def search(self, query: str) -> None:
    query = query.strip()
    if query == self._last_query:
      return
    self._last_query = query
    self.error = ""
    self._seq += 1
    seq = self._seq
    if len(query) < 2:
      with self._lock:
        self._results = []
      self._searching = False
      return
    # Coordinate entry works completely offline, including Home/Work setup.
    try:
      lat, lon = (float(v.strip()) for v in query.split(","))
      if -90 <= lat <= 90 and -180 <= lon <= 180:
        with self._lock:
          self._results = [SearchResult(query, "Coordinates", lat=lat, lon=lon, provider="coordinates")]
        self._searching = False
        return
    except ValueError:
      pass
    self._searching = True
    threading.Thread(target=self._do_search, args=(query, seq), daemon=True).start()

  def _do_search(self, query: str, seq: int) -> None:
    try:
      lat, lon, _, fix = current_or_last_gps_position(self._params)
      position = SimpleNamespace(latitude=lat, longitude=lon) if fix else None
      use_amap = _amap_client is not None and _amap_client.is_mainland_china_configured(self._params, position)
      if use_amap:
        key = _amap_client.get_key(self._params)
        if not self._amap_adcode and position is not None:
          self._amap_adcode = _amap_client.reverse_adcode(position, key)
        amap_results = _amap_client.autocomplete(
          query,
          key,
          city=self._amap_adcode,
          position=position,
        )
        try:
          self._params.put("AmapStatus", _amap_client.status())
        except Exception:
          pass
        results = [
          SearchResult(
            name=item.name,
            address=item.address,
            mapbox_id=item.provider_id,
            lat=item.latitude,
            lon=item.longitude,
            provider="amap",
          )
          for item in amap_results[:MAX_RESULTS]
        ]
        if seq == self._seq:
          with self._lock:
            self._results = results
        return

      token = resolve_mapbox_token(self._params)
      if not token:
        with self._lock:
          if seq == self._seq:
            self._results = []
            self.error = "Configure address search, enter latitude, longitude, or choose a point on the map."
        return
      params = {"q": query, "access_token": token, "session_token": self._session,
                "limit": MAX_RESULTS, "language": "en"}
      if fix:
        params["proximity"] = f"{lon},{lat}"
      resp = requests.get(f"{SEARCHBOX}/suggest", params=params, timeout=8)
      resp.raise_for_status()
      results = []
      for s in resp.json().get("suggestions", []):
        mid = s.get("mapbox_id")
        addr = s.get("full_address") or s.get("place_formatted") or ""
        # Skip brand/category refinement rows (e.g. "Walmart · Brand") — not a single routable place.
        if not mid or not addr or s.get("feature_type") in ("category", "brand"):
          continue
        results.append(SearchResult(name=s.get("name", ""), address=addr, mapbox_id=mid,
                                    distance_m=s.get("distance")))
      if seq == self._seq:
        with self._lock:
          self._results = results
    except Exception as e:
      cloudlog.event("nav_search.suggest_failed", error=str(e))
      if seq == self._seq:
        with self._lock:
          self._results = []
          self.error = "Search unavailable. Check your connection and search provider."
    finally:
      if seq == self._seq:
        self._searching = False

  def retrieve(self, result: SearchResult) -> SearchResult | None:
    """Resolve a suggestion's coordinates (Search Box suggest omits them by design)."""
    if result.has_coords:
      return result
    try:
      if result.provider == "amap":
        if _amap_client is None:
          return None
        item = _amap_client.place_detail(result.mapbox_id, _amap_client.get_key(self._params))
        try:
          self._params.put("AmapStatus", _amap_client.status())
        except Exception:
          pass
        if item is None or item.latitude is None or item.longitude is None:
          return None
        result.lat, result.lon = item.latitude, item.longitude
        result.name = item.name or result.name
        result.address = item.address or result.address
        return result

      token = resolve_mapbox_token(self._params)
      resp = requests.get(f"{SEARCHBOX}/retrieve/{result.mapbox_id}",
                          params={"access_token": token, "session_token": self._session}, timeout=8)
      resp.raise_for_status()
      feats = resp.json().get("features", [])
      if not feats:
        return None
      coords = feats[0]["geometry"]["coordinates"]
      props = feats[0].get("properties", {})
      result.lon, result.lat = float(coords[0]), float(coords[1])
      result.name = props.get("name") or result.name
      result.address = props.get("full_address") or result.address
      return result
    except Exception as e:
      cloudlog.event("nav_search.retrieve_failed", error=str(e))
      return None


# --- destination + favorites persistence -------------------------------------------------------

def set_destination(lat: float, lon: float, name: str) -> None:
  """Hand a destination to navd (it routes off NavigationDestination). Mirrors hephaestusd's
  setNavDestination so a fresh route is always recomputed."""
  params = Params()
  params.remove("AthenaNavigationRoute")
  params.put_bool("NavigationActive", False)
  # NavigationDestination is a JSON-typed param: pass the object, not a pre-serialized string.
  params.put("NavigationDestination", {"latitude": float(lat), "longitude": float(lon), "name": name or ""})


def cancel_navigation() -> None:
  """Clear the active route/destination so navd stops navigating."""
  params = Params()
  params.remove("NavigationDestination")
  params.remove("AthenaNavigationRoute")
  params.put_bool("NavigationActive", False)


def _load_favorites() -> dict:
  try:
    with open(FAVORITES_PATH) as f:
      data = json.load(f)
    return data if isinstance(data, dict) else {}
  except Exception:
    return {}


def _save_favorites(data: dict) -> None:
  try:
    tmp = FAVORITES_PATH + ".tmp"
    with open(tmp, "w") as f:
      json.dump(data, f)
    import os
    os.replace(tmp, FAVORITES_PATH)
  except Exception as e:
    cloudlog.event("nav_search.save_favorites_failed", error=str(e))


def _place_to_result(place: dict | None) -> SearchResult | None:
  if not place:
    return None
  try:
    return SearchResult(name=place.get("name", ""), address=place.get("address", ""),
                        lat=float(place["lat"]), lon=float(place["lon"]))
  except Exception:
    return None


def get_home() -> SearchResult | None:
  return _place_to_result(_load_favorites().get("home"))


def get_work() -> SearchResult | None:
  return _place_to_result(_load_favorites().get("work"))


def _place_dict(r: SearchResult) -> dict:
  return {"name": r.name, "address": r.address, "lat": r.lat, "lon": r.lon}


def save_home(r: SearchResult) -> None:
  data = _load_favorites()
  data["home"] = _place_dict(r)
  _save_favorites(data)


def save_work(r: SearchResult) -> None:
  data = _load_favorites()
  data["work"] = _place_dict(r)
  _save_favorites(data)


def remove_home() -> None:
  data = _load_favorites()
  data.pop("home", None)
  _save_favorites(data)


def remove_work() -> None:
  data = _load_favorites()
  data.pop("work", None)
  _save_favorites(data)


def remove_recent(r: SearchResult) -> None:
  data = _load_favorites()
  data["recents"] = [p for p in data.get("recents", [])
                     if not (abs(p.get("lat", 0) - (r.lat or 0)) < 1e-5 and abs(p.get("lon", 0) - (r.lon or 0)) < 1e-5)]
  _save_favorites(data)


def get_recents() -> list[SearchResult]:
  out = []
  for p in _load_favorites().get("recents", []):
    r = _place_to_result(p)
    if r is not None:
      out.append(r)
  return out


def add_recent(r: SearchResult) -> None:
  if not r.has_coords:
    return
  data = _load_favorites()
  recents = [p for p in data.get("recents", [])
             if not (abs(p.get("lat", 0) - r.lat) < 1e-5 and abs(p.get("lon", 0) - r.lon) < 1e-5)]
  recents.insert(0, _place_dict(r))
  data["recents"] = recents[:MAX_RECENTS]
  _save_favorites(data)
