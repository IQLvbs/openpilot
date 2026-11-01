"""
Copyright (c) IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos
"""
from types import SimpleNamespace

import pytest

from iqpilot.selfdrive.ui.lib import nav_search


class FakeParams:
  def get(self, key, *args, **kwargs):
    return {
      "AmapWebServiceKey": "amap-key",
      "OsmLocationName": "CN",
      "MapboxToken": "mapbox-key",
    }.get(key)


@pytest.mark.parametrize('enabled', [False, True])
def test_selecting_destination_preserves_navigation_master_switch(monkeypatch, enabled):
  values = {'NavigationEnabled': enabled, 'NavigationActive': True, 'AthenaNavigationRoute': {}}
  params = SimpleNamespace(put=values.__setitem__, put_bool=values.__setitem__, remove=lambda key: values.pop(key, None))
  monkeypatch.setattr(nav_search, 'Params', lambda: params)
  nav_search.set_destination(41.0, -87.0, 'Home')
  assert values['NavigationEnabled'] is enabled
  assert values['NavigationDestination'] == {'latitude': 41.0, 'longitude': -87.0, 'name': 'Home'}
  assert not values['NavigationActive']
  assert 'AthenaNavigationRoute' not in values


def test_china_search_uses_amap_without_calling_mapbox(monkeypatch):
  search = nav_search.NavSearch.__new__(nav_search.NavSearch)
  search._params = FakeParams()
  search._seq = 1
  search._results = []
  search._searching = True
  search._amap_adcode = ""
  import threading
  search._lock = threading.Lock()

  amap_client = SimpleNamespace(
    is_mainland_china_configured=lambda *args, **kwargs: True,
    get_key=lambda *args, **kwargs: "amap-key",
    status=lambda: "ok",
    reverse_adcode=lambda *args, **kwargs: "",
    autocomplete=lambda *args, **kwargs: [],
  )
  monkeypatch.setattr(nav_search, "_amap_client", amap_client)
  monkeypatch.setattr(nav_search, "current_or_last_gps_position", lambda *_: (39.9, 116.4, 0.0, True))
  monkeypatch.setattr(amap_client, "reverse_adcode", lambda *args, **kwargs: "110000")
  monkeypatch.setattr(amap_client, "autocomplete", lambda *args, **kwargs: [
    SimpleNamespace(
      name="Tiananmen",
      address="Dongcheng, Beijing",
      provider_id="B000A83M61",
      latitude=39.9087,
      longitude=116.3975,
    ),
  ])
  monkeypatch.setattr(nav_search.requests, "get", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Mapbox called")))

  search._do_search("Tiananmen", 1)

  assert len(search._results) == 1
  assert search._results[0].provider == "amap"
  assert search._results[0].has_coords


def test_non_china_search_preserves_mapbox_request(monkeypatch):
  search = nav_search.NavSearch.__new__(nav_search.NavSearch)
  search._params = FakeParams()
  search._session = "session"
  search._seq = 1
  search._results = []
  search._searching = True
  search._amap_adcode = ""
  import threading
  search._lock = threading.Lock()
  captured = {}

  class Response:
    def raise_for_status(self):
      return None

    def json(self):
      return {"suggestions": []}

  def get(url, *, params, timeout):
    captured.update(url=url, params=params, timeout=timeout)
    return Response()

  monkeypatch.setattr(nav_search, "current_or_last_gps_position", lambda *_: (41.3, -90.2, 0.0, True))
  monkeypatch.setattr(nav_search, "resolve_mapbox_token", lambda *_: "mapbox-key")
  monkeypatch.setattr(nav_search, "_amap_client", None)
  monkeypatch.setattr(nav_search.requests, "get", get)

  search._do_search("Home", 1)

  assert captured["url"] == f"{nav_search.SEARCHBOX}/suggest"
  assert captured["params"] == {
    "q": "Home",
    "access_token": "mapbox-key",
    "session_token": "session",
    "limit": nav_search.MAX_RESULTS,
    "language": "en",
    "proximity": "-90.2,41.3",
  }
  assert captured["timeout"] == 8


def test_saved_addresses_and_recents_round_trip(tmp_path, monkeypatch):
  monkeypatch.setattr(nav_search, "FAVORITES_PATH", str(tmp_path / "nav_favorites.json"))
  home = nav_search.SearchResult("Home", "Home address", lat=41.0, lon=-87.0)
  work = nav_search.SearchResult("Work", "Work address", lat=41.1, lon=-87.1)
  nav_search.save_home(home)
  nav_search.save_work(work)
  nav_search.add_recent(home)
  nav_search.add_recent(work)
  nav_search.add_recent(home)
  assert nav_search.get_home() == home
  assert nav_search.get_work() == work
  assert nav_search.get_recents() == [home, work]
  nav_search.remove_home()
  nav_search.remove_work()
  nav_search.remove_recent(home)
  assert nav_search.get_home() is None
  assert nav_search.get_work() is None
  assert nav_search.get_recents() == [work]
