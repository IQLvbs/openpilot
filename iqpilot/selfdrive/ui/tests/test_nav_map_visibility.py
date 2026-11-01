from types import SimpleNamespace

import pytest

from iqpilot.ui.onroad import nav_map_panel as nav


@pytest.fixture
def scene(monkeypatch, mocker):
  params = {'NavigationEnabled': True, 'OnScreenNavigation': True, 'OnlineOSMaps': True, 'OfflineOSMaps': False}
  panel = nav.NavMapPanel.__new__(nav.NavMapPanel)
  panel._params = SimpleNamespace(get_bool=lambda key: params.get(key, False), put_bool=params.__setitem__)
  panel._force_visible = False
  panel._last_params_refresh = -float('inf')
  panel.active = True
  panel._release_providers = mocker.Mock()
  clock = SimpleNamespace(now=10.0)
  sources = SimpleNamespace(downloaded=False)
  monkeypatch.setattr(nav.time, 'monotonic', lambda: clock.now)
  monkeypatch.setattr(nav, 'gui_app', SimpleNamespace(big_ui=lambda: True))
  monkeypatch.setattr(nav, 'find_offline_mbtiles_path', lambda: '/maps/offline.mbtiles' if sources.downloaded else None)
  monkeypatch.setattr(nav, 'find_offline_xyz_root', lambda: None)
  return SimpleNamespace(panel=panel, params=params, clock=clock, sources=sources)


@pytest.mark.parametrize('navigation_enabled', [False, True])
@pytest.mark.parametrize('maps_enabled', [False, True])
@pytest.mark.parametrize('downloaded', [False, True])
def test_map_visibility_requires_navigation_even_with_downloaded_maps(scene, navigation_enabled, maps_enabled, downloaded):
  scene.params['NavigationEnabled'] = navigation_enabled
  scene.params['OnScreenNavigation'] = maps_enabled
  scene.sources.downloaded = downloaded
  assert scene.panel.maps_enabled() is (navigation_enabled and (maps_enabled or downloaded))
  assert scene.params['NavigationEnabled'] is navigation_enabled
  if not navigation_enabled:
    assert scene.params['OnScreenNavigation'] is maps_enabled
    assert not scene.params['OfflineOSMaps']


def test_disabling_navigation_hides_existing_map_and_releases_tiles(scene):
  scene.sources.downloaded = True
  assert scene.panel.maps_enabled()
  scene.params['NavigationEnabled'] = False
  scene.clock.now += nav.PARAMS_REFRESH_S
  scene.panel.update()
  assert not scene.panel.active
  assert not scene.panel.maps_enabled()
  scene.panel._release_providers.assert_called_once()


def test_enabling_map_display_cannot_bypass_navigation_master_switch(scene):
  scene.params['NavigationEnabled'] = False
  scene.sources.downloaded = True
  scene.panel.set_maps_enabled(True)
  assert not scene.panel.maps_enabled()
  assert not scene.panel.active
  scene.panel._release_providers.assert_called_once()
