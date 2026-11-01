from types import SimpleNamespace

import pytest

from iqpilot.ui.mici.onroad import navigation


@pytest.fixture
def scene(monkeypatch):
  nav = SimpleNamespace(active=True, destinationValid=True, destinationLatitude=41.0, destinationLongitude=-87.0,
                        nextManeuverValid=True, nextManeuverDistance=3000.0, nextManeuverDescription='Turn left onto Main Street',
                        nextManeuverType=SimpleNamespace(raw=1), nextManeuverDirection=SimpleNamespace(raw=1))
  render = SimpleNamespace(active=True, destinationLatitude=41.0, destinationLongitude=-87.0,
                           nextManeuverLatitude=40.5, nextManeuverLongitude=-87.0)

  class Messages(dict):
    alive = {'iqNavState': True, 'iqNavRenderState': True}
    valid = {'iqNavState': True, 'iqNavRenderState': True}

  params = {'NavigationDestination': {'latitude': 41.0, 'longitude': -87.0}, 'NavigationActive': True, 'NavigationEnabled': True}
  sm = Messages(iqNavState=nav, iqNavRenderState=render)
  state = SimpleNamespace(started=True, is_metric=False, sm=sm,
                          params=SimpleNamespace(get=params.get, get_bool=lambda key: bool(params.get(key))))
  clock = SimpleNamespace(now=10.0)
  monkeypatch.setattr(navigation, 'ui_state', state)
  monkeypatch.setattr(navigation, 'gui_app', SimpleNamespace(font=lambda _: None))
  monkeypatch.setattr(navigation.time, 'monotonic', lambda: clock.now)
  screen = navigation.MiciNavigation()
  screen.update()
  return SimpleNamespace(screen=screen, nav=nav, render=render, sm=sm, state=state, params=params, clock=clock)


@pytest.mark.parametrize('invalid', ['missing', 'malformed', 'cancelled', 'disabled', 'offroad', 'stale', 'invalid_message',
                                     'inactive', 'wrong_destination', 'no_destination', 'no_maneuver', 'nan', 'negative'])
def test_invalid_guidance_returns_to_camera(scene, invalid):
  assert scene.screen.active
  if invalid == 'missing':
    scene.params.pop('NavigationDestination')
  elif invalid == 'malformed':
    scene.params['NavigationDestination'] = {'latitude': float('nan'), 'longitude': -87.0}
  elif invalid == 'cancelled':
    scene.params['NavigationActive'] = False
  elif invalid == 'disabled':
    scene.params['NavigationEnabled'] = False
  elif invalid == 'offroad':
    scene.state.started = False
  elif invalid == 'stale':
    scene.sm.alive['iqNavState'] = False
  elif invalid == 'invalid_message':
    scene.sm.valid['iqNavState'] = False
  else:
    field, value = {'inactive': ('active', False), 'wrong_destination': ('destinationLatitude', 42.0),
                    'no_destination': ('destinationValid', False), 'no_maneuver': ('nextManeuverValid', False),
                    'nan': ('nextManeuverDistance', float('nan')), 'negative': ('nextManeuverDistance', -1.0)}[invalid]
    setattr(scene.nav, field, value)
  scene.screen.update()
  assert not scene.screen.active


def dismiss(scene):
  scene.screen.tap()
  assert scene.screen.active
  scene.clock.now += 0.2
  scene.screen.tap()
  assert not scene.screen.active


def test_double_tap_hides_until_point_nine_miles(scene):
  dismiss(scene)
  scene.nav.nextManeuverDistance = navigation.RETURN_DISTANCE_M + 1
  scene.screen.update()
  assert not scene.screen.active
  scene.nav.nextManeuverDistance = navigation.RETURN_DISTANCE_M
  scene.screen.update()
  assert scene.screen.active


def test_nearby_maneuver_can_be_dismissed_until_following_turn(scene):
  scene.nav.nextManeuverDistance = 500
  scene.screen.update()
  dismiss(scene)
  scene.nav.nextManeuverDistance = 100
  scene.screen.update()
  assert not scene.screen.active
  # Next turn has the same instruction, but a different location.
  scene.render.nextManeuverLatitude += 0.001
  scene.nav.nextManeuverDistance = 180
  scene.screen.update()
  assert scene.screen.active


def test_render_stream_dropout_does_not_undo_dismissal(scene):
  scene.nav.nextManeuverDistance = 500
  scene.screen.update()
  dismiss(scene)
  scene.sm.alive['iqNavRenderState'] = False
  scene.screen.update()
  assert not scene.screen.active


def test_slow_taps_do_not_dismiss(scene):
  scene.screen.tap()
  scene.clock.now += 1
  scene.screen.tap()
  assert scene.screen.active


def test_new_destination_clears_dismissal_but_waits_for_matching_guidance(scene):
  dismiss(scene)
  scene.params['NavigationDestination']['latitude'] = 42
  scene.screen.update()
  assert not scene.screen.active
  scene.nav.destinationLatitude = 42
  scene.screen.update()
  assert scene.screen.active


def test_guidance_recovers_without_waiting_screen(scene):
  scene.sm.alive['iqNavState'] = False
  scene.screen.update()
  assert not scene.screen.active
  scene.sm.alive['iqNavState'] = True
  scene.screen.update()
  assert scene.screen.active
  assert scene.screen._street == 'Turn left onto Main Street'


def test_master_switch_hides_live_guidance_and_can_reenable_it(scene):
  scene.params['NavigationEnabled'] = False
  scene.screen.update()
  assert not scene.screen.active
  scene.params['NavigationEnabled'] = True
  scene.screen.update()
  assert scene.screen.active


def test_navigation_taps_are_consumed_before_home_callback(scene, mocker):
  import pyray as rl
  from iqpilot.selfdrive.ui.mici.onroad.augmented_road_view import AugmentedRoadView
  from iqpilot.system.ui.lib.application import MousePos

  view = SimpleNamespace(_navigation=scene.screen, _content_rect=rl.Rectangle(0, 0, 476, 240),
                         _bookmark_icon=mocker.Mock())
  AugmentedRoadView._handle_mouse_release(view, MousePos(200, 100))
  assert scene.screen.active
  scene.clock.now += 0.2
  AugmentedRoadView._handle_mouse_release(view, MousePos(200, 100))
  assert not scene.screen.active
  view._bookmark_icon.interacting.assert_not_called()
