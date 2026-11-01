"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos
"""

from datetime import datetime
from types import SimpleNamespace

import pytest

from iqpilot.cereal import car, custom
from iqpilot.common.constants import CV
from iqpilot.common.slc_variables import OFFSET_MAP_IMPERIAL
from iqpilot.selfdrive.car.cruise import VCruiseHelper
from iqpilot.selfdrive.controls.lib.iq_longitudinal_planner import LongitudinalPlannerIQ
from iqpilot.selfdrive.controls.lib.slc_vcruise import SLCVCruise, CRUISING_SPEED
from iqpilot.selfdrive.controls.lib.speed_limit_controller import SpeedLimitController, POLICY_MAP_DATA_PRIORITY, POLICY_COMBINED


class FakeParams:
  def __init__(self):
    self.values = {}

  def get(self, key, encoding=None):
    _ = encoding
    return self.values.get(key)

  def get_bool(self, key):
    return bool(self.values.get(key, False))

  def put_nonblocking(self, key, value):
    self.values[key] = value

  def put(self, key, value):
    self.values[key] = value


def _build_sm(v_cruise_cluster=100.0, v_ego_cluster=27.8, gas=False, enabled=True, iq_limit=0.0):
  # vCruiseCluster is in kph in carState.
  return {
    "carState": SimpleNamespace(vCruiseCluster=v_cruise_cluster, vEgoCluster=v_ego_cluster, gasPressed=gas,
                                steeringAngleDeg=0.0, buttonEvents=[]),
    "iqCarState": SimpleNamespace(speedLimit=iq_limit, accelPressed=False, decelPressed=False),
    "selfdriveState": SimpleNamespace(enabled=enabled),
    "vehicleParameters": SimpleNamespace(angleOffsetDeg=0.0),
  }


class _FakeSLC:
  def __init__(self):
    self.target = 0.0
    self.source = "None"
    self.active_target = 0.0
    self.active_source = "None"
    self.unconfirmed_speed_limit = 0.0
    self.overridden_speed = 0.0
    self.pending_events = []
    self.assist_state = None
    self.output_a_target = 0.0
    self.update_limits_calls = 0
    self.update_override_calls = 0
    self._offset = 0.0

  def update_limits(self, *_args, **_kwargs):
    self.update_limits_calls += 1

  def update_override(self, *_args, **_kwargs):
    self.update_override_calls += 1

  def reset_override(self, _sm):
    self.overridden_speed = 0.0

  def get_offset(self, _is_metric):
    return self._offset


def _base_slc_params_controller():
  return {
    "slc_policy": POLICY_MAP_DATA_PRIORITY,
    "slc_auto_confirm": False,
    "slc_fallback_previous_speed_limit": False,
    "slc_fallback_set_speed": False,
    "speed_limit_confirmation_higher": False,
    "speed_limit_confirmation_lower": False,
    "slc_online_filler": True,
    "map_speed_lookahead_higher": 5.0,
    "map_speed_lookahead_lower": 5.0,
  }


def test_speed_limit_controller_resolves_source_by_priority():
  params = FakeParams()
  controller = SpeedLimitController(params)
  controller.update_gps = lambda _sm: None
  controller._resolver.update_map_data = lambda *_args, **_kwargs: None
  controller.get_mapbox_speed_limit = lambda *_args, **_kwargs: None
  controller.mapbox_requests["total_requests"] = 0
  controller.mapbox_requests["max_requests"] = 999999
  controller.mapbox_limit = 22.0
  controller._resolver.map_speed_limit = 18.0  # map data wins in map_data_priority policy

  sm = _build_sm(iq_limit=25.0)
  slc_params = _base_slc_params_controller()
  slc_params["slc_policy"] = POLICY_MAP_DATA_PRIORITY

  controller.update_limits(25.0, datetime.now(), True, 30.0, 27.0, sm, slc_params)
  assert controller.active_source == "Map Data"
  assert controller.active_target == 18.0


def test_speed_limit_controller_combined_mode_prefers_smallest_limit():
  params = FakeParams()
  controller = SpeedLimitController(params)
  controller.update_gps = lambda _sm: None
  controller._resolver.update_map_data = lambda *_args, **_kwargs: None
  controller.get_mapbox_speed_limit = lambda *_args, **_kwargs: None
  controller.mapbox_requests["total_requests"] = 0
  controller.mapbox_requests["max_requests"] = 999999
  controller.mapbox_limit = 24.0
  controller._resolver.map_speed_limit = 16.0  # smallest of: dashboard=28, mapbox=24, map_data=16

  sm = _build_sm(iq_limit=28.0)
  slc_params = _base_slc_params_controller()
  slc_params["slc_policy"] = POLICY_COMBINED

  controller.update_limits(28.0, datetime.now(), True, 31.0, 27.0, sm, slc_params)
  assert controller.active_source == "Map Data"
  assert controller.active_target == 16.0


def test_slc_vcruise_applies_target_without_increasing_cruise():
  slc = SLCVCruise()
  slc.slc = _FakeSLC()
  slc.slc.target = 23.0
  slc.slc.source = "Dashboard"
  slc.slc.active_target = 23.0
  slc.slc.active_source = "Dashboard"
  slc.slc._offset = 1.0

  slc._get_slc_params = lambda: {
    "speed_limit_controller": True,
    "speed_limit_mode": 3,
    "show_speed_limits": False,
    "is_metric": True,
    "slc_policy": POLICY_MAP_DATA_PRIORITY,
    "slc_auto_confirm": False,
    "speed_limit_confirmation_higher": False,
    "speed_limit_confirmation_lower": False,
    "map_speed_lookahead_higher": 5.0,
    "map_speed_lookahead_lower": 5.0,
    "slc_fallback_experimental_mode": False,
    "slc_fallback_set_speed": False,
    "slc_fallback_previous_speed_limit": False,
    "speed_limit_controller_override_manual": True,
    "speed_limit_controller_override_set_speed": False,
    "slc_online_filler": False,
  }

  v_cruise = 30.0
  sm = _build_sm(v_cruise_cluster=v_cruise * CV.MS_TO_KPH, v_ego_cluster=27.0, iq_limit=23.0)
  out = slc.update(apply_enabled=True, now=None, time_validated=True, v_cruise=v_cruise, v_ego=27.0, sm=sm)

  assert slc.slc.update_limits_calls == 1
  assert slc.slc.update_override_calls == 1
  assert out <= v_cruise
  assert out >= CRUISING_SPEED


def test_slc_vcruise_show_only_does_not_modify_cruise():
  slc = SLCVCruise()
  slc.slc = _FakeSLC()
  slc.slc.target = 21.0
  slc.slc.source = "Map Data"
  slc.slc.active_target = 21.0
  slc.slc.active_source = "Map Data"
  slc._get_slc_params = lambda: {
    "speed_limit_controller": False,
    "speed_limit_mode": 1,
    "show_speed_limits": True,
    "is_metric": True,
    "slc_policy": POLICY_MAP_DATA_PRIORITY,
    "slc_auto_confirm": False,
    "speed_limit_confirmation_higher": False,
    "speed_limit_confirmation_lower": False,
    "map_speed_lookahead_higher": 5.0,
    "map_speed_lookahead_lower": 5.0,
    "slc_fallback_experimental_mode": False,
    "slc_fallback_set_speed": False,
    "slc_fallback_previous_speed_limit": False,
    "speed_limit_controller_override_manual": True,
    "speed_limit_controller_override_set_speed": False,
    "slc_online_filler": False,
  }

  v_cruise = 29.0
  sm = _build_sm(v_cruise_cluster=v_cruise * CV.MS_TO_KPH, v_ego_cluster=26.0, iq_limit=21.0)
  out = slc.update(apply_enabled=True, now=None, time_validated=True, v_cruise=v_cruise, v_ego=26.0, sm=sm)

  assert slc.slc.update_limits_calls == 1
  assert slc.slc.update_override_calls == 0
  assert out == v_cruise


def test_slc_vcruise_auto_raises_for_higher_limit_when_confirmation_disabled():
  slc = SLCVCruise()
  slc.slc = _FakeSLC()
  slc.slc.target = 20.0
  slc.slc.source = "Map Data"
  slc.slc.active_target = 20.0
  slc.slc.active_source = "Map Data"

  slc._get_slc_params = lambda: {
    "speed_limit_controller": True,
    "speed_limit_mode": 3,
    "show_speed_limits": False,
    "is_metric": True,
    "slc_policy": POLICY_MAP_DATA_PRIORITY,
    "slc_auto_confirm": False,
    "speed_limit_confirmation_higher": False,
    "speed_limit_confirmation_lower": False,
    "map_speed_lookahead_higher": 5.0,
    "map_speed_lookahead_lower": 5.0,
    "slc_fallback_experimental_mode": False,
    "slc_fallback_set_speed": False,
    "slc_fallback_previous_speed_limit": False,
    "speed_limit_controller_override_manual": True,
    "speed_limit_controller_override_set_speed": False,
    "slc_online_filler": False,
  }

  v_cruise = 13.5
  sm = _build_sm(v_cruise_cluster=v_cruise * CV.MS_TO_KPH, v_ego_cluster=13.5, iq_limit=20.0)
  out = slc.update(apply_enabled=True, now=None, time_validated=True, v_cruise=v_cruise, v_ego=13.5, sm=sm)

  assert out > v_cruise
  assert out == 20.0


def test_slc_vcruise_does_not_auto_raise_when_higher_confirmation_enabled():
  slc = SLCVCruise()
  slc.slc = _FakeSLC()
  slc.slc.target = 20.0
  slc.slc.source = "Map Data"
  slc.slc.active_target = 20.0
  slc.slc.active_source = "Map Data"

  slc._get_slc_params = lambda: {
    "speed_limit_controller": True,
    "speed_limit_mode": 3,
    "show_speed_limits": False,
    "is_metric": True,
    "slc_policy": POLICY_MAP_DATA_PRIORITY,
    "slc_auto_confirm": False,
    "speed_limit_confirmation_higher": True,
    "speed_limit_confirmation_lower": False,
    "map_speed_lookahead_higher": 5.0,
    "map_speed_lookahead_lower": 5.0,
    "slc_fallback_experimental_mode": False,
    "slc_fallback_set_speed": False,
    "slc_fallback_previous_speed_limit": False,
    "speed_limit_controller_override_manual": True,
    "speed_limit_controller_override_set_speed": False,
    "slc_online_filler": False,
  }

  v_cruise = 13.5
  sm = _build_sm(v_cruise_cluster=v_cruise * CV.MS_TO_KPH, v_ego_cluster=13.5, iq_limit=20.0)
  out = slc.update(apply_enabled=True, now=None, time_validated=True, v_cruise=v_cruise, v_ego=13.5, sm=sm)

  assert out == v_cruise


@pytest.fixture(params=[True, False], ids=["metric", "imperial"])
def set_speed_slc(request, monkeypatch):
  monkeypatch.setattr("iqpilot.selfdrive.controls.lib.slc_vcruise.Params", FakeParams)
  slc = SLCVCruise()
  slc._maybe_log_debug = lambda *_args: None
  slc.slc.update_gps = lambda _sm: None
  slc.slc._resolver.update_map_data = lambda *_args: None
  params = _base_slc_params_controller() | {
    "speed_limit_controller": True,
    "speed_limit_mode": 3,
    "show_speed_limits": False,
    "is_metric": request.param,
    "slc_online_filler": False,
    "slc_fallback_experimental_mode": False,
    "speed_limit_controller_override_manual": False,
    "speed_limit_controller_override_set_speed": True,
  }
  slc._get_slc_params = lambda: params
  unit = CV.KPH_TO_MS if request.param else CV.MPH_TO_MS
  slc.slc._resolver.map_speed_limit = 50 * unit
  sm = _FakeSM(_build_sm(v_ego_cluster=50 * unit))
  sm["iqCarState"].slcSetSpeedRequestId = 0
  sm["iqCarState"].slcSetSpeedGestureId = 0
  sm["iqCarState"].slcSetSpeedRequestKph = 0.0

  def step(speed, increase=False, new_gesture=False):
    sm["carState"].vCruiseCluster = speed * unit * CV.MS_TO_KPH
    if new_gesture:
      sm["iqCarState"].slcSetSpeedGestureId += 1
    if increase:
      sm["iqCarState"].slcSetSpeedRequestId += 1
      sm["iqCarState"].slcSetSpeedRequestKph = sm["carState"].vCruiseCluster
    target = slc.update(sm["selfdriveState"].enabled, None, True, speed * unit, 50 * unit, sm)
    return min(speed * unit, target) / unit

  step(50)
  return SimpleNamespace(slc=slc, params=params, sm=sm, unit=unit, step=step)


@pytest.mark.parametrize("confirm_higher", [False, True])
def test_set_speed_override_tracks_driver_adjustments(set_speed_slc, confirm_higher):
  system = set_speed_slc
  system.params["speed_limit_confirmation_higher"] = confirm_higher
  assert system.step(50, new_gesture=True) == pytest.approx(50)
  assert system.step(55, increase=True) == pytest.approx(55)
  assert system.step(60, increase=True) == pytest.approx(60)
  assert system.step(60) == pytest.approx(60)
  assert system.step(55) == pytest.approx(55)
  assert system.step(50) == pytest.approx(50)
  assert not system.slc.slc.override_slc
  assert system.step(45) == pytest.approx(45)
  assert system.step(60) == pytest.approx(50)
  assert system.step(61, increase=True, new_gesture=True) == pytest.approx(61)


def test_set_speed_override_ignores_automatic_speed_changes(set_speed_slc):
  system = set_speed_slc
  assert system.step(80) == pytest.approx(50)
  system.slc.slc._resolver.map_speed_limit = 60 * system.unit
  assert system.step(60) == pytest.approx(60)
  assert system.step(80) == pytest.approx(60)
  assert system.slc.slc.overridden_speed == 0


@pytest.mark.parametrize("limit", [40, 55])
def test_set_speed_override_resets_on_accepted_limit(set_speed_slc, limit):
  system = set_speed_slc
  assert system.step(60, increase=True, new_gesture=True) == pytest.approx(60)
  system.slc.slc._resolver.map_speed_limit = limit * system.unit
  assert system.step(65, increase=True) == pytest.approx(limit)
  assert system.step(70, increase=True) == pytest.approx(limit)
  assert system.step(71, increase=True, new_gesture=True) == pytest.approx(71)


@pytest.mark.parametrize("limit,button", [(40, "decelCruise"), (55, "accelCruise")])
def test_set_speed_override_does_not_reuse_confirmation_gesture(set_speed_slc, limit, button):
  system = set_speed_slc
  system.params["speed_limit_confirmation_higher"] = True
  system.params["speed_limit_confirmation_lower"] = True
  system.slc.slc._resolver.map_speed_limit = limit * system.unit
  system.step(50, new_gesture=True)
  assert system.slc.assist_state == custom.IQPlan.SpeedLimit.AssistState.preActive
  assert system.step(60, increase=True) == pytest.approx(50)
  system.sm["carState"].buttonEvents = [car.CarState.ButtonEvent(type=button, pressed=False)]
  assert system.step(61, increase=True) == pytest.approx(limit)
  system.sm["carState"].buttonEvents = []
  assert system.step(65, increase=True) == pytest.approx(limit)
  assert system.step(66, increase=True, new_gesture=True) == pytest.approx(66)


@pytest.mark.parametrize("reset", ["disengage", "information", "off", "missing_limit"])
def test_set_speed_override_cannot_survive_reset(set_speed_slc, reset):
  system = set_speed_slc
  assert system.step(60, increase=True, new_gesture=True) == pytest.approx(60)
  if reset == "disengage":
    system.sm["selfdriveState"].enabled = False
  elif reset in ("information", "off"):
    system.params["speed_limit_controller"] = False
    system.params["show_speed_limits"] = reset == "information"
  else:
    system.slc.slc._resolver.map_speed_limit = 0
  system.step(60)
  assert system.slc.slc.overridden_speed == 0
  system.sm["selfdriveState"].enabled = True
  system.params["speed_limit_controller"] = True
  system.slc.slc._resolver.map_speed_limit = 50 * system.unit
  assert system.step(60) == pytest.approx(50)
  assert system.step(65, increase=True) == pytest.approx(50)
  assert system.step(66, increase=True, new_gesture=True) == pytest.approx(66)


def test_set_speed_override_respects_offset(set_speed_slc):
  system = set_speed_slc
  system.slc.slc.params.put("speed_limit_offset1", 10)
  system.slc.slc.params.put("speed_limit_offset2", 10)
  system.slc.slc.params.put("speed_limit_offset3", 10)
  system.slc.slc._offset_cache.clear()
  system.step(50, new_gesture=True)
  assert system.step(52, increase=True) == pytest.approx(52)
  assert not system.slc.slc.override_slc
  assert system.step(56, increase=True) == pytest.approx(56)
  assert system.step(55) == pytest.approx(55)
  assert not system.slc.slc.override_slc


def test_manual_override_still_requires_accelerator(set_speed_slc):
  system = set_speed_slc
  system.params["speed_limit_controller_override_set_speed"] = False
  system.params["speed_limit_controller_override_manual"] = True
  assert system.step(60, increase=True, new_gesture=True) == pytest.approx(50)
  system.sm["carState"].gasPressed = True
  system.sm["carState"].vEgoCluster = 55 * system.unit
  system.slc.slc.update_override(60 * system.unit, 0, 55 * system.unit, 0, system.sm, system.params, system.params["is_metric"])
  assert system.slc.slc.overridden_speed == pytest.approx(55 * system.unit)
  system.sm["carState"].gasPressed = False
  system.slc.slc.update_override(60 * system.unit, 0, 55 * system.unit, 0, system.sm, system.params, system.params["is_metric"])
  assert system.slc.slc.overridden_speed == pytest.approx(55 * system.unit)


@pytest.mark.parametrize("pcm_cruise", [False, True])
def test_driver_increase_reaches_slc_without_transient_button_events(set_speed_slc, pcm_cruise):
  system = set_speed_slc
  helper = VCruiseHelper(car.CarParams(pcmCruise=pcm_cruise), custom.IQCarParams(pcmCruiseSpeed=True))
  helper.set_speed_to_limit = False
  helper.v_cruise_kph = helper.v_cruise_cluster_kph = 50 * system.unit * CV.MS_TO_KPH
  state = car.CarState(cruiseState={"available": True, "speed": 50 * system.unit, "speedCluster": 50 * system.unit})
  helper.update_v_cruise(state, True, system.params["is_metric"])
  state.buttonEvents = [car.CarState.ButtonEvent(type="accelCruise", pressed=True)]
  helper.update_v_cruise(state, True, system.params["is_metric"])
  state.buttonEvents = [car.CarState.ButtonEvent(type="accelCruise", pressed=False)]
  if pcm_cruise:
    state.cruiseState.speed = state.cruiseState.speedCluster = 51 * system.unit
  helper.update_v_cruise(state, True, system.params["is_metric"])
  state.buttonEvents = []
  for _ in range(5):
    helper.update_v_cruise(state, True, system.params["is_metric"])
  system.sm["iqCarState"] = custom.IQCarState.new_message(
    slcSetSpeedRequestId=helper.slc_set_speed_request_id,
    slcSetSpeedGestureId=helper.slc_set_speed_gesture_id,
    slcSetSpeedRequestKph=helper.slc_set_speed_request_kph,
  )
  requested = helper.v_cruise_kph * CV.KPH_TO_MS / system.unit
  assert requested > 50
  assert system.step(requested) == pytest.approx(requested)


def test_set_speed_override_keeps_navigation_constraint(set_speed_slc):
  system = set_speed_slc
  assert system.step(60, increase=True, new_gesture=True) == pytest.approx(60)
  planner = LongitudinalPlannerIQ.__new__(LongitudinalPlannerIQ)
  planner.slimit = system.slc
  planner.iq_dynamic = SimpleNamespace(
    set_slc_experimental_mode=lambda _mode: None, update=lambda _sm: None, force_stop_requested=lambda: False)
  planner.force_stop_timer = 0.0
  planner.override_force_stop_timer = 0.0
  planner.override_force_stop = False
  system.sm["iqNavState"] = SimpleNamespace(longitudinalEngaged=False, valid=False)
  assert planner.update_targets(system.sm, 50 * system.unit, 60 * system.unit) == pytest.approx(60 * system.unit)
  system.sm["iqNavState"] = SimpleNamespace(longitudinalEngaged=True, valid=True, speedTarget=40 * system.unit)
  assert planner.update_targets(system.sm, 50 * system.unit, 60 * system.unit) == pytest.approx(40 * system.unit)


def test_set_speed_override_cannot_bypass_construction_zone(set_speed_slc):
  system = set_speed_slc
  assert system.step(60, increase=True, new_gesture=True) == pytest.approx(60)
  system.params["construction_zone_assist"] = True
  system.params["construction_zone_speed"] = 40
  system.sm["iqConstructionZone"] = SimpleNamespace(active=True)
  system.sm.alive["iqConstructionZone"] = True
  assert system.step(60) == pytest.approx(40)
  assert system.step(65, increase=True, new_gesture=True) == pytest.approx(40)
  assert not system.slc.slc.override_slc


class _FakeSM(dict):
  def __init__(self, services, alive=None):
    super().__init__(services)
    self.alive = alive or {}


def _construction_sm(active=True, alive=True, iq_limit=0.0):
  sm = _FakeSM(_build_sm(iq_limit=iq_limit))
  sm["iqConstructionZone"] = SimpleNamespace(active=active, orangeFraction=0.001, secondsSinceHit=1.0)
  sm.alive = {"iqConstructionZone": alive}
  return sm


def _construction_controller():
  params = FakeParams()
  controller = SpeedLimitController(params)
  controller.update_gps = lambda _sm: None
  controller._resolver.update_map_data = lambda *_args, **_kwargs: None
  controller.get_mapbox_speed_limit = lambda *_args, **_kwargs: None
  controller.mapbox_requests["total_requests"] = 0
  controller.mapbox_requests["max_requests"] = 999999
  return controller


def _construction_slc_params():
  slc_params = _base_slc_params_controller()
  slc_params["slc_online_filler"] = False
  slc_params["construction_zone_assist"] = True
  slc_params["construction_zone_speed"] = 60.0
  slc_params["is_metric"] = False
  return slc_params


def test_construction_zone_clamps_higher_limit():
  controller = _construction_controller()
  controller._resolver.map_speed_limit = 31.3  # ~70 mph
  sm = _construction_sm()

  controller.update_limits(0.0, None, True, 33.0, 30.0, sm, _construction_slc_params())
  assert controller.active_source == "Construction"
  assert abs(controller.active_target - 60.0 * CV.MPH_TO_MS) < 1e-6


def test_construction_zone_does_not_raise_lower_limit():
  controller = _construction_controller()
  controller._resolver.map_speed_limit = 20.0  # below the 60 mph clamp
  sm = _construction_sm()

  controller.update_limits(0.0, None, True, 33.0, 30.0, sm, _construction_slc_params())
  assert controller.active_source == "Map Data"
  assert controller.active_target == 20.0


def test_construction_zone_applies_without_other_sources():
  controller = _construction_controller()
  controller._resolver.map_speed_limit = 0.0
  sm = _construction_sm()

  controller.update_limits(0.0, None, True, 33.0, 30.0, sm, _construction_slc_params())
  assert controller.active_source == "Construction"
  assert abs(controller.active_target - 60.0 * CV.MPH_TO_MS) < 1e-6


def test_construction_zone_ignored_when_not_alive_or_inactive_or_disabled():
  for kwargs, slc_toggle in (
    (dict(alive=False), True),
    (dict(active=False), True),
    (dict(), False),
  ):
    controller = _construction_controller()
    controller._resolver.map_speed_limit = 31.3
    sm = _construction_sm(**kwargs)
    slc_params = _construction_slc_params()
    slc_params["construction_zone_assist"] = slc_toggle

    controller.update_limits(0.0, None, True, 33.0, 30.0, sm, slc_params)
    assert controller.active_source == "Map Data"
    assert controller.active_target == 31.3


def test_construction_zone_metric_speed_units():
  controller = _construction_controller()
  controller._resolver.map_speed_limit = 33.0
  sm = _construction_sm()
  slc_params = _construction_slc_params()
  slc_params["is_metric"] = True
  slc_params["construction_zone_speed"] = 100.0  # kph

  controller.update_limits(0.0, None, True, 36.0, 33.0, sm, slc_params)
  assert controller.active_source == "Construction"
  assert abs(controller.active_target - 100.0 * CV.KPH_TO_MS) < 1e-6


def test_construction_zone_never_raises_cruise_even_with_auto_raise():
  slc = SLCVCruise()
  slc.slc = _FakeSLC()
  slc.slc.target = 60.0 * CV.MPH_TO_MS
  slc.slc.source = "Construction"
  slc.slc.active_target = slc.slc.target
  slc.slc.active_source = "Construction"
  slc.slc._offset = 2.0  # must be ignored for Construction

  slc._get_slc_params = lambda: {
    "speed_limit_controller": True,
    "speed_limit_mode": 3,
    "show_speed_limits": False,
    "is_metric": False,
    "slc_policy": POLICY_MAP_DATA_PRIORITY,
    "slc_auto_confirm": False,
    "speed_limit_confirmation_higher": False,  # auto-raise allowed
    "speed_limit_confirmation_lower": False,
    "map_speed_lookahead_higher": 5.0,
    "map_speed_lookahead_lower": 5.0,
    "slc_fallback_experimental_mode": False,
    "slc_fallback_set_speed": False,
    "slc_fallback_previous_speed_limit": False,
    "speed_limit_controller_override_manual": True,
    "speed_limit_controller_override_set_speed": False,
    "slc_online_filler": False,
    "construction_zone_assist": True,
    "construction_zone_speed": 60.0,
  }

  # user cruising below the construction clamp: must not be raised to it
  v_cruise = 22.0
  sm = _build_sm(v_cruise_cluster=v_cruise * CV.MS_TO_KPH, v_ego_cluster=22.0)
  out = slc.update(apply_enabled=True, now=None, time_validated=True, v_cruise=v_cruise, v_ego=22.0, sm=sm)
  assert out == v_cruise
  assert slc.slc_offset == 0

  # user cruising above it: clamped down
  v_cruise = 33.0
  sm = _build_sm(v_cruise_cluster=v_cruise * CV.MS_TO_KPH, v_ego_cluster=33.0)
  out = slc.update(apply_enabled=True, now=None, time_validated=True, v_cruise=v_cruise, v_ego=33.0, sm=sm)
  assert abs(out - 60.0 * CV.MPH_TO_MS) < 1e-6


def _offset_controller(pct1=10.0, pct2=5.0, pct3=8.0):
  params = FakeParams()
  params.put("speed_limit_offset1", pct1)
  params.put("speed_limit_offset2", pct2)
  params.put("speed_limit_offset3", pct3)
  controller = SpeedLimitController(params)
  controller._assist.source = "Map Data"
  return controller


def test_get_offset_percent_per_zone():
  controller = _offset_controller()

  controller._assist.target = 6.7  # ~15 mph -> zone 1
  assert abs(controller.get_offset(False) - 6.7 * 0.10) < 1e-9

  controller._assist.target = 13.4  # ~30 mph -> zone 2
  assert abs(controller.get_offset(False) - 13.4 * 0.05) < 1e-9

  controller._assist.target = 31.3  # ~70 mph -> zone 3 (open-ended)
  assert abs(controller.get_offset(False) - 31.3 * 0.08) < 1e-9


def test_get_offset_zone_lower_bound_inclusive():
  controller = _offset_controller()
  boundary = OFFSET_MAP_IMPERIAL[1][0]
  controller._assist.target = boundary
  assert abs(controller.get_offset(False) - boundary * 0.05) < 1e-9


def test_get_offset_zero_without_real_limit_source():
  for source in ("None", "Construction"):
    controller = _offset_controller()
    controller._assist.source = source
    controller._assist.target = 30.0
    assert controller.get_offset(False) == 0.0


def test_get_offset_percent_clamped():
  controller = _offset_controller(pct3=500.0)
  controller._assist.target = 30.0
  assert abs(controller.get_offset(False) - 30.0 * 0.50) < 1e-9


def test_construction_zone_fires_event_once_per_zone_entry():
  from iqpilot.cereal import custom
  event = custom.IQOnroadEvent.EventName.constructionZoneDetected

  controller = _construction_controller()
  controller._resolver.map_speed_limit = 31.3
  slc_params = _construction_slc_params()

  controller.update_limits(0.0, None, True, 33.0, 30.0, _construction_sm(), slc_params)
  assert event in controller.pending_events

  controller.update_limits(0.0, None, True, 33.0, 30.0, _construction_sm(), slc_params)
  assert event not in controller.pending_events

  # zone releases, then a new zone: fires again
  controller.update_limits(0.0, None, True, 33.0, 30.0, _construction_sm(active=False), slc_params)
  assert event not in controller.pending_events
  controller.update_limits(0.0, None, True, 33.0, 30.0, _construction_sm(), slc_params)
  assert event in controller.pending_events


@pytest.mark.parametrize("alive,valid,limit_valid,limit", [
  (False, True, True, 25.0), (True, False, True, 25.0), (True, True, False, 25.0),
  (True, True, True, 0.0), (True, True, True, float("nan")), (True, True, True, float("inf")),
])
def test_navigation_mapbox_limit_requires_fresh_valid_data(alive, valid, limit_valid, limit):
  controller = _construction_controller()
  controller.get_tomtom_speed_limit = lambda *_args: None
  controller.mapbox_limit = 20.0
  sm = _FakeSM(_build_sm())
  sm["iqNavState"] = custom.IQNavState.new_message(mapboxSpeedLimit=limit, mapboxSpeedLimitValid=limit_valid)
  sm.alive["iqNavState"] = alive
  sm.valid = {"iqNavState": valid}
  controller.update_limits(0, datetime.now(), True, 30, 20, sm, _base_slc_params_controller())
  assert controller.target == pytest.approx(20.0)
  assert controller.source == "Mapbox"


@pytest.mark.parametrize("policy,expected", [(0, 0.0), (1, 25.0), (2, 25.0)])
@pytest.mark.parametrize("online_filler", [False, True])
def test_navigation_mapbox_only_limit_obeys_slc_policy(policy, expected, online_filler):
  controller = _construction_controller()
  controller.get_tomtom_speed_limit = lambda *_args: None
  sm = _FakeSM(_build_sm())
  sm["iqNavState"] = custom.IQNavState.new_message(mapboxSpeedLimit=25.0, mapboxSpeedLimitValid=True)
  sm.alive["iqNavState"] = True
  sm.valid = {"iqNavState": True}
  params = _base_slc_params_controller() | {"slc_policy": policy, "slc_online_filler": online_filler}
  controller.update_limits(0, datetime.now(), True, 30, 20, sm, params)
  assert controller.target == pytest.approx(expected)


def test_navigation_mapbox_limit_requires_confirmation_before_override(set_speed_slc):
  system = set_speed_slc
  system.params["speed_limit_confirmation_higher"] = True
  system.slc.slc._resolver.map_speed_limit = 0
  system.sm["iqNavState"] = custom.IQNavState.new_message(mapboxSpeedLimit=60 * system.unit, mapboxSpeedLimitValid=True)
  system.sm.alive["iqNavState"] = True
  system.sm.valid = {"iqNavState": True}
  system.step(50, new_gesture=True)
  assert system.slc.assist_state == custom.IQPlan.SpeedLimit.AssistState.preActive
  assert system.step(70, increase=True) == pytest.approx(50)
  system.sm["carState"].buttonEvents = [car.CarState.ButtonEvent(type="accelCruise", pressed=False)]
  assert system.step(71, increase=True) == pytest.approx(60)
  system.sm["carState"].buttonEvents = []
  assert system.step(72, increase=True) == pytest.approx(60)
  assert system.step(73, increase=True, new_gesture=True) == pytest.approx(73)


@pytest.mark.parametrize("status, minimum_backoff", [(401, 3000.0), (None, 15.0)])
def test_mapbox_failure_backs_off_instead_of_retrying_every_second(monkeypatch, status, minimum_backoff):
  import iqpilot.selfdrive.controls.lib.speed_limit_controller as slc_module

  class InlineExecutor:
    def submit(self, fn):
      result = fn()
      return SimpleNamespace(result=lambda: result, add_done_callback=lambda callback: callback(SimpleNamespace(result=lambda: result)))

  calls = []

  def get(*_args, **_kwargs):
    calls.append(1)
    error = slc_module.requests.HTTPError("rejected") if status else slc_module.requests.ConnectionError("down")
    error.response = SimpleNamespace(status_code=status) if status else None
    raise error

  monkeypatch.setattr(slc_module, "is_url_pingable", lambda _url: True)
  controller = SpeedLimitController(FakeParams())
  controller.executor = InlineExecutor()
  controller.session = SimpleNamespace(get=get)
  controller.gps_valid = True
  controller.mapbox_token = "token"
  controller.gps_position = {"latitude": 40.0, "longitude": -73.0, "bearing": 0.0}
  sm = {"carState": SimpleNamespace(steeringAngleDeg=0.0), "vehicleParameters": SimpleNamespace(angleOffsetDeg=0.0)}

  for _ in range(200):
    controller.get_mapbox_speed_limit(datetime(2026, 9, 20), False, 25.0, sm)
  assert len(calls) == 1
  assert controller.mapbox_backoff_until - slc_module.time.monotonic() > minimum_backoff

  controller.mapbox_token = "replacement"
  controller.segment_distance = 0.0
  controller.get_mapbox_speed_limit(datetime(2026, 9, 20), False, 25.0, sm)
  assert len(calls) == 2


def test_hanging_request_does_not_queue_a_request_every_frame(monkeypatch):
  # On a dead network one provider's request hangs ~20 s on the shared single worker. Each queued request
  # must still count as in flight, or every 20 Hz planner frame queues another one and plannerd's heap
  # and its periodic full collection grow for the rest of the drive.
  import iqpilot.selfdrive.controls.lib.speed_limit_controller as slc_module

  class StalledExecutor:
    def __init__(self):
      self.queued = []

    def submit(self, fn):
      future = SimpleNamespace(callbacks=[], add_done_callback=lambda callback: future.callbacks.append(callback))
      self.queued.append((fn, future))
      return future

    def run_all(self):
      while self.queued:
        fn, future = self.queued.pop(0)
        result = fn()
        for callback in future.callbacks:
          callback(SimpleNamespace(result=lambda result=result: result))

  def get(*_args, **_kwargs):
    raise slc_module.requests.ConnectionError("down")

  monkeypatch.setattr(slc_module, "is_url_pingable", lambda _url: True)
  controller = SpeedLimitController(FakeParams())
  controller.executor = StalledExecutor()
  controller.session = SimpleNamespace(get=get)
  controller.gps_valid = True
  controller.mapbox_token = "token"
  controller.tomtom_token = "token"
  controller.gps_position = {"latitude": 40.0, "longitude": -73.0, "bearing": 0.0}
  sm = {"carState": SimpleNamespace(steeringAngleDeg=0.0), "vehicleParameters": SimpleNamespace(angleOffsetDeg=0.0)}

  for _ in range(400):
    controller.get_mapbox_speed_limit(datetime(2026, 10, 23), False, 15.0, sm)
    controller.get_tomtom_speed_limit(datetime(2026, 10, 23), False, 15.0, sm)
  assert len(controller.executor.queued) == 2

  controller.executor.run_all()
  assert not controller.calling_mapbox and not controller.calling_tomtom
  for _ in range(400):
    controller.get_mapbox_speed_limit(datetime(2026, 10, 23), False, 15.0, sm)
    controller.get_tomtom_speed_limit(datetime(2026, 10, 23), False, 15.0, sm)
  assert controller.executor.queued == []
