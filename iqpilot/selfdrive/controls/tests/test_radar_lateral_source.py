# Copyright (c) 2026 IQ.Lvbs. All rights reserved.
from types import SimpleNamespace

import pytest

from iqdbc.car.hyundai.values import HyundaiFlags, HyundaiFlagsIQ
from iqpilot.selfdrive.controls.radard import RadarD, get_lead
from iqpilot.selfdrive.controls.tests.test_scan_radar_leads import make_track, model_lead


@pytest.mark.parametrize('brand,flags,iq_flags,expected', [
  ('hyundai', 0, 0, False),
  ('hyundai', HyundaiFlags.CAMERA_SCC, 0, True),
  ('hyundai', HyundaiFlags.CANFD_CAMERA_SCC, 0, True),
  ('hyundai', 0, HyundaiFlagsIQ.ENHANCED_SCC, True),
  ('toyota', HyundaiFlags.CAMERA_SCC, HyundaiFlagsIQ.ENHANCED_SCC, False),
])
def test_vehicle_configuration_selects_lateral_source(brand, flags, iq_flags, expected, monkeypatch):
  monkeypatch.setattr('iqpilot.selfdrive.controls.radard.CustomStopDistance', lambda: None)
  radar = RadarD(SimpleNamespace(brand=brand, flags=flags), SimpleNamespace(flags=iq_flags))
  assert radar.model_lateral is expected


@pytest.mark.parametrize('model_lateral', [False, True])
def test_matched_lead_preserves_radar_motion_and_track_geometry(model_lateral):
  track = make_track(7, 28.48, y_rel=0.3)
  result = get_lead(10.0, True, {7: track}, model_lead(y=0.6), 10.0, model_lateral=model_lateral)
  expected = track.get_RadarState(0.9)
  expected['yRel'] = -0.6 if model_lateral else 0.3
  assert result == expected
  assert track.yRel == 0.3


@pytest.mark.parametrize('ready,prob', [(False, 0.9), (True, 0.0)])
def test_radar_only_lead_never_uses_unconfirmed_model_lateral(ready, prob):
  track = make_track(7, 8.0, y_rel=0.3, v_ego=2.0)
  result = get_lead(2.0, ready, {7: track}, model_lead(y=2.0, prob=prob), 2.0, model_lateral=True)
  assert result == track.get_RadarState()


def test_closer_radar_only_override_preserves_its_own_lateral():
  matched = make_track(7, 18.48, y_rel=0.3, v_ego=2.0)
  closer = make_track(8, 5.0, y_rel=0.5, v_ego=2.0)
  result = get_lead(2.0, True, {7: matched, 8: closer}, model_lead(x=20.0, y=0.6, v=2.0),
                    2.0, model_lateral=True)
  assert result == closer.get_RadarState()


def test_vision_only_lead_keeps_model_lateral():
  result = get_lead(10.0, True, {}, model_lead(y=0.6), 10.0, model_lateral=False)
  assert result['yRel'] == -0.6
  assert not result['radar']
