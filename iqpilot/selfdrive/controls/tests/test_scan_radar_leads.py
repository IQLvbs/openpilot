from types import SimpleNamespace

import pytest

from iqdbc.car.honda.values import CAR
from iqpilot.selfdrive.controls.radard import (SCAN_CHALLENGER_STALE_CYCLES, SCAN_DISTANCE_STALE_CYCLES,
                                               SCAN_LEAD_MIN_CYCLES, SCAN_LEAD_PROB, SCAN_SWEEP_DT,
                                               KalmanParams, RadarD, Track, get_lead,
                                               scan_low_speed_candidate, track_agrees_with_model,
                                               uses_scan_radar)

DT_MDL = 0.05


def honda_cp(fingerprint=CAR.HONDA_CIVIC_BOSCH, radar_unavailable=False, brand="honda"):
  return SimpleNamespace(brand=brand, carFingerprint=fingerprint, radarUnavailable=radar_unavailable, flags=0)


def model_lead(x=30.0, y=0.0, v=10.0, prob=0.9, x_std=2.0, y_std=0.5, v_std=1.0):
  return SimpleNamespace(x=[x], y=[y], v=[v], a=[0.0], prob=prob, xStd=[x_std], yStd=[y_std], vStd=[v_std])


def make_track(identifier, d_rel, y_rel=0.0, v_rel=0.0, v_ego=10.0, cycles=SCAN_LEAD_MIN_CYCLES, measured=True):
  track = Track(identifier, v_rel + v_ego, KalmanParams(SCAN_SWEEP_DT))
  for _ in range(cycles):
    track.update(d_rel, y_rel, v_rel, v_rel + v_ego, measured)
  return track


class TestScanRadarGating:
  def test_capable_verified_car_uses_scan(self):
    assert uses_scan_radar(honda_cp())

  def test_radar_unavailable_disables(self):
    assert not uses_scan_radar(honda_cp(radar_unavailable=True))

  def test_non_family_car_never_scans(self):
    assert not uses_scan_radar(honda_cp(fingerprint=CAR.HONDA_CIVIC))
    assert not uses_scan_radar(honda_cp(fingerprint=CAR.HONDA_CIVIC_2022))

  def test_other_brand_never_scans(self):
    assert not uses_scan_radar(honda_cp(brand="toyota"))


class TestTrackMeasurementGating:
  def test_repeated_payload_not_absorbed_twice(self):
    kp = KalmanParams(SCAN_SWEEP_DT)
    absorbed = Track(1, 10.0, kp)
    starved = Track(1, 10.0, kp)
    absorbed.update(30.0, 0.0, 5.0, 15.0, True)
    starved.update(30.0, 0.0, 5.0, 15.0, True)
    v_after_first = starved.vLeadK

    for _ in range(5):
      absorbed.update(30.0, 0.0, 5.0, 15.0, True, absorb_measurement=True)
      starved.update(30.0, 0.0, 5.0, 15.0, False, absorb_measurement=False)

    assert starved.vLeadK == v_after_first
    assert absorbed.vLeadK != v_after_first
    assert starved.cnt == absorbed.cnt


class TestLowSpeedCandidate:
  def test_needs_minimum_cycles(self):
    young = make_track(1, 10.0, v_ego=2.0, cycles=SCAN_LEAD_MIN_CYCLES - 1)
    mature = make_track(2, 10.0, v_ego=2.0)
    assert not scan_low_speed_candidate(young, 2.0)
    assert scan_low_speed_candidate(mature, 2.0)

  def test_geometry_still_applies(self):
    offset = make_track(3, 10.0, y_rel=2.0, v_ego=2.0)
    assert not scan_low_speed_candidate(offset, 2.0)


class TestModelAgreement:
  def test_strict_tighter_than_relaxed(self):
    track = make_track(1, 40.0, v_ego=10.0)
    # ~15m disagreement: beyond the strict 25% envelope, inside the relaxed 40% one
    lead = model_lead(x=56.5)
    assert not track_agrees_with_model(track, lead, 10.0, strict=True)
    assert track_agrees_with_model(track, lead, 10.0, strict=False)


class TestScanLeadSelection:
  def test_held_lead_survives_probability_dip(self):
    track = make_track(7, 10.0, v_ego=2.0)
    lead = model_lead(x=11.5, v=2.0, prob=0.05)
    # raw prob is below threshold, the filtered prob is passed in above it
    result = get_lead(2.0, True, {7: track}, lead, 2.0, low_speed_override=True,
                      scan_radar=True, filtered_prob=0.6, held_track_id=7)
    assert result['status'] and result['radarTrackId'] == 7

  def test_unrelated_closer_point_cannot_usurp_without_model_evidence(self):
    held = make_track(7, 10.0, v_ego=2.0)
    interloper = make_track(9, 4.0, y_rel=0.5, v_ego=2.0)
    lead = model_lead(x=11.5, v=2.0, prob=0.9)
    result = get_lead(2.0, True, {7: held, 9: interloper}, lead, 2.0, low_speed_override=True,
                      scan_radar=True, filtered_prob=0.9, held_track_id=7)
    assert result['radarTrackId'] == 7

  def test_matching_closer_point_takes_over(self):
    held = make_track(7, 10.0, v_ego=2.0)
    closer = make_track(9, 4.0, v_ego=2.0)
    lead = model_lead(x=5.5, v=2.0, prob=0.9)  # model agrees with the closer car
    result = get_lead(2.0, True, {7: held, 9: closer}, lead, 2.0, low_speed_override=True,
                      scan_radar=True, filtered_prob=0.9, held_track_id=7)
    assert result['radarTrackId'] == 9

  def test_radar_only_takeover_when_no_model_lead(self):
    track = make_track(4, 8.0, v_ego=2.0)
    lead = model_lead(prob=0.0)
    result = get_lead(2.0, True, {4: track}, lead, 2.0, low_speed_override=True,
                      scan_radar=True, filtered_prob=0.0, held_track_id=-1)
    assert result['status'] and result['radarTrackId'] == 4

  def test_young_track_cannot_lead_alone(self):
    track = make_track(4, 8.0, v_ego=2.0, cycles=1)
    lead = model_lead(prob=0.0)
    result = get_lead(2.0, True, {4: track}, lead, 2.0, low_speed_override=True,
                      scan_radar=True, filtered_prob=0.0, held_track_id=-1)
    assert not result['status']

  def test_non_scan_behavior_unchanged(self):
    track = make_track(4, 8.0, v_ego=2.0, cycles=1)
    lead = model_lead(prob=0.0)
    result = get_lead(2.0, True, {4: track}, lead, 2.0, low_speed_override=True)
    assert result['status']  # legacy path has no maturity gate


class TestHeldLeadStaleness:
  def make_radard(self, tracks):
    rd = object.__new__(RadarD)
    rd.tracks = tracks
    rd.ready = True
    rd.v_ego = 10.0
    rd.held_lead_ids = [7, -1]
    rd._held_evidence_ids = [7, -1]
    rd._challenger_stale_counts = [0, 0]
    rd._distance_stale_counts = [0, 0]
    return rd

  def test_relaxed_match_keeps_hold(self):
    held = make_track(7, 30.0, v_ego=10.0)
    rd = self.make_radard({7: held})
    for _ in range(5):
      rd._refresh_held_lead_evidence(0, model_lead(x=32.0), 0.9)
    assert rd.held_lead_ids[0] == 7

  def test_better_challenger_releases_hold(self):
    held = make_track(7, 80.0, v_ego=10.0)
    challenger = make_track(9, 30.0, v_ego=10.0)
    rd = self.make_radard({7: held, 9: challenger})
    lead = model_lead(x=31.5)
    for _ in range(SCAN_CHALLENGER_STALE_CYCLES):
      rd._refresh_held_lead_evidence(0, lead, 0.9)
    assert rd.held_lead_ids[0] == -1

  def test_gross_distance_disagreement_releases_hold(self):
    held = make_track(7, 80.0, v_ego=10.0)
    rd = self.make_radard({7: held})
    lead = model_lead(x=31.5, x_std=60.0, y_std=30.0, v_std=30.0)  # huge model uncertainty
    for _ in range(SCAN_DISTANCE_STALE_CYCLES):
      rd._refresh_held_lead_evidence(0, lead, 0.9)
    assert rd.held_lead_ids[0] == -1

  def test_low_probability_resets_evidence_without_release(self):
    held = make_track(7, 80.0, v_ego=10.0)
    rd = self.make_radard({7: held})
    for _ in range(10):
      rd._refresh_held_lead_evidence(0, model_lead(x=31.5), SCAN_LEAD_PROB)
    assert rd.held_lead_ids[0] == 7
    assert rd._distance_stale_counts[0] == 0


class TestProbFilterTiming:
  def test_rises_instantly_decays_slowly(self):
    from iqpilot.common.filter_simple import FirstOrderFilter
    f = FirstOrderFilter(0.0, 0.2, DT_MDL)
    f.x = max(f.x, 0.9)
    assert f.x == pytest.approx(0.9)
    f.update(0.0)
    assert f.x > 0.6
