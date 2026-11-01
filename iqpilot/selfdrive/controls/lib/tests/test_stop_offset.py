from dataclasses import replace
from types import SimpleNamespace

import pytest
import numpy as np

from iqpilot.selfdrive.controls.lib.stop_offset import (
  LeadStopTargets, StopFrame, StopOffset, StopTarget, minimum_stopping_distance, stopping_trajectory,
)

TIMES = (0.0, 1.0, 2.0, 4.0, 8.0, 10.0)


def frame(time=0.0, position=0.0, speed=10.0, **kwargs):
  return StopFrame(time, position, speed, 0.0, **kwargs)


def lead(speed=0.0, acceleration=0.0, distance=40.0, track=7, status=True):
  return SimpleNamespace(status=status, dRel=distance, vLead=speed, aLeadK=acceleration, radarTrackId=track)


@pytest.mark.parametrize('offset', (0.0, 1.0, 2.0, 4.0, 10.0))
def test_offset_is_applied_once_in_world_coordinates(offset):
  controller = StopOffset(offset, TIMES)
  target = StopTarget(1, 'lead0', 37.0, 0.0)
  for i in range(20):
    position, now = i * 0.5, i * 0.05
    plan = controller.update(frame(now, position), (replace(target, observed_at=now),))
    assert plan.requested_position == 37.0 - offset
    assert plan.remaining == 37.0 - offset - position
    assert plan.target.position == 37.0


def test_zero_tracks_the_unadjusted_target():
  plan = StopOffset(0.0, TIMES).update(frame(), (StopTarget(1, 'lead0', 37.0, 0.0),))
  assert plan.active
  assert plan.requested_position == plan.remaining == 37.0
  assert plan.offset == 0.0
  assert plan.trajectory is not None


def test_zero_holds_at_the_unadjusted_target():
  plan = StopOffset(0.0, TIMES).update(frame(position=37.0, speed=0.0), (StopTarget(1, 'lead0', 37.0, 0.0),))
  assert plan.state == 'holding'
  assert plan.remaining == 0.0
  assert plan.trajectory.should_stop


def test_passed_target_retains_signed_distance_and_requested_offset():
  plan = StopOffset(10.0, TIMES).update(frame(position=30.0), (StopTarget(1, 'lead0', 37.0, 0.0),))
  assert plan.offset == 10.0
  assert plan.remaining == -3.0
  assert plan.shortfall > 3.0


def test_anchor_refinement_cannot_ratchet_the_target_forward():
  controller = StopOffset(2.0, TIMES)
  for i in range(100):
    plan = controller.update(frame(i * 0.05), (StopTarget(1, 'model', 37.0 + i * 0.1, i * 0.05),))
    assert plan.requested_position <= 35.5


def test_dropout_preserves_committed_target_and_degrades_to_a_stop():
  controller = StopOffset(2.0, TIMES)
  controller.update(frame(), (StopTarget(1, 'lead0', 37.0, 0.0),))
  plan = controller.update(frame(0.2, 2.0), ())
  assert plan.active and plan.state == 'dropout'
  assert plan.remaining == 33.0
  assert plan.trajectory is not None
  assert controller.update(frame(0.4, 4.0), ()).state == 'degraded'


def test_uncommitted_stale_target_becomes_unavailable():
  controller = StopOffset(2.0, TIMES)
  assert controller.update(frame(speed=0.0), (StopTarget(1, 'lead0', 100.0, 0.0),)).trajectory is None
  assert not controller.update(frame(0.4, speed=0.0), ()).active


@pytest.mark.parametrize('changes', ({'override': True}, {'active': False}))
def test_driver_override_and_disengagement_clear_the_event(changes):
  controller = StopOffset(2.0, TIMES)
  controller.update(frame(), (StopTarget(1, 'lead0', 37.0, 0.0),))
  assert not controller.update(replace(frame(0.05), **changes), ()).active
  assert not controller.update(frame(0.1), ()).active


def test_explicit_release_is_immediate():
  controller = StopOffset(2.0, TIMES)
  target = StopTarget(1, 'lead0', 37.0, 0.0)
  controller.update(frame(), (target,))
  assert not controller.update(frame(0.05), (replace(target, released=True),)).active


@pytest.mark.parametrize('speed', (0.3, 2.0, 15.0, 30.0))
def test_steady_following_never_qualifies(speed):
  source = LeadStopTargets()
  controller = StopOffset(10.0, TIMES)
  for i in range(30):
    current = frame(i * 0.05, speed=speed)
    targets = source.update(current, (lead(speed=speed, distance=40.0 + speed * i * 0.05),))
    assert not controller.update(current, targets).active


def test_braking_lead_qualifies_well_before_low_speed_fade():
  source = LeadStopTargets()
  controller = StopOffset(4.0, TIMES)
  for i in range(4):
    now = i * 0.05
    current = frame(now, speed=15.0)
    targets = source.update(current, (lead(speed=12.0 - now * 2.0, acceleration=-2.0,
                                           distance=40.0 + 12.0 * now - now ** 2),))
    plan = controller.update(current, targets)
  assert plan.active
  assert targets[0].stop_time > 5.0
  assert plan.trajectory is not None


def test_lead_resuming_motion_releases_the_stop():
  source = LeadStopTargets()
  controller = StopOffset(2.0, TIMES)
  assert controller.update(frame(), source.update(frame(), (lead(),))).active
  next_frame = frame(0.05)
  assert not controller.update(next_frame, source.update(next_frame, (lead(speed=0.5, acceleration=1.0),))).active


def test_cut_in_replaces_event_and_target_instead_of_inheriting_anchor():
  source = LeadStopTargets()
  controller = StopOffset(2.0, TIMES)
  original = controller.update(frame(), source.update(frame(), (lead(),)))
  next_frame = frame(0.05)
  cut_in = controller.update(next_frame, source.update(next_frame, (lead(distance=10.0, track=9),)))
  assert cut_in.target.event_id != original.target.event_id
  assert cut_in.remaining == 5.0


def test_reachable_bound_accounts_for_delay_and_jerk():
  instantaneous = 10.0 ** 2 / (2.0 * 3.5)
  ramped = minimum_stopping_distance(10.0, 0.0, 0.0, 3.5, 2.5)
  delayed = minimum_stopping_distance(10.0, 0.0, 0.5, 3.5, 2.5)
  assert delayed > ramped > instantaneous
  assert delayed - ramped == pytest.approx(5.0)


@pytest.mark.parametrize('acceleration', (-2.0, -0.5, 0.0, 0.5))
def test_stop_trajectory_reaches_target_with_zero_speed_and_acceleration(acceleration):
  current = replace(frame(speed=12.0), acceleration=acceleration, commanded_acceleration=acceleration)
  times = np.linspace(0.0, 40.0, 4001)
  trajectory = stopping_trajectory(current, 60.0, times, 0.5, 2.5, 1.5, 0.05)
  prefix = 12.0 * 0.5 + 0.5 * acceleration * 0.5 ** 2
  assert prefix + np.trapezoid(trajectory.speeds, times) == pytest.approx(60.0, abs=0.01)
  assert trajectory.speeds[-1] == trajectory.accelerations[-1] == 0.0
  assert min(trajectory.speeds) >= 0.0
  assert min(trajectory.accelerations) >= -2.5
  assert max(abs(x) for x in trajectory.jerks) <= 1.5
  assert abs(trajectory.acceleration - acceleration) <= 1.5 * 0.05 + 1e-9


@pytest.mark.parametrize('track', (7, -1))
def test_track_identity_survives_lead_slot_handoff_and_releases(track):
  source, controller = LeadStopTargets(), StopOffset(2.0, TIMES)
  original = controller.update(frame(), source.update(frame(), (lead(track=track), lead(status=False))))
  next_frame = frame(0.05)
  moved = controller.update(next_frame, source.update(next_frame, (lead(status=False), lead(track=track))))
  assert moved.target.event_id == original.target.event_id
  assert moved.target.source == 'lead1'
  release = frame(0.1)
  assert not controller.update(release, source.update(release, (lead(status=False), lead(track=track, speed=1.0, acceleration=1.0)))).active


def test_duplicate_vision_observations_do_not_leave_a_second_stop_event():
  source = LeadStopTargets()
  targets = source.update(frame(), (lead(track=-1), lead(track=-1, distance=40.1)))
  assert len(targets) == 1


def test_fresh_physical_lead_cancels_odometry_drift():
  controller = StopOffset(2.0, TIMES)
  original = controller.update(frame(), (StopTarget(1, 'lead0', 37.0, 0.0),))
  drifted = controller.update(frame(0.05, position=2.0), (StopTarget(1, 'lead0', 39.0, 0.05),))
  assert drifted.remaining == original.remaining


def test_stationary_reacquisition_preserves_committed_hold():
  source, controller = LeadStopTargets(), StopOffset(2.0, TIMES)
  original = controller.update(frame(), source.update(frame(), (lead(),)))
  reacquired_frame = frame(0.8, position=5.0, speed=0.0)
  reacquired = controller.update(reacquired_frame, source.update(reacquired_frame, (lead(distance=35.0),)))
  assert reacquired.target.event_id == original.target.event_id
  assert reacquired.trajectory.should_stop


@pytest.mark.parametrize('track', (-1, 7))
def test_stopped_lead_speed_noise_does_not_release_approach(track):
  # Route f9364b6c2f, t=1570.5..1574: a stopped vision lead briefly
  # estimates 0.11..0.18 m/s while ego creeps through its 3 m target.
  source, controller = LeadStopTargets(), StopOffset(0.0, TIMES)
  samples = ((1.754, 5.689, 0.065, -0.043),
             (1.334, 4.812, 0.129, -0.031),
             (1.041, 4.163, 0.175, -0.019),
             (0.818, 3.862, 0.110, -0.012))
  position = 0.0
  for i, (speed, distance, lead_speed, acceleration) in enumerate(samples):
    # Intermediate frames preserve real 20 Hz observation continuity.
    for j in range(10):
      current = frame((10 * i + j) * 0.05, position, speed=speed)
      targets = source.update(current, (lead(lead_speed, acceleration, distance, track),))
      plan = controller.update(current, targets)
      assert plan.active
      assert plan.trajectory is not None
      position += speed * 0.05


@pytest.mark.parametrize('speed,acceleration', ((0.6, 0.0), (0.2, 0.4), (1.0, 1.0)))
def test_lead_resume_releases_after_speed_noise_hysteresis(speed, acceleration):
  source, controller = LeadStopTargets(), StopOffset(0.0, TIMES)
  first = frame(speed=1.0)
  assert controller.update(first, source.update(first, (lead(distance=5.0),))).active
  current = frame(0.05, speed=1.0)
  assert not controller.update(current, source.update(current, (lead(speed, acceleration, 5.0),))).active
