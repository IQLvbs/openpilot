import pytest
import numpy as np

from iqpilot.selfdrive.test.longitudinal_maneuvers.stop_offset import run_stop


def test_braking_lead_receives_the_full_requested_offset():
  baseline = run_stop(0, moving_lead=True, distance=35.0)
  adjusted = run_stop(2, moving_lead=True, distance=35.0)
  assert baseline[-1, 2] == adjusted[-1, 2] == 0.0
  assert adjusted[-1, 5] - baseline[-1, 5] == pytest.approx(2.0, abs=0.05)
  assert adjusted[-1, 5] == pytest.approx(5.0, abs=0.3)


@pytest.mark.parametrize('offset', (0, 1, 2, 4, 10))
@pytest.mark.parametrize('moving_lead', (False, True))
@pytest.mark.parametrize('e2e', (False, True))
@pytest.mark.parametrize('new_lead_mpc', (False, True))
def test_final_gap_for_supported_lead_planners_and_modes(offset, moving_lead, e2e, new_lead_mpc):
  trace = run_stop(offset, moving_lead=moving_lead, distance=35.0 if moving_lead else 90.0,
                   e2e=e2e, new_lead_mpc=new_lead_mpc)
  assert trace[-1, 2] == 0.0
  assert trace[-1, 5] == pytest.approx(3.0 + offset, abs=0.5)
  assert min(trace[:, 5]) > 0.0
  assert max(trace[:, 9]) == offset


@pytest.mark.parametrize('vehicle,dead_time,lag', (('honda', 0.0, 0.0), ('ev6', 0.15, 0.7), ('elantra', 0.4, 1.1)))
def test_actuator_delay_and_vehicle_control_loops(vehicle, dead_time, lag):
  trace = run_stop(2, vehicle=vehicle, dead_time=dead_time, lag=lag)
  assert trace[-1, 2] == 0.0
  assert trace[-1, 5] == pytest.approx(5.0, abs=0.3 if lag == 0.0 else 0.5)


@pytest.mark.parametrize('speed', (0.3, 12.0))
def test_steady_following_commands_match_zero_offset(speed):
  baseline = run_stop(0, steady_lead=True, speed=speed, distance=30.0, duration=12.0)
  adjusted = run_stop(10, steady_lead=True, speed=speed, distance=30.0, duration=12.0)
  np.testing.assert_array_equal(adjusted[:, :6], baseline[:, :6])
  assert not np.any(adjusted[:, 7])


@pytest.mark.parametrize('visibility', (lambda now: not 8.0 < now < 8.2, lambda now: now < 8.0))
def test_committed_stop_survives_short_and_persistent_perception_loss(visibility):
  trace = run_stop(2, e2e=True, visibility=visibility)
  assert trace[-1, 2] == 0.0
  assert trace[-1, 5] == pytest.approx(5.0, abs=0.5)
  assert trace[-1, 12] == 1.0


def test_mode_handoff_keeps_the_same_event_and_offset():
  trace = run_stop(2, e2e=True, handoff_at=7.0)
  assert trace[-1, 5] == pytest.approx(5.0, abs=0.5)
  assert len(set(trace[trace[:, 7] == 1.0, 8])) == 1


def test_resume_releases_hold():
  trace = run_stop(2, resume_at=22.0, duration=30.0)
  assert trace[400, 2] == 0.0
  assert trace[-1, 2] > 2.0
  assert trace[-1, 7] == 0.0


def test_second_lead_uses_the_same_stopping_preference():
  trace = run_stop(4, only_lead2=True)
  assert trace[-1, 5] == pytest.approx(7.0, abs=0.5)


def test_perception_bias_is_reported_in_the_true_gap():
  trace = run_stop(2, distance_bias=0.4)
  assert trace[-1, 5] == pytest.approx(4.6, abs=0.3)
  assert trace[-1, 6] == pytest.approx(0.0, abs=0.3)


def test_bounded_noise_and_range_dependent_bias_preserve_the_target():
  trace = run_stop(2, distance_bias=lambda now, distance: 0.05 * np.sin(8.0 * now) + 0.002 * distance)
  assert trace[-1, 2] == 0.0
  assert trace[-1, 5] == pytest.approx(5.0, abs=0.5)
  assert max(abs(np.diff(trace[:, 3]))) / 0.05 < 3.0
  assert len(set(trace[trace[:, 7] == 1.0, 8])) == 1


def test_oversized_late_request_reports_shortfall_instead_of_shrinking_offset():
  trace = run_stop(10, speed=12.0, distance=20.0, duration=15.0)
  assert np.any(trace[:, 10] > 0.0)
  assert np.all(trace[:, 9] == 10.0)
  assert np.all(np.isfinite(trace))
  assert min(trace[:, 4]) >= -4.0
  assert min(trace[:, 2]) >= 0.0


@pytest.mark.parametrize('moving_lead', (False, True))
def test_default_setting_controls_the_nominal_stopping_position(moving_lead):
  trace = run_stop(None, moving_lead=moving_lead, distance=35.0 if moving_lead else 90.0)
  assert trace[-1, 2] == 0.0
  assert trace[-1, 5] == pytest.approx(3.0, abs=0.3)
  assert trace[-1, 5] - trace[-1, 6] == pytest.approx(3.0, abs=1e-5)
  assert np.any(trace[:, 7])
  assert np.all(trace[:, 9] == 0.0)


@pytest.mark.parametrize('speed,distance,moving_lead', ((30.0, 260.0, False), (30.0, 65.0, True), (0.0, 20.0, False)))
def test_road_speed_approach_and_initial_launch(speed, distance, moving_lead):
  trace = run_stop(10, speed=speed, distance=distance, moving_lead=moving_lead, duration=50.0)
  assert trace[-1, 2] == 0.0
  assert trace[-1, 5] == pytest.approx(13.0, abs=0.5)


@pytest.mark.parametrize('moving_lead', (False, True))
@pytest.mark.parametrize('offset', (0, 2))
def test_iq_dynamic_keeps_the_same_stop_target(moving_lead, offset):
  trace = run_stop(offset, moving_lead=moving_lead, distance=35.0 if moving_lead else 90.0, dynamic=True)
  assert trace[-1, 2] == 0.0
  assert trace[-1, 5] == pytest.approx(3.0 + offset, abs=0.5)
  assert len(set(trace[trace[:, 7] == 1.0, 8])) == 1
