"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos
"""

import numpy as np
import pytest

from iqpilot.selfdrive.state_estimation import ModelDefinition, StateEstimator


def linear_model(process_noise: float = 0.2, observation_noise: float = 0.5) -> ModelDefinition:
  return ModelDefinition(
    state_size=2,
    error_size=2,
    transition=lambda state, dt, _: np.array([state[0] + dt * state[1], state[1]]),
    measurements={1: lambda state, _: state[:1]},
    process_noise=np.eye(2) * process_noise,
    observation_noise={1: np.array([[observation_noise]])},
  )


def test_linear_prediction_matches_closed_form() -> None:
  estimator = StateEstimator(linear_model(), np.array([2.0, 3.0]), np.diag([4.0, 5.0]))
  estimator.init_state(np.array([2.0, 3.0]), np.diag([4.0, 5.0]), 1.0)
  estimator.predict(1.25)
  transition = np.array([[1.0, 0.25], [0.0, 1.0]])
  np.testing.assert_allclose(estimator.x, np.array([2.75, 3.0]), atol=1e-10)
  np.testing.assert_allclose(estimator.P, transition @ np.diag([4.0, 5.0]) @ transition.T + 0.25 * np.eye(2) * 0.2, atol=1e-10)


def test_linear_update_matches_closed_form() -> None:
  estimator = StateEstimator(linear_model(), np.array([0.0, 0.0]), np.diag([2.0, 3.0]))
  estimator.init_state(np.array([0.0, 0.0]), np.diag([2.0, 3.0]), 0.0)
  estimator.predict_and_observe(0.0, 1, np.array([4.0]))
  gain = 2.0 / 2.5
  np.testing.assert_allclose(estimator.x, np.array([gain * 4.0, 0.0]), atol=1e-10)
  np.testing.assert_allclose(estimator.P, np.diag([(1.0 - gain) * 2.0, 3.0]), atol=1e-10)


def test_zero_innovation_does_not_change_state() -> None:
  estimator = StateEstimator(linear_model(), np.array([4.0, 2.0]), np.eye(2))
  estimator.init_state(np.array([4.0, 2.0]), np.eye(2), 0.0)
  estimator.predict_and_observe(0.0, 1, np.array([4.0]))
  np.testing.assert_array_equal(estimator.x, np.array([4.0, 2.0]))


def test_larger_observation_noise_reduces_correction() -> None:
  low = StateEstimator(linear_model(observation_noise=0.1), np.zeros(2), np.eye(2))
  high = StateEstimator(linear_model(observation_noise=10.0), np.zeros(2), np.eye(2))
  low.predict_and_observe(0.0, 1, np.array([1.0]))
  high.predict_and_observe(0.0, 1, np.array([1.0]))
  assert abs(low.x[0]) > abs(high.x[0])


def test_larger_process_noise_increases_uncertainty() -> None:
  low = StateEstimator(linear_model(process_noise=0.1), np.zeros(2), np.eye(2))
  high = StateEstimator(linear_model(process_noise=2.0), np.zeros(2), np.eye(2))
  low.init_state(np.zeros(2), np.eye(2), 0.0)
  high.init_state(np.zeros(2), np.eye(2), 0.0)
  low.predict(1.0)
  high.predict(1.0)
  assert np.all(np.diag(high.P) > np.diag(low.P))


def test_batch_update_preserves_covariance_properties() -> None:
  estimator = StateEstimator(linear_model(), np.zeros(2), np.eye(2))
  estimator.predict_and_observe(0.0, 1, np.array([[1.0], [0.5], [-0.2]]))
  np.testing.assert_allclose(estimator.P, estimator.P.T, atol=1e-12)
  assert np.linalg.eigvalsh(estimator.P).min() >= -1e-10
  assert np.isfinite(estimator.x).all()
  assert np.isfinite(estimator.P).all()


def test_delayed_observation_replays_deterministically() -> None:
  chronological = StateEstimator(linear_model(), np.zeros(2), np.eye(2), max_rewind_age=2.0)
  delayed = StateEstimator(linear_model(), np.zeros(2), np.eye(2), max_rewind_age=2.0)
  chronological.init_state(np.zeros(2), np.eye(2), 0.0)
  delayed.init_state(np.zeros(2), np.eye(2), 0.0)
  chronological.predict_and_observe(0.5, 1, np.array([1.0]))
  chronological.predict_and_observe(1.0, 1, np.array([2.0]))
  delayed.predict_and_observe(1.0, 1, np.array([2.0]))
  delayed.predict_and_observe(0.5, 1, np.array([1.0]))
  np.testing.assert_allclose(delayed.x, chronological.x, atol=1e-10)
  np.testing.assert_allclose(delayed.P, chronological.P, atol=1e-10)


def test_invalid_dimensions_fail_deterministically() -> None:
  with pytest.raises(ValueError, match="state dimension mismatch"):
    StateEstimator(linear_model(), np.zeros(3), np.eye(2))
  estimator = StateEstimator(linear_model(), np.zeros(2), np.eye(2))
  with pytest.raises(ValueError, match="measurement dimension mismatch"):
    estimator.predict_and_observe(0.0, 1, np.zeros(2))
