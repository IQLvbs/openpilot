"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos
"""
import pytest

from iqdbc.car.common.conversions import Conversions as CV
from iqdbc.car.volkswagen.cluster_ratio import ClusterRatioEstimator

CLUSTER_STEP = 0.32 * CV.KPH_TO_MS
TRUE_RATIO = 1.021


def quantized_cluster(v_ego):
  return round(v_ego * TRUE_RATIO / CLUSTER_STEP) * CLUSTER_STEP


def drive(estimator, speeds, accel=0.0):
  return [estimator.update(v, accel, quantized_cluster(v)) for v in speeds]


def test_learns_the_cluster_ratio_at_cruising_speed():
  est = ClusterRatioEstimator()
  drive(est, [25.0] * 3000)
  assert est.ratio == pytest.approx(TRUE_RATIO, abs=0.004)


def test_low_speed_quantization_does_not_move_the_ratio():
  est = ClusterRatioEstimator()
  drive(est, [25.0] * 3000)
  learned = est.ratio
  ratios = drive(est, [1.1 + 0.01 * i for i in range(400)])
  assert all(r == learned for r in ratios)


def test_set_speed_stays_flat_through_a_pull_away():
  est = ClusterRatioEstimator()
  drive(est, [25.0] * 3000)
  speeds = [min(25.0, 1.0 + 0.02 * i) for i in range(1300)]
  shown = [119.0 * r for r in drive(est, speeds, accel=2.0)]
  assert max(shown) - min(shown) < 0.1


def test_hard_acceleration_is_not_learned_from():
  est = ClusterRatioEstimator()
  drive(est, [25.0] * 3000)
  learned = est.ratio
  for v in [10.0 + 0.1 * i for i in range(100)]:
    est.update(v, 3.0, v * 0.9)
  assert est.ratio == learned


def test_ratio_is_clamped_to_a_plausible_band():
  est = ClusterRatioEstimator()
  for _ in range(5000):
    est.update(25.0, 0.0, 25.0 * 1.5)
  assert est.ratio == pytest.approx(ClusterRatioEstimator.MAX_RATIO, abs=1e-3)
  est = ClusterRatioEstimator()
  for _ in range(5000):
    est.update(25.0, 0.0, 25.0 * 0.5)
  assert est.ratio == pytest.approx(ClusterRatioEstimator.MIN_RATIO, abs=1e-3)


def test_missing_cluster_speed_is_ignored():
  est = ClusterRatioEstimator()
  drive(est, [25.0] * 3000)
  learned = est.ratio
  assert est.update(25.0, 0.0, 0.0) == learned
