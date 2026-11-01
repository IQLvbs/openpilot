import numpy as np
import pytest

from iqpilot.selfdrive.locationd.helpers import ParameterEstimator, PointBuckets, fft_next_good_size


class ScalarBuckets(PointBuckets):
  def add_point(self, x, y):
    for bounds, bucket in self.buckets.items():
      if bounds[0] <= x < bounds[1]:
        bucket.append([x, y])
        return
    raise ValueError(x)


def test_fft_next_good_size_small_input():
  assert fft_next_good_size(6) == 6


def test_point_buckets_base_requires_an_insertion_policy():
  buckets = PointBuckets([(0, 1)], [1], 1, 2, 2)
  with pytest.raises(NotImplementedError):
    buckets.add_point(0.5, 1.0)


def test_point_buckets_load_and_retrieve():
  buckets = ScalarBuckets([(0, 1), (1, 2)], [1, 1], 2, 3, 2)
  points = [[0.25, 10.0], [1.25, 20.0], [0.75, 30.0]]
  buckets.load_points(points)
  assert buckets.is_valid()
  assert buckets.is_calculable()
  np.testing.assert_allclose(buckets.get_points(), [[0.25, 10.0], [0.75, 30.0], [1.25, 20.0]])
  assert buckets.get_points(2).shape == (2, 2)


@pytest.mark.parametrize("method,args", [
  ("reset", ()),
  ("handle_log", (0, "carState", None)),
  ("get_msg", (True, False)),
])
def test_parameter_estimator_requires_an_implementation(method, args):
  with pytest.raises(NotImplementedError):
    getattr(ParameterEstimator(), method)(*args)
