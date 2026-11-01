import pytest

from iqdbc.car.values import PLATFORMS
from iqdbc.car.tests.routes import route_exempt_cars, routes


@pytest.mark.parametrize("platform", PLATFORMS.keys())
def test_test_route_present(platform):
  tested_platforms = [r.car_model for r in routes]
  assert platform in set(tested_platforms) | set(route_exempt_cars), \
    f"Missing test route for {platform}. Add a route to iqdbc/car/tests/routes.py"


def test_route_exemptions_are_current():
  tested_platforms = {r.car_model for r in routes}
  assert len(route_exempt_cars) == len(set(route_exempt_cars))
  assert not tested_platforms & set(route_exempt_cars)
  assert set(route_exempt_cars) <= set(PLATFORMS)
