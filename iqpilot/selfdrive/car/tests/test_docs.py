from iqdbc.lvbs.car.car_catalog import build_car_catalog
from iqpilot.selfdrive.car.vehicle_catalog import load_catalog


class TestCarDocs:
  def test_vehicle_catalog(self):
    assert load_catalog() == build_car_catalog()
