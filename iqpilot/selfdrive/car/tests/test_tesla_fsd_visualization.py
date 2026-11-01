from iqdbc.car import structs
from iqdbc.lvbs.car.interfaces import apply_iq_car_config
from iqdbc.lvbs.car.tesla.values import TeslaFlagsIQ, TeslaSafetyFlagsIQ
from iqpilot.selfdrive.car.interfaces import initialize_params


class ParamStore:
  def get(self, name, return_default=False):
    return name == "IQTeslaFsdVisualization"


class CarInterface:
  def get_longitudinal_tuning_iq(self, CP, CP_IQ):
    return None


def test_fsd_visualization_is_snapshotted():
  snapshot = initialize_params(ParamStore())
  params = {key: value for item in snapshot for key, value in item.items()}
  assert params["IQTeslaFsdVisualization"] is True

  CP = structs.CarParams(brand="tesla")
  CP_IQ = structs.IQCarParams()
  apply_iq_car_config(CarInterface(), CP, CP_IQ, snapshot)
  assert CP_IQ.flags & TeslaFlagsIQ.FSD_VISUALIZATION
  assert CP_IQ.iqSafetyFlags & TeslaSafetyFlagsIQ.FSD_VISUALIZATION
