"""Exercise controller construction as well as the imports used at manager startup."""
import pytest

from iqdbc.car import DT_CTRL, structs
from iqdbc.car.volkswagen.interface import CarInterface
from iqdbc.car.volkswagen.values import CAR


@pytest.mark.parametrize("candidate", [
  CAR.VOLKSWAGEN_JETTA_MK6,
  CAR.VOLKSWAGEN_GOLF_MK7,
  CAR.AUDI_Q5_MK1,
  CAR.VOLKSWAGEN_ID4_MK1,
  CAR.VOLKSWAGEN_ID4_MK2,
  CAR.VOLKSWAGEN_GOLF_MK8,
])
def test_controller_starts_and_runs_without_private_alc(candidate, monkeypatch, tmp_path):
  from iqpilot.system.proprietary_runtime import _verified_import

  def missing_bundle(*args):
    raise RuntimeError("bundle not found: iqpilot_alc_private")

  monkeypatch.setattr(_verified_import, "import_verified_module", missing_bundle)
  monkeypatch.setenv("PARAMS_ROOT", str(tmp_path))
  fingerprint = {bus: {} for bus in range(7)}
  cp = CarInterface.get_params(candidate, fingerprint, [], alpha_long=False, is_release=False, docs=False)
  cp_iq = CarInterface.get_params_iq(cp, candidate, fingerprint, [], alpha_long=False, is_release_iq=False, docs=False)
  interface = CarInterface(cp, cp_iq)
  assert interface.CC._iq_lvbs_alc is None

  cc_iq = structs.IQCarControl()
  now = 0
  # Cover both curvature update and lateral handoff, not just a successful import.
  for enabled in (False, True, False):
    cc = structs.CarControl()
    cc.enabled = enabled
    cc.latActive = enabled
    cc.actuators.curvature = 0.001
    for _ in range(10):
      now += int(DT_CTRL * 1e9)
      interface.update([])
      interface.apply(cc.as_reader(), cc_iq, now)
