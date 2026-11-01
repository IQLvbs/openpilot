import pytest

from iqdbc.car import Bus, gen_empty_fingerprint
from iqdbc.car.hyundai.interface import CarInterface
from iqdbc.car.hyundai.values import CAR, HyundaiFlags
from iqpilot.common.params import Params

FCA11 = 0x38D
SCC_FRAMES = (0x420, 0x421, 0x50A, 0x389)
RADAR_FRAMES = (FCA11, *SCC_FRAMES)


def build(candidate, alpha_long, camera_scc, monkeypatch, tmp_path):
  monkeypatch.setenv("PARAMS_ROOT", str(tmp_path))
  Params().put_bool("ControlsReady", True)
  fingerprint = gen_empty_fingerprint()
  for addr in RADAR_FRAMES:
    fingerprint[0][addr] = 8
  cp = CarInterface.get_params(candidate, fingerprint, [], alpha_long, False, False)
  if camera_scc:
    cp.flags |= HyundaiFlags.CAMERA_SCC.value
  cp_iq = CarInterface.get_params_iq(cp, candidate, fingerprint, [], alpha_long, False, False)
  return CarInterface(cp, cp_iq)


def run_registration_ladder(interface):
  parsers = interface.can_parsers
  for parser in (parsers[Bus.pt], parsers[Bus.cam]):
    parser.seen_addresses |= set(RADAR_FRAMES)
  interface.CS.controls_ready_count = 103
  for _ in range(3):
    interface.CS.monitor_fingerprint(parsers, False)
  return parsers[Bus.pt].addresses, parsers[Bus.cam].addresses


@pytest.mark.parametrize("candidate", (CAR.HYUNDAI_ELANTRA_2021, CAR.KIA_K5_2021, CAR.HYUNDAI_SONATA))
def test_openpilot_longitudinal_never_waits_on_the_silenced_radar(candidate, monkeypatch, tmp_path):
  interface = build(candidate, True, False, monkeypatch, tmp_path)
  assert interface.CP.openpilotLongitudinalControl
  pt, cam = run_registration_ladder(interface)
  assert not set(RADAR_FRAMES) & (pt | cam)


def test_stock_longitudinal_still_reads_the_radar_frames(monkeypatch, tmp_path):
  interface = build(CAR.HYUNDAI_ELANTRA_2021, False, False, monkeypatch, tmp_path)
  assert not interface.CP.openpilotLongitudinalControl
  pt, _ = run_registration_ladder(interface)
  assert set(RADAR_FRAMES) <= pt
  assert interface.CS.fca11 is not None


def test_camera_scc_keeps_reading_the_camera_frames(monkeypatch, tmp_path):
  interface = build(CAR.HYUNDAI_ELANTRA_2021, True, True, monkeypatch, tmp_path)
  _, cam = run_registration_ladder(interface)
  assert set(RADAR_FRAMES) <= cam
  assert interface.CS.fca11 is not None
