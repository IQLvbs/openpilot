import ast

from iqdbc.car import gen_empty_fingerprint, structs
from iqdbc.car.can_definitions import CanData
from iqdbc.car.hyundai.carstate import CarState
from iqdbc.car.hyundai.interface import CarInterface
from iqdbc.car.hyundai.values import CAR
from iqpilot.common.params import Params

BCM_PO_11 = 913
CRUISE_BUTTON_ALT = 1007
ButtonType = structs.CarState.ButtonEvent.Type


def elantra_fingerprint():
  fingerprint = gen_empty_fingerprint()
  for addr in (BCM_PO_11, CRUISE_BUTTON_ALT, 0x4F1, 0x386, 0x251):
    fingerprint[0][addr] = 8
  return fingerprint


def build(monkeypatch, tmp_path, write_param=True):
  monkeypatch.setenv("PARAMS_ROOT", str(tmp_path))
  fingerprint = elantra_fingerprint()
  if write_param:
    Params().put("FingerPrints", str({bus: dict(msgs) for bus, msgs in fingerprint.items()}))
  cp = CarInterface.get_params(CAR.HYUNDAI_ELANTRA_2021, fingerprint, [], False, False, False)
  cp_iq = CarInterface.get_params_iq(cp, CAR.HYUNDAI_ELANTRA_2021, fingerprint, [], False, False, False)
  return CarState(cp, cp_iq)


def test_get_car_publishes_the_fingerprint_carstate_reads(monkeypatch, tmp_path):
  monkeypatch.setenv("PARAMS_ROOT", str(tmp_path))
  from iqdbc.car.car_helpers import get_car

  frames = [[CanData(BCM_PO_11, b"\x00" * 8, 0), CanData(CRUISE_BUTTON_ALT, b"\x00" * 8, 0)]]

  def can_recv(wait_for_one=False):
    return frames

  get_car(can_recv, lambda msgs: None, lambda obd: None, False, False,
          fixed_fingerprint=CAR.HYUNDAI_ELANTRA_2021)

  published = ast.literal_eval(Params().get("FingerPrints"))
  assert BCM_PO_11 in published[0]


def test_lfa_button_is_detected_from_the_written_fingerprint(monkeypatch, tmp_path):
  cs = build(monkeypatch, tmp_path)
  assert cs.HAS_LFA_BUTTON
  assert cs.CRUISE_BUTTON_ALT


def test_optional_message_flags_are_off_without_the_fingerprint(monkeypatch, tmp_path):
  cs = build(monkeypatch, tmp_path, write_param=False)
  assert not cs.HAS_LFA_BUTTON
  assert not cs.CRUISE_BUTTON_ALT


def test_written_fingerprint_round_trips(monkeypatch, tmp_path):
  monkeypatch.setenv("PARAMS_ROOT", str(tmp_path))
  fingerprint = elantra_fingerprint()
  Params().put("FingerPrints", str({bus: dict(msgs) for bus, msgs in fingerprint.items()}))
  parsed = ast.literal_eval(Params().get("FingerPrints"))
  assert BCM_PO_11 in parsed[0]


def test_lfa_button_maps_to_a_lateral_toggle_event():
  from iqdbc.car.hyundai.carstate import BUTTONS_DICT
  from iqdbc.car.hyundai.values import Buttons

  assert BUTTONS_DICT[Buttons.LFA_BUTTON] == ButtonType.lfaButton
