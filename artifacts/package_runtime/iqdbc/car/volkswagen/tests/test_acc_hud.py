# Copyright © IQ.Lvbs, part of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
import pytest

from iqdbc.can import CANParser
from iqdbc.car import structs
from iqdbc.car.volkswagen.hud import AccHudState
from iqdbc.car.volkswagen.interface import CarInterface
from iqdbc.car.volkswagen.values import CAR


@pytest.fixture(params=[
  (CAR.VOLKSWAGEN_GOLF_MK7, "vw_mqb", "ACC_02", "ACC_Status_Anzeige", "ACC_Display_Prio", "ACC_Abstandsindex"),
  (CAR.VOLKSWAGEN_PASSAT_NMS, "vw_pq", "ACC_GRA_Anzeige", "ACA_StaACC", "ACA_PrioDisp", "ACA_gemZeitl"),
  (CAR.AUDI_A4_MK4, "vw_mlb", "ACC_02", "ACC_Status_Anzeige", "ACC_Display_Prio", "ACC_Relevantes_Objekt"),
])
def vw_hud(request):
  candidate, dbc, message, status, priority, lead = request.param
  fingerprint = {bus: {} for bus in range(7)}
  cp = CarInterface.get_params(candidate, fingerprint, [], alpha_long=True, is_release=False, docs=False)
  cp_iq = CarInterface.get_params_iq(cp, candidate, fingerprint, [], alpha_long=True, is_release_iq=False, docs=False)
  car = CarInterface(cp, cp_iq)
  car.update([(0, [])])
  car.CS.out = structs.CarState()
  car.CS.out.cruiseState.available = True
  car.CS.acc_type = 1
  parser = CANParser(dbc, [(message, 0)], car.CC.CAN.pt)
  return car, parser, message, status, priority, lead


def apply_hud(vw_hud, *, enabled=False, active=False, gas=False, override=False, lead=False, fcw=False):
  car, parser, message, *_ = vw_hud
  car.CS.out.gasPressed = gas
  cc = structs.CarControl()
  cc.enabled, cc.longActive = enabled, active
  cc.cruiseControl.override = override
  cc.hudControl.leadVisible = lead
  cc.hudControl.leadDistance = 12.0
  cc.hudControl.leadDistanceBars = 2
  cc.hudControl.setSpeed = 255 / 3.6
  cc.hudControl.visualAlert = structs.CarControl.HUDControl.VisualAlert.fcw if fcw else structs.CarControl.HUDControl.VisualAlert.none
  now = (car.CC.frame + 1) * 10_000_000
  _, messages = car.apply(cc.as_reader(), structs.IQCarControl(), now)
  parser.update([(now, messages)])
  return parser.vl[message]


@pytest.mark.parametrize("available,expected", [(False, 0), (True, 2)])
def test_disengaged_gas_keeps_off_or_standby(vw_hud, available, expected):
  car, _, _, status, priority, lead = vw_hud
  car.CS.out.cruiseState.available = available
  values = apply_hud(vw_hud, gas=True, lead=True, override=True)
  assert values[status] == expected
  assert values[priority] == 3
  assert values[lead] == 0


@pytest.mark.parametrize("active", [False, True])
def test_engaged_gas_keeps_override_display(vw_hud, active):
  *_, status, _, _ = vw_hud
  values = apply_hud(vw_hud, enabled=True, active=active, gas=True)
  assert values[status] == 4


def test_fault_outranks_gas_and_active_display(vw_hud):
  car, _, _, status, _, _ = vw_hud
  car.CS.out.accFaulted = True
  values = apply_hud(vw_hud, enabled=True, active=True, gas=True)
  assert values[status] == 6


def test_continuous_lead_expires_without_changing_input(vw_hud):
  car, _, _, status, priority, lead = vw_hud
  values = apply_hud(vw_hud, enabled=True, active=True, lead=True)
  assert values[lead] > 0
  assert values[priority] == 1
  for _ in range(AccHudState.DISPLAY_FRAMES + car.CC.CCP.ACC_HUD_STEP):
    values = apply_hud(vw_hud, enabled=True, active=True, lead=True)
  assert values[status] == 3
  assert values[priority] == 2
  assert values[lead] == 0


def test_fcw_bypasses_expired_lead_display(vw_hud):
  car, _, _, _, priority, lead = vw_hud
  for _ in range(AccHudState.DISPLAY_FRAMES + car.CC.CCP.ACC_HUD_STEP):
    apply_hud(vw_hud, enabled=True, active=True, lead=True)
  for _ in range(car.CC.CCP.ACC_HUD_STEP):
    values = apply_hud(vw_hud, enabled=True, active=True, lead=True, fcw=True)
  assert values[priority] == 0
  assert values[lead] > 0


def test_lead_flicker_does_not_extend_display():
  state = AccHudState()
  for frame in range(400):
    visible, _ = state.update(frame, True, frame % 3 != 1, 2)
    if frame >= state.DISPLAY_FRAMES:
      assert not visible
  for frame in range(400, 400 + state.LEAD_CLEAR_FRAMES):
    state.update(frame, True, False, 2)
  assert state.update(400 + state.LEAD_CLEAR_FRAMES, True, True, 2) == (True, True)


def test_disengaged_lead_never_requests_display():
  state = AccHudState()
  for frame in range(400):
    assert state.update(frame, False, frame % 3 != 1, 2) == (False, False)


def test_disengagement_clears_lead_and_times_out_notification():
  state = AccHudState()
  state.update(0, True, True, 2)
  assert state.update(1, False, True, 2) == (False, True)
  assert state.update(1 + state.DISPLAY_FRAMES, False, True, 2) == (False, False)
