import pytest

from iqdbc.can.parser import get_raw_value
from iqdbc.car import Bus, gen_empty_fingerprint, structs
from iqdbc.car.hyundai.hyundaicanfd import CanBus
from iqdbc.car.hyundai.interface import CarInterface
from iqdbc.car.hyundai.values import CAR, Buttons, HyundaiFlags
from iqpilot.common.params import Params

CRUISE_BUTTONS = 0x1CF
CAMERA_SCC_EV6 = HyundaiFlags.CANFD | HyundaiFlags.CANFD_CAMERA_SCC | HyundaiFlags.EV


def camera_scc_ev6(monkeypatch, tmp_path):
  monkeypatch.setenv("PARAMS_ROOT", str(tmp_path))
  Params().put_bool("ControlsReady", True)
  fingerprint = gen_empty_fingerprint()
  cp = CarInterface.get_params(CAR.KIA_EV6, fingerprint, [], True, False, False)
  cp.flags = int(CAMERA_SCC_EV6)
  cp.openpilotLongitudinalControl = True
  cp_iq = CarInterface.get_params_iq(cp, CAR.KIA_EV6, fingerprint, [], True, False, False)
  interface = CarInterface(cp, cp_iq)
  interface.update([(10_000_000, [])])
  buttons = {sig: 0 for sig in interface.can_parsers[Bus.pt].dbc.name_to_msg["CRUISE_BUTTONS"].sigs}
  interface.CS.cruise_buttons_msg = buttons
  interface.CS.cruise_btns_msg_canfd = "CRUISE_BUTTONS"
  interface.CS.MainMode_ACC = True
  interface.CS.ACCMode = 1
  return interface


def buttons_sent_to_camera(interface, enabled, driver_button, frames=40):
  cc = structs.CarControl()
  cc.enabled = enabled
  cc = cc.as_reader()
  cc_iq = structs.IQCarControl()
  interface.CS.cruise_buttons_msg["CRUISE_BUTTONS"] = driver_button
  signal = interface.can_parsers[Bus.pt].dbc.name_to_msg["CRUISE_BUTTONS"].sigs["CRUISE_BUTTONS"]
  camera_bus = CanBus(interface.CP).CAM
  seen = []
  t = 10_000_000
  for _ in range(frames):
    t += 10_000_000
    _, sends = interface.apply(cc, cc_iq, t)
    seen += [get_raw_value(dat, signal) for addr, dat, bus in sends if addr == CRUISE_BUTTONS and bus == camera_bus]
  return seen


@pytest.mark.parametrize("held", [Buttons.SET_DECEL, Buttons.RES_ACCEL])
def test_engaged_camera_scc_never_sees_a_held_set_speed_button(held, monkeypatch, tmp_path):
  interface = camera_scc_ev6(monkeypatch, tmp_path)
  seen = buttons_sent_to_camera(interface, True, held)
  assert seen
  assert held not in seen


@pytest.mark.parametrize("button", [Buttons.CANCEL, Buttons.GAP_DIST])
def test_engaged_camera_scc_still_gets_cancel_and_gap(button, monkeypatch, tmp_path):
  interface = camera_scc_ev6(monkeypatch, tmp_path)
  assert set(buttons_sent_to_camera(interface, True, button)) == {button}


def test_disengaged_camera_scc_gets_the_driver_buttons(monkeypatch, tmp_path):
  interface = camera_scc_ev6(monkeypatch, tmp_path)
  assert Buttons.SET_DECEL in buttons_sent_to_camera(interface, False, Buttons.SET_DECEL)


def test_engaged_camera_scc_keeps_openpilots_own_rearm_press(monkeypatch, tmp_path):
  interface = camera_scc_ev6(monkeypatch, tmp_path)
  interface.CS.ACCMode = 4
  interface.CS.out.vEgo = 20.0
  assert Buttons.SET_DECEL in buttons_sent_to_camera(interface, True, Buttons.NONE, frames=220)
