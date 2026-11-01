"""Keep AOL steering independent of cruise without reviving stale HCA torque."""
import pytest

from iqdbc.can import CANParser
from iqdbc.car import DT_CTRL, structs
from iqdbc.car.volkswagen.interface import CarInterface
from iqdbc.car.volkswagen.values import CAR


@pytest.fixture(params=[CAR.VOLKSWAGEN_JETTA_MK6, CAR.VOLKSWAGEN_GOLF_MK7, CAR.AUDI_Q5_MK1])
def hca_frames(request, monkeypatch, tmp_path):
  monkeypatch.setenv("PARAMS_ROOT", str(tmp_path))
  fingerprint = {bus: {} for bus in range(7)}
  candidate = request.param
  cp = CarInterface.get_params(candidate, fingerprint, [], alpha_long=False, is_release=False, docs=False)
  cp_iq = CarInterface.get_params_iq(cp, candidate, fingerprint, [], alpha_long=False, is_release_iq=False, docs=False)
  interface = CarInterface(cp, cp_iq)
  controller = interface.CC
  pq = candidate == CAR.VOLKSWAGEN_JETTA_MK6
  message = "HCA_1" if pq else "HCA_01"
  torque_signal = "LM_Offset" if pq else "HCA_01_LM_Offset"
  status_signal = "HCA_Status" if pq else "HCA_01_Status_HCA"
  parser = CANParser(controller.packer_pt.dbc.name, [(message, 50)], controller._pt_tx_bus)
  address = parser.dbc.name_to_msg[message].address
  now = 0

  def run(*, enabled, lat_active=True, aol_available=False, aol_enabled=False, aol_active=False,
          driver_torque=0, steps=20):
    nonlocal now
    cc = structs.CarControl()
    cc.enabled = enabled
    cc.latActive = lat_active
    cc.actuators.torque = 1.0
    cc_iq = structs.IQCarControl()
    cc_iq.aol.available = aol_available
    cc_iq.aol.enabled = aol_enabled
    cc_iq.aol.active = aol_active
    frames = []
    for _ in range(steps):
      now += int(DT_CTRL * 1e9)
      interface.update([])
      interface.CS.out.vEgo = 20.0
      interface.CS.out.steeringTorque = driver_torque
      interface.CS.out.steeringPressed = driver_torque != 0
      output, can_sends = interface.apply(cc.as_reader(), cc_iq, now)
      if any(addr == address and bus == parser.bus for addr, _, bus in can_sends):
        assert address in parser.update([(now, can_sends)])
        values = parser.vl[message]
        frames.append((values[torque_signal], values[status_signal], output.torqueOutputCan))
    assert frames, "The controller must emit real HCA CAN frames"
    return frames

  return run


ACTIVE_AOL = {"aol_available": True, "aol_enabled": True, "aol_active": True}


def assert_inactive(frames):
  # Check the first transmitted frame too: a late ramp-down can trip the EPS alarm.
  assert all(torque == 0 and status == 3 and output == 0 for torque, status, output in frames)


def assert_active(frames):
  assert all(torque > 0 and status in (5, 7) and output > 0 for torque, status, output in frames)


@pytest.mark.parametrize("enabled,aol", [(True, {}), (False, ACTIVE_AOL), (True, ACTIVE_AOL)])
def test_cruise_and_active_aol_can_each_authorize_torque(hca_frames, enabled, aol):
  assert_active(hca_frames(enabled=enabled, **aol))


@pytest.mark.parametrize("aol", [
  {},
  {"aol_available": True},
  {"aol_available": True, "aol_enabled": True},  # paused, not torque-authorized
  {"aol_available": True, "aol_active": True},  # inconsistent, not enabled
  {"aol_enabled": True, "aol_active": True},  # inconsistent, preference unavailable
])
def test_lingering_lateral_flag_does_not_authorize_torque(hca_frames, aol):
  assert_active(hca_frames(enabled=True))
  assert_inactive(hca_frames(enabled=False, lat_active=True, steps=25, **aol))


@pytest.mark.parametrize("enabled", [False, True])
def test_lateral_cutoff_overrides_both_authorities(hca_frames, enabled):
  assert_active(hca_frames(enabled=enabled, **ACTIVE_AOL))
  assert_inactive(hca_frames(enabled=enabled, lat_active=False, **ACTIVE_AOL))


@pytest.mark.parametrize("was_cruise_enabled", [False, True])
@pytest.mark.parametrize("keep_aol_enabled", [False, True], ids=["cancelled", "paused"])
def test_aol_cancel_or_pause_stops_hca_immediately(hca_frames, was_cruise_enabled, keep_aol_enabled):
  assert_active(hca_frames(enabled=was_cruise_enabled, **ACTIVE_AOL))
  assert_inactive(hca_frames(enabled=False, lat_active=True, aol_available=True,
                             aol_enabled=keep_aol_enabled, aol_active=False, steps=25))


def test_cruise_to_active_aol_handoff_retains_steering(hca_frames):
  assert_active(hca_frames(enabled=True, **ACTIVE_AOL))
  assert_active(hca_frames(enabled=False, **ACTIVE_AOL))


def test_aol_still_yields_to_opposing_driver_torque(hca_frames):
  assert_inactive(hca_frames(enabled=False, driver_torque=-5000, **ACTIVE_AOL))
