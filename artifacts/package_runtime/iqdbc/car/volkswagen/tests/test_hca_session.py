# Copyright © IQ.Lvbs, part of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
import pytest

from iqdbc.car import DT_CTRL, structs
from iqdbc.car.volkswagen.hca_session import EpsSessionClock
from iqdbc.car.volkswagen.interface import CarInterface
from iqdbc.car.volkswagen.values import CAR, CarControllerParams


def run(clock, active, seconds):
  for _ in range(round(seconds / DT_CTRL)):
    clock.update(active)


def test_session_age_counts_while_active():
  clock = EpsSessionClock(1.1)
  run(clock, True, 10.0)
  assert clock.age_s == pytest.approx(10.0, abs=0.02)
  assert clock.resets == 0


def test_short_gap_keeps_session_and_counts_through_it():
  clock = EpsSessionClock(1.1)
  run(clock, True, 5.0)
  run(clock, False, 0.5)
  run(clock, True, 5.0)
  assert clock.age_s == pytest.approx(10.5, abs=0.03)
  assert clock.resets == 0


def test_gap_of_reset_duration_closes_session():
  clock = EpsSessionClock(1.1)
  run(clock, True, 300.0)
  run(clock, False, 1.2)
  assert clock.age_s == 0.0
  assert clock.resets == 1
  run(clock, True, 2.0)
  assert clock.age_s == pytest.approx(2.0, abs=0.02)


def test_idle_does_not_count_resets():
  clock = EpsSessionClock(1.1)
  run(clock, False, 30.0)
  assert clock.resets == 0
  assert clock.age_s == 0.0


@pytest.mark.parametrize("candidate", [CAR.AUDI_Q5_MK1])
def test_mlb_dbc_maps_both_active_statuses(candidate, monkeypatch, tmp_path):
  monkeypatch.setenv("PARAMS_ROOT", str(tmp_path))
  ccp = CarControllerParams(CarInterface.get_params(candidate, {bus: {} for bus in range(7)}, [], alpha_long=False, is_release=False, docs=False))
  assert ccp.hca_status_values[5] == "ACTIVE"
  assert ccp.hca_status_values[7] == "ACTIVE"
  assert ccp.hca_status_values[4] == "REJECTED"


def make_interface(candidate, monkeypatch, tmp_path):
  from iqpilot.system.proprietary_runtime import _verified_import

  def missing_bundle(*args):
    raise RuntimeError("bundle not found")

  monkeypatch.setattr(_verified_import, "import_verified_module", missing_bundle)
  monkeypatch.setenv("PARAMS_ROOT", str(tmp_path))
  fingerprint = {bus: {} for bus in range(7)}
  cp = CarInterface.get_params(candidate, fingerprint, [], alpha_long=False, is_release=False, docs=False)
  cp_iq = CarInterface.get_params_iq(cp, candidate, fingerprint, [], alpha_long=False, is_release_iq=False, docs=False)
  return CarInterface(cp, cp_iq)


def drive(interface, lat_active, frames, torque=0.5, start=0):
  cc_iq = structs.IQCarControl()
  now = start
  for _ in range(frames):
    cc = structs.CarControl()
    cc.enabled = lat_active
    cc.latActive = lat_active
    cc.actuators.torque = torque
    now += int(DT_CTRL * 1e9)
    interface.update([])
    interface.apply(cc.as_reader(), cc_iq, now)
  return now


def test_controller_alert_follows_its_own_timer(monkeypatch, tmp_path):
  interface = make_interface(CAR.AUDI_Q5_MK1, monkeypatch, tmp_path)
  controller = interface.CC
  assert controller.eps_timer_workaround
  controller.hca_frame_timer_running = round(300 / DT_CTRL)
  drive(interface, True, 20)
  assert not controller.eps_timer_soft_disable_alert
  assert controller.hca_session_sec == pytest.approx(300.2, abs=0.05)
  controller.hca_frame_timer_running = round(351 / DT_CTRL)
  drive(interface, True, 4)
  assert controller.eps_timer_soft_disable_alert


def test_controller_alert_follows_eps_feedback_clock(monkeypatch, tmp_path):
  interface = make_interface(CAR.AUDI_Q5_MK1, monkeypatch, tmp_path)
  interface.CS.eps_session.age_frames = round(351 / DT_CTRL)
  drive(interface, True, 4)
  assert interface.CC.eps_timer_soft_disable_alert


def test_controller_counts_timer_reset_after_hca_gap(monkeypatch, tmp_path):
  interface = make_interface(CAR.AUDI_Q5_MK1, monkeypatch, tmp_path)
  controller = interface.CC
  controller.hca_frame_timer_running = round(100 / DT_CTRL)
  drive(interface, False, round(1.3 / DT_CTRL))
  assert controller.hca_reset_count == 1
  assert controller.hca_session_sec == 0.0
