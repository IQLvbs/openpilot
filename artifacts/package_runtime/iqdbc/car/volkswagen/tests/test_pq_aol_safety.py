"""Exercise the shipped no-ALC controller against Panda's compiled safety hooks."""
import pytest

from iqdbc.can import CANPacker, CANParser
from iqdbc.car import DT_CTRL, structs
from iqdbc.car.volkswagen.interface import CarInterface
from iqdbc.car.volkswagen.values import CAR
from iqdbc.safety import ALTERNATIVE_EXPERIENCE
from iqdbc.safety.tests.libsafety import libsafety_py


APD = 0x3D6
HCA = 0xD2


@pytest.fixture(params=[False, True], ids=["gateway", "lowline"])
def pq_without_alc(request, monkeypatch, tmp_path):
  from iqpilot.system.proprietary_runtime import _verified_import

  def missing_bundle(*args):
    raise RuntimeError("bundle not found: iqpilot_alc_private")

  monkeypatch.setattr(_verified_import, "import_verified_module", missing_bundle)
  monkeypatch.setenv("PARAMS_ROOT", str(tmp_path))
  fingerprint = {bus: {} for bus in range(7)}
  if not request.param:
    fingerprint[0] = {0x368: 8, 0x56A: 8, 0x440: 8}
  fingerprint[1] = {0x1A0: 8}
  cp = CarInterface.get_params(CAR.VOLKSWAGEN_JETTA_MK6, fingerprint, [], False, False, False)
  cp.alternativeExperience |= ALTERNATIVE_EXPERIENCE.ENABLE_AOL
  cp_iq = CarInterface.get_params_iq(cp, CAR.VOLKSWAGEN_JETTA_MK6, fingerprint, [], False, False, False)
  interface = CarInterface(cp, cp_iq)
  assert interface.CC._iq_lvbs_alc is None

  safety = libsafety_py.libsafety
  config = cp.safetyConfigs[0]
  assert safety.set_safety_hooks(config.safetyModel.raw, config.safetyParam) == 0
  safety.init_tests()
  safety.set_aol_params(True, False, False)
  safety.set_heartbeat_engaged_aol(True)
  rx_packer = CANPacker("vw_pq")

  def rx(message, values):
    addr, data, bus = rx_packer.make_can_msg(message, 1, values)
    assert safety.safety_rx_hook(libsafety_py.make_CANPacket(addr, bus, data))

  # Enable AOL through the real ACC-main RX path, without granting cruise authority.
  rx("Motor_5", {"MO5_GRA_Hauptsch": 1})
  assert safety.get_enable_aol()
  assert safety.get_controls_allowed_lat()
  assert not safety.get_controls_allowed()
  now = 0

  def send(*, cruise=False, aol_active=True, lat_active=True, aol_enabled=True, aol_available=True, frames=20):
    nonlocal now
    cc = structs.CarControl()
    cc.enabled = cruise
    cc.latActive = lat_active
    cc.actuators.torque = 0.5
    cc_iq = structs.IQCarControl()
    cc_iq.aol.available = aol_available
    cc_iq.aol.enabled = aol_enabled
    cc_iq.aol.active = aol_active
    cc_iq.angleOffsetDeg = 1.25
    result = []
    for _ in range(frames):
      now += int(DT_CTRL * 1e9)
      safety.set_timer(now // 1000)
      rx("Motor_2", {"MO2_Sta_GRA": int(cruise)})
      assert bool(safety.get_controls_allowed()) == cruise
      interface.update([])
      interface.CS.out.vEgo = 20.0
      interface.CS.out.steeringTorque = 0.0
      _, sends = interface.apply(cc.as_reader(), cc_iq, now)
      for addr, data, bus in sends:
        # Preserve controller ordering; an authorization sent after HCA is too late.
        accepted = bool(safety.safety_tx_hook(libsafety_py.make_CANPacket(addr, bus, data)))
        result.append((addr, data, bus, accepted))
    return result

  yield interface, safety, send
  safety.set_safety_hooks(config.safetyModel.raw, config.safetyParam)
  safety.init_tests()


def steering(messages):
  frames = [msg for msg in messages if msg[0] == HCA]
  assert frames, "The controller must emit HCA steering frames"
  return frames


def apd(messages):
  return [msg for msg in messages if msg[0] == APD]


def torque(message):
  return (int.from_bytes(message[1][2:4], "little") & 0x7FFF) // 32


def test_aol_commands_reach_panda_without_cruise_or_private_bundle(pq_without_alc):
  interface, safety, send = pq_without_alc
  messages = send(frames=100)
  hca = steering(messages)
  assert hca and all(msg[3] for msg in hca), "Panda rejected AOL steering despite application activation"
  assert all(torque(msg) > 0 for msg in hca)
  assert len(apd(messages)) == len(hca) == 50
  assert all(msg[2] == 1 and msg[3] for msg in apd(messages))
  assert all(msg[2] == interface.CC._pt_tx_bus for msg in hca)
  assert [msg[1][1] & 0xF for msg in apd(messages)] == [i % 16 for i in range(50)]
  assert not safety.get_controls_allowed(), "The test must not borrow cruise authorization"


def test_cruise_cancel_preserves_authorized_aol(pq_without_alc):
  _, safety, send = pq_without_alc
  assert all(msg[3] for msg in steering(send(cruise=True)))
  messages = send(cruise=False)
  assert all(msg[3] and torque(msg) > 0 for msg in steering(messages))
  assert not safety.get_controls_allowed()


@pytest.mark.parametrize("aol_available,aol_enabled,aol_active,lat_active", [
  (True, False, False, True),  # full cancellation, with a lingering lateral-active bit
  (True, True, False, True),  # paused AOL stays enabled but must not authorize torque
  (True, True, True, False),  # a lateral fault/cutoff overrides a still-active AOL state
  (False, True, True, True),  # an unavailable preference cannot authorize steering
  (True, False, True, True),  # an inconsistent active bit cannot revive disabled AOL
])
def test_cutoff_clears_panda_authority_before_inactive_hca(pq_without_alc, aol_available, aol_enabled, aol_active, lat_active):
  _, safety, send = pq_without_alc
  active_hca = steering(send())[-1]
  assert active_hca[3] and torque(active_hca) > 0
  messages = send(aol_available=aol_available, aol_enabled=aol_enabled, aol_active=aol_active, lat_active=lat_active)
  assert apd(messages), "Panda needs an explicit authorization clear, not silence"
  assert all(not (msg[1][2] & 8) and msg[3] for msg in apd(messages))
  assert all(torque(msg) == 0 and msg[1][1] >> 4 == 3 and msg[3] for msg in steering(messages))
  addr, data, bus, _ = active_hca
  assert not safety.safety_tx_hook(libsafety_py.make_CANPacket(addr, bus, data))


def test_public_apd_preserves_vehicle_geometry_and_calibration(pq_without_alc):
  interface, _, send = pq_without_alc
  parser = CANParser("vw_pq", [("APD_1", 50)], 1)
  messages = apd(send())
  assert messages
  for i, (addr, data, bus, accepted) in enumerate(messages):
    assert accepted
    assert APD in parser.update([(int(i * DT_CTRL * 2e9), [(addr, data, bus)])])
    values = parser.vl["APD_1"]
    assert values["APD_Version"] == 1
    assert values["APD_Flags"] == 9
    assert values["APD_SteerRatio"] == pytest.approx(interface.CP.steerRatio, abs=0.005)
    assert values["APD_Wheelbase"] == pytest.approx(interface.CP.wheelbase * 1000, abs=0.5)
    assert values["APD_AngleOffSet"] == pytest.approx(1.25, abs=0.005)


def test_plain_cruise_and_full_disengagement_still_work(pq_without_alc):
  _, safety, send = pq_without_alc
  messages = send(cruise=True, aol_active=False, aol_enabled=False)
  assert all(msg[3] and torque(msg) > 0 for msg in steering(messages))
  assert all(not (msg[1][2] & 8) for msg in apd(messages))
  active_hca = steering(messages)[-1]
  messages = send(cruise=False, aol_active=False, aol_enabled=False)
  assert all(msg[3] and torque(msg) == 0 and msg[1][1] >> 4 == 3 for msg in steering(messages))
  addr, data, bus, _ = active_hca
  assert not safety.safety_tx_hook(libsafety_py.make_CANPacket(addr, bus, data))


def test_private_sender_does_not_get_a_duplicate_fallback(pq_without_alc):
  interface, _, send = pq_without_alc

  class PrivateAlc:
    calls = 0

    @staticmethod
    def angle_lateral_control_enabled(*args):
      return False

    @staticmethod
    def update_vw_alc(*args):
      pass

    @staticmethod
    def update_turn_signals(*args):
      pass

    @classmethod
    def append_private_apd(cls, controller, cc_iq, messages):
      cls.calls += 1

  interface.CC._iq_lvbs_alc = PrivateAlc()
  assert not apd(send())
  assert PrivateAlc.calls == 10
