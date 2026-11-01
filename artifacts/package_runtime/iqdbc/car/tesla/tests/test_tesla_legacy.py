"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos
"""
import pytest

from iqdbc.can import CANPacker, CANParser
from iqdbc.car import Bus, gen_empty_fingerprint, structs
from iqdbc.car.fw_versions import match_fw_to_car
from iqdbc.car.structs import CarParams
from iqdbc.car.tesla.carcontroller import CarController
from iqdbc.car.tesla.carstate import CarState
from iqdbc.car.tesla.fingerprints import FW_VERSIONS
from iqdbc.car.tesla.interface import CarInterface
from iqdbc.car.tesla.radar_interface import BOSCH_TRIGGER_MSG, RadarInterface
from iqdbc.car.tesla.teslacan_legacy import TeslaCANLegacy
from iqdbc.car.tesla.values import (CAR, DBC, LEGACY_CARS, LEGACY_HW1_CARS, LEGACY_HW2_CARS, TeslaFlags,
                                    TeslaLegacySafetyFlags, get_legacy_canbus)

Ecu = CarParams.Ecu
NANOS = 1_000_000_000


def make_params(candidate, fingerprint=None):
  fingerprint = fingerprint or gen_empty_fingerprint()
  CP = CarInterface.get_params(candidate, fingerprint, [], False, False, False)
  CP_IQ = CarInterface.get_params_iq(CP, candidate, fingerprint, [], False, False, False)
  return CP, CP_IQ


def checksum(addr, dat):
  return ((addr & 0xFF) + ((addr >> 8) & 0xFF) + sum(dat)) & 0xFF


class TestLegacyParams:
  @pytest.mark.parametrize("candidate", LEGACY_CARS)
  def test_safety_configs(self, candidate):
    CP, CP_IQ = make_params(candidate)

    assert CP.brand == "tesla"
    assert CP.steerControlType == structs.CarParams.SteerControlType.angle
    assert CP.openpilotLongitudinalControl
    assert all(cfg.safetyModel == CarParams.SafetyModel.teslaLegacy for cfg in CP.safetyConfigs)
    assert CP.safetyConfigs[0].safetyParam & TeslaLegacySafetyFlags.LONG_CONTROL
    assert CP_IQ.flags == 0
    assert not CP.enableBsm

    if candidate in LEGACY_HW1_CARS:
      assert len(CP.safetyConfigs) == 1
      assert CP.safetyConfigs[0].safetyParam & TeslaLegacySafetyFlags.HW1
      assert not CP.safetyConfigs[0].safetyParam & TeslaLegacySafetyFlags.EXTERNAL_PANDA
    else:
      hw = TeslaLegacySafetyFlags.HW2 if candidate in LEGACY_HW2_CARS else TeslaLegacySafetyFlags.HW3
      assert len(CP.safetyConfigs) == 2
      assert CP.safetyConfigs[0].safetyParam & hw
      assert not CP.safetyConfigs[0].safetyParam & TeslaLegacySafetyFlags.EXTERNAL_PANDA
      assert CP.safetyConfigs[1].safetyParam & hw
      assert CP.safetyConfigs[1].safetyParam & TeslaLegacySafetyFlags.EXTERNAL_PANDA

  def test_hw3_safety_param_values_match_safety_header(self):
    CP, _ = make_params(CAR.TESLA_MODEL_S_HW3)
    assert [cfg.safetyParam for cfg in CP.safetyConfigs] == [33, 36]

  def test_no_sdm1_flag(self):
    fingerprint = gen_empty_fingerprint()
    CP, _ = make_params(CAR.TESLA_MODEL_S_HW3, fingerprint)
    assert CP.flags & TeslaFlags.NO_SDM1

    fingerprint[0][0x201] = 5
    CP, _ = make_params(CAR.TESLA_MODEL_S_HW3, fingerprint)
    assert not CP.flags & TeslaFlags.NO_SDM1

  def test_radar_unavailable_matches_xnor(self):
    unavailable = {c for c in LEGACY_CARS if make_params(c)[0].radarUnavailable}
    assert unavailable == {CAR.TESLA_MODEL_S_HW2}

  def test_modern_platforms_unchanged(self):
    CP, _ = make_params(CAR.TESLA_MODEL_Y)
    assert CP.safetyConfigs[0].safetyModel == CarParams.SafetyModel.tesla
    assert len(CP.safetyConfigs) == 1


class TestLegacyBuses:
  @pytest.mark.parametrize("candidate, expected", [
    (CAR.TESLA_MODEL_S_HW1, {Bus.party: 0, Bus.ap_party: 2, Bus.pt: 0, Bus.ap_pt: 2, Bus.chassis: 0}),
    (CAR.TESLA_MODEL_X_HW1, {Bus.party: 0, Bus.ap_party: 2, Bus.pt: 0, Bus.ap_pt: 2, Bus.chassis: 0}),
    (CAR.TESLA_MODEL_S_HW2, {Bus.party: 0, Bus.ap_party: 2, Bus.pt: 4, Bus.ap_pt: 6, Bus.chassis: 0}),
    (CAR.TESLA_MODEL_X_HW2, {Bus.party: 0, Bus.ap_party: 2, Bus.pt: 4, Bus.ap_pt: 6, Bus.chassis: 0}),
    (CAR.TESLA_MODEL_S_HW3, {Bus.party: 0, Bus.ap_party: 2, Bus.pt: 4, Bus.ap_pt: 6, Bus.chassis: 1}),
    (CAR.TESLA_MODEL_X_HW3, {Bus.party: 0, Bus.ap_party: 2, Bus.pt: 4, Bus.ap_pt: 6, Bus.chassis: 1}),
  ])
  def test_parser_buses(self, candidate, expected):
    CP, CP_IQ = make_params(candidate)
    parsers = CarState.get_can_parsers(CP, CP_IQ)
    assert {bus: parser.bus for bus, parser in parsers.items()} == expected

  def test_hw3_radar_is_on_the_external_panda(self):
    assert get_legacy_canbus(CAR.TESLA_MODEL_S_HW3).radar == 5
    assert get_legacy_canbus(CAR.TESLA_MODEL_X_HW3).radar == 5
    assert get_legacy_canbus(CAR.TESLA_MODEL_S_HW2).radar == 1

  def test_dbc_map(self):
    assert DBC[CAR.TESLA_MODEL_S_HW3][Bus.party] == 'tesla_raven_party'
    assert DBC[CAR.TESLA_MODEL_S_HW3][Bus.pt] == 'tesla_powertrain'
    assert DBC[CAR.TESLA_MODEL_S_HW3][Bus.chassis] == 'tesla_can'
    assert DBC[CAR.TESLA_MODEL_S_HW3][Bus.radar] == 'tesla_radar_continental_generated'
    assert DBC[CAR.TESLA_MODEL_S_HW1][Bus.pt] == 'tesla_can'
    assert DBC[CAR.TESLA_MODEL_S_HW2][Bus.pt] == 'tesla_powertrain'
    assert DBC[CAR.TESLA_MODEL_S_HW1][Bus.radar] == 'tesla_radar_bosch_generated'


class TestLegacyFingerprint:
  @pytest.mark.parametrize("candidate", [c for c in LEGACY_CARS if c in FW_VERSIONS])
  def test_exact_match(self, candidate):
    for fw in FW_VERSIONS[candidate][(Ecu.eps, 0x730, None)]:
      car_fw = [CarParams.CarFw(ecu=Ecu.eps, fwVersion=fw, address=0x730, subAddress=0, brand='tesla')]
      exact, matches = match_fw_to_car(car_fw, '0' * 17, log=False)
      assert exact
      assert matches == {candidate}

  def test_hw3_fw_is_distinct_from_modern_platforms(self):
    modern = set()
    for car, ecus in FW_VERSIONS.items():
      if car not in LEGACY_CARS:
        modern |= {fw for fws in ecus.values() for fw in fws}
    for fw in FW_VERSIONS[CAR.TESLA_MODEL_S_HW3][(Ecu.eps, 0x730, None)]:
      assert fw not in modern


def feed(parsers, *groups):
  messages = [msg for group in groups for msg in group]
  for parser in parsers.values():
    parser.update([(NANOS, messages)])


def pack(dbc, bus, **messages):
  packer = CANPacker(dbc)
  return [packer.make_can_msg(name, bus, values) for name, values in messages.items()]


class TestLegacyCarState:
  @staticmethod
  def car_state(candidate, fingerprint=None):
    CP, CP_IQ = make_params(candidate, fingerprint)
    cs = CarState(CP, CP_IQ)
    parsers = CarState.get_can_parsers(CP, CP_IQ)
    cs.update(parsers)
    return cs, parsers

  def test_hw3_decodes_every_source(self):
    cs, parsers = self.car_state(CAR.TESLA_MODEL_S_HW3)

    feed(
      parsers,
      pack('tesla_raven_party', 0, EPAS_sysStatus={
        "EPAS_internalSAS": 10.0, "EPAS_handsOnLevel": 1, "EPAS_torsionBarTorque": 0.5, "EPAS_eacStatus": 2}),
      pack('tesla_can', 1,
           ESP_B={"ESP_vehicleSpeed": 72.0},
           BrakeMessage={"driverBrakeStatus": 2},
           DI_state={"DI_cruiseState": 2, "DI_speedUnits": 1, "DI_digitalSpeed": 90},
           DI_torque2={"DI_gear": 4},
           GTW_carState={"BC_indicatorLStatus": 1, "DOOR_STATE_FL": 0},
           STW_ANGLHP_STAT={"StW_AnglHP_Spd": 12.0},
           RCM_status={"RCM_buckleDriverStatus": 1}),
      pack('tesla_powertrain', 4, DI_torque1={"DI_pedalPos": 40}),
      pack('tesla_powertrain', 6, DAS_control={"DAS_aebEvent": 1}),
      pack('tesla_raven_party', 2, DAS_steeringControl={"DAS_steeringControlType": 2}),
    )
    ret, ret_iq = cs.update(parsers)

    assert ret.vEgoRaw == pytest.approx(20.0, abs=0.01)
    assert ret.brakePressed
    assert ret.gasPressed
    assert ret.steeringAngleDeg == pytest.approx(-10.0, abs=0.1)
    assert ret.steeringTorque == pytest.approx(-0.5, abs=0.01)
    assert ret.steeringRateDeg == pytest.approx(-12.0, abs=0.5)
    assert ret.cruiseState.enabled
    assert ret.cruiseState.available
    assert ret.cruiseState.speed == pytest.approx(25.0, abs=0.01)
    assert ret.gearShifter == structs.CarState.GearShifter.drive
    assert ret.leftBlinker
    assert not ret.rightBlinker
    assert not ret.doorOpen
    assert not ret.seatbeltUnlatched
    assert ret.stockAeb
    assert ret.stockLkas
    assert cs.hands_on_level == 1
    assert cs.das_control["DAS_aebEvent"] == 1

  def test_model_x_hw3_reads_the_same_sources_as_model_s_hw3(self):
    cs, parsers = self.car_state(CAR.TESLA_MODEL_X_HW3)
    feed(
      parsers,
      pack('tesla_raven_party', 0, EPAS_sysStatus={"EPAS_internalSAS": 4.0, "EPAS_handsOnLevel": 0, "EPAS_eacStatus": 2}),
      pack('tesla_can', 1, ESP_B={"ESP_vehicleSpeed": 36.0}, DI_state={"DI_cruiseState": 2, "DI_speedUnits": 1, "DI_digitalSpeed": 40}),
      pack('tesla_powertrain', 4, DI_torque1={"DI_pedalPos": 0}),
    )
    ret, _ = cs.update(parsers)
    assert ret.vEgoRaw == pytest.approx(10.0, abs=0.01)
    assert ret.steeringAngleDeg == pytest.approx(-4.0, abs=0.1)
    assert ret.cruiseState.enabled

  def test_hw3_door_open(self):
    cs, parsers = self.car_state(CAR.TESLA_MODEL_S_HW3)
    for door in ("DOOR_STATE_FL", "DOOR_STATE_FR", "DOOR_STATE_RL", "DOOR_STATE_RR", "DOOR_STATE_FrontTrunk", "BOOT_STATE"):
      feed(parsers, pack('tesla_can', 1, GTW_carState={door: 1}))
      ret, _ = cs.update(parsers)
      assert ret.doorOpen, door

    feed(parsers, pack('tesla_can', 1, GTW_carState={"DOOR_STATE_FL": 0}))
    ret, _ = cs.update(parsers)
    assert not ret.doorOpen

  def test_hw3_cruise_states(self):
    cs, parsers = self.car_state(CAR.TESLA_MODEL_S_HW3)
    expectations = {1: (False, True, False), 2: (True, True, False), 3: (True, True, True), 5: (False, False, False)}
    for state, (enabled, available, standstill) in expectations.items():
      feed(parsers, pack('tesla_can', 1, DI_state={"DI_cruiseState": state, "DI_speedUnits": 0, "DI_digitalSpeed": 50}))
      ret, _ = cs.update(parsers)
      assert ret.cruiseState.enabled == enabled, state
      assert ret.cruiseState.available == available, state
      assert ret.standstill == standstill, state
      assert ret.accFaulted == (state == 5)
      assert ret.cruiseState.speed == pytest.approx(50 * 0.44704, abs=0.01)

  def test_hw3_steer_faults(self):
    cs, parsers = self.car_state(CAR.TESLA_MODEL_S_HW3)
    feed(parsers, pack('tesla_raven_party', 0, EPAS_sysStatus={"EPAS_eacStatus": 3, "EPAS_handsOnLevel": 0}))
    ret, _ = cs.update(parsers)
    assert ret.steerFaultPermanent and not ret.steerFaultTemporary

    feed(parsers, pack('tesla_raven_party', 0, EPAS_sysStatus={"EPAS_eacStatus": 0, "EPAS_eacErrorCode": 9}))
    ret, _ = cs.update(parsers)
    assert ret.steerFaultTemporary
    assert ret.steeringDisengage

    feed(parsers, pack('tesla_raven_party', 0, EPAS_sysStatus={"EPAS_eacStatus": 2, "EPAS_handsOnLevel": 3}))
    ret, _ = cs.update(parsers)
    assert ret.steeringDisengage

  def test_hw3_seatbelt_with_sdm1(self):
    fingerprint = gen_empty_fingerprint()
    fingerprint[1][0x201] = 5
    cs, parsers = self.car_state(CAR.TESLA_MODEL_S_HW3, fingerprint)
    feed(parsers, pack('tesla_can', 1, SDM1={"SDM_bcklDrivStatus": 0}))
    ret, _ = cs.update(parsers)
    assert ret.seatbeltUnlatched

    feed(parsers, pack('tesla_can', 1, SDM1={"SDM_bcklDrivStatus": 1}))
    ret, _ = cs.update(parsers)
    assert not ret.seatbeltUnlatched

  def test_hw3_seatbelt_without_sdm1(self):
    cs, parsers = self.car_state(CAR.TESLA_MODEL_S_HW3)
    feed(parsers, pack('tesla_can', 1, RCM_status={"RCM_buckleDriverStatus": 0}))
    ret, _ = cs.update(parsers)
    assert ret.seatbeltUnlatched

    feed(parsers, pack('tesla_can', 1, RCM_status={"RCM_buckleDriverStatus": 1}))
    ret, _ = cs.update(parsers)
    assert not ret.seatbeltUnlatched

  def test_hw1_reads_everything_from_party_bus(self):
    cs, parsers = self.car_state(CAR.TESLA_MODEL_S_HW1)
    feed(
      parsers,
      pack('tesla_can', 0,
           EPAS_sysStatus={"EPAS_internalSAS": -5.0, "EPAS_handsOnLevel": 0, "EPAS_eacStatus": 2},
           ESP_B={"ESP_vehicleSpeed": 36.0},
           BrakeMessage={"driverBrakeStatus": 1},
           DI_state={"DI_cruiseState": 2, "DI_speedUnits": 1, "DI_digitalSpeed": 40},
           DI_torque1={"DI_pedalPos": 0},
           DI_torque2={"DI_gear": 1},
           GTW_carState={"BC_indicatorRStatus": 1},
           SDM1={"SDM_bcklDrivStatus": 1},
           STW_ANGLHP_STAT={"StW_AnglHP_Spd": 0}),
      pack('tesla_can', 2,
           DAS_control={"DAS_aebEvent": 0},
           DAS_steeringControl={"DAS_steeringControlType": 0}),
    )
    ret, _ = cs.update(parsers)

    assert ret.vEgoRaw == pytest.approx(10.0, abs=0.01)
    assert not ret.brakePressed
    assert not ret.gasPressed
    assert ret.steeringAngleDeg == pytest.approx(5.0, abs=0.1)
    assert ret.gearShifter == structs.CarState.GearShifter.park
    assert ret.rightBlinker and not ret.leftBlinker
    assert not ret.stockAeb and not ret.stockLkas

  def test_model_x_hw1_blinkers_come_from_stalk(self):
    cs, parsers = self.car_state(CAR.TESLA_MODEL_X_HW1)
    feed(parsers, pack('tesla_can', 0, STW_ACTN_RQ={"TurnIndLvr_Stat": 2}))
    ret, _ = cs.update(parsers)
    assert ret.rightBlinker and not ret.leftBlinker

    feed(parsers, pack('tesla_can', 0, STW_ACTN_RQ={"TurnIndLvr_Stat": 1}))
    ret, _ = cs.update(parsers)
    assert ret.leftBlinker and not ret.rightBlinker

  def test_hw2_gas_comes_from_powertrain_bus(self):
    cs, parsers = self.car_state(CAR.TESLA_MODEL_S_HW2)
    feed(parsers, pack('tesla_powertrain', 4, DI_torque1={"DI_pedalPos": 20}))
    ret, _ = cs.update(parsers)
    assert ret.gasPressed

    feed(parsers, pack('tesla_powertrain', 0, DI_torque1={"DI_pedalPos": 0}))
    ret, _ = cs.update(parsers)
    assert ret.gasPressed


class FakeActuators:
  def __init__(self, angle=0.0, accel=0.0):
    self.steeringAngleDeg = angle
    self.accel = accel

  def as_builder(self):
    return self


class FakeCarControl:
  def __init__(self, lat_active=True, long_active=True, cancel=False, angle=0.0, accel=0.0):
    self.actuators = FakeActuators(angle, accel)
    self.latActive = lat_active
    self.longActive = long_active
    self.cruiseControl = type("Cruise", (), {"cancel": cancel})()


class FakeCarState:
  hands_on_level = 0
  das_control = {"DAS_controlCounter": 0}
  out = type("Out", (), {"vEgoRaw": 20.0, "steeringAngleDeg": 0.0, "vEgo": 20.0, "gasPressed": False})()


def make_controller(candidate):
  CP, CP_IQ = make_params(candidate)
  parsers = CarState.get_can_parsers(CP, CP_IQ)
  dbc_names = {bus: parser.dbc_name for bus, parser in parsers.items()}
  return CarController(dbc_names, CP, CP_IQ)


class TestLegacyCarController:
  @staticmethod
  def run_frames(controller, frames, cc):
    sent = {}
    for frame in range(frames):
      _, can_sends = controller.update(cc, structs.IQCarControl(), FakeCarState(), 0)
      sent[frame] = can_sends
    return sent

  def test_hw3_message_addresses_and_buses(self):
    controller = make_controller(CAR.TESLA_MODEL_S_HW3)
    sent = self.run_frames(controller, 20, FakeCarControl())

    steering = [m for f in sent.values() for m in f if m[0] == 0x488]
    eac = [m for f in sent.values() for m in f if m[0] == 0x27d]
    long = [m for f in sent.values() for m in f if m[0] == 0x2bf]

    assert len(steering) == 10
    assert len(eac) == 2
    assert len(long) == 5
    assert {m[2] for m in steering} == {0}
    assert {m[2] for m in eac} == {0}
    assert {m[2] for m in long} == {4}
    assert not [m for f in sent.values() for m in f if m[0] == 0x2b9]

  def test_hw1_sends_long_on_party_bus_without_eac_monitor(self):
    controller = make_controller(CAR.TESLA_MODEL_S_HW1)
    sent = self.run_frames(controller, 20, FakeCarControl())

    addrs = {m[0] for f in sent.values() for m in f}
    assert addrs == {0x488, 0x2b9}
    assert {m[2] for f in sent.values() for m in f} == {0}

  def test_hw2_message_addresses_and_buses(self):
    controller = make_controller(CAR.TESLA_MODEL_X_HW2)
    sent = self.run_frames(controller, 20, FakeCarControl())

    by_addr = {}
    for f in sent.values():
      for m in f:
        by_addr.setdefault(m[0], set()).add(m[2])
    assert by_addr == {0x488: {0}, 0x27d: {0}, 0x2bf: {4}}

  def test_steering_checksum_and_counter(self):
    controller = make_controller(CAR.TESLA_MODEL_S_HW3)
    sent = self.run_frames(controller, 8, FakeCarControl(angle=5.0))
    steering = [m for f in sent.values() for m in f if m[0] == 0x488]

    parser = CANParser('tesla_raven_party', [("DAS_steeringControl", 50)], 0)
    counters = []
    for i, (addr, dat, bus) in enumerate(steering):
      assert dat[3] == checksum(addr, dat[:3])
      parser.update([(NANOS + i * 20_000_000, [(addr, dat, bus)])])
      counters.append(int(parser.vl["DAS_steeringControl"]["DAS_steeringControlCounter"]))
    assert counters == [0, 1, 2, 3]

  def test_steering_is_idle_when_lateral_inactive(self):
    controller = make_controller(CAR.TESLA_MODEL_S_HW3)
    sent = self.run_frames(controller, 4, FakeCarControl(lat_active=False, angle=5.0))
    steering = [m for f in sent.values() for m in f if m[0] == 0x488]

    parser = CANParser('tesla_raven_party', [("DAS_steeringControl", 50)], 0)
    for i, (addr, dat, bus) in enumerate(steering):
      parser.update([(NANOS + i * 20_000_000, [(addr, dat, bus)])])
      assert parser.vl["DAS_steeringControl"]["DAS_steeringControlType"] == 0

  def test_steering_angle_is_negated_and_rate_limited(self):
    controller = make_controller(CAR.TESLA_MODEL_S_HW3)
    angles = []
    parser = CANParser('tesla_raven_party', [("DAS_steeringControl", 50)], 0)
    for frame in range(20):
      _, can_sends = controller.update(FakeCarControl(angle=30.0), structs.IQCarControl(), FakeCarState(), 0)
      for addr, dat, bus in can_sends:
        if addr == 0x488:
          parser.update([(NANOS + frame * 10_000_000, [(addr, dat, bus)])])
          angles.append(parser.vl["DAS_steeringControl"]["DAS_steeringAngleRequest"])

    assert angles[0] <= 0.0
    assert angles[-1] < angles[0]
    assert all(abs(b - a) <= 5.0 + 0.1 for a, b in zip(angles, angles[1:], strict=False))

  def test_longitudinal_checksum_and_state(self):
    controller = make_controller(CAR.TESLA_MODEL_S_HW3)
    _, can_sends = controller.update(FakeCarControl(accel=1.0), structs.IQCarControl(), FakeCarState(), 0)
    long = [m for m in can_sends if m[0] == 0x2bf]
    assert len(long) == 1

    addr, dat, bus = long[0]
    assert dat[7] == checksum(0x2b9, dat[:7])

    parser = CANParser('tesla_powertrain', [("DAS_control", 25)], 4)
    parser.update([(NANOS, [(addr, dat, bus)])])
    values = parser.vl["DAS_control"]
    assert values["DAS_accState"] == 4
    assert values["DAS_accelMax"] == pytest.approx(1.0, abs=0.05)
    assert values["DAS_accelMin"] == pytest.approx(1.0, abs=0.05)
    assert values["DAS_aebEvent"] == 0

  def test_cancel_sends_silent_cancel_state(self):
    controller = make_controller(CAR.TESLA_MODEL_S_HW3)
    _, can_sends = controller.update(FakeCarControl(cancel=True, long_active=False), structs.IQCarControl(), FakeCarState(), 0)
    addr, dat, bus = next(m for m in can_sends if m[0] == 0x2bf)

    parser = CANParser('tesla_powertrain', [("DAS_control", 25)], 4)
    parser.update([(NANOS, [(addr, dat, bus)])])
    assert parser.vl["DAS_control"]["DAS_accState"] == 13

  def test_accel_is_clipped_to_safety_limits(self):
    controller = make_controller(CAR.TESLA_MODEL_S_HW3)
    _, can_sends = controller.update(FakeCarControl(accel=10.0), structs.IQCarControl(), FakeCarState(), 0)
    addr, dat, bus = next(m for m in can_sends if m[0] == 0x2bf)

    parser = CANParser('tesla_powertrain', [("DAS_control", 25)], 4)
    parser.update([(NANOS, [(addr, dat, bus)])])
    assert parser.vl["DAS_control"]["DAS_accelMax"] <= 2.0 + 0.05


class TestTeslaCanLegacy:
  @staticmethod
  def make(candidate=CAR.TESLA_MODEL_S_HW3):
    canbus = get_legacy_canbus(candidate)
    packers = {canbus.party: CANPacker(DBC[candidate][Bus.party]), canbus.powertrain: CANPacker(DBC[candidate][Bus.pt])}
    return TeslaCANLegacy(packers, canbus)

  def test_jerk_limits_collapse_on_gas_and_ramp_back(self):
    can = self.make()
    can.create_longitudinal_command(4, 0.5, 0, 20.0, True, True)
    assert can.jerk_upper == 0.0
    assert can.jerk_lower == 0.0

    for i in range(1, 6):
      can.create_longitudinal_command(4, 0.5, i, 20.0, True, False)
    assert 0.0 < can.jerk_upper < 4.9
    assert -4.9 < can.jerk_lower < 0.0
    assert can.jerk_upper == pytest.approx(-can.jerk_lower)

    for i in range(1000):
      can.create_longitudinal_command(4, 0.5, i % 8, 20.0, True, False)
    assert can.jerk_upper == pytest.approx(4.9)
    assert can.jerk_lower == pytest.approx(-4.9)

  def test_set_speed_follows_accel_sign_when_active(self):
    can = self.make()
    parser = CANParser('tesla_powertrain', [("DAS_control", 25)], 4)

    for accel, expect_zero in ((-0.5, True), (0.5, False)):
      addr, dat, bus = can.create_longitudinal_command(4, accel, 0, 20.0, True, False)
      parser.update([(NANOS, [(addr, dat, bus)])])
      set_speed = parser.vl["DAS_control"]["DAS_setSpeed"]
      assert (set_speed == 0) == expect_zero

  def test_idle_set_speed_tracks_ego(self):
    can = self.make()
    parser = CANParser('tesla_powertrain', [("DAS_control", 25)], 4)
    addr, dat, bus = can.create_longitudinal_command(4, 0.0, 0, 20.0, False, False)
    parser.update([(NANOS, [(addr, dat, bus)])])
    assert parser.vl["DAS_control"]["DAS_setSpeed"] == pytest.approx(72.0, abs=0.5)


class TestLegacyRadar:
  def test_hw3_uses_continental_radar_on_external_panda_bus(self):
    CP, CP_IQ = make_params(CAR.TESLA_MODEL_S_HW3)
    ri = RadarInterface(CP, CP_IQ)
    assert not ri.bosch_radar
    assert ri.rcp.bus == 5
    assert ri.trigger_msg == 0x410 + 80 - 1

  @pytest.mark.parametrize("candidate", [CAR.TESLA_MODEL_S_HW1, CAR.TESLA_MODEL_X_HW1, CAR.TESLA_MODEL_X_HW2])
  def test_bosch_radar_parser(self, candidate):
    CP, CP_IQ = make_params(candidate)
    ri = RadarInterface(CP, CP_IQ)
    assert ri.bosch_radar
    assert ri.rcp.bus == 1
    assert ri.trigger_msg == BOSCH_TRIGGER_MSG
    assert ri.num_points == 32

  def test_s_hw2_radar_is_off(self):
    CP, CP_IQ = make_params(CAR.TESLA_MODEL_S_HW2)
    ri = RadarInterface(CP, CP_IQ)
    assert ri.radar_off_can

  def test_bosch_points_are_filtered_and_held_between_frames(self):
    CP, CP_IQ = make_params(CAR.TESLA_MODEL_S_HW1)
    ri = RadarInterface(CP, CP_IQ)
    dbc = DBC[CAR.TESLA_MODEL_S_HW1][Bus.radar]
    packer = CANPacker(dbc)

    def frame(good_dist, exist, nanos):
      msgs = [packer.make_can_msg("TeslaRadarSguInfo", 1, {"RADC_HWFail": 0})]
      for i in range(32):
        a = {"Index": i % 2, "Tracked": 0}
        b = {"Index2": i % 2}
        if i == 0:
          a = {"LongDist": good_dist, "LatDist": 1.0, "LongSpeed": -2.0, "ProbExist": exist, "Tracked": 1,
               "Meas": 1, "Index": 1, "LongAccel": 0.5}
          b = {"LatSpeed": 0.25, "Index2": 1}
        msgs.append(packer.make_can_msg(f"RadarPoint{i}_A", 1, a))
        msgs.append(packer.make_can_msg(f"RadarPoint{i}_B", 1, b))
      return [(nanos, msgs)]

    ri.update(frame(40.0, 90.0, NANOS))
    ri.update(frame(40.0, 90.0, NANOS + 125_000_000))
    rr = ri.update(frame(40.0, 90.0, NANOS + 250_000_000))
    assert rr is not None
    assert len(rr.points) == 1
    point = rr.points[0]
    assert point.dRel == pytest.approx(40.0, abs=0.1)
    assert point.yRel == pytest.approx(1.0, abs=0.2)
    assert point.vRel == pytest.approx(-2.0, abs=0.1)

    rr = ri.update(frame(0.0, 90.0, NANOS + 375_000_000))
    assert rr is not None and len(rr.points) == 0

    rr = ri.update(frame(40.0, 10.0, NANOS + 500_000_000))
    assert rr is not None and len(rr.points) == 0


@pytest.mark.parametrize("candidate", LEGACY_CARS)
def test_modern_tesla_options_are_not_applied(candidate):
  from iqdbc.lvbs.car.interfaces import _apply_tesla_options
  CP, CP_IQ = make_params(candidate)
  _apply_tesla_options(CP, CP_IQ, {"IQTeslaTorqueBlend": "1", "IQTeslaFsdVisualization": "1"})
  assert CP_IQ.flags == 0
  assert CP_IQ.iqSafetyFlags == 0
