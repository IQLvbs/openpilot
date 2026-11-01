import sys
import types
from collections import defaultdict

import pytest

from iqdbc.car import Bus, gen_empty_fingerprint, structs
from iqdbc.car.structs import CarParams
from iqdbc.car.fw_versions import match_fw_to_car
from iqdbc.car.tesla.fingerprints import FW_VERSIONS
from iqdbc.car.tesla.interface import CarInterface
from iqdbc.car.tesla.teslacan import TeslaCAN
from iqdbc.car.tesla.radar_interface import RADAR_START_ADDR
from iqdbc.car.tesla.carcontroller import CarController
from iqdbc.car.tesla.carstate import CarState, stock_autosteer_invalid
from iqdbc.car.tesla.values import (CANBUS, CAR, FW_PATTERN, LEGACY_CARS, LEGACY_DAS_STEERING_FW, TeslaFlags, TeslaSafetyFlags,
                                    get_platform_codes, is_ascii_version_fw, is_hw4_gen2_fw, is_legacy_das_steering)
from iqdbc.lvbs.car.interfaces import _apply_tesla_options
from iqdbc.lvbs.car.tesla.values import TeslaFlagsIQ, TeslaSafetyFlagsIQ
from iqdbc.can import CANPacker, CANParser
from iqdbc.car.common.conversions import Conversions as CV

Ecu = CarParams.Ecu
EPS_ADDR = 0x730

HW4_GEN2_BINARY_FW = b'\x01\x0113\x04\x00\x01\x00\x00\x01\x01\x00\x00\x00\x00\x00\x00\xff\xff'
HW4_GEN2_ASCII_FW = b'TeMYG4_Main_0.0.0 (78),Y4OC.E80.003.07.0'


def hw4_gen2_fingerprint(vehicle_bus: bool = False):
  fingerprint = gen_empty_fingerprint()
  fingerprint[CANBUS.autopilot_party][0x399] = 8
  if vehicle_bus:
    fingerprint[CANBUS.vehicle][0x3F5] = 8
  return fingerprint


def fw_match(fw: bytes):
  car_fw = [CarParams.CarFw(ecu=Ecu.eps, fwVersion=fw, address=EPS_ADDR, subAddress=0, brand='tesla')]
  exact, matches = match_fw_to_car(car_fw, '0' * 17, log=False)
  return exact, matches


class TestTeslaFwPattern:
  def test_all_known_fw_parses(self):
    for car, ecus in FW_VERSIONS.items():
      if car in LEGACY_CARS:
        continue
      for fws in ecus.values():
        for fw in fws:
          if not is_ascii_version_fw(fw):
            continue
          assert FW_PATTERN.match(fw) is not None, f'{car}: unparsed FW version: {fw}'

  def test_only_hw4_gen2_ships_a_binary_fw(self):
    # a binary FW can only ever match exactly, so it must not creep into other platforms
    for car, ecus in FW_VERSIONS.items():
      if car in LEGACY_CARS:
        continue
      for fws in ecus.values():
        for fw in fws:
          if is_ascii_version_fw(fw):
            continue
          assert car == CAR.TESLA_MODEL_Y, f'{car}: unexpected binary FW version: {fw}'
          assert is_hw4_gen2_fw(fw)

  def test_hw4_gen2_ascii_fw_parses(self):
    match = FW_PATTERN.match(HW4_GEN2_ASCII_FW)
    assert match is not None
    assert match.group('model') == b'Y4'
    assert match.group('gen') == b'E80'
    assert match.group('series') == b'003'
    assert match.group('version') == b'07.0'
    assert is_hw4_gen2_fw(HW4_GEN2_ASCII_FW)

  def test_hw4_gen2_binary_fw_is_not_parsed(self):
    assert FW_PATTERN.match(HW4_GEN2_BINARY_FW) is None
    assert get_platform_codes([HW4_GEN2_BINARY_FW]) == set()
    assert is_hw4_gen2_fw(HW4_GEN2_BINARY_FW)

  @pytest.mark.parametrize("fw", [HW4_GEN2_BINARY_FW, HW4_GEN2_ASCII_FW])
  def test_hw4_gen2_fw_matches_model_y(self, fw):
    exact, matches = fw_match(fw)
    assert exact
    assert matches == {CAR.TESLA_MODEL_Y}

  def test_unknown_hw4_gen2_fw_fuzzy_matches_model_y(self):
    # an EPS bump on a 2026+ Model Y still carries the Y4 model code
    exact, matches = fw_match(b'TeMYG4_Main_0.0.0 (91),Y4OC.E80.003.12.0')
    assert not exact
    assert matches == {CAR.TESLA_MODEL_Y}

  def test_model_code_identifies_one_platform(self):
    # a new platform reusing an existing model code would silently misfingerprint
    platforms = defaultdict(set)
    for car, ecus in FW_VERSIONS.items():
      for fws in ecus.values():
        for model, _, _ in get_platform_codes(fws):
          platforms[model].add(car)

    for model, cars in platforms.items():
      assert len(cars) == 1, f'model code {model} maps to multiple platforms: {cars}'

  def test_exact_match_still_wins(self):
    for car, ecus in FW_VERSIONS.items():
      for fws in ecus.values():
        for fw in fws:
          exact, matches = fw_match(fw)
          assert exact, f'{fw} fell back to fuzzy matching'
          assert matches == {car}, f'{fw} matched {matches}, expected {car}'

  @pytest.mark.parametrize("fw, expected", [
    # a firmware bump within a known series, the case that used to fingerprint as MOCK
    (b'TeMYG4_Main_0.0.0 (99),Y4003.14.0', CAR.TESLA_MODEL_Y),
    (b'TeMYG4_Main_0.0.0 (99),E4H015.09.0', CAR.TESLA_MODEL_3),
    (b'TeM3_SP_XP002p2_0.0.0 (40),XPR003.12.0', CAR.TESLA_MODEL_X),
    # Tesla bumps the series within a platform (E4014 -> E4015, Y4002 -> Y4003)
    (b'TeMYG4_Main_0.0.0 (12),Y4004.01.0', CAR.TESLA_MODEL_Y),
    # an unknown model code is a car we don't support
    (b'TeCT_Main_0.0.0 (1),CT001.01.0', None),
    (b'garbage', None),
  ])
  def test_unknown_fw_fuzzy_match(self, fw, expected):
    exact, matches = fw_match(fw)
    if expected is None:
      assert matches == set(), f'{fw} unexpectedly matched {matches}'
    else:
      assert not exact
      assert matches == {expected}


class TestTeslaLegacyDasSteering:
  def test_reproduces_known_table(self):
    for car, ecus in FW_VERSIONS.items():
      for fws in ecus.values():
        for fw in fws:
          expected = fw in LEGACY_DAS_STEERING_FW.get(car, [])
          assert is_legacy_das_steering(car, fw) == expected, f'{car}: wrong legacy DAS verdict for {fw}'

  @pytest.mark.parametrize("car, fw, expected", [
    # below a family's known modern cutoff (Y4/003 splits at 003.04.0)
    (CAR.TESLA_MODEL_Y, b'TeMYG4_Legacy3Y_0.0.0 (5),Y4003.03.9', True),
    (CAR.TESLA_MODEL_Y, b'TeMYG4_Main_0.0.0 (99),Y4003.14.0', False),
    # E4/015 splits at 015.04.5
    (CAR.TESLA_MODEL_3, b'TeMYG4_Main_0.0.0 (68),E4H015.03.9', True),
    (CAR.TESLA_MODEL_3, b'TeMYG4_Main_0.0.0 (99),E4H015.09.0', False),
    # families with no known modern FW: interpolate legacy, extrapolate modern
    (CAR.TESLA_MODEL_3, b'TeM3_E014p10_0.0.0 (16),E014.18.00', True),
    (CAR.TESLA_MODEL_3, b'TeM3_E014p10_0.0.0 (30),E014.22.0', False),
    # numeric, not lexical, version compare (XPR003.10.0 > XPR003.6.0)
    (CAR.TESLA_MODEL_X, b'TeM3_SP_XP002p2_0.0.0 (30),XPR003.9.0', True),
    # an unknown series has no history to compare against
    (CAR.TESLA_MODEL_Y, b'TeMYG4_Main_0.0.0 (12),Y4004.01.0', False),
    (CAR.TESLA_MODEL_Y, b'garbage', False),
  ])
  def test_unknown_fw(self, car, fw, expected):
    assert is_legacy_das_steering(car, fw) == expected

  @pytest.mark.parametrize("fw", [
    HW4_GEN2_BINARY_FW,
    HW4_GEN2_ASCII_FW,
    # a HW4 gen2 version number that would fall below the Y4/003 legacy cutoff if it were compared
    b'TeMYG4_Main_0.0.0 (70),Y4OC.E80.003.01.0',
  ])
  def test_hw4_gen2_is_never_legacy(self, fw):
    assert not is_legacy_das_steering(CAR.TESLA_MODEL_Y, fw)

  def test_hw4_gen2_fw_does_not_move_the_cutoff(self):
    # the shared Y4/003 family splits at 003.04.0 and must stay there
    assert is_legacy_das_steering(CAR.TESLA_MODEL_Y, b'TeMYG4_Legacy3Y_0.0.0 (5),Y4003.03.9')
    assert not is_legacy_das_steering(CAR.TESLA_MODEL_Y, b'TeMYG4_Legacy3Y_0.0.0 (6),Y4003.04.0')

  def test_flag_set_from_fuzzy_match(self):
    for fw, legacy in ((b'TeMYG4_Legacy3Y_0.0.0 (5),Y4003.03.9', True),
                       (b'TeMYG4_Main_0.0.0 (99),Y4003.14.0', False)):
      car_fw = [CarParams.CarFw(ecu=Ecu.eps, fwVersion=fw, address=EPS_ADDR, subAddress=0, brand='tesla')]
      CP = CarInterface.get_params(CAR.TESLA_MODEL_Y, gen_empty_fingerprint(), car_fw, False, False, False)
      assert bool(CP.flags & TeslaFlags.LEGACY_DAS_STEERING) == legacy
      assert bool(CP.safetyConfigs[0].safetyParam & TeslaSafetyFlags.LEGACY_DAS_STEERING) == legacy


def test_fsd_visualization_params():
  CP = structs.CarParams(brand="tesla")
  CP_IQ = structs.IQCarParams()
  _apply_tesla_options(CP, CP_IQ, {"IQTeslaFsdVisualization": "1"})
  assert CP_IQ.flags & TeslaFlagsIQ.FSD_VISUALIZATION
  assert CP_IQ.iqSafetyFlags & TeslaSafetyFlagsIQ.FSD_VISUALIZATION


@pytest.mark.parametrize("enabled, expected", [(False, True), (True, False)])
def test_fsd_visualization_invalid_lkas(enabled, expected):
  CP = structs.CarParams()
  CP_IQ = structs.IQCarParams(iqSafetyFlags=TeslaSafetyFlagsIQ.FSD_VISUALIZATION if enabled else 0)
  assert stock_autosteer_invalid(CP, CP_IQ, 3) == expected


class TestStockAutosteerInvalid:
  @pytest.mark.parametrize("autopilot_state, expected", [(0, False), (1, False), (2, False), (3, True), (6, True)])
  def test_reads_autopilot_state_off_gen2(self, autopilot_state, expected):
    CP = structs.CarParams()
    CP_IQ = structs.IQCarParams()
    # the setting is ignored while DAS_status is available
    assert stock_autosteer_invalid(CP, CP_IQ, autopilot_state, 1) == expected

  @pytest.mark.parametrize("autosteer_enabled, expected", [(0, False), (1, True)])
  def test_falls_back_to_the_setting_on_gen2(self, autosteer_enabled, expected):
    CP = structs.CarParams(flags=int(TeslaFlags.HW4_GEN2))
    CP_IQ = structs.IQCarParams()
    # DAS_status is never sent, so a stale autopilot state must not be trusted
    assert stock_autosteer_invalid(CP, CP_IQ, 6, autosteer_enabled) == expected

  def test_fsd_visualization_still_wins_on_gen2(self):
    CP = structs.CarParams(flags=int(TeslaFlags.HW4_GEN2))
    CP_IQ = structs.IQCarParams(iqSafetyFlags=TeslaSafetyFlagsIQ.FSD_VISUALIZATION)
    assert not stock_autosteer_invalid(CP, CP_IQ, 6, 1)

  def test_missing_das_settings_still_wins(self):
    CP = structs.CarParams(flags=int(TeslaFlags.MISSING_DAS_SETTINGS))
    CP_IQ = structs.IQCarParams()
    assert not stock_autosteer_invalid(CP, CP_IQ, 6, 1)


class TestTeslaFingerprint:
  @pytest.mark.parametrize("candidate", [CAR.TESLA_MODEL_3, CAR.TESLA_MODEL_Y, CAR.TESLA_MODEL_X])
  @pytest.mark.parametrize("bus", [CANBUS.party, CANBUS.vehicle, CANBUS.autopilot_party])
  @pytest.mark.parametrize("marker", [0x3DF, 0x3B6])
  def test_vehicle_bus_addon_requires_marker_on_vehicle_bus(self, candidate, bus, marker):
    fingerprint = gen_empty_fingerprint()
    fingerprint[bus][marker] = 8
    CP = CarInterface.get_params(candidate, fingerprint, [], False, False, False)
    CP_IQ = CarInterface.get_params_iq(CP, candidate, fingerprint, [], False, False, False)
    addon_present = bus == CANBUS.vehicle

    assert bool(CP_IQ.flags & TeslaFlagsIQ.HAS_VEHICLE_BUS) == addon_present
    assert bool(CP_IQ.iqSafetyFlags & TeslaSafetyFlagsIQ.HAS_VEHICLE_BUS) == addon_present
    assert (Bus.adas in CarState.get_can_parsers(CP, CP_IQ)) == addon_present

  def test_standard_harness_with_mirrored_odometer_has_no_vehicle_bus_parser(self):
    fingerprint = gen_empty_fingerprint()
    for bus in (CANBUS.party, CANBUS.autopilot_party):
      fingerprint[bus][0x3B6] = 6
    CP = CarInterface.get_params(CAR.TESLA_MODEL_3, fingerprint, [], False, False, False)
    CP_IQ = CarInterface.get_params_iq(CP, CAR.TESLA_MODEL_3, fingerprint, [], False, False, False)

    assert not CP_IQ.flags & TeslaFlagsIQ.HAS_VEHICLE_BUS
    assert not CP_IQ.iqSafetyFlags & TeslaSafetyFlagsIQ.HAS_VEHICLE_BUS
    assert Bus.adas not in CarState.get_can_parsers(CP, CP_IQ)

  def test_radar_detection(self):
    # Test radar availability detection for cars with radar DBC defined
    for radar in (True, False):
      fingerprint = gen_empty_fingerprint()
      if radar:
        fingerprint[1][RADAR_START_ADDR] = 8
      CP = CarInterface.get_params(CAR.TESLA_MODEL_3, fingerprint, [], False, False, False)
      assert CP.radarUnavailable != radar

  def test_no_radar_car(self):
    # Model X doesn't have radar DBC defined, should always be unavailable
    for radar in (True, False):
      fingerprint = gen_empty_fingerprint()
      if radar:
        fingerprint[1][RADAR_START_ADDR] = 8
      CP = CarInterface.get_params(CAR.TESLA_MODEL_X, fingerprint, [], False, False, False)
      assert CP.radarUnavailable  # Always unavailable since no radar DBC


class TestTeslaHW4Gen2Fingerprint:
  def test_not_detected_without_das_status_gen2(self):
    CP = CarInterface.get_params(CAR.TESLA_MODEL_Y, gen_empty_fingerprint(), [], False, False, False)
    assert not CP.flags & TeslaFlags.HW4_GEN2
    assert not CP.flags & TeslaFlags.HW4_GEN2_VEHICLE_BUS
    assert not CP.safetyConfigs[0].safetyParam & TeslaSafetyFlags.HW4_GEN2

  def test_detected_from_das_status_gen2(self):
    CP = CarInterface.get_params(CAR.TESLA_MODEL_Y, hw4_gen2_fingerprint(), [], False, False, False)
    assert CP.flags & TeslaFlags.HW4_GEN2
    assert CP.safetyConfigs[0].safetyParam & TeslaSafetyFlags.HW4_GEN2
    # DAS_settings is on the party bus here, so it's not missing
    assert not CP.flags & TeslaFlags.MISSING_DAS_SETTINGS
    # blinkers and the seatbelt buckle need the VEHICLE bus tapped
    assert not CP.flags & TeslaFlags.HW4_GEN2_VEHICLE_BUS

  def test_vehicle_bus_detected_from_vcfront_lighting(self):
    CP = CarInterface.get_params(CAR.TESLA_MODEL_Y, hw4_gen2_fingerprint(vehicle_bus=True), [], False, False, False)
    assert CP.flags & TeslaFlags.HW4_GEN2
    assert CP.flags & TeslaFlags.HW4_GEN2_VEHICLE_BUS

  def test_radar_stays_unavailable_on_gen2(self):
    # the tapped VEHICLE bus carries an unrelated RADAR_START_ADDR
    fingerprint = hw4_gen2_fingerprint(vehicle_bus=True)
    fingerprint[CANBUS.vehicle][RADAR_START_ADDR] = 8
    CP = CarInterface.get_params(CAR.TESLA_MODEL_Y, fingerprint, [], False, False, False)
    assert CP.radarUnavailable

  def test_missing_das_settings_unaffected(self):
    # HW2.5 and Model X are still detected by the absence of DAS_settings on the autopilot party bus
    CP = CarInterface.get_params(CAR.TESLA_MODEL_Y, gen_empty_fingerprint(), [], False, False, False)
    assert CP.flags & TeslaFlags.MISSING_DAS_SETTINGS

    fingerprint = gen_empty_fingerprint()
    fingerprint[CANBUS.autopilot_party][0x293] = 8
    CP = CarInterface.get_params(CAR.TESLA_MODEL_Y, fingerprint, [], False, False, False)
    assert not CP.flags & TeslaFlags.MISSING_DAS_SETTINGS

  def test_adas_parser_only_with_the_vehicle_bus(self):
    CP = CarInterface.get_params(CAR.TESLA_MODEL_Y, hw4_gen2_fingerprint(), [], False, False, False)
    assert Bus.adas not in CarState.get_can_parsers(CP, structs.IQCarParams())

    CP = CarInterface.get_params(CAR.TESLA_MODEL_Y, hw4_gen2_fingerprint(vehicle_bus=True), [], False, False, False)
    assert Bus.adas in CarState.get_can_parsers(CP, structs.IQCarParams())


class TestTeslaCan:
  class DummyPacker:
    def make_can_msg(self, name, bus, values):
      return name, bus, values

  def test_vehicle_bus_odometer_decodes_kilometers(self):
    packer = CANPacker("tesla_model3_vehicle")
    parser = CANParser("tesla_model3_vehicle", [("ID3B6UI_odometer", 1)], 1)

    message = packer.make_can_msg("ID3B6UI_odometer", 1, {
      "UI_odometer": 29150.377,
      "UI_odometerCounter": 1,
      "UI_odometerChecksum": 0,
    })
    parser.update([1_000_000_000, [message]])

    assert parser.vl["ID3B6UI_odometer"]["UI_odometer"] == 29150.377

  def test_longitudinal_command_does_not_reference_missing_jerk_attr(self):
    CP = CarInterface.get_non_essential_params(CAR.TESLA_MODEL_3)
    tesla_can = TeslaCAN(CP, self.DummyPacker())

    name, bus, values = tesla_can.create_longitudinal_command(4, 1.0, 0, 20.0, True, False)

    assert name == "DAS_control"
    assert bus == 0
    assert values["DAS_jerkMax"] <= 4.9
    assert values["DAS_jerkMax"] >= 0.0

  def test_longitudinal_jerk_accumulates_and_resets(self):
    CP = CarInterface.get_non_essential_params(CAR.TESLA_MODEL_3)
    tesla_can = TeslaCAN(CP, self.DummyPacker())

    first = tesla_can.create_longitudinal_command(4, 1.0, 0, 20.0, True, False)[2]
    second = tesla_can.create_longitudinal_command(4, 1.0, 1, 20.0, True, False)[2]
    inactive = tesla_can.create_longitudinal_command(4, 0.0, 2, 20.0, False, False)[2]

    assert second["DAS_jerkMax"] > first["DAS_jerkMax"] > 0.0
    assert inactive["DAS_jerkMax"] == 0.0

  def test_longitudinal_comfort_mode_hysteresis(self):
    CP = CarInterface.get_non_essential_params(CAR.TESLA_MODEL_3)
    tesla_can = TeslaCAN(CP, self.DummyPacker())

    coast = tesla_can.create_longitudinal_command(4, 0.0, 0, 20.0, True, False, set_speed_kph=64.0, comfort_mode=True)[2]
    drive = tesla_can.create_longitudinal_command(4, 0.08, 1, 20.0, True, False, set_speed_kph=64.0, comfort_mode=True)[2]
    drive_held = tesla_can.create_longitudinal_command(4, 0.04, 2, 20.0, True, False, set_speed_kph=64.0, comfort_mode=True)[2]
    coast_again = tesla_can.create_longitudinal_command(4, 0.03, 3, 20.0, True, False, set_speed_kph=64.0, comfort_mode=True)[2]
    regen = tesla_can.create_longitudinal_command(4, -0.08, 4, 20.0, True, False, set_speed_kph=64.0, comfort_mode=True)[2]
    regen_held = tesla_can.create_longitudinal_command(4, -0.04, 5, 20.0, True, False, set_speed_kph=64.0, comfort_mode=True)[2]
    final_coast = tesla_can.create_longitudinal_command(4, -0.03, 6, 20.0, True, False, set_speed_kph=64.0, comfort_mode=True)[2]

    assert coast["DAS_setSpeed"] == 72.0
    assert coast["DAS_jerkMin"] == -0.7
    assert coast["DAS_jerkMax"] == 0.7
    assert drive["DAS_setSpeed"] == 64.0
    assert drive_held["DAS_setSpeed"] == 64.0
    assert coast_again["DAS_setSpeed"] == 72.0
    assert regen["DAS_setSpeed"] == 0
    assert regen_held["DAS_setSpeed"] == 0
    assert final_coast["DAS_setSpeed"] == 72.0

  def test_longitudinal_comfort_mode_crosses_directly_between_drive_and_regen(self):
    CP = CarInterface.get_non_essential_params(CAR.TESLA_MODEL_3)
    tesla_can = TeslaCAN(CP, self.DummyPacker())

    tesla_can.create_longitudinal_command(4, 0.08, 0, 20.0, True, False, comfort_mode=True)
    regen = tesla_can.create_longitudinal_command(4, -0.08, 1, 20.0, True, False, comfort_mode=True)[2]
    drive = tesla_can.create_longitudinal_command(4, 0.08, 2, 20.0, True, False, set_speed_kph=64.0, comfort_mode=True)[2]

    assert regen["DAS_setSpeed"] == 0
    assert drive["DAS_setSpeed"] == 64.0

  def test_longitudinal_comfort_mode_stopping_bypasses_coast(self):
    CP = CarInterface.get_non_essential_params(CAR.TESLA_MODEL_3)
    tesla_can = TeslaCAN(CP, self.DummyPacker())

    values = tesla_can.create_longitudinal_command(4, -0.01, 0, 20.0, True, False, comfort_mode=True, stopping=True)[2]

    assert values["DAS_setSpeed"] == 0
    assert values["DAS_jerkMin"] == -4.9

  def test_route_zero_crossing_commands_stay_in_coast(self):
    CP = CarInterface.get_non_essential_params(CAR.TESLA_MODEL_3)
    tesla_can = TeslaCAN(CP, self.DummyPacker())
    commands = [0.001925, -0.007605, -0.003657, 0.001745, 0.014604, -0.000265, -0.008631, 0.003343,
                0.000620, -0.001185, -0.000301, 0.003632, 0.006865, -0.003242, -0.000854, 0.016680]

    set_speeds = [
      tesla_can.create_longitudinal_command(4, accel, counter % 8, 11.18, True, False, comfort_mode=True)[2]["DAS_setSpeed"]
      for counter, accel in enumerate(commands)
    ]

    assert set(set_speeds) == {11.18 * 3.6}

  def test_longitudinal_command_uses_explicit_set_speed(self):
    CP = CarInterface.get_non_essential_params(CAR.TESLA_MODEL_3)
    tesla_can = TeslaCAN(CP, self.DummyPacker())

    name, bus, values = tesla_can.create_longitudinal_command(4, 1.0, 0, 20.0, True, False, set_speed_kph=64.0)

    assert name == "DAS_control"
    assert bus == 0
    assert values["DAS_setSpeed"] == 64.0

  def test_longitudinal_command_preserves_decel_when_explicit_set_speed_present(self):
    CP = CarInterface.get_non_essential_params(CAR.TESLA_MODEL_3)
    tesla_can = TeslaCAN(CP, self.DummyPacker())

    name, bus, values = tesla_can.create_longitudinal_command(4, -0.5, 0, 20.0, True, False, set_speed_kph=64.0)

    assert name == "DAS_control"
    assert bus == 0
    assert values["DAS_setSpeed"] == 0
    assert values["DAS_accelMin"] < 0


class TestTeslaCarControllerIQParams:
  def test_iq_params_override_set_speed(self):
    CP = CarInterface.get_non_essential_params(CAR.TESLA_MODEL_3)
    CP.openpilotLongitudinalControl = True
    controller = CarController(CAR.TESLA_MODEL_3.config.dbc_dict, CP, structs.IQCarParams())

    class DummyCruise:
      cancel = False

    class DummyActuators:
      steeringAngleDeg = 0.0
      accel = 1.0
      longControlState = structs.CarControl.Actuators.LongControlState.pid

      def as_builder(self):
        return self

    class DummyCarControl:
      actuators = DummyActuators()
      latActive = False
      longActive = True
      cruiseControl = DummyCruise()

    class DummyCarState:
      hands_on_level = 0
      out = type("Out", (), {"vEgoRaw": 20.0, "steeringAngleDeg": 0.0, "steeringRateDeg": 0.0, "steeringTorque": 0.0, "vEgo": 20.0})()
      das_accCancel = False
      cruise_override = False
      das_control = {"DAS_controlCounter": 0}

    cc_iq = structs.IQCarControl(params=[
      structs.IQCarControl.Param(
        key="enhancedStockLongitudinalControl.setSpeedKph",
        type="float",
        value=b"64.0",
      )
    ])

    captured = {}

    def fake_longitudinal_command(state, accel, cntr, v_ego, active, cruise_override, set_speed_kph=None, **kwargs):
      captured["set_speed_kph"] = set_speed_kph
      return ("DAS_control", 0, {"DAS_setSpeed": set_speed_kph})

    controller.tesla_can.create_longitudinal_command = fake_longitudinal_command

    new_actuators, _ = controller.update(DummyCarControl(), cc_iq, DummyCarState(), 0)
    assert captured["set_speed_kph"] == 64.0
    assert new_actuators.accel == 1.0


class TestTeslaOdometerStoreMissing:
  @staticmethod
  def _car_state(fingerprint):
    CP = CarInterface.get_params(CAR.TESLA_MODEL_Y, fingerprint, [], False, False, False)
    CP_IQ = CarInterface.get_params_iq(CP, CAR.TESLA_MODEL_Y, fingerprint, [], False, False, False)
    car_state = CarState(CP, CP_IQ)
    parsers = CarState.get_can_parsers(CP, CP_IQ)
    car_state.update(parsers)
    return car_state, parsers

  def test_update_passes_through_odometer_when_store_is_unavailable(self):
    fingerprint = hw4_gen2_fingerprint(vehicle_bus=True)
    fingerprint[CANBUS.vehicle][0x3B6] = 8
    car_state, parsers = self._car_state(fingerprint)
    assert bool(car_state.CP_IQ.flags & TeslaFlagsIQ.HAS_VEHICLE_BUS)
    car_state._odometer_store = None

    packer = CANPacker("tesla_model3_vehicle")
    odo_msg = packer.make_can_msg("ID3B6UI_odometer", 1, {
      "UI_odometer": 29150.377,
      "UI_odometerCounter": 1,
      "UI_odometerChecksum": 0,
    })
    for parser in parsers.values():
      parser.update([1_000_000_000, [odo_msg]])
    ret, _ = car_state.update(parsers)

    assert ret.odometer == pytest.approx(29150.377, abs=0.01)


class TestTeslaStalklessCancel:
  @staticmethod
  def _send_acc_state(car_state, parsers, acc_state, t):
    packer = CANPacker("tesla_model3_party")
    msg = packer.make_can_msg("DAS_control", CANBUS.autopilot_party, {"DAS_accState": acc_state})
    parsers[Bus.ap_party].update([t, [msg]])
    ret, _ = car_state.update(parsers)
    return ret

  def test_acc_cancel_state_raises_cancel_button(self):
    fingerprint = gen_empty_fingerprint()
    CP = CarInterface.get_params(CAR.TESLA_MODEL_Y, fingerprint, [], False, False, False)
    CP_IQ = CarInterface.get_params_iq(CP, CAR.TESLA_MODEL_Y, fingerprint, [], False, False, False)
    car_state = CarState(CP, CP_IQ)
    parsers = CarState.get_can_parsers(CP, CP_IQ)

    ret = self._send_acc_state(car_state, parsers, 4, 1_000_000_000)
    assert not ret.buttonEvents
    ret = self._send_acc_state(car_state, parsers, 3, 1_500_000_000)
    assert not ret.buttonEvents

    ret = self._send_acc_state(car_state, parsers, 13, 2_000_000_000)
    assert [(b.type, b.pressed) for b in ret.buttonEvents] == [(structs.CarState.ButtonEvent.Type.cancel, True)]

    ret = self._send_acc_state(car_state, parsers, 4, 3_000_000_000)
    assert [(b.type, b.pressed) for b in ret.buttonEvents] == [(structs.CarState.ButtonEvent.Type.cancel, False)]


@pytest.fixture
def stub_odometer_store():
  # the odometer store lives in a signed runtime that isn't importable off-device
  name = "iqpilot.system.proprietary_runtime._verified_import"
  previous = sys.modules.get(name)
  module = types.ModuleType(name)
  alc = types.SimpleNamespace(create_vehicle_odometer_store=lambda CP, params: types.SimpleNamespace(record=lambda km: km))
  module.import_verified_module = lambda *_args, **_kwargs: alc
  sys.modules[name] = module
  yield
  if previous is None:
    del sys.modules[name]
  else:
    sys.modules[name] = previous


@pytest.mark.usefixtures("stub_odometer_store")
class TestTeslaCarStateSignalRouting:
  @staticmethod
  def _car_state(fingerprint):
    CP = CarInterface.get_params(CAR.TESLA_MODEL_Y, fingerprint, [], False, False, False)
    CP_IQ = structs.IQCarParams()
    car_state = CarState(CP, CP_IQ)
    parsers = CarState.get_can_parsers(CP, CP_IQ)
    # a CANParser only starts decoding a message once carstate has read it once
    car_state.update(parsers)
    return car_state, parsers

  @staticmethod
  def _update(car_state, parsers, messages):
    for parser in parsers.values():
      parser.update([1_000_000_000, messages])
    return car_state.update(parsers)

  @staticmethod
  def _party(**messages):
    packer = CANPacker("tesla_model3_party")
    return [packer.make_can_msg(name, bus, values) for name, (bus, values) in messages.items()]

  @staticmethod
  def _vehicle(**messages):
    packer = CANPacker("tesla_model3_vehicle")
    return [packer.make_can_msg(name, 1, values) for name, values in messages.items()]

  def test_gen2_reads_doors_blindspot_and_autosteer(self):
    car_state, parsers = self._car_state(hw4_gen2_fingerprint())
    ret, _ = self._update(car_state, parsers, self._party(
      VehicleStatus=(CANBUS.party, {"allDoorsClosed": 0}),
      DAS_statusGen2=(CANBUS.autopilot_party, {"DAS_blindSpotRearLeft": 1, "DAS_blindSpotRearRight": 0}),
      DAS_settings=(CANBUS.party, {"DAS_autosteerEnabled": 1}),
    ))
    assert ret.doorOpen
    assert ret.leftBlindspot
    assert not ret.rightBlindspot
    assert ret.invalidLkasSetting

    ret, _ = self._update(car_state, parsers, self._party(
      VehicleStatus=(CANBUS.party, {"allDoorsClosed": 1}),
      DAS_statusGen2=(CANBUS.autopilot_party, {"DAS_blindSpotRearLeft": 0, "DAS_blindSpotRearRight": 2}),
      DAS_settings=(CANBUS.party, {"DAS_autosteerEnabled": 0}),
    ))
    assert not ret.doorOpen
    assert not ret.leftBlindspot
    assert ret.rightBlindspot
    assert not ret.invalidLkasSetting

  def test_gen2_never_reads_das_status(self):
    # DAS_status is never sent on gen2, so an engaged-looking value must not leak in
    car_state, parsers = self._car_state(hw4_gen2_fingerprint())
    ret, _ = self._update(car_state, parsers, self._party(
      DAS_status=(CANBUS.autopilot_party, {"DAS_autopilotState": 6}),
      DAS_settings=(CANBUS.party, {"DAS_autosteerEnabled": 0}),
    ))
    assert 0x39b not in parsers[Bus.ap_party].addresses
    assert not ret.invalidLkasSetting

  def test_gen2_reads_blinkers_and_seatbelt_from_the_vehicle_bus(self):
    car_state, parsers = self._car_state(hw4_gen2_fingerprint(vehicle_bus=True))
    ret, _ = self._update(car_state, parsers, self._vehicle(
      VCFRONT_lighting={"VCFRONT_indicatorLeftRequest": 2, "VCFRONT_indicatorRightRequest": 0},
      SeatBeltStatus={"driverBuckleStatus": 0},
    ))
    assert ret.leftBlinker
    assert not ret.rightBlinker
    assert ret.seatbeltUnlatched

    ret, _ = self._update(car_state, parsers, self._vehicle(
      VCFRONT_lighting={"VCFRONT_indicatorLeftRequest": 0, "VCFRONT_indicatorRightRequest": 1},
      SeatBeltStatus={"driverBuckleStatus": 1},
    ))
    assert not ret.leftBlinker
    assert ret.rightBlinker
    assert not ret.seatbeltUnlatched

  def test_gen2_without_the_vehicle_bus_reports_no_blinkers(self):
    car_state, parsers = self._car_state(hw4_gen2_fingerprint())
    assert Bus.adas not in parsers
    ret, _ = self._update(car_state, parsers, self._party(
      VehicleStatus=(CANBUS.party, {"allDoorsClosed": 1}),
    ))
    assert not ret.leftBlinker
    assert not ret.rightBlinker
    assert not ret.seatbeltUnlatched

  def test_gen2_reports_no_speed_limit(self):
    car_state, parsers = self._car_state(hw4_gen2_fingerprint())
    _, ret_iq = self._update(car_state, parsers, self._party(
      DI_state=(CANBUS.party, {"DI_speedUnits": 0}),
      DAS_status=(CANBUS.autopilot_party, {"DAS_fusedSpeedLimit": 55}),
    ))
    assert ret_iq.speedLimit == 0

  def test_hw4_still_reads_ui_warning_and_das_status(self):
    fingerprint = gen_empty_fingerprint()
    fingerprint[CANBUS.autopilot_party][0x293] = 8
    car_state, parsers = self._car_state(fingerprint)
    assert Bus.adas not in parsers

    ret, ret_iq = self._update(car_state, parsers, self._party(
      UI_warning=(CANBUS.party, {"anyDoorOpen": 1, "leftBlinkerBlinking": 1,
                                 "rightBlinkerBlinking": 0, "buckleStatus": 0}),
      DAS_status=(CANBUS.autopilot_party, {"DAS_blindSpotRearLeft": 0, "DAS_blindSpotRearRight": 1,
                                           "DAS_autopilotState": 6, "DAS_fusedSpeedLimit": 55}),
      DI_state=(CANBUS.party, {"DI_speedUnits": 0}),
      DAS_settings=(CANBUS.party, {"DAS_autosteerEnabled": 0}),
    ))
    assert ret.doorOpen
    assert ret.leftBlinker
    assert not ret.rightBlinker
    assert ret.seatbeltUnlatched
    assert not ret.leftBlindspot
    assert ret.rightBlindspot
    # the setting is 0, so this can only come from DAS_autopilotState
    assert ret.invalidLkasSetting
    assert ret_iq.speedLimit == pytest.approx(55 * CV.MPH_TO_MS)

  # a CANParser reports its bus invalid until every message carstate has read arrives, so the
  # set of messages each configuration touches is part of the contract, not an implementation detail
  PARTY_MSGS = {0x118, 0x145, 0x155, 0x257, 0x286, 0x370}
  AP_PARTY_MSGS = {0x129, 0x2b9, 0x488}

  @pytest.mark.parametrize("fingerprint, iq_flags, expected", [
    # HW3/HW4: UI_warning and DAS_status, and DAS_settings is on the autopilot party bus
    (gen_empty_fingerprint(), 0,
     {Bus.party: PARTY_MSGS | {0x311}, Bus.ap_party: AP_PARTY_MSGS | {0x39b}}),
    # the IQ vehicle bus tap adds the odometer, battery and touch messages
    (gen_empty_fingerprint(), TeslaFlagsIQ.HAS_VEHICLE_BUS,
     {Bus.party: PARTY_MSGS | {0x311}, Bus.ap_party: AP_PARTY_MSGS | {0x39b},
      Bus.adas: {0x132, 0x292, 0x33a, 0x3b6, 0x3df}}),
    # HW4 gen2: VehicleStatus and DAS_settings on the party bus, DAS_statusGen2 in place of DAS_status
    (hw4_gen2_fingerprint(), 0,
     {Bus.party: PARTY_MSGS | {0x1e5, 0x293}, Bus.ap_party: AP_PARTY_MSGS | {0x399}}),
    # HW4 gen2 taps the VEHICLE bus for blinkers and the buckle switch alone
    (hw4_gen2_fingerprint(vehicle_bus=True), 0,
     {Bus.party: PARTY_MSGS | {0x1e5, 0x293}, Bus.ap_party: AP_PARTY_MSGS | {0x399},
      Bus.adas: {0x3f5, 0x40a}}),
  ])
  def test_only_reads_messages_the_hardware_sends(self, fingerprint, iq_flags, expected):
    CP = CarInterface.get_params(CAR.TESLA_MODEL_Y, fingerprint, [], False, False, False)
    CP_IQ = structs.IQCarParams(flags=int(iq_flags))
    car_state = CarState(CP, CP_IQ)
    parsers = CarState.get_can_parsers(CP, CP_IQ)
    for _ in range(2):
      car_state.update(parsers)

    assert set(parsers) == set(expected)
    for bus, addresses in expected.items():
      assert parsers[bus].addresses == addresses, f'{bus}: {sorted(map(hex, parsers[bus].addresses))}'
