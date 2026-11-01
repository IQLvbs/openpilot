import random
import re
from types import SimpleNamespace

import pytest

from iqdbc.can import CANPacker
from iqdbc.car import Bus, DT_CTRL, structs
from iqdbc.car.volkswagen.hca_session import EpsSessionClock
from iqdbc.car.volkswagen.carstate import CarState
from iqdbc.car.structs import CarParams
from iqdbc.car.volkswagen.interface import CarInterface
from iqdbc.car.volkswagen.values import (CAR, FW_QUERY_CONFIG, MLB_ACC_COORDINATOR_MSGS, MLB_GEARBOX_MSGS, MLB_MSG_ACC_01,
                                         MLB_MSG_ACC_02, MLB_MSG_ACC_10,
                                         MLB_MSG_GATEWAY_05, MLB_MSG_GETRIEBE_01, MLB_MSG_LH_EPS_01, MLB_MSG_LH_EPS_03,
                                         WMI, VolkswagenFlags, VolkswagenFlagsIQ, VolkswagenSafetyFlags)
from iqdbc.car.volkswagen.fingerprints import FW_VERSIONS

Ecu = CarParams.Ecu

CHASSIS_CODE_PATTERN = re.compile('[A-Z0-9]{2}')
AWAITING_VIN_CAPTURE = {CAR.FORD_CAPRI_MK1}
# TODO: determine the unknown groups
SPARE_PART_FW_PATTERN = re.compile(b'\xf1\x87(?P<gateway>[0-9][0-9A-Z]{2})(?P<unknown>[0-9][0-9A-Z][0-9])(?P<unknown2>[0-9A-Z]{2}[0-9])([A-Z0-9]| )')


class TestVolkswagenPlatformConfigs:
  def test_spare_part_fw_pattern(self, subtests):
    # Relied on for determining if a FW is likely VW
    for platform, ecus in FW_VERSIONS.items():
      with subtests.test(platform=platform.value):
        for fws in ecus.values():
          for fw in fws:
            assert SPARE_PART_FW_PATTERN.match(fw) is not None, f"Bad FW: {fw}"

  def test_chassis_codes(self, subtests):
    for platform in CAR:
      with subtests.test(platform=platform.value):
        assert len(platform.config.wmis) > 0, "WMIs not set"
        if platform in AWAITING_VIN_CAPTURE:
          assert len(platform.config.chassis_codes) == 0, "Remove from AWAITING_VIN_CAPTURE once the chassis code is known"
          continue
        assert len(platform.config.chassis_codes) > 0, "Chassis codes not set"
        assert all(CHASSIS_CODE_PATTERN.match(cc) for cc in
                   platform.config.chassis_codes), "Bad chassis codes"

        # No two platforms should share chassis codes
        for comp in CAR:
          if platform == comp:
            continue

          shared_chassis_codes = platform.config.chassis_codes & comp.config.chassis_codes
          if len(shared_chassis_codes) == 0:
            continue

          # A shared chassis code is unambiguous when the VIN WMI separates the candidates.
          if platform.config.wmis.isdisjoint(comp.config.wmis):
            continue

          platform_model_years = getattr(platform.config, "model_years", set())
          comp_model_years = getattr(comp.config, "model_years", set())
          if platform_model_years and comp_model_years and platform_model_years.isdisjoint(comp_model_years):
            continue

          radar_ecu = (Ecu.fwdRadar, 0x757, None)
          platform_radar_fw = set(FW_VERSIONS.get(platform, {}).get(radar_ecu, []))
          comp_radar_fw = set(FW_VERSIONS.get(comp, {}).get(radar_ecu, []))
          if not platform_radar_fw or not comp_radar_fw:
            continue
          assert platform_radar_fw.isdisjoint(comp_radar_fw), f"Ambiguous VIN and radar firmware: {comp}"

  def test_custom_fuzzy_fingerprinting(self, subtests):
    all_radar_fw = list({fw for ecus in FW_VERSIONS.values() for fw in ecus.get((Ecu.fwdRadar, 0x757, None), [])})

    for platform in CAR:
      with subtests.test(platform=platform.name):
        for wmi in WMI:
          for chassis_code in platform.config.chassis_codes | {"00"}:
            platform_model_years = getattr(platform.config, "model_years", set())
            model_years = platform_model_years if platform_model_years else {"0"}
            for model_year in model_years | {"0"}:
              vin = ["0"] * 17
              vin[0:3] = wmi
              vin[6:8] = chassis_code
              vin[9] = model_year
              vin = "".join(vin)

              for radar_fw in random.sample(all_radar_fw, 5) + [b'\xf1\x875Q0907572G \xf1\x890571', b'\xf1\x877H9907572AA\xf1\x890396']:
                live_fws = {(0x757, None): [radar_fw]}
                matches = FW_QUERY_CONFIG.match_fw_to_car_fuzzy(live_fws, vin, FW_VERSIONS)

                expected_matches = set()
                for candidate in CAR:
                  candidate_model_years = getattr(candidate.config, "model_years", set())
                  candidate_has_radar = (Ecu.fwdRadar, 0x757, None) in FW_VERSIONS.get(candidate, {})
                  model_year_match = not candidate_model_years or model_year in candidate_model_years
                  if (wmi in candidate.config.wmis and chassis_code in candidate.config.chassis_codes and model_year_match and
                      radar_fw in all_radar_fw and candidate_has_radar):
                    expected_matches.add(candidate)
                assert expected_matches == matches, "Bad match"

  @pytest.mark.parametrize("candidate, expected", (
    (CAR.VOLKSWAGEN_PASSAT_B7, True),
    (CAR.SEAT_ALHAMBRA_MK1, True),
    (CAR.VOLKSWAGEN_GOLF_MK7, False),
  ))
  def test_pq_acc_fts_epb_flags(self, candidate, expected):
    params = CarInterface.get_params(candidate, {bus: {} for bus in range(7)}, [], alpha_long=False, is_release=False, docs=False)
    assert bool(params.flags & VolkswagenFlagsIQ.IQ_PQ_ACC_FTS_EPB) is expected
    assert bool(params.safetyConfigs[-1].safetyParam & VolkswagenSafetyFlags.PQ_ACC_FTS_EPB) is expected

  @pytest.mark.parametrize("candidate, alpha_long, expected", (
    (CAR.VOLKSWAGEN_PASSAT_B7, False, False),
    (CAR.VOLKSWAGEN_PASSAT_B7, True, True),
    (CAR.VOLKSWAGEN_GOLF_MK7, False, False),
    (CAR.VOLKSWAGEN_GOLF_MK7, True, True),
    (CAR.AUDI_Q5_MK1, True, False),
    (CAR.VOLKSWAGEN_ID4_MK1, True, False),
    (CAR.VOLKSWAGEN_GOLF_MK8, True, False),
  ))
  def test_supported_vw_longitudinal_stays_active_during_gas_override(self, candidate, alpha_long, expected):
    fingerprints = {bus: {} for bus in range(7)}
    params = CarInterface.get_params(candidate, fingerprints, [], alpha_long=alpha_long, is_release=False, docs=False)
    params_iq = CarInterface.get_params_iq(params, candidate, fingerprints, [], alpha_long=alpha_long, is_release_iq=False, docs=False)
    assert params_iq.longActiveWithGasOverride is expected
    if expected:
      assert bool(params.safetyConfigs[-1].safetyParam & VolkswagenSafetyFlags.ALLOW_LONG_ACCEL_WITH_GAS_PRESSED) is alpha_long


def _mlb_fingerprint(bus, msgs):
  fingerprints = {b: {} for b in range(7)}
  fingerprints[bus] = {msg: 8 for msg in msgs}
  return fingerprints


A4_MK4_BUS_1 = (0x040, 0x080, 0x081, 0x086, 0x100, 0x101, 0x103, 0x104, 0x105, 0x106, 0x107, 0x10B,
                0x10C, 0x10E, 0x114, 0x11D, 0x309, 0x30B, 0x30E, 0x312, 0x391, 0x392, 0x39C, 0x3BF,
                0x3C0, 0x440, 0x471, 0x520, 0x585, 0x590, 0x5F0, 0x640, 0x641, 0x643, 0x644, 0x647,
                0x670, 0x6B2, 0x6B4, 0x6B7, 0x6B8, 0x6C0, 0x6C1)

MLB_ECAN_GATEWAY = (0x086, 0x09F, 0x102, 0x103, 0x105, 0x106, 0x10B, 0x10C, 0x30B)
MLB_ECAN_CAMERA = (0x109, 0x10D, 0x117, 0x30C, 0x30F)


def test_mlb_no_ecan_car_moves_to_the_powertrain_bus():
  fingerprints = _mlb_fingerprint(1, A4_MK4_BUS_1)
  params = CarInterface.get_params(CAR.AUDI_A4_MK4, fingerprints, [], alpha_long=True, is_release=False, docs=False)

  assert params.flags & VolkswagenFlagsIQ.IQ_MLB_NO_ECAN
  assert params.safetyConfigs[-1].safetyParam & VolkswagenSafetyFlags.MLB_NO_ECAN
  assert params.flags & VolkswagenFlagsIQ.IQ_CC_ONLY_NO_RADAR
  assert params.flags & VolkswagenFlagsIQ.IQ_MLB_NO_HCA_EPS
  assert params.transmissionType == CarParams.TransmissionType.manual
  assert not params.alphaLongitudinalAvailable
  assert not params.openpilotLongitudinalControl
  assert params.dashcamOnly


def test_mlb_without_an_extended_can_is_not_controllable():
  for msgs in (A4_MK4_BUS_1, A4_MK4_BUS_1 + (0x9F,)):
    params = CarInterface.get_params(CAR.AUDI_A4_MK4, _mlb_fingerprint(1, msgs), [],
                                     alpha_long=False, is_release=False, docs=False)
    assert params.dashcamOnly


def test_mlb_gateway_car_keeps_the_extended_can():
  fingerprints = _mlb_fingerprint(0, MLB_ECAN_GATEWAY)
  fingerprints[2] = {msg: 8 for msg in MLB_ECAN_CAMERA}
  params = CarInterface.get_params(CAR.AUDI_Q5_MK1, fingerprints, [], alpha_long=False, is_release=False, docs=False)

  assert not params.flags & VolkswagenFlagsIQ.IQ_MLB_NO_ECAN
  assert not params.flags & (VolkswagenFlagsIQ.IQ_CC_ONLY | VolkswagenFlagsIQ.IQ_CC_ONLY_NO_RADAR)
  assert not params.flags & VolkswagenFlagsIQ.IQ_MLB_NO_HCA_EPS
  assert params.transmissionType == CarParams.TransmissionType.automatic
  assert params.alphaLongitudinalAvailable
  assert not params.dashcamOnly


def test_mlb_flags_are_not_inferred_without_a_fingerprint():
  fingerprints = {bus: {} for bus in range(7)}
  for platform in (CAR.AUDI_Q5_MK1, CAR.PORSCHE_MACAN_MK1, CAR.AUDI_A4_MK4):
    params = CarInterface.get_params(platform, fingerprints, [], alpha_long=False, is_release=False, docs=True)
    assert not params.flags & VolkswagenFlagsIQ.IQ_MLB_NO_ECAN
    assert not params.flags & (VolkswagenFlagsIQ.IQ_CC_ONLY | VolkswagenFlagsIQ.IQ_CC_ONLY_NO_RADAR)
    assert not params.flags & VolkswagenFlagsIQ.IQ_MLB_NO_HCA_EPS
    assert params.transmissionType == CarParams.TransmissionType.automatic


def _build_mlb_car(platform, fingerprints):
  CP = CarInterface.get_params(platform, fingerprints, [], alpha_long=False, is_release=False, docs=False)
  CP_IQ = CarInterface.get_params_iq(CP, platform, fingerprints, [], alpha_long=False, is_release_iq=False, docs=False)
  return CarInterface(CP, CP_IQ)


def _a4_mk4_car():
  return _build_mlb_car(CAR.AUDI_A4_MK4, _mlb_fingerprint(1, A4_MK4_BUS_1))


def _q5_mk1_car():
  fingerprints = _mlb_fingerprint(0, MLB_ECAN_GATEWAY)
  fingerprints[2] = {msg: 8 for msg in MLB_ECAN_CAMERA}
  return _build_mlb_car(CAR.AUDI_Q5_MK1, fingerprints)


def _a4_mk4_frames(packer, reverse=False, eps_torque=None):
  msgs = [
    packer.make_can_msg("ESP_03", 1, {"ESP_%s_Radgeschw" % s: 30.0 for s in ("VL", "VR", "HL", "HR")}),
    packer.make_can_msg("Motor_03", 1, {"MO_Fahrpedalrohwert_01": 0, "MO_BLS": 0}),
    packer.make_can_msg("ESP_05", 1, {"ESP_Bremsdruck": 0, "ESP_Fahrer_bremst": 0}),
    packer.make_can_msg("ESP_01", 1, {"ESP_Tastung_passiv": 0}),
    packer.make_can_msg("ESP_02", 1, {"ESP_Stillstandsflag": 0}),
    packer.make_can_msg("TSK_02", 1, {"TSK_Status": 0}),
    packer.make_can_msg("LS_01", 1, {"LS_Hauptschalter": 1, "LS_Codierung": 2}),
    packer.make_can_msg("LWI_01", 1, {"LWI_Lenkradwinkel": 12.0, "LWI_Lenkradw_Geschw": 4.0}),
    packer.make_can_msg("Kombi_01", 1, {"KBI_angez_Geschw": 30.0, "KBI_Handbremse": 0}),
    packer.make_can_msg("Kombi_02", 1, {"KBI_Inhalt_Tank": 40, "KBI_Kilometerstand": 100000}),
    packer.make_can_msg("Airbag_02", 1, {"AB_Gurtschloss_FA": 3}),
    packer.make_can_msg("Gateway_05", 1, {"BCM1_Rueckfahrlicht_Schalter": int(reverse)}),
  ]
  if eps_torque is not None:
    msgs.append(packer.make_can_msg("LH_EPS_03", 1, {"EPS_Lenkmoment": abs(eps_torque),
                                                     "EPS_VZ_Lenkmoment": eps_torque < 0,
                                                     "EPS_HCA_Status": 3}))
  return msgs


def _run(car, build, frames=25):
  nanos = 0
  for _ in range(frames):
    nanos += int(DT_CTRL * 1e9)
    ret, _ = car.update([(nanos, build())])
  return ret


def test_mlb_no_ecan_parsers_read_the_powertrain_bus():
  a4 = _a4_mk4_car()
  assert a4.CS.get_can_parsers(a4.CP, a4.CP_IQ)[Bus.pt].bus == 1

  q5 = _q5_mk1_car()
  assert q5.CS.get_can_parsers(q5.CP, q5.CP_IQ)[Bus.pt].bus == 0


def test_mlb_no_ecan_transmits_on_the_powertrain_bus():
  a4 = _a4_mk4_car()
  packer = CANPacker("vw_mlb")

  CC = structs.CarControl()
  CC.enabled = True
  CC.latActive = True
  CC.cruiseControl.cancel = True
  CC = CC.as_reader()
  CC_IQ = structs.IQCarControl()

  sent = []
  nanos = 0
  for _ in range(20):
    nanos += int(DT_CTRL * 1e9)
    a4.update([(nanos, _a4_mk4_frames(packer, eps_torque=0))])
    _, can_sends = a4.apply(CC, CC_IQ, nanos)
    sent.extend(can_sends)

  assert {addr for addr, _, _ in sent} == {0x126, 0x397, 0x10B}
  assert {bus for _, _, bus in sent} == {1}


def test_hca_torque_stops_the_moment_openpilot_disengages():
  a4 = _a4_mk4_car()
  packer = CANPacker("vw_mlb")
  nanos = 0

  def run(enabled, lat_active, steps=20):
    nonlocal nanos
    CC = structs.CarControl()
    CC.enabled = enabled
    CC.latActive = lat_active
    CC.actuators.torque = 1.0
    CC = CC.as_reader()
    CC_IQ = structs.IQCarControl()
    for _ in range(steps):
      nanos += int(DT_CTRL * 1e9)
      a4.update([(nanos, _a4_mk4_frames(packer, eps_torque=0))])
      a4.apply(CC, CC_IQ, nanos)
    return a4.CC.HCA_Status, a4.CC.apply_torque_last

  engaged_status, engaged_torque = run(enabled=True, lat_active=True)
  assert engaged_status in (5, 7)
  assert engaged_torque != 0

  disengaged_status, disengaged_torque = run(enabled=False, lat_active=True)
  assert disengaged_status == 3
  assert disengaged_torque == 0


def test_mlb_no_hca_eps_reports_a_steer_fault_without_a_can_fault():
  a4 = _a4_mk4_car()
  assert a4.CP.flags & VolkswagenFlagsIQ.IQ_MLB_NO_HCA_EPS

  packer = CANPacker("vw_mlb")
  ret = _run(a4, lambda: _a4_mk4_frames(packer))

  assert ret.steerFaultPermanent
  assert not ret.steerFaultTemporary
  assert ret.steeringTorque == 0.0
  assert not ret.steeringPressed
  assert ret.steeringAngleDeg == pytest.approx(12.0, abs=0.2)

  pt = a4.can_parsers[Bus.pt]
  assert pt.message_states[MLB_MSG_LH_EPS_03].ignore_alive
  assert all(parser.can_valid for parser in a4.can_parsers.values())


def test_mlb_no_hca_eps_survives_the_parser_aliveness_timeout():
  a4 = _a4_mk4_car()
  packer = CANPacker("vw_mlb")
  pt = a4.can_parsers[Bus.pt]

  nanos = 0
  for _ in range(int(15.0 / DT_CTRL)):
    nanos += int(DT_CTRL * 1e9)
    ret, _ = a4.update([(nanos, _a4_mk4_frames(packer))])
    pt.vl["LH_EPS_03"]["EPS_HCA_Status"]

  assert ret.steerFaultPermanent
  assert all(parser.can_valid for parser in a4.can_parsers.values())
  assert not any(parser.bus_timeout for parser in a4.can_parsers.values())


def test_mlb_lane_assist_eps_restores_the_normal_torque_path():
  fingerprints = _mlb_fingerprint(1, A4_MK4_BUS_1 + (0x9F,))
  a4 = _build_mlb_car(CAR.AUDI_A4_MK4, fingerprints)
  assert not a4.CP.flags & VolkswagenFlagsIQ.IQ_MLB_NO_HCA_EPS

  packer = CANPacker("vw_mlb")
  ret = _run(a4, lambda: _a4_mk4_frames(packer, eps_torque=-140))

  assert ret.steeringTorque == pytest.approx(-140, abs=1)
  assert ret.steeringPressed
  assert not ret.steerFaultPermanent
  assert not a4.can_parsers[Bus.pt].message_states[MLB_MSG_LH_EPS_03].ignore_alive


def test_mlb_cc_only_never_reads_the_acc_coordinator():
  a4 = _a4_mk4_car()
  assert a4.CP.flags & VolkswagenFlagsIQ.IQ_CC_ONLY_NO_RADAR

  packer = CANPacker("vw_mlb")
  ret = _run(a4, lambda: _a4_mk4_frames(packer))

  assert ret.cruiseState.available
  assert not ret.cruiseState.enabled
  assert ret.cruiseState.speed == 0.0

  parsers = a4.can_parsers
  for msg in MLB_ACC_COORDINATOR_MSGS + (MLB_MSG_ACC_10,):
    for parser in parsers.values():
      assert msg not in parser.addresses


def test_mlb_manual_gear_follows_the_reverse_light_switch():
  a4 = _a4_mk4_car()
  assert a4.CP.transmissionType == CarParams.TransmissionType.manual

  packer = CANPacker("vw_mlb")
  assert _run(a4, lambda: _a4_mk4_frames(packer, reverse=False)).gearShifter == structs.CarState.GearShifter.drive
  assert _run(a4, lambda: _a4_mk4_frames(packer, reverse=True)).gearShifter == structs.CarState.GearShifter.reverse
  assert _run(a4, lambda: _a4_mk4_frames(packer, reverse=False)).gearShifter == structs.CarState.GearShifter.drive


def _q5_params(bus0, bus1=(), car_fw=()):
  fingerprints = _mlb_fingerprint(0, bus0)
  fingerprints[1] = {msg: 8 for msg in bus1}
  fingerprints[2] = {msg: 8 for msg in MLB_ECAN_CAMERA}
  return CarInterface.get_params(CAR.AUDI_Q5_MK1, fingerprints, list(car_fw), alpha_long=False, is_release=False, docs=False)


MLB_ECAN_GATEWAY_NO_GEARBOX = tuple(msg for msg in MLB_ECAN_GATEWAY if msg not in MLB_GEARBOX_MSGS)


def test_mlb_gearbox_on_the_powertrain_bus_keeps_automatic():
  params = _q5_params(MLB_ECAN_GATEWAY_NO_GEARBOX, bus1=(MLB_MSG_GETRIEBE_01, MLB_MSG_GATEWAY_05))
  assert params.transmissionType == CarParams.TransmissionType.automatic


def test_mlb_transmission_ecu_firmware_keeps_automatic():
  transmission = CarParams.CarFw.new_message(ecu=Ecu.transmission, address=0x7e1)
  params = _q5_params(MLB_ECAN_GATEWAY_NO_GEARBOX, bus1=(MLB_MSG_GATEWAY_05,), car_fw=(transmission,))
  assert params.transmissionType == CarParams.TransmissionType.automatic


def test_mlb_reverse_switch_on_the_extended_can_alone_keeps_automatic():
  params = _q5_params(MLB_ECAN_GATEWAY_NO_GEARBOX + (MLB_MSG_GATEWAY_05,))
  assert params.transmissionType == CarParams.TransmissionType.automatic


def test_mlb_manual_needs_no_gearbox_anywhere_and_a_reverse_switch_on_the_powertrain_bus():
  params = _q5_params(MLB_ECAN_GATEWAY_NO_GEARBOX, bus1=(MLB_MSG_GATEWAY_05,))
  assert params.transmissionType == CarParams.TransmissionType.manual


def test_mlb_manual_gateway_car_reads_reverse_from_the_powertrain_bus():
  fingerprints = _mlb_fingerprint(0, MLB_ECAN_GATEWAY_NO_GEARBOX)
  fingerprints[1] = {MLB_MSG_GATEWAY_05: 8}
  fingerprints[2] = {msg: 8 for msg in MLB_ECAN_CAMERA}
  q5 = _build_mlb_car(CAR.AUDI_Q5_MK1, fingerprints)
  assert q5.CP.transmissionType == CarParams.TransmissionType.manual

  packer = CANPacker("vw_mlb")
  frames = lambda reverse: [packer.make_can_msg("Gateway_05", 1, {"BCM1_Rueckfahrlicht_Schalter": int(reverse)})]
  assert _run(q5, lambda: frames(True)).gearShifter == structs.CarState.GearShifter.reverse
  assert _run(q5, lambda: frames(False)).gearShifter == structs.CarState.GearShifter.drive
  assert MLB_MSG_GATEWAY_05 not in q5.can_parsers[Bus.pt].addresses


def test_mlb_automatic_never_subscribes_gateway_05():
  q5 = _q5_mk1_car()
  q5.update([(int(DT_CTRL * 1e9), [])])
  assert MLB_MSG_GATEWAY_05 not in q5.can_parsers[Bus.pt].addresses
  assert MLB_MSG_GATEWAY_05 not in q5.can_parsers[Bus.aux].addresses


@pytest.mark.parametrize("button_type", (
  structs.CarState.ButtonEvent.Type.setCruise,
  structs.CarState.ButtonEvent.Type.resumeCruise,
))
def test_button_enable_is_blocked_while_cruise_fault_lateral_mode_is_still_faulted(button_type):
  state = object.__new__(CarState)
  state.CP = SimpleNamespace(pcmCruise=False)
  state.cruise_fault_lateral_active = True
  state.cruise_faulted = True

  button_events = [structs.CarState.ButtonEvent(pressed=False, type=button_type)]

  assert not state.update_button_enable(button_events)


def test_button_enable_recovers_once_cruise_fault_clears():
  state = object.__new__(CarState)
  state.CP = SimpleNamespace(pcmCruise=False)
  state.cruise_fault_lateral_active = True
  state.cruise_faulted = False

  button_events = [structs.CarState.ButtonEvent(
    pressed=False,
    type=structs.CarState.ButtonEvent.Type.setCruise,
  )]

  assert state.update_button_enable(button_events)


def test_pq_hca_ready_does_not_complete_eps_initialization():
  state = object.__new__(CarState)
  state.eps_init_complete = False
  state.eps_session = EpsSessionClock(1.1)

  state.frame = 0
  assert state.update_hca_state("READY", ready_confirms_init=False) == (True, False)

  state.frame = 317
  assert state.update_hca_state("FAULT", ready_confirms_init=False) == (True, False)

  state.frame = 1001
  assert state.update_hca_state("FAULT", ready_confirms_init=False) == (False, True)


@pytest.mark.parametrize("platform", (CAR.AUDI_Q4_MK1, CAR.VOLKSWAGEN_ID4_MK2))
def test_meb_does_not_infer_mqb_cluster_from_address(platform):
  fingerprints = {bus: {} for bus in range(7)}
  fingerprints[0][0x30B] = 8
  params = CarInterface.get_params(platform, fingerprints, [], alpha_long=False, is_release=False, docs=False)
  assert not params.flags & VolkswagenFlags.KOMBI_PRESENT



def _subscribed_addresses(car, updates=5):
  nanos = 0
  for _ in range(updates):
    nanos += int(DT_CTRL * 1e9)
    car.update([(nanos, [])])
  return {bus: set(parser.addresses) for bus, parser in car.can_parsers.items()}


def _frames_for(car, packer, subscribed):
  frames = []
  for bus, parser in car.can_parsers.items():
    for addr in sorted(subscribed[bus]):
      frames.append(packer.make_can_msg(parser.dbc.addr_to_msg[addr].name, parser.bus, {}))
  return frames


def _run_past_aliveness_timeout(car, build, seconds=15.0):
  nanos = 0
  ret = None
  for _ in range(int(seconds / DT_CTRL)):
    nanos += int(DT_CTRL * 1e9)
    ret, _ = car.update([(nanos, build())])
  return ret


def _sparse_q5(car_fw=(), bus1=()):
  fingerprints = _mlb_fingerprint(0, MLB_ECAN_GATEWAY_NO_GEARBOX)
  fingerprints[1] = {msg: 8 for msg in bus1}
  fingerprints[2] = {msg: 8 for msg in MLB_ECAN_CAMERA}
  CP = CarInterface.get_params(CAR.AUDI_Q5_MK1, fingerprints, list(car_fw), alpha_long=False, is_release=False, docs=False)
  CP_IQ = CarInterface.get_params_iq(CP, CAR.AUDI_Q5_MK1, fingerprints, list(car_fw), alpha_long=False, is_release_iq=False, docs=False)
  return CarInterface(CP, CP_IQ)


@pytest.mark.parametrize("car_fw", (
  (),
  (CarParams.CarFw.new_message(ecu=Ecu.transmission, address=0x7e1),),
), ids=("no_fw", "transmission_fw"))
def test_mlb_gateway_car_with_a_sparse_fingerprint_snapshot_reads_only_what_an_automatic_reads(car_fw):
  automatic = _subscribed_addresses(_q5_mk1_car())
  assert all(MLB_MSG_GATEWAY_05 not in addrs for addrs in automatic.values())

  q5 = _sparse_q5(car_fw)
  packer = CANPacker("vw_mlb")
  ret = _run_past_aliveness_timeout(q5, lambda: _frames_for(q5, packer, automatic))

  assert ret.canValid
  assert all(parser.can_valid for parser in q5.can_parsers.values())
  assert not any(parser.bus_timeout for parser in q5.can_parsers.values())
  assert {bus: set(parser.addresses) for bus, parser in q5.can_parsers.items()} == automatic
  assert q5.CP.transmissionType == CarParams.TransmissionType.automatic


def _jetta_camera_harness_car():
  fingerprints = {bus: {} for bus in range(7)}
  fingerprints[2][0x126] = 8
  car_fw = [CarParams.CarFw.new_message(ecu=Ecu.fwdCamera, address=0x74f)]
  CP = CarInterface.get_params(CAR.VOLKSWAGEN_JETTA_MK7, fingerprints, car_fw, alpha_long=False, is_release=False, docs=False)
  CP_IQ = CarInterface.get_params_iq(CP, CAR.VOLKSWAGEN_JETTA_MK7, fingerprints, car_fw, alpha_long=False, is_release_iq=False, docs=False)
  return CarInterface(CP, CP_IQ)


def test_mqb_camera_harness_keeps_can_valid_when_bus_1_never_carries_blinkmodi_02():
  car = _jetta_camera_harness_car()
  assert car.CP.networkLocation == CarParams.NetworkLocation.fwdCamera

  subscribed = _subscribed_addresses(car)
  assert 0x366 in subscribed[Bus.pt]
  assert set(car.can_parsers) == {Bus.pt, Bus.cam}

  packer = CANPacker(CAR.VOLKSWAGEN_JETTA_MK7.config.dbc_dict[Bus.pt])
  ret = _run_past_aliveness_timeout(car, lambda: _frames_for(car, packer, subscribed))

  assert ret.canValid
  assert not ret.canTimeout
  assert all(parser.can_valid for parser in car.can_parsers.values())


def test_mqb_still_reports_can_invalid_when_the_powertrain_bus_loses_blinkmodi_02():
  car = _jetta_camera_harness_car()
  subscribed = _subscribed_addresses(car)
  on_the_wire = {bus: set(addrs) for bus, addrs in subscribed.items()}
  on_the_wire[Bus.pt] -= {0x366}

  packer = CANPacker(CAR.VOLKSWAGEN_JETTA_MK7.config.dbc_dict[Bus.pt])
  ret = _run_past_aliveness_timeout(car, lambda: _frames_for(car, packer, on_the_wire))

  assert not ret.canValid


def test_mlb_manual_gateway_car_adds_only_the_reverse_switch_on_the_powertrain_bus():
  automatic = _subscribed_addresses(_q5_mk1_car())
  manual = _subscribed_addresses(_sparse_q5(bus1=(MLB_MSG_GATEWAY_05,)))

  assert manual[Bus.pt] == automatic[Bus.pt]
  assert manual[Bus.cam] == automatic[Bus.cam]
  assert manual[Bus.aux] == automatic[Bus.aux] | {MLB_MSG_GATEWAY_05}


def _read_like_the_alc_runtime(parser, name):
  # the closed-source ALC reads its key slot through parser.vl, which subscribes the message on first access
  if name not in parser.dat:
    parser.vl[name]
  return parser.dat.get(name, b"")


@pytest.mark.parametrize("build", (_a4_mk4_car, _q5_mk1_car), ids=("a4_mk4", "q5_mk1"))
def test_mlb_alc_key_slot_read_keeps_can_valid_on_a_rack_that_never_sends_it(build):
  car = build()
  subscribed = _subscribed_addresses(car)
  never_sent = {bus: addrs - {MLB_MSG_LH_EPS_01} for bus, addrs in subscribed.items()}

  packer = CANPacker("vw_mlb")

  def build_frames():
    _read_like_the_alc_runtime(car.can_parsers[Bus.pt], "LH_EPS_01")
    return _frames_for(car, packer, never_sent)

  ret = _run_past_aliveness_timeout(car, build_frames)

  assert ret.canValid
  assert all(parser.can_valid for parser in car.can_parsers.values())
  assert not any(parser.bus_timeout for parser in car.can_parsers.values())


@pytest.mark.parametrize("build", (_a4_mk4_car, _q5_mk1_car), ids=("a4_mk4", "q5_mk1"))
def test_mlb_carstate_subscriptions_settle_and_stay_alive(build):
  car = build()
  subscribed = _subscribed_addresses(car)

  packer = CANPacker("vw_mlb")
  ret = _run_past_aliveness_timeout(car, lambda: _frames_for(car, packer, subscribed))

  assert {bus: set(parser.addresses) for bus, parser in car.can_parsers.items()} == subscribed
  assert ret.canValid
  assert all(parser.can_valid for parser in car.can_parsers.values())
  assert not any(parser.bus_timeout for parser in car.can_parsers.values())


def _steering_state(flags, lwi=(0.0, 0), eps=(0.0, 0)):
  state = object.__new__(CarState)
  state.CP = SimpleNamespace(flags=flags)
  state.CCP = SimpleNamespace(STEER_DRIVER_ALLOWANCE=100, hca_status_values={5: "ACTIVE"})
  state.eps_init_complete = True
  state.frame = 0
  state.eps_session = EpsSessionClock(1.1)
  pt_cp = SimpleNamespace(vl={
    "LWI_01": {"LWI_Lenkradwinkel": lwi[0], "LWI_VZ_Lenkradwinkel": lwi[1],
               "LWI_Lenkradw_Geschw": 4.0, "LWI_VZ_Lenkradw_Geschw": 0},
    "LH_EPS_03": {"EPS_Berechneter_LW": eps[0], "EPS_VZ_BLW": eps[1],
                  "EPS_Lenkmoment": 0.0, "EPS_VZ_Lenkmoment": 0, "EPS_HCA_Status": 5},
  })
  ret = structs.CarState()
  state.parse_mlb_mqb_steering_state(ret, pt_cp)
  return ret


def test_mlb_takes_the_steering_angle_from_the_eps():
  ret = _steering_state(VolkswagenFlags.MLB, lwi=(12.0, 0), eps=(9.6, 0))
  assert ret.steeringAngleDeg == pytest.approx(9.6)
  assert ret.steeringRateDeg == pytest.approx(4.0)


def test_mlb_eps_angle_honours_its_own_sign_bit():
  ret = _steering_state(VolkswagenFlags.MLB, lwi=(12.0, 0), eps=(9.6, 1))
  assert ret.steeringAngleDeg == pytest.approx(-9.6)


def test_mlb_without_hca_eps_keeps_the_steering_wheel_sensor():
  flags = VolkswagenFlags.MLB | VolkswagenFlagsIQ.IQ_MLB_NO_HCA_EPS
  ret = _steering_state(flags, lwi=(12.0, 1), eps=(9.6, 0))
  assert ret.steeringAngleDeg == pytest.approx(-12.0)


def test_mqb_keeps_the_steering_wheel_sensor():
  ret = _steering_state(0, lwi=(12.0, 0), eps=(9.6, 0))
  assert ret.steeringAngleDeg == pytest.approx(12.0)


def _mlb_long_car(platform):
  fingerprints = _mlb_fingerprint(0, MLB_ECAN_GATEWAY)
  fingerprints[2] = {msg: 8 for msg in MLB_ECAN_CAMERA}
  CP = CarInterface.get_params(platform, fingerprints, [], alpha_long=True, is_release=False, docs=False)
  CP_IQ = CarInterface.get_params_iq(CP, platform, fingerprints, [], alpha_long=True, is_release_iq=False, docs=False)
  assert CP.openpilotLongitudinalControl
  return CarInterface(CP, CP_IQ)


def _run_long_controller(car, build, seconds=15.0):
  CC = structs.CarControl().as_reader()
  CC_IQ = structs.IQCarControl()
  nanos = 0
  ret = None
  for _ in range(int(seconds / DT_CTRL)):
    nanos += int(DT_CTRL * 1e9)
    ret, _ = car.update([(nanos, build())])
    car.apply(CC, CC_IQ, nanos)
  return ret


@pytest.mark.parametrize("platform", (CAR.PORSCHE_MACAN_MK1, CAR.AUDI_Q5_MK1), ids=("macan_mk1", "q5_mk1"))
def test_mlb_long_keeps_can_valid_when_the_acc_ecu_never_sends_acc_01(platform):
  car = _mlb_long_car(platform)
  subscribed = _subscribed_addresses(car)
  never_sent = {bus: addrs - {MLB_MSG_ACC_01} for bus, addrs in subscribed.items()}

  packer = CANPacker("vw_mlb")
  ret = _run_long_controller(car, lambda: _frames_for(car, packer, never_sent))

  assert ret.canValid
  assert all(parser.can_valid for parser in car.can_parsers.values())
  assert not any(parser.bus_timeout for parser in car.can_parsers.values())
  assert "ACC_01" not in car.CS.acc_stock_counters
  assert car.CC.acc_counters_seeded == {"ACC_02"}


def test_mlb_long_seeds_acc_01_from_the_stock_frame_when_it_is_sent():
  car = _mlb_long_car(CAR.AUDI_Q5_MK1)
  subscribed = _subscribed_addresses(car)
  assert MLB_MSG_ACC_01 in subscribed[Bus.cam]

  packer = CANPacker("vw_mlb")
  ret = _run_long_controller(car, lambda: _frames_for(car, packer, subscribed), seconds=1.0)

  assert ret.canValid
  assert car.CC.acc_counters_seeded == {"ACC_01", "ACC_02"}
  assert car.CS.acc_stock_counters["ACC_01"] == car.can_parsers[Bus.cam].vl["ACC_01"]["COUNTER"]
  assert MLB_MSG_ACC_02 in subscribed[Bus.cam]
