import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, IntFlag
from functools import cache
from iqdbc.car import ACCELERATION_DUE_TO_GRAVITY, Bus, CarSpecs, DbcDict, PlatformConfig, Platforms
from iqdbc.car.lateral import AngleSteeringLimits, ISO_LATERAL_ACCEL
from iqdbc.car.structs import CarParams, CarState
from iqdbc.car.docs_definitions import CarDocs, CarFootnote, CarHarness, CarParts, Column, SupportType
from iqdbc.car.fw_query_definitions import FwQueryConfig, LiveFwVersions, OfflineFwVersions, Request, StdQueries

Ecu = CarParams.Ecu


class Footnote(Enum):
  HW_TYPE = CarFootnote(
    "Some 2023 model years have HW4. To check which hardware type your vehicle has, look for " +
    "<b>Autopilot computer</b> under <b>Software -> Additional Vehicle Information</b> on your vehicle's touchscreen. </br></br>" +
    "See <a href=\"https://www.notateslaapp.com/news/2173/how-to-check-if-your-tesla-has-hardware-4-ai4-or-hardware-3\">this page</a> for more information.",
    Column.MODEL)

  SETUP = CarFootnote(
    "See more setup details for <a href=\"https://github.com/commaai/openpilot/wiki/tesla\" target=\"_blank\">Tesla</a>.",
    Column.MAKE, setup_note=True)


@dataclass
class TeslaCarDocsHW3(CarDocs):
  package: str = "All"
  car_parts: CarParts = field(default_factory=CarParts.common([CarHarness.tesla_a]))
  footnotes: list[Enum] = field(default_factory=lambda: [Footnote.HW_TYPE, Footnote.SETUP])


@dataclass
class TeslaCarDocsHW4(CarDocs):
  package: str = "All"
  car_parts: CarParts = field(default_factory=CarParts.common([CarHarness.tesla_b]))
  footnotes: list[Enum] = field(default_factory=lambda: [Footnote.HW_TYPE, Footnote.SETUP])

@dataclass
class TeslaCarDocsHW4Gen2(CarDocs):
  package: str = "All"
  car_parts: CarParts = field(default_factory=CarParts.common([CarHarness.tesla_c]))
  footnotes: list[Enum] = field(default_factory=lambda: [Footnote.HW_TYPE, Footnote.SETUP])


@dataclass
class TeslaCarHW4ModelSXDocs(TeslaCarDocsHW4):
  support_type: SupportType = SupportType.COMMUNITY
  support_link: str = "community"


@dataclass
class TeslaPlatformConfig(PlatformConfig):
  dbc_dict: DbcDict = field(default_factory=lambda: {Bus.party: 'tesla_model3_party', Bus.adas: 'tesla_model3_vehicle'})


class CAR(Platforms):
  TESLA_MODEL_3 = TeslaPlatformConfig(
    [
      # TODO: do we support 2017? It's HW3
      TeslaCarDocsHW3("Tesla Model 3 (with HW3) 2019-23"),
      TeslaCarDocsHW4("Tesla Model 3 (with HW4) 2024-25"),
    ],
    CarSpecs(mass=1899., wheelbase=2.875, steerRatio=12.0),
    {Bus.party: 'tesla_model3_party', Bus.radar: 'tesla_radar_continental_generated', Bus.adas: 'tesla_model3_vehicle'},
  )
  TESLA_MODEL_Y = TeslaPlatformConfig(
    [
      TeslaCarDocsHW3("Tesla Model Y (with HW3) 2020-23"),
      TeslaCarDocsHW4("Tesla Model Y (with HW4) 2024-25"),
      TeslaCarDocsHW4Gen2("Tesla Model Y (with HW4) 2026"),
    ],
    CarSpecs(mass=2072., wheelbase=2.890, steerRatio=12.0),
    {Bus.party: 'tesla_model3_party', Bus.radar: 'tesla_radar_continental_generated', Bus.adas: 'tesla_model3_vehicle'},
  )
  TESLA_MODEL_X = TeslaPlatformConfig(
    [TeslaCarHW4ModelSXDocs("Tesla Model X (with HW4) 2024")],
    CarSpecs(mass=2495., wheelbase=2.960, steerRatio=12.0),
  )
  TESLA_MODEL_X_HW1 = TeslaPlatformConfig(
    [CarDocs("Tesla Model X (with HW1) 2014-16", "All", car_parts=CarParts.common([CarHarness.tesla_model_x_hw1]))],
    CarSpecs(mass=2447., wheelbase=2.960, steerRatio=15.0),
    {
      Bus.chassis: 'tesla_can',
      Bus.party: 'tesla_can',
      Bus.pt: 'tesla_can',
      Bus.radar: 'tesla_radar_bosch_generated',
    },
  )
  TESLA_MODEL_X_HW2 = TeslaPlatformConfig(
    [CarDocs("Tesla Model X (with HW2) 2016-19", "All", car_parts=CarParts.common([CarHarness.tesla_model_sx_hw2]))],
    CarSpecs(mass=2100., wheelbase=2.960, steerRatio=15.0),
    {
      Bus.chassis: 'tesla_can',
      Bus.party: 'tesla_can',
      Bus.pt: 'tesla_powertrain',
      Bus.radar: 'tesla_radar_bosch_generated',
    },
  )
  TESLA_MODEL_S_HW1 = TeslaPlatformConfig(
    [CarDocs("Tesla Model S (with HW1) 2014-16", "All", car_parts=CarParts.common([CarHarness.tesla_model_s_hw1]))],
    CarSpecs(mass=2100., wheelbase=2.960, steerRatio=15.0),
    {
      Bus.chassis: 'tesla_can',
      Bus.party: 'tesla_can',
      Bus.pt: 'tesla_can',
      Bus.radar: 'tesla_radar_bosch_generated',
    },
  )
  TESLA_MODEL_S_HW2 = TeslaPlatformConfig(
    [CarDocs("Tesla Model S (with HW2) 2017-19", "All", car_parts=CarParts.common([CarHarness.tesla_model_sx_hw2]))],
    CarSpecs(mass=2100., wheelbase=2.960, steerRatio=15.0),
    {
      Bus.chassis: 'tesla_can',
      Bus.party: 'tesla_can',
      Bus.pt: 'tesla_powertrain',
      Bus.radar: 'tesla_radar_bosch_generated',
    },
  )
  TESLA_MODEL_X_HW3 = TeslaPlatformConfig(
    [CarDocs("Tesla Model X (with HW3) 2021-23", "All", car_parts=CarParts.common([CarHarness.tesla_model_sx_hw3]))],
    CarSpecs(mass=2468., wheelbase=2.960, steerRatio=15.0),
    {
      Bus.chassis: 'tesla_can',
      Bus.party: 'tesla_raven_party',
      Bus.pt: 'tesla_powertrain',
      Bus.radar: 'tesla_radar_continental_generated',
    },
  )
  TESLA_MODEL_S_HW3 = TeslaPlatformConfig(
    [CarDocs("Tesla Model S (with HW3) 2020-23", "All", car_parts=CarParts.common([CarHarness.tesla_model_sx_hw3]))],
    CarSpecs(mass=2100., wheelbase=2.960, steerRatio=15.0),
    {
      Bus.chassis: 'tesla_can',
      Bus.party: 'tesla_raven_party',
      Bus.pt: 'tesla_powertrain',
      Bus.radar: 'tesla_radar_continental_generated',
    },
  )


LEGACY_CARS = (CAR.TESLA_MODEL_S_HW1, CAR.TESLA_MODEL_S_HW2, CAR.TESLA_MODEL_S_HW3, CAR.TESLA_MODEL_X_HW1, CAR.TESLA_MODEL_X_HW2,
               CAR.TESLA_MODEL_X_HW3)
LEGACY_HW1_CARS = (CAR.TESLA_MODEL_S_HW1, CAR.TESLA_MODEL_X_HW1)
LEGACY_HW2_CARS = (CAR.TESLA_MODEL_S_HW2, CAR.TESLA_MODEL_X_HW2)
LEGACY_HW3_CARS = (CAR.TESLA_MODEL_S_HW3, CAR.TESLA_MODEL_X_HW3)
LEGACY_BOSCH_RADAR_CARS = LEGACY_HW1_CARS + LEGACY_HW2_CARS


# Cars with this EPS FW have a 2-bit DAS_steeringControlType and use TeslaFlags.LEGACY_DAS_STEERING
LEGACY_DAS_STEERING_FW = {
  CAR.TESLA_MODEL_3: [
    b'TeM3_E014p10_0.0.0 (16),E014.17.00',
    b'TeM3_E014p10_0.0.0 (16),EL014.17.00',
    b'TeM3_ES014p11_0.0.0 (25),ES014.19.0',
    b'TeMYG4_DCS_Update_0.0.0 (13),E4014.28.1',
    b'TeMYG4_DCS_Update_0.0.0 (9),E4014.26.0',
    b'TeMYG4_Legacy3Y_0.0.0 (2),E4015.02.0',
    b'TeMYG4_Legacy3Y_0.0.0 (5),E4015.03.2',
    b'TeMYG4_Legacy3Y_0.0.0 (5),E4L015.03.2',
    b'TeMYG4_Main_0.0.0 (59),E4H014.29.0',
    b'TeMYG4_Main_0.0.0 (65),E4H015.01.0',
    b'TeMYG4_Main_0.0.0 (67),E4H015.02.1',
    b'TeMYG4_SingleECU_0.0.0 (33),E4S014.27',
  ],
  CAR.TESLA_MODEL_Y: [
    b'TeM3_E014p10_0.0.0 (16),Y002.18.00',
    b'TeM3_E014p10_0.0.0 (16),YP002.18.00',
    b'TeM3_ES014p11_0.0.0 (16),YS002.17',
    b'TeM3_ES014p11_0.0.0 (25),YS002.19.0',
    b'TeMYG4_DCS_Update_0.0.0 (13),Y4002.27.1',
    b'TeMYG4_DCS_Update_0.0.0 (13),Y4P002.27.1',
    b'TeMYG4_DCS_Update_0.0.0 (9),Y4P002.25.0',
    b'TeMYG4_Legacy3Y_0.0.0 (2),Y4003.02.0',
    b'TeMYG4_Legacy3Y_0.0.0 (2),Y4P003.02.0',
    b'TeMYG4_Legacy3Y_0.0.0 (5),Y4003.03.2',
    b'TeMYG4_Legacy3Y_0.0.0 (5),Y4P003.03.2',
    b'TeMYG4_SingleECU_0.0.0 (28),Y4S002.23.0',
    b'TeMYG4_SingleECU_0.0.0 (33),Y4S002.26',
  ],
  CAR.TESLA_MODEL_X: [
    b'TeM3_SP_XP002p2_0.0.0 (23),XPR003.6.0',
    b'TeM3_SP_XP002p2_0.0.0 (36),XPR003.10.0',
  ],
}

# e.g. TeMYG4_Main_0.0.0 (87),Y4003.09.3
#      11111_22222_______33____45__666666
# 1 = EPS firmware program, 2 = build lineage, 3 = build number, 4 = model code,
# 5 = trim/hardware variant, 6 = series and software version
#
# Only the model code identifies the vehicle: 1 and 2 are shared across models (Model 3 and
# Model Y both ship TeM3_ and TeMYG4_ firmware) and 3 is only monotone within one lineage.
#
# HW4 gen2 inserts a hardware generation between the trim and the series:
# TeMYG4_Main_0.0.0 (78),Y4OC.E80.003.07.0
#                        44_55_gggg_666666
FW_PATTERN = re.compile(rb'^Te[A-Z0-9]+_[A-Za-z0-9_]+_0\.0\.0 \(\d+\),' +
                        rb'(?P<model>E4|E|Y4|Y|XP)[A-Z]{0,2}(?:\.(?P<gen>[A-Z]\d{2})\.)?' +
                        rb'(?P<series>\d{3})\.(?P<version>\d+(?:\.\d+)*)$')

# The EPS on a 2026+ Model Y answers 0xF181 with a binary application software id instead of an
# ASCII version string, so it can only ever be matched exactly.
HW4_GEN2_FW_PREFIX = b'\x01\x01'


def is_ascii_version_fw(fw: bytes) -> bool:
  return not fw.startswith(HW4_GEN2_FW_PREFIX)


def is_hw4_gen2_fw(fw: bytes) -> bool:
  if fw.startswith(HW4_GEN2_FW_PREFIX):
    return True

  match = FW_PATTERN.match(fw)
  return match is not None and match.group('gen') is not None


def get_platform_codes(fw_versions: list[bytes] | set[bytes]) -> set[tuple[bytes, bytes, tuple[int, ...]]]:
  codes = set()
  for fw in fw_versions:
    match = FW_PATTERN.match(fw)
    if match is not None:
      codes.add((match.group('model'), match.group('series'),
                 tuple(int(v) for v in match.group('version').split(b'.'))))

  return codes


@cache
def _das_steering_cutoffs() -> dict[tuple[str, bytes, bytes], tuple[tuple[int, ...] | None, tuple[int, ...] | None]]:
  """Per (platform, model code, series) family, the oldest known modern version and the newest
  known legacy version. Tesla only ever moves a family forward, so these bound the split."""
  # imported here because fingerprints.py imports this module
  from iqdbc.car.tesla.fingerprints import FW_VERSIONS

  legacy: defaultdict[tuple, set] = defaultdict(set)
  modern: defaultdict[tuple, set] = defaultdict(set)
  for platform, ecus in FW_VERSIONS.items():
    known_legacy = LEGACY_DAS_STEERING_FW.get(platform, [])
    for fws in ecus.values():
      for fw in fws:
        if is_hw4_gen2_fw(fw):
          continue
        for model, series, version in get_platform_codes([fw]):
          (legacy if fw in known_legacy else modern)[(platform, model, series)].add(version)

  return {k: (min(modern[k]) if k in modern else None, max(legacy[k]) if k in legacy else None)
          for k in set(legacy) | set(modern)}


def is_legacy_das_steering(candidate: str, fw: bytes) -> bool:
  """Whether an EPS FW uses the 2-bit DAS_steeringControlType. Unknown firmware newer than
  anything in a family is treated as modern: cars only move forward, and someone left behind
  on legacy software can force the platform with CarPlatformBundle."""
  if fw in LEGACY_DAS_STEERING_FW.get(candidate, []):
    return True

  # HW4 gen2 shipped long after the 3-bit DAS_steeringControlType and shares a series with
  # firmware that predates it, so it must not be compared against that series' cutoff
  if is_hw4_gen2_fw(fw):
    return False

  codes = get_platform_codes([fw])
  if not len(codes):
    return False

  model, series, version = next(iter(codes))
  first_modern, last_legacy = _das_steering_cutoffs().get((candidate, model, series), (None, None))
  if first_modern is not None:
    return version < first_modern

  return last_legacy is not None and version <= last_legacy


def match_fw_to_car_fuzzy(live_fw_versions: LiveFwVersions, vin: str, offline_fw_versions: OfflineFwVersions) -> set[str]:
  # Tesla fingerprints on the EPS alone and Ecu.eps is in FUZZY_EXCLUDE_ECUS, so the generic fuzzy
  # matcher can never match a Tesla. Match on the model code, which survives the EPS version bumps
  # that ship with Tesla software updates. The series is deliberately not required to be known:
  # Tesla bumps it within a platform (E4014 -> E4015, Y4002 -> Y4003).
  offline_codes: defaultdict[bytes, set[str]] = defaultdict(set)
  for candidate, ecus in offline_fw_versions.items():
    for fws in ecus.values():
      for model, _, _ in get_platform_codes(fws):
        offline_codes[model].add(candidate)

  candidates: set[str] = set()
  for ecu, addr, sub_addr in {e for ecus in offline_fw_versions.values() for e in ecus}:
    if ecu != Ecu.eps:
      continue
    for model, _, _ in get_platform_codes(live_fw_versions.get((addr, sub_addr), set())):
      candidates |= offline_codes[model]

  return candidates if len(candidates) == 1 else set()


FW_QUERY_CONFIG = FwQueryConfig(
  requests=[
    Request(
      [StdQueries.TESTER_PRESENT_REQUEST, StdQueries.SUPPLIER_SOFTWARE_VERSION_REQUEST],
      [StdQueries.TESTER_PRESENT_RESPONSE, StdQueries.SUPPLIER_SOFTWARE_VERSION_RESPONSE],
      bus=0,
    ),
    # the HW4 gen2 (2026+ Model Y) EPS doesn't answer 0xF195, only the binary application software id at 0xF181
    Request(
      [StdQueries.TESTER_PRESENT_REQUEST, StdQueries.UDS_VERSION_REQUEST],
      [StdQueries.TESTER_PRESENT_RESPONSE, StdQueries.UDS_VERSION_RESPONSE],
      bus=0,
    ),
    Request(
      [StdQueries.TESTER_PRESENT_REQUEST, StdQueries.MANUFACTURER_SOFTWARE_VERSION_REQUEST],
      [StdQueries.TESTER_PRESENT_RESPONSE, StdQueries.MANUFACTURER_SOFTWARE_VERSION_RESPONSE],
      bus=0,
    ),
  ],
  match_fw_to_car_fuzzy=match_fw_to_car_fuzzy,
)


class CANBUS:
  party = 0
  vehicle = 1
  autopilot_party = 2


@dataclass(frozen=True)
class LegacyCanBus:
  party: int
  chassis: int
  powertrain: int
  radar: int
  autopilot_party: int
  autopilot_powertrain: int


def get_legacy_canbus(candidate: str) -> LegacyCanBus:
  if candidate in LEGACY_HW1_CARS:
    return LegacyCanBus(party=0, chassis=0, powertrain=0, radar=1, autopilot_party=2, autopilot_powertrain=2)
  if candidate in LEGACY_HW3_CARS:
    return LegacyCanBus(party=0, chassis=1, powertrain=4, radar=5, autopilot_party=2, autopilot_powertrain=6)
  return LegacyCanBus(party=0, chassis=0, powertrain=4, radar=1, autopilot_party=2, autopilot_powertrain=6)


GEAR_MAP = {
  "DI_GEAR_INVALID": CarState.GearShifter.unknown,
  "DI_GEAR_P": CarState.GearShifter.park,
  "DI_GEAR_R": CarState.GearShifter.reverse,
  "DI_GEAR_N": CarState.GearShifter.neutral,
  "DI_GEAR_D": CarState.GearShifter.drive,
  "DI_GEAR_SNA": CarState.GearShifter.unknown,
}


# Add extra tolerance for average banked road since safety doesn't have the roll
AVERAGE_ROAD_ROLL = 0.06  # ~3.4 degrees, 6% superelevation. higher actual roll lowers lateral acceleration


class CarControllerParams:
  ANGLE_LIMITS: AngleSteeringLimits = AngleSteeringLimits(
    # EPAS faults above this angle
    360,  # deg
    # Tesla uses a vehicle model instead, check carcontroller.py for details
    ([], []),
    ([], []),

    # Vehicle model angle limits
    # Add extra tolerance for average banked road since safety doesn't have the roll
    MAX_LATERAL_ACCEL=ISO_LATERAL_ACCEL + (ACCELERATION_DUE_TO_GRAVITY * AVERAGE_ROAD_ROLL),  # ~3.6 m/s^2
    MAX_LATERAL_JERK=3.0 + (ACCELERATION_DUE_TO_GRAVITY * AVERAGE_ROAD_ROLL),  # ~3.6 m/s^3

    # limit angle rate to both prevent a fault and for low speed comfort (~12 mph rate down to 0 mph)
    MAX_ANGLE_RATE=5,  # deg/20ms frame, EPS faults at 12 at a standstill
  )

  STEER_STEP = 2  # Angle command is sent at 50 Hz
  ACCEL_MAX = 2.0    # m/s^2
  ACCEL_MIN = -3.48  # m/s^2
  JERK_LIMIT_MAX = 4.9  # m/s^3, ACC faults at 5.0
  JERK_LIMIT_MIN = -4.9  # m/s^3, ACC faults at 5.0
  JERK_UP = 1.0  # m/s^3
  JERK_RAMP_RATE = JERK_LIMIT_MAX * 0.002


class TeslaSafetyFlags(IntFlag):
  LONG_CONTROL = 1
  LEGACY_DAS_STEERING = 2
  HW4_GEN2 = 4


class TeslaLegacySafetyFlags(IntFlag):
  LONG_CONTROL = 1
  EXTERNAL_PANDA = 4
  HW1 = 8
  HW2 = 16
  HW3 = 32


class TeslaFlags(IntFlag):
  LONG_CONTROL = 1
  LEGACY_DAS_STEERING = 2
  MISSING_DAS_SETTINGS = 4
  # 2026+ Model Y (Juniper): DAS_status moved from 0x39b to 0x399 and UI_warning (0x311) is gone
  HW4_GEN2 = 8
  # blinkers and the seatbelt buckle are only on the VEHICLE bus, which needs its CAN lines tapped
  HW4_GEN2_VEHICLE_BUS = 16
  NO_SDM1 = 32


DBC = CAR.create_dbc_map()

STEER_THRESHOLD = 1
