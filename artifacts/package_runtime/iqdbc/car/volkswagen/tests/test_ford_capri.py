# Copyright © IQ.Lvbs, part of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
import pytest

from iqdbc.car import Bus, structs
from iqdbc.car.fw_versions import match_fw_to_car
from iqdbc.car.volkswagen.interface import CarInterface
from iqdbc.car.volkswagen.values import CAR, DBC, VolkswagenFlags

GEOMETRY_AND_IDENTITY = {
  "carFingerprint", "mass", "rotationalInertia", "centerToFront", "steerRatio",
  "tireStiffnessFront", "tireStiffnessRear", "wheelbase",
}

MEB_STEER_RATIOS = {
  CAR.VOLKSWAGEN_ID3_MK1: 18.4,
  CAR.VOLKSWAGEN_ID3_MK2: 18.4,
  CAR.VOLKSWAGEN_ID4_MK1: 18.4,
  CAR.VOLKSWAGEN_ID4_MK2: 18.4,
  CAR.AUDI_Q4_MK1: 18.4,
  CAR.AUDI_Q4_MK2: 18.4,
  CAR.CUPRA_BORN_MK1: 18.9,
  CAR.SKODA_ENYAQ_MK1: 18.4,
  CAR.SKODA_ENYAQ_MK2: 18.4,
  CAR.FORD_CAPRI_MK1: 18.7,
}


def car_params(candidate):
  fingerprint = {bus: {} for bus in range(7)}
  return CarInterface.get_params(candidate, fingerprint, [], alpha_long=False, is_release=False, docs=False).to_dict()


def test_capri_is_its_own_gen2_platform():
  config = CAR.FORD_CAPRI_MK1.config
  assert config.flags & VolkswagenFlags.MEB
  assert config.flags & VolkswagenFlags.MEB_GEN2
  assert DBC[CAR.FORD_CAPRI_MK1][Bus.pt] == DBC[CAR.VOLKSWAGEN_ID4_MK2][Bus.pt]
  assert {doc.name for doc in config.car_docs} == {"Ford Capri 2024-25", "Ford Explorer EV 2024-25"}
  assert car_params(CAR.FORD_CAPRI_MK1)["carFingerprint"] == "FORD_CAPRI_MK1"


def test_capri_behaves_like_every_other_gen2_platform_apart_from_geometry():
  reference = car_params(CAR.VOLKSWAGEN_ID4_MK2)
  capri = car_params(CAR.FORD_CAPRI_MK1)
  keys = (set(reference) | set(capri)) - GEOMETRY_AND_IDENTITY
  assert {key: (reference.get(key), capri.get(key)) for key in keys if reference.get(key) != capri.get(key)} == {}


@pytest.mark.parametrize("platform,expected", MEB_STEER_RATIOS.items(), ids=lambda value: str(value))
def test_steer_ratios_are_pinned_per_platform(platform, expected):
  assert car_params(platform)["steerRatio"] == pytest.approx(expected, abs=1e-4)


def test_every_meb_platform_has_a_pinned_steer_ratio():
  meb = {platform for platform in CAR if platform.config.flags & VolkswagenFlags.MEB}
  assert meb == set(MEB_STEER_RATIOS)



def live_radar_only(version):
  radar = structs.CarParams.CarFw()
  radar.ecu = structs.CarParams.Ecu.fwdRadar
  radar.address = 0x757
  radar.fwVersion = version
  radar.brand = "volkswagen"
  return [radar]


def test_the_capri_is_recognized_from_the_only_ecu_its_camera_harness_answers():
  matched, candidates = match_fw_to_car(live_radar_only(b"\xf1\x871EE907567  \xf1\x890683"), "0" * 17, log=False)
  assert matched
  assert candidates == {str(CAR.FORD_CAPRI_MK1)}


def test_a_different_radar_version_does_not_become_a_capri():
  matched, candidates = match_fw_to_car(live_radar_only(b"\xf1\x871EE907567  \xf1\x890999"), "0" * 17, log=False)
  assert str(CAR.FORD_CAPRI_MK1) not in candidates
