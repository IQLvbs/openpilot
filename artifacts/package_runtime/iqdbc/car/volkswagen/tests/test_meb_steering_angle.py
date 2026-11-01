# Copyright © IQ.Lvbs, part of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
import json
import os

import pytest

from iqdbc.can import CANParser
from iqdbc.car import Bus
from iqdbc.car.volkswagen.values import CAR, VolkswagenFlags

with open(os.path.join(os.path.dirname(__file__), "meb_angle_frames.json")) as f:
  FRAMES = json.load(f)
MEB_PLATFORMS = [p for p in CAR if p.config.flags & VolkswagenFlags.MEB]
CASES = [(name, pair) for name, group in FRAMES.items() for pair in group["frames"]]
EPS_AGREEMENT = 0.02
EPS_ABS_TOLERANCE_DEG = 0.6


def signed(vl, message, magnitude, sign):
  return vl[message][magnitude] * (-1 if vl[message][sign] else 1)


def decode(platform, lwi_hex, eps_hex):
  parser = CANParser(platform.config.dbc_dict[Bus.pt], [("LWI_01", 0), ("LH_EPS_03", 0)], 0)
  parser.update([(0, [(0x86, bytes.fromhex(lwi_hex), 0), (0x9F, bytes.fromhex(eps_hex), 0)])])
  lwi = signed(parser.vl, "LWI_01", "LWI_Lenkradwinkel", "LWI_VZ_Lenkradwinkel")
  eps = signed(parser.vl, "LH_EPS_03", "EPS_Berechneter_LW", "EPS_VZ_BLW")
  return lwi, eps


@pytest.mark.parametrize("platform", MEB_PLATFORMS, ids=lambda p: str(p))
def test_lwi_scale_matches_the_eps_computed_angle_on_real_frames(platform):
  for _, (lwi_hex, eps_hex) in CASES:
    lwi, eps = decode(platform, lwi_hex, eps_hex)
    assert abs(lwi - eps) <= max(EPS_ABS_TOLERANCE_DEG, EPS_AGREEMENT * abs(eps)), (lwi_hex, eps_hex, lwi, eps)


def test_fixtures_cover_both_generations_and_large_angles():
  assert {"gen1", "gen2", "gen2_id4"} <= set(FRAMES)
  largest = max(
    abs(decode(CAR.SKODA_ENYAQ_MK1, lwi_hex, eps_hex)[0]) for lwi_hex, eps_hex in FRAMES["gen1"]["frames"]
  )
  assert largest > 300


def test_a_wrong_scale_would_be_caught():
  lwi, eps = decode(CAR.SKODA_ENYAQ_MK1, *FRAMES["gen1"]["frames"][0])
  assert abs(lwi * 0.0843 / 0.1 - eps) > max(EPS_ABS_TOLERANCE_DEG, EPS_AGREEMENT * abs(eps))
