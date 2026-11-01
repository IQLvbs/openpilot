"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "tools", "scripts", "iqemac"))

from make_big_selector import WIRE_CONTRACTS, _contracts_from

AUX = {"main_deg": None, "main_pp": "calibrated", "scale": 1,
       "wide_deg": None, "wide_pp": "calibrated", "wide_proj": "rectilinear"}


def test_contracts_are_taken_from_the_checkpoint_not_written_by_hand():
  # The runner compares the entry against the metadata it reads off the onnx and refuses the
  # handshake on any difference, so every one of these has to come from the same place.
  props = {
    "rh_aux": json.dumps(AUX),
    "rh_hires": json.dumps({"design": "emac_full", "shape": [1, 6, 604, 964]}),
    "map_tile_style": "iqmodel-map-raster-v3",
    "speed_limit_contract": "rh-speed-limit-v1",
    "drive_profile_contract": "rh-drive-profile-v1",
    "model_checkpoint": "iqmodel/whatever",
  }
  got = _contracts_from(props)
  assert set(got) == set(WIRE_CONTRACTS)
  assert got["rh_aux"] == AUX
  assert got["rh_hires"]["shape"] == [1, 6, 604, 964]
  assert got["map_tile_style"] == "iqmodel-map-raster-v3"


def test_a_null_view_angle_survives_the_round_trip():
  # main_deg/wide_deg being null is what selects the device-rendered aux over the full-FOV path,
  # so a truthiness test here would silently change which aux the model is fed.
  got = _contracts_from({"rh_aux": json.dumps(AUX)})
  assert "main_deg" in got["rh_aux"]
  assert got["rh_aux"]["main_deg"] is None
  assert got["rh_aux"]["wide_deg"] is None


def test_contracts_the_checkpoint_does_not_declare_are_left_out():
  assert _contracts_from({"model_checkpoint": "x"}) == {}
  got = _contracts_from({"map_tile_style": "iqmodel-map-raster-v1"})
  assert got == {"map_tile_style": "iqmodel-map-raster-v1"}
