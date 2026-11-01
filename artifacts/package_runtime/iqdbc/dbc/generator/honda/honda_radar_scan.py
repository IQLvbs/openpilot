#!/usr/bin/env python3
"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos
"""
import os

SCAN_SLOTS = 16

FRAME_SIGNALS = {
  "POS": [
    ("SCAN_STATE", 15, 4),
    ("CYCLE", 27, 4),
    ("DIST_RAW", 23, 12),
    ("BEARING_RAW", 39, 11),
    ("DIST_SIGMA_RAW", 7, 7),
  ],
  "SHAPE": [
    ("CYCLE", 28, 4),
    ("PRESENCE_RAW", 46, 7),
  ],
  "LIFE": [
    ("CYCLE", 11, 4),
    ("AGE_RAW", 7, 12),
  ],
  "IDENT": [
    ("CYCLE", 12, 4),
    ("OBJECT_HANDLE", 55, 8),
  ],
  "MOTION": [
    ("CYCLE", 12, 4),
    ("CLOSING_SPEED_RAW", 7, 11),
    ("CLOSING_SPEED_SIGMA_RAW", 23, 10),
    ("DIST_RATIO_RAW", 55, 10),
  ],
}

QUARTET_KINDS = ("POS", "SHAPE", "LIFE", "IDENT")


def quartet_base_address(slot: int) -> int:
  return 0x280 + 4 * slot if slot < 4 else 0x2D0 + 4 * (slot - 4)


def motion_address(slot: int) -> int:
  return 0x2C8 + slot if slot < 8 else 0x290 + (slot - 8)


def frame_address(slot: int, kind: str) -> int:
  if kind == "MOTION":
    return motion_address(slot)
  return quartet_base_address(slot) + QUARTET_KINDS.index(kind)


HEADER = """VERSION "honda_radar_scan -- Honda Bosch radar object scan: 16 object slots, each a POS/SHAPE/LIFE/IDENT \
frame quartet plus a synchronized MOTION companion. Raw counts only; physical conversion lives in \
iqdbc/car/honda/radar_scan.py"

NS_ :
\tNS_DESC_
\tCM_
\tBA_DEF_
\tBA_
\tVAL_
\tCAT_DEF_
\tCAT_
\tFILTER
\tBA_DEF_DEF_
\tEV_DATA_
\tENVVAR_DATA_
\tSGTYPE_
\tSGTYPE_VAL_
\tBA_DEF_SGTYPE_
\tBA_SGTYPE_
\tSIG_TYPE_REF_
\tVAL_TABLE_
\tSIG_GROUP_
\tSIG_VALTYPE_
\tSIGTYPE_VALTYPE_
\tBO_TX_BU_
\tBA_DEF_REL_
\tBA_REL_
\tBA_DEF_DEF_REL_
\tBU_SG_REL_
\tBU_EV_REL_
\tBU_BO_REL_
\tSG_MUL_VAL_

BS_:

BU_: SCANNER XXX

"""


def render() -> str:
  parts = [HEADER]
  for slot in range(SCAN_SLOTS):
    for kind in (*QUARTET_KINDS, "MOTION"):
      addr = frame_address(slot, kind)
      parts.append(f"\nBO_ {addr} RADAR_SCAN_{slot:02d}_{kind}: 8 SCANNER\n")
      for name, start_bit, size in FRAME_SIGNALS[kind]:
        parts.append(f" SG_ {name} : {start_bit}|{size}@0+ (1,0) [0|{(1 << size) - 1}] \"\" XXX\n")
  return "".join(parts)


if __name__ == "__main__":
  out_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "honda_radar_scan.dbc")
  with open(out_path, "w", encoding="utf-8") as f:
    f.write(render())
