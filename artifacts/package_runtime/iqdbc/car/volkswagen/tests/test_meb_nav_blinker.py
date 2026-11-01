"""Copyright (c) IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved."""

from iqdbc.can import CANPacker, CANParser
from iqdbc.car.volkswagen import mebcan
from iqdbc.car.volkswagen.carcontroller import ea_blinker_command, ea_send_ready, next_ea_counter


EA_HUD_VALUES = {
  "COUNTER": 0,
  "EA_Texte": 0,
  "ACF_Lampe_Hands_Off": 0,
  "EA_Infotainment_Anf": 0,
  "EA_Tueren_Anf": 0,
  "EA_Innenraumlicht_Anf": 0,
  "zFAS_Warnblinken": 0,
  "STP_Primaeranz": 0,
  "EA_Bremslichtblinken": 0,
  "EA_Blinken": 0,
  "EA_Unknown": 0,
}
EA_CONTROL_VALUES = {"EA_Funktionsstatus": 2}


def blinker_values(dbc, requests):
  packer = CANPacker(dbc)
  parser = CANParser(dbc, [("EA_02", 50)], 0)
  values = []
  for counter, stock_blinker, left, right in requests:
    ea_hud_values = {**EA_HUD_VALUES, "COUNTER": counter, "EA_Blinken": stock_blinker}
    msg = mebcan.create_blinker_control(
      packer, 0, ea_hud_values, EA_CONTROL_VALUES, left, right, False,
    )
    parser.update([0, [msg]])
    values.append((int(parser.vl["EA_02"]["COUNTER"]), int(parser.vl["EA_02"]["EA_Blinken"])))
  return values


def blinker_value(dbc, counter, stock_blinker, left, right, override_counter=None):
  packer = CANPacker(dbc)
  parser = CANParser(dbc, [("EA_02", 50)], 0)
  ea_hud_values = {**EA_HUD_VALUES, "COUNTER": counter, "EA_Blinken": stock_blinker}
  msg = mebcan.create_blinker_control(
    packer, 0, ea_hud_values, EA_CONTROL_VALUES, left, right, False, override_counter,
  )
  parser.update([0, [msg]])
  return int(parser.vl["EA_02"]["COUNTER"]), int(parser.vl["EA_02"]["EA_Blinken"])


def test_nav_blinker_request_remains_asserted_until_released():
  requests = [(counter % 16, 0, True, False) for counter in range(100)] + [(4, 0, False, False)]
  assert blinker_values("vw_meb", requests) == [(counter % 16, 1) for counter in range(100)] + [(4, 0)]


def test_nav_blinker_directions_on_meb_variants():
  requests = [(7, 0, True, False), (8, 0, False, True), (9, 0, False, False)]
  for dbc in ("vw_meb", "vw_meb_2024", "vw_mqbevo"):
    assert blinker_values(dbc, requests) == [(7, 1), (8, 2), (9, 0)]


def test_stock_blinker_has_priority_over_nav_request():
  requests = [(10, 1, False, True), (11, 2, True, False), (12, 3, True, False)]
  assert blinker_values("vw_meb", requests) == [(10, 1), (11, 2), (12, 3)]


def test_ea_frame_is_sent_once_per_stock_counter():
  last_counter = None
  sends = []
  for counter in (5, 5, 5, 6, 6, 7):
    stock_values = {"COUNTER": counter}
    if ea_send_ready(stock_values, last_counter):
      sends.append(counter)
      last_counter = counter
  assert sends == [5, 6, 7]


def test_blinker_counter_can_continue_between_stock_frames():
  assert blinker_value("vw_meb", 8, 0, True, False, 9) == (9, 1)
  assert blinker_value("vw_meb", 8, 0, True, False, 10) == (10, 1)


def test_stock_blinker_priority_survives_counter_override():
  assert blinker_value("vw_meb", 8, 2, True, False, 9) == (9, 2)


def test_nav_blinker_retriggers_only_between_lamp_cycles():
  assert ea_blinker_command(True, False, False, False) == (True, False)
  assert ea_blinker_command(True, False, True, False) == (False, False)
  assert ea_blinker_command(True, False, False, False) == (True, False)
  assert ea_blinker_command(False, True, False, True) == (False, False)


def test_blinker_counter_runs_at_20_ms():
  tx_counter = None
  counters = []
  for _ in range(10):
    tx_counter = next_ea_counter(tx_counter, 14)
    counters.append(tx_counter)
  assert counters == [15, 0, 1, 2, 3, 4, 5, 6, 7, 8]
