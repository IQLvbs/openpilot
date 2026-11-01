"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""
import socket
import time
from types import SimpleNamespace

import pytest

from iqpilot.system.ui.lib.cellular_info import (
  MM_MODEM, MM_MODEM_3GPP, CellularInfo, CellularInfoPoller, parse_modem, pick_addresses, read_cellular_info, technology_name,
)


def modem(name="T-Mobile", code="310260", tech=1 << 14, quality=(78, True)):
  return {
    "/org/freedesktop/ModemManager1/Modem/0": {
      MM_MODEM: {"AccessTechnologies": ("u", tech), "SignalQuality": ("(ub)", quality)},
      MM_MODEM_3GPP: {"OperatorName": ("s", name), "OperatorCode": ("s", code)},
    },
  }


@pytest.mark.parametrize("mask,name", [
  ((1 << 15) | (1 << 14), "5G"), (1 << 14, "LTE"), (1 << 9, "HSPA"), (1 << 5, "3G"), (1 << 4, "EDGE"), (1 << 1, "2G"), (0, ""),
])
def test_technology_names_pick_the_best_bearer(mask, name):
  assert technology_name(mask) == name


def test_parses_carrier_technology_and_signal():
  assert parse_modem(modem()) == ("T-Mobile", "LTE", 78)


def test_falls_back_to_the_operator_code_and_hides_stale_signal():
  assert parse_modem(modem(name="", quality=(40, False))) == ("MCC-MNC 310260", "LTE", None)
  assert parse_modem(modem(name="", code="")) == ("", "LTE", 78)


def test_no_modem_reads_empty():
  assert parse_modem({"/org/freedesktop/ModemManager1": {}}) == ("", "", None)


def addr(family, address):
  return SimpleNamespace(family=family, address=address)


def test_skips_link_local_ipv6():
  addrs = [addr(socket.AF_INET6, "fe80::1%wwan0"), addr(socket.AF_INET, "100.64.12.9"), addr(socket.AF_INET6, "2607:fb90:1::5")]
  assert pick_addresses(addrs) == ("100.64.12.9", "2607:fb90:1::5")
  assert pick_addresses([]) == ("", "")


def test_read_survives_a_missing_modem_and_interface():
  def broken():
    raise OSError("no system bus")
  assert read_cellular_info(broken, broken) == CellularInfo()
  info = read_cellular_info(lambda: modem(), lambda: ("100.64.12.9", ""))
  assert info.carrier_text == "T-Mobile \u2022 LTE" and info.ipv4 == "100.64.12.9"


def test_poller_refreshes_off_the_render_thread_and_throttles():
  calls = []
  poller = CellularInfoPoller(lambda: calls.append(1) or CellularInfo("Visible", "5G"), interval=60)
  assert poller.poll() == CellularInfo()
  deadline = time.monotonic() + 2
  while poller.info == CellularInfo() and time.monotonic() < deadline:
    time.sleep(0.01)
  assert poller.info.carrier_text == "Visible \u2022 5G"
  poller.poll()
  assert len(calls) == 1
