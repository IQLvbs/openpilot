"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""
from types import SimpleNamespace

import pytest

from iqpilot.selfdrive.ui.mici.layouts.settings.network import network_layout
from iqpilot.selfdrive.ui.mici.layouts.settings.network.network_layout import NetworkLayoutMici
from iqpilot.system.ui.lib.cellular_info import CellularInfo
from iqpilot.system.ui.lib.speed_test import Phase, SpeedTestState
from iqpilot.system.wireguard.tunnel import TunnelState, TunnelStatus


class Card:
  def __init__(self):
    self.value = ""
    self.writes = 0
    self.checked = None

  def get_value(self):
    return self.value

  def set_value(self, value):
    self.value = value
    self.writes += 1

  def set_checked(self, checked):
    self.checked = checked


@pytest.fixture
def layout(monkeypatch):
  wg = SimpleNamespace(installed=lambda: True, enabled=lambda: True,
                       read_status=lambda: TunnelStatus(TunnelState.CONNECTED, "10.8.0.7/32", handshake=1.0),
                       describe_compact=network_layout.wireguard.describe_compact)
  monkeypatch.setattr(network_layout, "wireguard", wg)
  view = object.__new__(NetworkLayoutMici)
  view._cellular = SimpleNamespace(poll=lambda: CellularInfo("T-Mobile", "LTE", 70, "100.64.12.9", "2607:fb90::5"))
  view._speed_test = SimpleNamespace(state=SpeedTestState(Phase.DONE, 1.0, 31.6, 84.2, 12.7))
  for name in ("_carrier_btn", "_cell_ip_btn", "_cell_ipv6_btn", "_speed_test_btn", "_wireguard_btn"):
    setattr(view, name, Card())
  view._esim_profile_frame = 0
  view._wireguard_installed = False
  return view, wg


def test_cards_show_carrier_addresses_speed_and_tunnel(layout):
  view, _ = layout
  view._update_status_cards()
  assert view._carrier_btn.value == "T-Mobile • LTE"
  assert view._cell_ip_btn.value == "100.64.12.9"
  assert view._cell_ipv6_btn.value == "2607:fb90::5"
  assert view._speed_test_btn.value == "84.2 / 12.7 Mbps"
  assert view._wireguard_installed and view._wireguard_btn.checked is True
  assert view._wireguard_btn.value == "10.8.0.7"


def test_cards_only_relayout_when_the_text_changes(layout):
  view, _ = layout
  view._update_status_cards()
  view._esim_profile_frame = 1
  view._update_status_cards()
  assert view._carrier_btn.writes == 1 and view._speed_test_btn.writes == 1


def test_empty_modem_and_no_result_fall_back(layout):
  view, wg = layout
  view._cellular = SimpleNamespace(poll=lambda: CellularInfo())
  view._speed_test = SimpleNamespace(state=SpeedTestState())
  wg.installed = lambda: False
  view._update_status_cards()
  assert view._carrier_btn.value == "no modem"
  assert view._cell_ip_btn.value == "not connected"
  assert view._speed_test_btn.value == "tap to run"
  assert not view._wireguard_installed and view._wireguard_btn.value == ""
