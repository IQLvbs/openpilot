# Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
from pathlib import Path

from iqpilot.selfdrive.selfdrived.selfdrived import NON_BLOCKING_PROCESSES
from iqpilot.system.manager.process_config import managed_processes

WIREGUARD_GO = Path(__file__).resolve().parents[3] / "third_party" / "wireguard-go" / "larch64" / "wireguard-go"


def test_tunnel_daemon_never_blocks_engagement():
  assert "k3wgd" in NON_BLOCKING_PROCESSES


def test_tunnel_daemon_runs_under_hephaestusd_not_manager():
  # k3wgd moved off the manager process list to run under hephaestusd, inheriting its CAP_NET_ADMIN/CAP_NET_RAW; the manager must not double-launch it.
  assert "k3wgd" not in managed_processes


def test_wireguard_go_ships_with_the_tree():
  assert WIREGUARD_GO.is_file()
  assert WIREGUARD_GO.stat().st_size > 1_000_000
