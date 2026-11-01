"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""
import json
from pathlib import Path

import pytest

from iqpilot.system.wireguard import tunnel as wg
from iqpilot.system.wireguard import wireguardd
from iqpilot.system.wireguard.tunnel import TunnelState, TunnelStatus
from iqpilot.system.wireguard.wireguardd import HANDSHAKE_CHECK_S, HANDSHAKE_STALE_S, MAX_RETRY_S, RETRY_S, REUP_S, Tunnel

ENV = """
# hub
WG_ADDRESS=10.8.0.7/32
WG_PEER_PUBKEY=hubkey=
WG_ENDPOINT=hub.teal.test:51820
WG_ALLOWED_IPS=10.8.0.0/24, 0.0.0.0/0
WG_KEEPALIVE=15
"""
HUB_IP = "203.0.113.9"


class FakeParams:
  def __init__(self, values=None):
    self.values = dict(values or {})

  def get(self, key, return_default=False):
    if key not in self.values and return_default and key == wg.PARAM_ENABLED:
      return True
    return self.values.get(key)

  def put_bool(self, key, value):
    self.values[key] = bool(value)


class FakeHost:
  def __init__(self):
    self.params = FakeParams()
    self.now = 1000.0
    self.ifaces: set[str] = set()
    self.routes: dict[str, str] = {}
    self.handshake = 0
    self.calls: list[list[str]] = []
    self.fail: str | None = None
    self.endpoint_ips: tuple[str, ...] = (HUB_IP,)
    self.setconf = ""

  def run(self, args):
    self.calls.append(args)
    if self.fail and self.fail in " ".join(args):
      return 1, "boom"
    if args[0] == "env":
      binary, iface = args[-2], args[-1]
      assert binary.endswith("/wireguard-go") and args[args.index("nice") + 3] == binary
      self.ifaces.add(iface)
      return 0, ""
    if args[:3] == ["ip", "link", "del"]:
      self.ifaces.discard(args[3])
      self.routes = {r: d for r, d in self.routes.items() if d != args[3]}
    if args[:3] == ["ip", "route", "add"]:
      if args[3] in self.routes:
        return 2, "RTNETLINK answers: File exists"
      self.routes[args[3]] = args[5]
    if args[0].endswith("/wg") and args[1] == "setconf":
      self.setconf = Path(args[3]).read_text()
    if args[0].endswith("/wg") and args[1] == "show":
      return 0, f"hubkey=\t{self.handshake}"
    return 0, ""

  def launches(self):
    return sum(c[0] == "env" for c in self.calls)

  def handshake_reads(self):
    return sum(c[0].endswith("/wg") and c[1] == "show" for c in self.calls)

  def tunnel(self, root):
    return Tunnel(root, run=self.run, clock=lambda: self.now, has_iface=lambda i: i in self.ifaces,
                  resolver=lambda _h: self.endpoint_ips, sleep=lambda s: setattr(self, "now", self.now + s), params=self.params)


@pytest.fixture
def root(tmp_path):
  root = tmp_path / "wg"
  root.mkdir()
  for name in ("wireguard-go", "wg"):
    (root / name).touch()
  (root / "privatekey").write_text("devicekey=\n")
  (root / "publickey").write_text("devicepub=\n")
  (root / "wg.env").write_text(ENV)
  return root


@pytest.fixture
def host():
  return FakeHost()


def test_config_parsing(root):
  config = wg.load_config(root, FakeParams())
  assert config.endpoint_host == "hub.teal.test" and config.endpoint_port == 51820
  assert config.allowed_ips == ("10.8.0.0/24", "0.0.0.0/0")
  assert config.keepalive == 15


def test_default_routes_and_routes_covering_the_hub_are_never_installed():
  config = wg.WireGuardConfig("10.8.0.7/32", "k", "hub:51820",
                              ("10.8.0.0/24", "0.0.0.0/0", "::/0", "203.0.113.0/24", "fd00::/8", "junk"))
  assert config.routes((HUB_IP,)) == (("10.8.0.0/24", "fd00::/8"), ("0.0.0.0/0", "::/0", "203.0.113.0/24", "junk"))


@pytest.mark.parametrize("env", ["", wg.PLACEHOLDERS[0], "WG_ADDRESS=10.8.0.7/32\n", ENV.replace(":51820", ":notaport")])
def test_example_or_incomplete_config_is_unconfigured(root, host, env):
  (root / "wg.env").write_text(env if "WG_" in env else ENV.replace("hubkey=", env))
  status = host.tunnel(root).step()
  assert status.state == TunnelState.UNCONFIGURED
  assert status.public_key == "devicepub="
  assert host.launches() == 0


def test_brings_the_tunnel_up_niced_and_memory_capped(root, host):
  t = host.tunnel(root)
  assert t.step().state == TunnelState.CONNECTING
  launch = next(c for c in host.calls if c[0] == "env")
  assert launch[:6] == ["env", "GOMAXPROCS=2", "GOMEMLIMIT=48MiB", "nice", "-n", "10"]
  assert "PrivateKey = devicekey=" in host.setconf and "AllowedIPs = 10.8.0.0/24, 0.0.0.0/0" in host.setconf
  assert not (root / "wg0.setconf").exists()
  assert host.routes == {"10.8.0.0/24": "wg0"}
  assert not any(c[:3] == ["ip", "route", "replace"] for c in host.calls)

  host.handshake = int(host.now)
  host.now += 5
  status = t.step()
  assert status.state == TunnelState.CONNECTED
  assert status.address == "10.8.0.7/32" and status.handshake == host.handshake
  assert "0.0.0.0/0" in status.error


def test_an_existing_lan_route_is_left_alone(root, host):
  host.routes["10.8.0.0/24"] = "wlan0"
  status = host.tunnel(root).step()
  assert host.routes == {"10.8.0.0/24": "wlan0"}
  assert "already routed elsewhere: 10.8.0.0/24" in status.error


def test_only_our_own_wireguard_go_is_stopped(root, host):
  t = host.tunnel(root)
  t.step()
  wg.set_enabled(False, host.params)
  t.step()
  kills = [c for c in host.calls if c[0] == "pkill"]
  assert kills and all(c == ["pkill", "-xf", f"{root}/wireguard-go wg0"] for c in kills)


def test_handshake_is_read_rarely_once_connected(root, host):
  t = host.tunnel(root)
  t.step()
  host.handshake = int(host.now)
  for _ in range(12):
    host.now += 5
    assert t.step().state == TunnelState.CONNECTED
  assert host.handshake_reads() <= 60 / HANDSHAKE_CHECK_S + 2
  assert not any(c[0] == "pgrep" for c in host.calls)


def test_stale_handshake_re_ups_after_the_grace_period(root, host):
  t = host.tunnel(root)
  t.step()
  host.handshake = int(host.now)
  host.now += 1
  assert t.step().state == TunnelState.CONNECTED
  host.now += HANDSHAKE_STALE_S + HANDSHAKE_CHECK_S
  assert t.step().state == TunnelState.STALE
  assert host.launches() == 2
  host.now += REUP_S - 1
  assert t.step().state == TunnelState.STALE
  assert host.launches() == 2
  host.now += 1
  assert t.step().state == TunnelState.STALE
  assert host.launches() == 3
  host.handshake = int(host.now)
  host.now += 5
  assert t.step().state == TunnelState.CONNECTED


def test_crashed_tunnel_is_restarted(root, host):
  t = host.tunnel(root)
  t.step()
  host.ifaces.clear()
  host.now += 1
  assert t.step().state == TunnelState.CONNECTING
  assert host.launches() == 1
  host.now += RETRY_S
  t.step()
  assert host.launches() == 2 and "wg0" in host.ifaces


def test_repeated_failures_back_off_to_five_minutes(root, host):
  host.fail = "setconf"
  t = host.tunnel(root)
  gaps = []
  last = None
  for _ in range(4000):
    before = host.launches()
    t.step()
    if host.launches() != before:
      if last is not None:
        gaps.append(host.now - last)
      last = host.now
    host.now += 5
  assert gaps[0] >= RETRY_S * 2 - 5
  assert gaps == sorted(gaps)
  assert max(gaps) <= MAX_RETRY_S + 5
  assert not host.ifaces


def test_waits_for_the_network_before_dialing(root, host):
  host.endpoint_ips = ()
  status = host.tunnel(root).step()
  assert status.state == TunnelState.WAITING and "hub.teal.test" in status.error
  assert host.launches() == 0


def test_failed_bring_up_tears_down_and_reports_the_error(root, host):
  host.fail = "setconf"
  status = host.tunnel(root).step()
  assert status.state == TunnelState.ERROR and "boom" in status.error
  assert "wg0" not in host.ifaces
  assert not (root / "wg0.setconf").exists()


def test_disable_flag_takes_the_tunnel_down(root, host):
  t = host.tunnel(root)
  t.step()
  assert wg.set_enabled(False, host.params)
  assert t.step().state == TunnelState.OFF
  assert "wg0" not in host.ifaces
  assert wg.set_enabled(True, host.params)
  host.now += RETRY_S
  assert t.step().state == TunnelState.CONNECTING and "wg0" in host.ifaces


def test_toggle_never_raises_when_params_fail():
  class Broken:
    def get(self, *a, **k):
      raise OSError("params unavailable")

    def put_bool(self, *a):
      raise OSError("params unavailable")
  assert wg.set_enabled(False, Broken()) is False
  assert wg.enabled(Broken()) is True


def test_enabled_defaults_on():
  assert wg.enabled(FakeParams()) is True
  assert wg.enabled(FakeParams({wg.PARAM_ENABLED: False})) is False


def test_app_config_param_takes_precedence_over_wg_env(root, host):
  host.params.values[wg.PARAM_CONFIG] = {"address": "10.9.0.4/32", "peerPublicKey": "apphub=", "endpoint": "app.hub.test:4500",
                                         "allowedIps": ["10.9.0.0/24", "10.10.0.0/16"], "keepalive": 20}
  status = host.tunnel(root).step()
  assert status.address == "10.9.0.4/32" and status.endpoint == "app.hub.test:4500"
  assert "PublicKey = apphub=" in host.setconf and "PersistentKeepalive = 20" in host.setconf
  assert host.routes == {"10.9.0.0/24": "wg0", "10.10.0.0/16": "wg0"}


@pytest.mark.parametrize("config,problem", [
  ({"address": "10.9.0.4/32", "endpoint": "hub:1", "allowedIps": ["10.9.0.0/24"]}, "missing hub public key"),
  ({"address": "nope", "peerPublicKey": "k=", "endpoint": "hub:1", "allowedIps": ["10.9.0.0/24"]}, "invalid value"),
  ({"address": "10.9.0.4/32", "peerPublicKey": "k=", "endpoint": "hub", "allowedIps": ["10.9.0.0/24"]}, "invalid value"),
  ({"address": "10.9.0.4/32", "peerPublicKey": "k=", "endpoint": "hub:70000", "allowedIps": ["10.9.0.0/24"]}, "endpoint must be host:port"),
  ({"address": "10.9.0.4/32", "peerPublicKey": "k=", "endpoint": "hub:1", "allowedIps": ["10.9.0.0/24"], "keepalive": 99999},
   "keepalive must be"),
])
def test_bad_app_config_is_unconfigured_with_a_reason(root, host, config, problem):
  host.params.values[wg.PARAM_CONFIG] = config
  status = host.tunnel(root).step()
  assert status.state == TunnelState.UNCONFIGURED and problem in status.error
  assert host.launches() == 0


def test_restarted_supervisor_adopts_a_running_tunnel(root, host):
  host.tunnel(root).step()
  host.handshake = int(host.now)
  launches = host.launches()
  assert host.tunnel(root).step().state == TunnelState.CONNECTED
  assert host.launches() == launches


def test_status_is_published_on_change_or_heartbeat_only(tmp_path):
  path = tmp_path / "status.json"
  first = TunnelStatus(TunnelState.CONNECTED, "10.8.0.7/32", updated=100.0)
  assert wireguardd.publish(first, None, path)
  assert not wireguardd.publish(TunnelStatus(TunnelState.CONNECTED, "10.8.0.7/32", updated=105.0), first, path)
  assert wireguardd.publish(TunnelStatus(TunnelState.CONNECTED, "10.8.0.7/32", updated=100.0 + wireguardd.HEARTBEAT_S), first, path)
  assert wireguardd.publish(TunnelStatus(TunnelState.STALE, "10.8.0.7/32", updated=101.0), first, path)
  assert wireguardd.HEARTBEAT_S < wg.STATUS_STALE_S
  assert str(wg.WG_STATUS).startswith("/dev/shm/")


def test_ui_reads_status_and_flags_a_dead_supervisor(root, tmp_path):
  path = tmp_path / "status.json"
  status = TunnelStatus(TunnelState.CONNECTED, "10.8.0.7/32", "hub.teal.test:51820", 990.0, "devicepub=", updated=1000.0)
  wg.write_status(status, path)
  assert json.loads(path.read_text())["state"] == "connected"
  params = FakeParams()
  assert wg.read_status(root, now=1005.0, path=path, params=params) == status
  assert wg.read_status(root, now=1000.0 + wg.STATUS_STALE_S + 1, path=path, params=params).state == TunnelState.ERROR
  wg.set_enabled(False, params)
  assert wg.read_status(root, now=1005.0, path=path, params=params).state == TunnelState.OFF
  (root / "wg").unlink()
  assert wg.read_status(root, path=path, params=params).state == TunnelState.NOT_INSTALLED


@pytest.mark.parametrize("status,full,compact", [
  (TunnelStatus(TunnelState.CONNECTED, "10.8.0.7/32", handshake=986.0), "connected • 10.8.0.7 • 14s ago", "10.8.0.7"),
  (TunnelStatus(TunnelState.STALE, "10.8.0.7/32", handshake=700.0), "reconnecting • last 5m ago", "reconnecting"),
  (TunnelStatus(TunnelState.UNCONFIGURED), "needs a hub", "needs a hub"),
  (TunnelStatus(TunnelState.UNCONFIGURED, error="missing endpoint"), "missing endpoint", "missing endpoint"),
  (TunnelStatus(TunnelState.WAITING), "waiting for network", "waiting for network"),
  (TunnelStatus(TunnelState.ERROR, error="wg0 was not created"), "wg0 was not created", "wg0 was not created"),
  (TunnelStatus(TunnelState.OFF), "off", "off"),
])
def test_status_text(status, full, compact):
  assert wg.describe(status, now=1000.0) == full
  assert wg.describe_compact(status) == compact


def test_details_list_the_hub_address_and_device_key():
  status = TunnelStatus(TunnelState.CONNECTED, "10.8.0.7/32", "hub.teal.test:51820", public_key="devicepub=")
  assert wg.details(status) == "Hub hub.teal.test:51820\nAddress 10.8.0.7/32\nDevice key devicepub="
