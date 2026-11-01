"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""
from __future__ import annotations

import ipaddress
import json
import os
import time
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path

WG_ROOT = Path(os.environ.get("IQ_WIREGUARD_ROOT", "/data/wg"))
WG_STATUS = Path(os.environ.get("IQ_WIREGUARD_STATUS", "/dev/shm/iq_wireguard.json"))
HANDSHAKE_STALE_S = 180.0
STATUS_STALE_S = 30.0
DEFAULT_IFACE = "wg0"
DEFAULT_KEEPALIVE = 25
REQUIRED_KEYS = ("WG_ADDRESS", "WG_PEER_PUBKEY", "WG_ENDPOINT", "WG_ALLOWED_IPS")
PLACEHOLDERS = ("REPLACE_WITH_HUB_PUBLIC_KEY", "hub.example.com")
FULL_TUNNEL_ROUTES = ("0.0.0.0/0", "::/0")
PARAM_ENABLED = "IQWireGuardEnabled"
PARAM_CONFIG = "IQWireGuardConfig"
CONFIG_FIELDS = {
  "address": "WG_ADDRESS",
  "peerPublicKey": "WG_PEER_PUBKEY",
  "endpoint": "WG_ENDPOINT",
  "allowedIps": "WG_ALLOWED_IPS",
  "keepalive": "WG_KEEPALIVE",
}
CONFIG_LABELS = {"WG_ADDRESS": "address", "WG_PEER_PUBKEY": "hub public key", "WG_ENDPOINT": "endpoint", "WG_ALLOWED_IPS": "allowed IPs"}

wall_clock = time.time  # noqa: TID251


class TunnelState(StrEnum):
  NOT_INSTALLED = "not_installed"
  OFF = "off"
  UNCONFIGURED = "unconfigured"
  WAITING = "waiting"
  CONNECTING = "connecting"
  CONNECTED = "connected"
  STALE = "stale"
  ERROR = "error"


@dataclass(frozen=True)
class WireGuardConfig:
  address: str
  peer_public_key: str
  endpoint: str
  allowed_ips: tuple[str, ...]
  keepalive: int = DEFAULT_KEEPALIVE
  iface: str = DEFAULT_IFACE

  @property
  def endpoint_host(self) -> str:
    host, _, _ = self.endpoint.rpartition(":")
    return host.strip("[]")

  @property
  def endpoint_port(self) -> int:
    return int(self.endpoint.rpartition(":")[2])

  def routes(self, endpoint_ips: tuple[str, ...] = ()) -> tuple[tuple[str, ...], tuple[str, ...]]:
    hub = [ipaddress.ip_address(ip) for ip in endpoint_ips]
    keep, skipped = [], []
    for route in self.allowed_ips:
      try:
        network = ipaddress.ip_network(route, strict=False)
      except ValueError:
        skipped.append(route)
        continue
      if route in FULL_TUNNEL_ROUTES or network.prefixlen == 0 or any(ip.version == network.version and ip in network for ip in hub):
        skipped.append(route)
      else:
        keep.append(str(network))
    return tuple(keep), tuple(skipped)

  def setconf(self, private_key: str) -> str:
    return "\n".join([
      "[Interface]", f"PrivateKey = {private_key}", "",
      "[Peer]", f"PublicKey = {self.peer_public_key}", f"Endpoint = {self.endpoint}",
      f"AllowedIPs = {', '.join(self.allowed_ips)}", f"PersistentKeepalive = {self.keepalive}", "",
    ])


@dataclass(frozen=True)
class TunnelStatus:
  state: TunnelState
  address: str = ""
  endpoint: str = ""
  handshake: float = 0.0
  public_key: str = ""
  error: str = ""
  updated: float = field(default_factory=wall_clock)

  def handshake_age(self, now: float | None = None) -> float | None:
    if self.handshake <= 0:
      return None
    return max(0.0, (wall_clock() if now is None else now) - self.handshake)


def parse_env(text: str) -> dict[str, str]:
  values: dict[str, str] = {}
  for raw in text.splitlines():
    line = raw.strip()
    if not line or line.startswith("#") or "=" not in line:
      continue
    key, _, value = line.partition("=")
    value = value.split(" #", 1)[0].strip().strip("'\"")
    values[key.strip().removeprefix("export ").strip()] = value
  return values


def _default_params():
  from iqpilot.common.params import Params
  return Params()


def config_values(root: Path = WG_ROOT, params=None) -> dict[str, str]:
  try:
    raw = (params or _default_params()).get(PARAM_CONFIG)
  except Exception:
    raw = None
  if isinstance(raw, dict) and raw:
    values = {}
    for field, key in CONFIG_FIELDS.items():
      value = raw.get(field)
      if isinstance(value, list):
        value = ",".join(str(v) for v in value)
      if value not in (None, ""):
        values[key] = str(value)
    return values
  try:
    return parse_env((root / "wg.env").read_text())
  except OSError:
    return {}


def config_from_values(values: dict[str, str]) -> tuple[WireGuardConfig | None, str]:
  missing = [CONFIG_LABELS[k] for k in REQUIRED_KEYS if not values.get(k)]
  if missing:
    return None, f"missing {', '.join(missing)}"
  if any(p in values[k] for k in REQUIRED_KEYS for p in PLACEHOLDERS):
    return None, "still the example config"
  try:
    config = WireGuardConfig(
      address=values["WG_ADDRESS"],
      peer_public_key=values["WG_PEER_PUBKEY"],
      endpoint=values["WG_ENDPOINT"],
      allowed_ips=tuple(ip.strip() for ip in values["WG_ALLOWED_IPS"].split(",") if ip.strip()),
      keepalive=int(values.get("WG_KEEPALIVE") or DEFAULT_KEEPALIVE),
      iface=values.get("WG_IFACE") or DEFAULT_IFACE,
    )
    ipaddress.ip_interface(config.address)
    port = config.endpoint_port
  except ValueError as e:
    return None, f"invalid value: {e}"
  if not config.endpoint_host or not 0 < port < 65536:
    return None, "endpoint must be host:port"
  if not config.allowed_ips:
    return None, "missing allowed IPs"
  if not 0 <= config.keepalive <= 65535:
    return None, "keepalive must be 0-65535 seconds"
  return config, ""


def load_config(root: Path = WG_ROOT, params=None) -> WireGuardConfig | None:
  return config_from_values(config_values(root, params))[0]


def installed(root: Path = WG_ROOT) -> bool:
  return (root / "wireguard-go").is_file() and (root / "wg").is_file()


def enabled(params=None) -> bool:
  try:
    return bool((params or _default_params()).get(PARAM_ENABLED, return_default=True))
  except Exception:
    return True


def set_enabled(on: bool, params=None) -> bool:
  try:
    (params or _default_params()).put_bool(PARAM_ENABLED, bool(on))
  except Exception:
    return False
  return True


def public_key(root: Path = WG_ROOT) -> str:
  try:
    return (root / "publickey").read_text().strip()
  except OSError:
    return ""


def write_status(status: TunnelStatus, path: Path = WG_STATUS) -> None:
  tmp = path.with_name(path.name + ".tmp")
  tmp.write_text(json.dumps(asdict(status)))
  os.replace(tmp, path)


def read_status(root: Path = WG_ROOT, now: float | None = None, path: Path = WG_STATUS, params=None) -> TunnelStatus:
  if not installed(root):
    return TunnelStatus(TunnelState.NOT_INSTALLED)
  if not enabled(params):
    return TunnelStatus(TunnelState.OFF, public_key=public_key(root))
  try:
    data = json.loads(path.read_text())
    status = TunnelStatus(**{**data, "state": TunnelState(data["state"])})
  except (OSError, ValueError, KeyError, TypeError):
    return TunnelStatus(TunnelState.CONNECTING, public_key=public_key(root))
  if ((wall_clock() if now is None else now) - status.updated) > STATUS_STALE_S:
    return TunnelStatus(TunnelState.ERROR, status.address, status.endpoint, status.handshake, status.public_key,
                        "wireguardd is not running")
  return status


def _age(seconds: float) -> str:
  if seconds < 60:
    return f"{seconds:.0f}s ago"
  if seconds < 3600:
    return f"{seconds / 60:.0f}m ago"
  return f"{seconds / 3600:.0f}h ago"


def describe(status: TunnelStatus, now: float | None = None) -> str:
  address = status.address.split("/", 1)[0]
  age = status.handshake_age(now)
  if status.state == TunnelState.CONNECTED:
    return " \u2022 ".join(p for p in ("connected", address, _age(age) if age is not None else "") if p)
  if status.state == TunnelState.STALE:
    return f"reconnecting \u2022 last {_age(age)}" if age is not None else "reconnecting"
  return {
    TunnelState.NOT_INSTALLED: "not installed",
    TunnelState.OFF: "off",
    TunnelState.UNCONFIGURED: status.error or "needs a hub",
    TunnelState.WAITING: "waiting for network",
    TunnelState.CONNECTING: "connecting",
    TunnelState.ERROR: status.error or "error",
  }[status.state]


def details(status: TunnelStatus) -> str:
  lines = []
  if status.endpoint:
    lines.append(f"Hub {status.endpoint}")
  if status.address:
    lines.append(f"Address {status.address}")
  if status.public_key:
    lines.append(f"Device key {status.public_key}")
  if status.error and status.state != TunnelState.ERROR:
    lines.append(status.error)
  return "\n".join(lines)


def describe_compact(status: TunnelStatus) -> str:
  if status.state == TunnelState.CONNECTED and status.address:
    return status.address.split("/", 1)[0]
  if status.state == TunnelState.STALE:
    return "reconnecting"
  return describe(status)
