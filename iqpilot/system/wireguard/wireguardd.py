#!/usr/bin/env python3
"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""
from __future__ import annotations

import os
import socket
import subprocess
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from iqpilot.system.wireguard.tunnel import (
  HANDSHAKE_STALE_S, WG_ROOT, WG_STATUS, TunnelState, TunnelStatus, WireGuardConfig, config_from_values, config_values, enabled,
  installed, public_key, wall_clock, write_status,
)

POLL_S = 5.0
HEARTBEAT_S = 10.0
HANDSHAKE_CHECK_S = 30.0
RETRY_S = 30.0
MAX_RETRY_S = 300.0
REUP_S = 120.0
IFACE_WAIT_S = 10.0
COMMAND_TIMEOUT_S = 15
DAEMON_NICE = 19
TUNNEL_NICE = 10
TUNNEL_ENV = ("GOMAXPROCS=2", "GOMEMLIMIT=48MiB")

Runner = Callable[[list[str]], tuple[int, str]]


class TunnelError(RuntimeError):
  pass


def sudo_run(args: list[str]) -> tuple[int, str]:
  try:
    proc = subprocess.run(["sudo", "-n", *args], capture_output=True, text=True, timeout=COMMAND_TIMEOUT_S)
  except subprocess.TimeoutExpired:
    return 124, f"timed out after {COMMAND_TIMEOUT_S}s"
  return proc.returncode, (proc.stdout or proc.stderr).strip()


def iface_exists(iface: str) -> bool:
  return os.path.exists(f"/sys/class/net/{iface}")


def resolve(host: str) -> tuple[str, ...]:
  try:
    infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_UDP)
  except OSError:
    return ()
  return tuple(dict.fromkeys(str(info[4][0]) for info in infos))


class Tunnel:
  def __init__(self, root: Path = WG_ROOT, run: Runner = sudo_run, clock: Callable[[], float] = wall_clock,
               has_iface: Callable[[str], bool] = iface_exists, resolver: Callable[[str], tuple[str, ...]] = resolve,
               sleep: Callable[[float], None] = time.sleep, params=None):
    self.root = root
    self._params = params
    self._run = run
    self._clock = clock
    self._has_iface = has_iface
    self._resolve = resolver
    self._sleep = sleep
    self._iface = ""
    self._up_at = 0.0
    self._last_attempt = float("-inf")
    self._retry_s = RETRY_S
    self._handshake = 0.0
    self._handshake_checked = float("-inf")
    self._note = ""

  @property
  def _binary(self) -> str:
    return str(self.root / "wireguard-go")

  def _check(self, args: list[str]) -> str:
    rc, out = self._run(args)
    if rc != 0:
      raise TunnelError(f"{' '.join(args)}: {out or f'exit {rc}'}")
    return out

  def latest_handshake(self, iface: str) -> float:
    rc, out = self._run([str(self.root / "wg"), "show", iface, "latest-handshakes"])
    if rc != 0:
      return 0.0
    stamps = [float(parts[1]) for parts in (line.split() for line in out.splitlines()) if len(parts) == 2 and parts[1].isdigit()]
    return max(stamps, default=0.0)

  def _stop(self, iface: str) -> None:
    if self._has_iface(iface):
      self._run(["ip", "link", "del", iface])
    self._run(["pkill", "-xf", f"{self._binary} {iface}"])

  def down(self) -> None:
    if self._iface:
      self._stop(self._iface)
    self._iface = ""
    self._up_at = 0.0
    self._handshake = 0.0

  def up(self, config: WireGuardConfig, endpoint_ips: tuple[str, ...]) -> None:
    self._last_attempt = self._clock()
    self._iface = config.iface
    self._handshake, self._handshake_checked = 0.0, float("-inf")
    self._stop(config.iface)
    self._check(["mkdir", "-p", "/var/run/wireguard"])
    self._check(["env", *TUNNEL_ENV, "nice", "-n", str(TUNNEL_NICE), self._binary, config.iface])
    deadline = self._clock() + IFACE_WAIT_S
    while not self._has_iface(config.iface):
      if self._clock() >= deadline:
        raise TunnelError(f"{config.iface} was not created")
      self._sleep(0.25)
    key = (self.root / "privatekey").read_text().strip()
    setconf = self.root / "wg0.setconf"
    fd = os.open(setconf, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
      f.write(config.setconf(key))
    try:
      self._check([str(self.root / "wg"), "setconf", config.iface, str(setconf)])
    finally:
      setconf.unlink(missing_ok=True)
    self._check(["ip", "address", "replace", config.address, "dev", config.iface])
    self._check(["ip", "link", "set", config.iface, "up"])
    routes, skipped = config.routes(endpoint_ips)
    taken = [route for route in routes if self._run(["ip", "route", "add", route, "dev", config.iface])[0] != 0]
    notes = []
    if skipped:
      notes.append(f"not routed: {', '.join(skipped)}")
    if taken:
      notes.append(f"already routed elsewhere: {', '.join(taken)}")
    self._note = "; ".join(notes)
    self._up_at = self._clock()
    self._retry_s = RETRY_S

  def _handshake_now(self, iface: str, now: float) -> float:
    interval = HANDSHAKE_CHECK_S if self._handshake > 0 else POLL_S
    if now - self._handshake_checked >= interval:
      self._handshake = self.latest_handshake(iface)
      self._handshake_checked = now
    return self._handshake

  def step(self) -> TunnelStatus:
    now = self._clock()
    key = public_key(self.root)
    if not installed(self.root):
      return TunnelStatus(TunnelState.NOT_INSTALLED, updated=now)
    if not enabled(self._params):
      self.down()
      return TunnelStatus(TunnelState.OFF, public_key=key, updated=now)
    config, problem = config_from_values(config_values(self.root, self._params))
    if config is None:
      self.down()
      return TunnelStatus(TunnelState.UNCONFIGURED, public_key=key, error=problem, updated=now)

    def status(state: TunnelState, handshake: float = 0.0, error: str = "") -> TunnelStatus:
      return TunnelStatus(state, config.address, config.endpoint, handshake, key, error or self._note, now)

    if not self._iface and self._has_iface(config.iface):
      self._iface, self._up_at, self._last_attempt = config.iface, now, now
    running = self._iface == config.iface and self._has_iface(config.iface)
    handshake = 0.0
    if running:
      handshake = self._handshake_now(config.iface, now)
      if handshake > 0 and now - handshake <= HANDSHAKE_STALE_S:
        return status(TunnelState.CONNECTED, handshake)
      if handshake <= 0 and now - self._up_at <= HANDSHAKE_STALE_S:
        return status(TunnelState.CONNECTING)
      if now - self._last_attempt < REUP_S:
        return status(TunnelState.STALE, handshake)
    elif now - self._last_attempt < self._retry_s:
      return status(TunnelState.CONNECTING)

    endpoint_ips = self._resolve(config.endpoint_host)
    if not endpoint_ips:
      return status(TunnelState.WAITING, handshake, f"cannot resolve {config.endpoint_host}")
    try:
      self.up(config, endpoint_ips)
    except (TunnelError, OSError, subprocess.SubprocessError) as e:
      self.down()
      self._retry_s = min(self._retry_s * 2, MAX_RETRY_S)
      return status(TunnelState.ERROR, error=str(e))
    return status(TunnelState.STALE if running else TunnelState.CONNECTING, handshake)


def publish(current: TunnelStatus, previous: TunnelStatus | None, path: Path = WG_STATUS) -> bool:
  unchanged = previous is not None and replace(previous, updated=current.updated) == current
  if unchanged and current.updated - previous.updated < HEARTBEAT_S:
    return False
  write_status(current, path)
  return True


def main() -> None:
  from iqpilot.common.swaglog import cloudlog
  from iqpilot.common.params import Params
  os.nice(DAEMON_NICE)
  tunnel = Tunnel(params=Params())
  published: TunnelStatus | None = None
  while True:
    try:
      current = tunnel.step()
      if published is None or current.state != published.state or current.error != published.error:
        cloudlog.event("wireguardd.state", state=str(current.state), error=current.error)
      if publish(current, published):
        published = current
    except Exception:
      cloudlog.exception("wireguardd step failed")
    time.sleep(POLL_S)


if __name__ == "__main__":
  main()
