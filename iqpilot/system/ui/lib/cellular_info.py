"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""
from __future__ import annotations

import ipaddress
import socket
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

MM = "org.freedesktop.ModemManager1"
MM_MODEM = MM + ".Modem"
MM_MODEM_3GPP = MM + ".Modem.Modem3gpp"
CELL_IFACE = "wwan0"
REFRESH_S = 5.0

ACCESS_TECHNOLOGIES = (
  (1 << 15, "5G"),
  (1 << 14, "LTE"),
  (1 << 16, "LTE-M"),
  (1 << 17, "NB-IoT"),
  ((1 << 9) | (1 << 8) | (1 << 7) | (1 << 6), "HSPA"),
  (1 << 5, "3G"),
  (1 << 4, "EDGE"),
  ((1 << 3) | (1 << 2) | (1 << 1), "2G"),
)


@dataclass(frozen=True)
class CellularInfo:
  carrier: str = ""
  technology: str = ""
  signal_percent: int | None = None
  ipv4: str = ""
  ipv6: str = ""

  @property
  def carrier_text(self) -> str:
    return " \u2022 ".join(part for part in (self.carrier, self.technology) if part)


def technology_name(mask: int) -> str:
  return next((name for bits, name in ACCESS_TECHNOLOGIES if mask & bits), "")


def _value(variant):
  return variant[1] if isinstance(variant, tuple) and len(variant) == 2 and isinstance(variant[0], str) else variant


def parse_modem(objects: dict) -> tuple[str, str, int | None]:
  for interfaces in objects.values():
    modem = interfaces.get(MM_MODEM)
    if modem is None:
      continue
    three_gpp = interfaces.get(MM_MODEM_3GPP, {})
    carrier = str(_value(three_gpp.get("OperatorName", ("s", ""))) or "").strip()
    if not carrier:
      code = str(_value(three_gpp.get("OperatorCode", ("s", ""))) or "").strip()
      carrier = f"MCC-MNC {code}" if code else ""
    technology = technology_name(int(_value(modem.get("AccessTechnologies", ("u", 0))) or 0))
    quality = _value(modem.get("SignalQuality", ("(ub)", (0, False))))
    signal = int(quality[0]) if isinstance(quality, (tuple, list)) and quality and quality[1] else None
    return carrier, technology, signal
  return "", "", None


def pick_addresses(addrs: list) -> tuple[str, str]:
  ipv4 = ipv6 = ""
  for addr in addrs:
    if addr.family == socket.AF_INET and not ipv4:
      ipv4 = addr.address
    elif addr.family == socket.AF_INET6 and not ipv6:
      ip = ipaddress.ip_address(addr.address.split("%", 1)[0])
      if not ip.is_link_local:
        ipv6 = str(ip)
  return ipv4, ipv6


def interface_addresses(iface: str = CELL_IFACE) -> tuple[str, str]:
  import psutil
  return pick_addresses(psutil.net_if_addrs().get(iface, []))


def modem_objects() -> dict:
  from jeepney import DBusAddress, new_method_call
  from jeepney.io.blocking import open_dbus_connection
  with open_dbus_connection(bus="SYSTEM") as conn:
    address = DBusAddress("/org/freedesktop/ModemManager1", bus_name=MM, interface="org.freedesktop.DBus.ObjectManager")
    reply = conn.send_and_get_reply(new_method_call(address, "GetManagedObjects"), timeout=2)
  return reply.body[0] if reply.body else {}


def read_cellular_info(objects: Callable[[], dict] = modem_objects,
                       addresses: Callable[[], tuple[str, str]] = interface_addresses) -> CellularInfo:
  try:
    carrier, technology, signal = parse_modem(objects())
  except Exception:
    carrier, technology, signal = "", "", None
  try:
    ipv4, ipv6 = addresses()
  except Exception:
    ipv4, ipv6 = "", ""
  return CellularInfo(carrier, technology, signal, ipv4, ipv6)


class CellularInfoPoller:
  def __init__(self, reader: Callable[[], CellularInfo] = read_cellular_info, interval: float = REFRESH_S):
    self._reader = reader
    self._interval = interval
    self._info = CellularInfo()
    self._last = float("-inf")
    self._lock = threading.Lock()
    self._busy = False

  @property
  def info(self) -> CellularInfo:
    return self._info

  def poll(self) -> CellularInfo:
    now = time.monotonic()
    with self._lock:
      if self._busy or now - self._last < self._interval:
        return self._info
      self._busy, self._last = True, now
      current = self._info
    threading.Thread(target=self._refresh, daemon=True, name="cellular-info").start()
    return current

  def _refresh(self) -> None:
    try:
      self._info = self._reader()
    finally:
      self._busy = False
