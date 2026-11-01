# Copyright (c) 2026 IQ.Lvbs. All rights reserved.

import math
import secrets
from dataclasses import dataclass

MAX_AGE_S = 2.0
MAX_RECEIPT_AGE_S = 0.65
MAX_CLOCK_ERROR_S = 0.25
SESSION_TTL_S = 20.0


def number(value, name, minimum, maximum):
  if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not minimum <= value <= maximum:
    raise ValueError(f"invalid {name}")
  return float(value)


@dataclass(frozen=True)
class PhoneFix:
  measured_at: float
  received_at: float
  sequence: int
  gps: dict


class PhoneGPSReceiver:
  def __init__(self):
    self.session = ""
    self.owner = ""
    self.expires_at = 0.0
    self.sequence = -1
    self.fix = None
    self.error = ""

  def clear(self):
    self.session = ""
    self.owner = ""
    self.expires_at = 0.0
    self.sequence = -1
    self.fix = None
    self.error = ""

  def current(self, now):
    return self.fix if self.fix and 0 <= now - self.fix.measured_at <= MAX_AGE_S and now < self.expires_at else None

  def begin(self, owner, now):
    if not isinstance(owner, str) or not 8 <= len(owner) <= 128:
      raise ValueError("invalid clientId")
    if self.owner != owner and self.current(now):
      raise ValueError("another phone is supplying GPS")
    if self.owner != owner or now >= self.expires_at:
      self.clear()
      self.owner = owner
      self.session = secrets.token_urlsafe(24)
    self.expires_at = now + SESSION_TTL_S
    return {"success": 1, "sessionId": self.session, "deviceMonotonicMs": now * 1000, "expiresInMs": SESSION_TTL_S * 1000}

  def accept(self, payload, now):
    if not isinstance(payload, dict) or payload.get("sessionId") != self.session or not self.session or now >= self.expires_at:
      raise ValueError("phone GPS session expired")
    sequence = payload.get("sequence")
    if isinstance(sequence, bool) or not isinstance(sequence, int) or not self.sequence < sequence < 2**53:
      raise ValueError("out-of-order phone GPS fix")
    measured_at = number(payload.get("deviceMonotonicMs"), "measurement time", 0, 1e15) / 1000
    uncertainty = number(payload.get("clockUncertaintyMs"), "clock uncertainty", 0, MAX_CLOCK_ERROR_S * 1000) / 1000
    if now - measured_at + uncertainty > MAX_RECEIPT_AGE_S or measured_at - now > uncertainty or measured_at < self.expires_at - SESSION_TTL_S - MAX_AGE_S:
      raise ValueError("stale or future phone GPS fix")
    if self.fix and measured_at <= self.fix.measured_at:
      raise ValueError("out-of-order phone GPS timestamp")
    if payload.get("mocked") is True:
      raise ValueError("mock locations are not accepted")
    platform = payload.get("platform")
    if platform not in ("ios", "android"):
      raise ValueError("invalid phone platform")
    lat = number(payload.get("latitude"), "latitude", -90, 90)
    lon = number(payload.get("longitude"), "longitude", -180, 180)
    accuracy = number(payload.get("accuracy"), "horizontal accuracy", 0.1, 50)
    altitude = number(payload.get("altitude"), "altitude", -1000, 10000)
    vertical_accuracy = number(payload.get("altitudeAccuracy"), "vertical accuracy", 0.1, 100)
    speed = number(payload.get("speed"), "speed", 0, 100)
    heading = number(payload.get("heading"), "heading", 0, 360)
    timestamp = number(payload.get("timestamp"), "UTC timestamp", 1e12, 1e14)
    if self.fix and now - self.fix.measured_at < 5:
      previous = self.fix.gps
      dy = math.radians(lat - previous["latitude"]) * 6371000
      dx = math.radians(lon - previous["longitude"]) * 6371000 * math.cos(math.radians(lat))
      allowance = 100 * (measured_at - self.fix.measured_at) + 3 * (accuracy + previous["horizontalAccuracy"])
      if math.hypot(dx, dy) > allowance:
        raise ValueError("implausible phone GPS jump")
    radians = math.radians(heading)
    gps = {
      "latitude": lat, "longitude": lon, "altitude": altitude,
      "horizontalAccuracy": max(accuracy, 3) + speed * uncertainty,
      "verticalAccuracy": max(vertical_accuracy, 5), "speed": speed,
      "bearingDeg": heading, "unixTimestampMillis": int(timestamp),
      "source": "iOS" if platform == "ios" else "android",
      "vNED": [speed * math.cos(radians), speed * math.sin(radians), 0.0],
      "speedAccuracy": max(3.0, speed * 0.15), "bearingAccuracyDeg": 30.0 if speed >= 2 else 180.0,
      "hasFix": True, "flags": 1,
    }
    self.sequence = sequence
    self.fix = PhoneFix(min(now, measured_at), now, sequence, gps)
    self.error = ""
    return {"success": 1, "sequence": sequence, "ageMs": max(0, now - measured_at) * 1000}

  def status(self, now, enabled=True):
    fix = self.current(now) if enabled else None
    return {
      "supported": True, "enabled": enabled, "activeForDrive": True, "onroad": True,
      "source": "phone" if fix else "device", "state": "active" if fix else ("waiting" if enabled else "off"),
      "ageMs": (now - fix.measured_at) * 1000 if fix else None,
      "accuracyMeters": fix.gps["horizontalAccuracy"] if fix else None,
      "error": self.error,
    }
