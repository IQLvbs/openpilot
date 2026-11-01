"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos
"""
import math
from collections import deque
from dataclasses import dataclass, field

from iqdbc.can import CANParser
from iqdbc.car import Bus, structs
from iqdbc.car.honda.hondacan import CanBus
from iqdbc.car.honda.values import DBC
from iqdbc.dbc.generator.honda.honda_radar_scan import QUARTET_KINDS, SCAN_SLOTS, frame_address

SCAN_DBC_NAME = 'honda_radar_scan_generated'
SWEEP_HZ = 15

SLOT_ADDRS = [tuple(frame_address(slot, kind) for kind in (*QUARTET_KINDS, "MOTION")) for slot in range(SCAN_SLOTS)]
ALL_SCAN_ADDRS = [addr for addrs in SLOT_ADDRS for addr in addrs]
# slot 15's IDENT frame closes every observed sweep's quartet family; its MOTION companion follows
# but must never gate an otherwise valid sweep, so the quartet frame stays the trigger
SWEEP_TRIGGER_ADDR = SLOT_ADDRS[SCAN_SLOTS - 1][3]

DIST_LSB_M = 0.05712
DIST_BIAS_M = -3.0
BEARING_LSB_RAD = 1.0 / 2048.0
BEARING_ZERO = 1024

STATE_INVALID = 0xF
DIST_RAW_INVALID = 0xFFF
BEARING_RAW_INVALID = 0x7FF
AGE_RAW_INVALID = 0xFFF
HANDLE_MIN = 1
HANDLE_MAX = 0x3F

CLOSING_SPEED_RAW_INVALID = 0x7FE
CLOSING_SPEED_RAW_MIN = 0
CLOSING_SPEED_RAW_MAX = 1728
CLOSING_SPEED_RAW_ZERO = 864
CLOSING_SPEED_LSB_MPS = 1.0 / 64.0
# replay-derived quality gate: the native speed field degrades gradually with its sigma companion;
# above this the field is no longer authoritative and the decoder coasts instead
CLOSING_SPEED_SIGMA_TRUST_MAX = 511

DIST_RATIO_RAW_INVALID = 0x3FF
DIST_RATIO_LSB = 0.001
DIST_RATIO_BIAS = 0.5

# replay-derived acceptance gates, not recovered firmware constants; innovation is measured from the
# previous ACCEPTED observation so a reset can never become the baseline for following sweeps
DIST_SIGMA_DEGRADED_RAW = 4
DIST_INNOVATION_SOFT_M = 2.0
DIST_INNOVATION_HARD_M = 5.0
RAW_RATE_LIMIT_MPS = 50.0

HISTORY_LEN = 8
QUIET_TIMEOUT_S = 0.20


def decode_closing_speed(raw, sigma_raw=None):
  if raw is None:
    return None
  raw = int(raw)
  if raw == CLOSING_SPEED_RAW_INVALID or not CLOSING_SPEED_RAW_MIN <= raw <= CLOSING_SPEED_RAW_MAX:
    return None
  if sigma_raw is not None and int(sigma_raw) > CLOSING_SPEED_SIGMA_TRUST_MAX:
    return None
  return (raw - CLOSING_SPEED_RAW_ZERO) * CLOSING_SPEED_LSB_MPS


def decode_dist_ratio(raw):
  if raw is None:
    return None
  raw = int(raw)
  if raw == DIST_RATIO_RAW_INVALID or not 0 <= raw < DIST_RATIO_RAW_INVALID:
    return None
  return DIST_RATIO_BIAS + DIST_RATIO_LSB * raw


def ratio_implied_rate(raw, dist, dt):
  ratio = decode_dist_ratio(raw)
  if ratio is None or dt <= 0.0 or not math.isfinite(dist):
    return None
  return dist * (1.0 - ratio) / dt


def reading_degraded(dist_sigma_raw, presence_raw, speed_sigma_raw):
  geometry_bad = dist_sigma_raw >= DIST_SIGMA_DEGRADED_RAW or presence_raw in (0, 0x7F)
  motion_bad = speed_sigma_raw is not None and speed_sigma_raw > CLOSING_SPEED_SIGMA_TRUST_MAX
  return geometry_bad or motion_bad


@dataclass
class SlotReading:
  slot: int
  cycle: int
  age: int
  handle: int
  handle_ok: bool
  coherent: bool
  dist_raw: int = 0
  bearing_raw: int = 0
  dist_sigma_raw: int = 0
  presence_raw: int = 0
  speed_raw: int | None = None
  speed_sigma_raw: int | None = None
  ratio_raw: int | None = None


@dataclass
class ObjectLedger:
  handle: int
  prev_cycle: int | None = None
  prev_age: int | None = None
  last_seen_nanos: int | None = None
  wire_slot: int | None = None
  history: deque = field(default_factory=lambda: deque(maxlen=HISTORY_LEN))
  held_speed: float | None = None
  held_speed_nanos: int | None = None

  def continues_incarnation(self, cycle: int, age: int) -> bool:
    if self.prev_cycle is None or self.prev_age is None:
      return False
    cycle_delta = (cycle - self.prev_cycle) & 0xF
    age_delta = (age - self.prev_age) & 0xFFF
    return age_delta == 2 * cycle_delta

  def restart_incarnation(self):
    self.history.clear()
    self.held_speed = None
    self.held_speed_nanos = None

  def held_speed_fresh(self, now: int) -> bool:
    return (self.held_speed is not None and self.held_speed_nanos is not None and
            (now - self.held_speed_nanos) * 1e-9 <= QUIET_TIMEOUT_S)


def create_scan_parser(CP) -> CANParser:
  # the object scan is physically on the camera-side ACC-CAN
  return CANParser(DBC[CP.carFingerprint][Bus.radar], [(addr, SWEEP_HZ) for addr in ALL_SCAN_ADDRS], CanBus(CP).camera)


class HondaRadarScanner:
  def __init__(self, CP):
    self.rcp = create_scan_parser(CP)
    self.trigger_msg = SWEEP_TRIGGER_ADDR
    self.pts: dict[int, structs.RadarData.RadarPoint] = {}
    self._ledgers: dict[int, ObjectLedger] = {}
    self._slot_handles: list[int | None] = [None] * SCAN_SLOTS
    self._last_sweep_nanos = -1

  def sweep_overdue(self) -> bool:
    if self._last_sweep_nanos < 0:
      return False
    return (self.rcp._last_update_nanos - self._last_sweep_nanos) * 1e-9 > QUIET_TIMEOUT_S

  def quiet_bus_radardata(self) -> structs.RadarData:
    # whole-bus silence: drop everything and emit an EMPTY RadarData (not None) so radard sheds any
    # lead within a cycle instead of freezing a phantom
    self.pts.clear()
    self._ledgers.clear()
    self._slot_handles = [None] * SCAN_SLOTS
    self._last_sweep_nanos = -1
    ret = structs.RadarData()
    if not self.rcp.can_valid:
      ret.errors.canError = True
    ret.errors.radarUnavailableTemporary = True
    return ret

  def _drop_ledger(self, handle: int):
    self._ledgers.pop(handle, None)
    self.pts.pop(handle, None)
    for slot, bound in enumerate(self._slot_handles):
      if bound == handle:
        self._slot_handles[slot] = None

  def _drop_expired_ledgers(self, now: int):
    for handle, ledger in list(self._ledgers.items()):
      if ledger.last_seen_nanos is not None and (now - ledger.last_seen_nanos) * 1e-9 > QUIET_TIMEOUT_S:
        self._drop_ledger(handle)

  def _read_slot(self, slot: int, updated_messages) -> SlotReading | None:
    pos, shape, life, ident, motion = SLOT_ADDRS[slot]
    if not all(addr in updated_messages for addr in (pos, shape, life, ident)):
      # a missing CAN frame is not a lifecycle event; ledgers expire on their own staleness only
      return None

    v_pos, v_shape, v_life, v_ident = (self.rcp.vl[a] for a in (pos, shape, life, ident))
    cycle = int(v_pos['CYCLE'])
    if not (cycle == int(v_shape['CYCLE']) == int(v_life['CYCLE']) == int(v_ident['CYCLE'])):
      # the quartet doesn't share one radar cycle: not a coherent observation this window
      return None

    state = int(v_pos['SCAN_STATE'])
    dist_raw = int(v_pos['DIST_RAW'])
    bearing_raw = int(v_pos['BEARING_RAW'])
    age = int(v_life['AGE_RAW'])
    handle = int(v_ident['OBJECT_HANDLE'])
    reading = SlotReading(
      slot=slot,
      cycle=cycle,
      age=age,
      handle=handle,
      handle_ok=HANDLE_MIN <= handle <= HANDLE_MAX,
      coherent=(state != STATE_INVALID and dist_raw != DIST_RAW_INVALID and
                bearing_raw != BEARING_RAW_INVALID and age != AGE_RAW_INVALID),
      dist_raw=dist_raw,
      bearing_raw=bearing_raw,
      dist_sigma_raw=int(v_pos['DIST_SIGMA_RAW']),
      presence_raw=int(v_shape['PRESENCE_RAW']),
    )

    # the MOTION companion only contributes when it rides the same cycle; its absence never
    # invalidates the quartet, it only removes independent motion evidence
    if motion in updated_messages:
      v_motion = self.rcp.vl[motion]
      if int(v_motion['CYCLE']) == cycle:
        reading.speed_raw = int(v_motion['CLOSING_SPEED_RAW'])
        reading.speed_sigma_raw = int(v_motion['CLOSING_SPEED_SIGMA_RAW'])
        reading.ratio_raw = int(v_motion['DIST_RATIO_RAW'])
    return reading

  def _elect_by_handle(self, readings: list[SlotReading]) -> dict[int, SlotReading]:
    # one CAN identity can never yield two points; ties prefer the wire slot already bound to the
    # ledger, then the lower slot for deterministic handling of a malformed duplicate
    elected: dict[int, SlotReading] = {}
    for reading in readings:
      if not (reading.coherent and reading.handle_ok):
        # an invalid observation ends publication for the slot's current occupant without destroying
        # persistent state; the object may be multiplexed elsewhere or return before its deadline
        hidden = {self._slot_handles[reading.slot]}
        if reading.handle_ok:
          hidden.add(reading.handle)
        for handle in hidden - {None}:
          self.pts.pop(handle, None)
        continue
      current = elected.get(reading.handle)
      if current is None:
        elected[reading.handle] = reading
        continue
      bound_slot = self._ledgers[reading.handle].wire_slot if reading.handle in self._ledgers else None
      current_rank = (0 if bound_slot == current.slot else 1, current.slot)
      candidate_rank = (0 if bound_slot == reading.slot else 1, reading.slot)
      if candidate_rank < current_rank:
        elected[reading.handle] = reading
    return elected

  def _bind_slot(self, ledger: ObjectLedger, reading: SlotReading, now: int):
    ledger.prev_cycle = reading.cycle
    ledger.prev_age = reading.age
    ledger.last_seen_nanos = now
    ledger.wire_slot = reading.slot
    for slot, bound in enumerate(self._slot_handles):
      if slot != reading.slot and bound == ledger.handle:
        self._slot_handles[slot] = None
    self._slot_handles[reading.slot] = ledger.handle

  def _coast_point(self, ledger: ObjectLedger, now: int, dist: float, y_rel: float):
    # coasting keeps trustworthy geometry visible with the last authoritative motion, unmeasured,
    # instead of publishing a synthesized rate; without fresh held motion the point drops
    if ledger.held_speed_fresh(now):
      point = self.pts.get(ledger.handle)
      if point is not None:
        point.dRel = dist
        point.yRel = y_rel
        point.vRel = ledger.held_speed
        point.measured = False
      return
    ledger.held_speed = None
    ledger.held_speed_nanos = None
    self.pts.pop(ledger.handle, None)

  def process_sweep(self, updated_messages) -> structs.RadarData:
    ret = structs.RadarData()
    if not self.rcp.can_valid:
      ret.errors.canError = True

    now = self.rcp._last_update_nanos
    self._last_sweep_nanos = now
    self._drop_expired_ledgers(now)

    readings = [r for r in (self._read_slot(slot, updated_messages) for slot in range(SCAN_SLOTS)) if r is not None]
    elected = self._elect_by_handle(readings)
    elected_handles = set(elected)

    for handle, reading in sorted(elected.items(), key=lambda item: item[1].slot):
      # a wire-slot replacement ends publication for the old occupant, but not its persistent state
      old_handle = self._slot_handles[reading.slot]
      if old_handle is not None and old_handle != handle and old_handle not in elected_handles:
        self.pts.pop(old_handle, None)

      ledger = self._ledgers.get(handle)
      if ledger is None:
        ledger = ObjectLedger(handle=handle)
        self._ledgers[handle] = ledger

      if not ledger.continues_incarnation(reading.cycle, reading.age):
        # the CAN identity stays the external key, but a lifecycle discontinuity starts a new
        # incarnation and must not inherit the previous object's range-rate history
        ledger.restart_incarnation()
        self.pts.pop(handle, None)

      dist = DIST_LSB_M * reading.dist_raw + DIST_BIAS_M
      bearing = BEARING_LSB_RAD * (reading.bearing_raw - BEARING_ZERO)
      # the radar consumes range as a forward-axis quantity; positive bearing is left of center,
      # matching the RadarPoint.yRel sign contract
      y_rel = dist * math.tan(bearing)

      now_s = now * 1e-9
      native_speed = decode_closing_speed(reading.speed_raw, reading.speed_sigma_raw)
      unqualified_speed = decode_closing_speed(reading.speed_raw)
      # a live native speed rejected only by its sigma companion: geometry acceptance is still decided
      # on the same terms as every other sweep (high sigma correlates with bad/discontinuous range),
      # and only then does the point coast instead of publishing a synthesized rate
      sigma_veto = (native_speed is None and unqualified_speed is not None and
                    reading.speed_sigma_raw is not None and
                    reading.speed_sigma_raw > CLOSING_SPEED_SIGMA_TRUST_MAX)

      degraded = reading_degraded(reading.dist_sigma_raw, reading.presence_raw, reading.speed_sigma_raw)
      previous = ledger.history[-1] if ledger.history else None
      ratio_rate = None
      dist_rejected = False
      if previous is not None:
        prev_time, prev_dist = previous
        dt = now_s - prev_time
        if dt <= 0.0:
          dist_rejected = True
        else:
          ratio = decode_dist_ratio(reading.ratio_raw)
          ratio_rate = ratio_implied_rate(reading.ratio_raw, dist, dt)

          residuals_m = []
          if native_speed is not None:
            residuals_m.append(abs(dist - (prev_dist + native_speed * dt)))
          if ratio is not None:
            residuals_m.append(abs(prev_dist - dist * ratio))

          if residuals_m:
            innovation_m = min(residuals_m)
            dist_rejected = (innovation_m > DIST_INNOVATION_HARD_M or
                             (degraded and innovation_m > DIST_INNOVATION_SOFT_M))
          else:
            dist_rejected = abs((dist - prev_dist) / dt) > RAW_RATE_LIMIT_MPS

      if dist_rejected:
        # keep the last accepted point briefly as an unmeasured coast; rejected geometry is never
        # published and never becomes the baseline for a later derivative
        accepted_fresh = previous is not None and now_s - previous[0] <= QUIET_TIMEOUT_S
        point = self.pts.get(handle)
        if accepted_fresh and point is not None:
          point.measured = False
        else:
          self.pts.pop(handle, None)
        self._bind_slot(ledger, reading, now)
        continue

      if sigma_veto:
        self._coast_point(ledger, now, dist, y_rel)
        self._bind_slot(ledger, reading, now)
        continue

      # with neither a qualified native speed nor a usable ratio, the only remaining source is the
      # raw one-sweep derivative, which must never become an authoritative measurement: coast instead.
      # A true birth (no previous accepted sample) cannot mature this cycle regardless, so only
      # intercept once a derivative would have something to poison
      if native_speed is None and (ratio_rate is None or degraded) and previous is not None:
        self._coast_point(ledger, now, dist, y_rel)
        self._bind_slot(ledger, reading, now)
        continue

      ledger.history.append((now_s, dist))

      speed = native_speed if native_speed is not None else ratio_rate
      ledger.held_speed = speed
      ledger.held_speed_nanos = now

      # a birth observation has no range rate yet: keep it as history, publish only once a second
      # coherent observation of the same identity supplies a finite rate
      matured = len(ledger.history) >= 2 and math.isfinite(speed)
      if matured:
        if handle not in self.pts:
          point = structs.RadarData.RadarPoint()
          point.trackId = handle
          point.aRel = float('nan')
          point.yvRel = float('nan')
          self.pts[handle] = point
        self.pts[handle].dRel = dist
        self.pts[handle].yRel = y_rel
        self.pts[handle].vRel = speed
        self.pts[handle].measured = True
      else:
        self.pts.pop(handle, None)

      self._bind_slot(ledger, reading, now)

    ret.points = [self.pts[handle] for handle in sorted(self.pts)]
    return ret
