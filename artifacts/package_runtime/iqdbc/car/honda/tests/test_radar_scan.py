import math

import pytest

from iqdbc.can import CANParser
from iqdbc.car.honda.radar_scan import (AGE_RAW_INVALID, ALL_SCAN_ADDRS, BEARING_RAW_INVALID, BEARING_ZERO,
                                        CLOSING_SPEED_RAW_INVALID, CLOSING_SPEED_RAW_ZERO,
                                        CLOSING_SPEED_SIGMA_TRUST_MAX, DIST_BIAS_M, DIST_LSB_M,
                                        DIST_RATIO_RAW_INVALID, DIST_RAW_INVALID, HondaRadarScanner,
                                        QUIET_TIMEOUT_S, SCAN_DBC_NAME, SCAN_SLOTS, STATE_INVALID,
                                        SWEEP_TRIGGER_ADDR, decode_closing_speed, decode_dist_ratio)
from iqdbc.dbc.generator.honda.honda_radar_scan import FRAME_SIGNALS, frame_address

BUS = 2
SWEEP_DT_NS = 66_000_000


def set_bits(data, start_bit, size, value):
  value = int(value) & ((1 << size) - 1)
  pos = start_bit
  for i in range(size):
    bit = (value >> (size - 1 - i)) & 1
    byte_i, bit_i = pos // 8, pos % 8
    if bit:
      data[byte_i] |= (1 << bit_i)
    pos = pos - 1 if bit_i > 0 else pos + 15


GEOMETRY = {kind: {name: (start, size) for name, start, size in sigs} for kind, sigs in FRAME_SIGNALS.items()}


def build_frame(slot, kind, **fields):
  data = bytearray(8)
  for name, value in fields.items():
    set_bits(data, *GEOMETRY[kind][name], value)
  return (frame_address(slot, kind), bytes(data), BUS)


def quartet(slot, cycle, dist_raw=1000, bearing_raw=BEARING_ZERO, state=1, dist_sigma=0, presence=40,
            age=100, handle=5):
  return [
    build_frame(slot, "POS", SCAN_STATE=state, CYCLE=cycle, DIST_RAW=dist_raw, BEARING_RAW=bearing_raw,
                DIST_SIGMA_RAW=dist_sigma),
    build_frame(slot, "SHAPE", CYCLE=cycle, PRESENCE_RAW=presence),
    build_frame(slot, "LIFE", CYCLE=cycle, AGE_RAW=age),
    build_frame(slot, "IDENT", CYCLE=cycle, OBJECT_HANDLE=handle),
  ]


def motion_frame(slot, cycle, speed_raw=CLOSING_SPEED_RAW_ZERO, sigma_raw=0, ratio_raw=500):
  return build_frame(slot, "MOTION", CYCLE=cycle, CLOSING_SPEED_RAW=speed_raw,
                     CLOSING_SPEED_SIGMA_RAW=sigma_raw, DIST_RATIO_RAW=ratio_raw)


def closing_sweep(slot_msgs, cycle):
  # slot 15's quartet closes every sweep so the trigger fires
  msgs = list(slot_msgs)
  if not any(m[0] == SWEEP_TRIGGER_ADDR for m in msgs):
    msgs += quartet(15, cycle, state=STATE_INVALID, dist_raw=DIST_RAW_INVALID,
                    bearing_raw=BEARING_RAW_INVALID, age=AGE_RAW_INVALID, handle=0)
  return msgs


class ScanHarness:
  def __init__(self):
    self.scanner = object.__new__(HondaRadarScanner)
    self.scanner.rcp = CANParser(SCAN_DBC_NAME, [(a, 15) for a in ALL_SCAN_ADDRS], BUS)
    self.scanner.trigger_msg = SWEEP_TRIGGER_ADDR
    self.scanner.pts = {}
    self.scanner._ledgers = {}
    self.scanner._slot_handles = [None] * SCAN_SLOTS
    self.scanner._last_sweep_nanos = -1
    self.updated = set()
    self.nanos = 0
    self.cycle = 0

  def feed(self, msgs, dt_ns=SWEEP_DT_NS):
    self.nanos += dt_ns
    vls = self.scanner.rcp.update([self.nanos, list(msgs)])
    self.updated.update(vls)
    if self.scanner.trigger_msg not in self.updated:
      if self.scanner.sweep_overdue():
        return self.scanner.quiet_bus_radardata()
      return None
    result = self.scanner.process_sweep(self.updated)
    self.updated.clear()
    return result

  def sweep(self, slot_msgs=(), cycle_step=1, dt_ns=SWEEP_DT_NS):
    self.cycle = (self.cycle + cycle_step) & 0xF
    return self.feed(closing_sweep(slot_msgs, self.cycle), dt_ns=dt_ns)

  def object_sweep(self, slot=0, handle=5, dist_raw=1000, with_motion=True, cycle_step=1, age_step=None,
                   dt_ns=SWEEP_DT_NS, **kwargs):
    if age_step is None:
      age_step = 2 * cycle_step
    self._age = (getattr(self, "_age", 100) + age_step) & 0xFFF
    cycle = (self.cycle + cycle_step) & 0xF
    msgs = quartet(slot, cycle, dist_raw=dist_raw, age=self._age, handle=handle, **kwargs)
    if with_motion:
      msgs.append(motion_frame(slot, cycle))
    return self.sweep(msgs, cycle_step=cycle_step, dt_ns=dt_ns)


class TestFieldDecoding:
  def test_dist_conversion(self):
    assert DIST_LSB_M * 1000 + DIST_BIAS_M == pytest.approx(54.12)

  def test_closing_speed_decode_and_domain(self):
    assert decode_closing_speed(CLOSING_SPEED_RAW_ZERO) == 0.0
    assert decode_closing_speed(CLOSING_SPEED_RAW_ZERO + 64) == 1.0
    assert decode_closing_speed(CLOSING_SPEED_RAW_INVALID) is None
    assert decode_closing_speed(1729) is None
    assert decode_closing_speed(None) is None

  def test_closing_speed_sigma_veto(self):
    assert decode_closing_speed(CLOSING_SPEED_RAW_ZERO, CLOSING_SPEED_SIGMA_TRUST_MAX) == 0.0
    assert decode_closing_speed(CLOSING_SPEED_RAW_ZERO, CLOSING_SPEED_SIGMA_TRUST_MAX + 1) is None

  def test_dist_ratio_decode(self):
    assert decode_dist_ratio(500) == pytest.approx(1.0)
    assert decode_dist_ratio(DIST_RATIO_RAW_INVALID) is None
    assert decode_dist_ratio(None) is None

  def test_bearing_sign_convention(self):
    h = ScanHarness()
    h.object_sweep(bearing_raw=BEARING_ZERO + 100)
    result = h.object_sweep(bearing_raw=BEARING_ZERO + 100)
    assert result.points[0].yRel > 0  # left of center is positive
    dist = result.points[0].dRel
    assert result.points[0].yRel == pytest.approx(dist * math.tan(100 / 2048))

  def test_bearing_right_of_center_is_negative(self):
    h = ScanHarness()
    h.object_sweep(bearing_raw=BEARING_ZERO - 100)
    result = h.object_sweep(bearing_raw=BEARING_ZERO - 100)
    assert result.points[0].yRel < 0

  def test_boresight_is_zero(self):
    h = ScanHarness()
    h.object_sweep(bearing_raw=BEARING_ZERO)
    result = h.object_sweep(bearing_raw=BEARING_ZERO)
    assert result.points[0].yRel == 0.0


class TestPublicationRules:
  def test_birth_is_withheld_until_second_observation(self):
    h = ScanHarness()
    result = h.object_sweep()
    assert len(result.points) == 0
    result = h.object_sweep()
    assert len(result.points) == 1
    point = result.points[0]
    assert point.trackId == 5
    assert point.measured
    assert math.isnan(point.aRel) and math.isnan(point.yvRel)

  def test_handle_is_wire_identity_not_synthetic(self):
    h = ScanHarness()
    h.object_sweep(handle=0x22)
    result = h.object_sweep(handle=0x22)
    assert result.points[0].trackId == 0x22

  @pytest.mark.parametrize("field,value", [("state", STATE_INVALID), ("dist_raw", DIST_RAW_INVALID),
                                           ("bearing_raw", BEARING_RAW_INVALID)])
  def test_sentinels_invalidate_observation(self, field, value):
    h = ScanHarness()
    h.object_sweep()
    h.object_sweep()
    kwargs = {field: value}
    result = h.object_sweep(**kwargs)
    assert len(result.points) == 0

  def test_age_sentinel_invalidates_observation(self):
    h = ScanHarness()
    h.object_sweep()
    h.object_sweep()
    cycle = (h.cycle + 1) & 0xF
    msgs = quartet(0, cycle, age=AGE_RAW_INVALID, handle=5) + [motion_frame(0, cycle)]
    result = h.sweep(msgs)
    assert len(result.points) == 0

  @pytest.mark.parametrize("handle", [0, 0x40, 0xFF])
  def test_out_of_range_handle_invalidates(self, handle):
    h = ScanHarness()
    h.object_sweep()
    h.object_sweep()
    result = h.object_sweep(handle=handle)
    assert len(result.points) == 0

  def test_incomplete_quartet_is_not_an_observation(self):
    h = ScanHarness()
    h.object_sweep()
    h.object_sweep()
    cycle = (h.cycle + 1) & 0xF
    h._age = (h._age + 2) & 0xFFF
    msgs = quartet(0, cycle, age=h._age, handle=5)[:3]  # drop IDENT
    result = h.sweep(msgs)
    # a dropped CAN frame is not a lifecycle event: the published point persists untouched
    assert len(result.points) == 1
    result = h.object_sweep()
    assert len(result.points) == 1
    assert result.points[0].measured

  def test_cycle_mismatch_across_quartet_is_incoherent(self):
    h = ScanHarness()
    h.object_sweep()
    h.object_sweep()
    cycle = (h.cycle + 1) & 0xF
    msgs = quartet(0, cycle, age=200, handle=5)
    bad_life = build_frame(0, "LIFE", CYCLE=(cycle + 1) & 0xF, AGE_RAW=200)
    msgs[2] = bad_life
    result = h.sweep(msgs)
    # an incoherent quartet is not an observation: the published point persists untouched
    assert len(result.points) == 1
    assert result.points[0].measured


class TestLifecycle:
  def test_age_advances_two_per_cycle_keeps_identity(self):
    h = ScanHarness()
    h.object_sweep()
    h.object_sweep()
    result = h.object_sweep()
    assert len(result.points) == 1

  def test_continuity_across_skipped_cycles(self):
    h = ScanHarness()
    h.object_sweep()
    h.object_sweep()
    result = h.object_sweep(cycle_step=3, age_step=6)
    assert len(result.points) == 1

  def test_cycle_and_age_wraparound_stay_same_incarnation(self):
    h = ScanHarness()
    h.cycle = 14
    h._age = 4094
    h.object_sweep()  # cycle 15, age 4094+2 wraps
    h.object_sweep()  # cycle 0
    result = h.object_sweep()
    assert len(result.points) == 1

  def test_lifecycle_break_starts_new_incarnation(self):
    h = ScanHarness()
    h.object_sweep()
    h.object_sweep()
    # same handle, age jumps arbitrarily: history must not carry over, so no publication this sweep
    result = h.object_sweep(age_step=500)
    assert len(result.points) == 0
    result = h.object_sweep()
    assert len(result.points) == 1

  def test_death_then_rebirth_reuses_handle_with_clean_history(self):
    h = ScanHarness()
    h.object_sweep()
    h.object_sweep()
    for _ in range(4):
      h.sweep()  # object absent long enough to expire its ledger
    result = h.object_sweep()
    assert len(result.points) == 0
    result = h.object_sweep()
    assert len(result.points) == 1


class TestMotionPolicy:
  def test_native_speed_is_published(self):
    h = ScanHarness()
    speed_raw = CLOSING_SPEED_RAW_ZERO + 128
    cycle = (h.cycle + 1) & 0xF
    h.sweep(quartet(0, cycle, age=100, handle=5) + [motion_frame(0, cycle, speed_raw=speed_raw)])
    cycle = (h.cycle + 1) & 0xF
    result = h.sweep(quartet(0, cycle, age=102, handle=5) + [motion_frame(0, cycle, speed_raw=speed_raw)])
    assert result.points[0].vRel == pytest.approx(2.0)
    assert result.points[0].measured

  def test_missing_motion_frame_never_invalidates_geometry(self):
    h = ScanHarness()
    h.object_sweep(with_motion=False)
    result = h.object_sweep(with_motion=False)
    # without any motion source and no held speed, the point is withheld rather than synthesized
    assert len(result.points) == 0

  def test_stale_motion_cycle_is_ignored(self):
    h = ScanHarness()
    h.object_sweep()
    h.object_sweep()
    cycle = (h.cycle + 1) & 0xF
    h._age = (h._age + 2) & 0xFFF
    msgs = quartet(0, cycle, age=h._age, handle=5) + [motion_frame(0, (cycle - 1) & 0xF)]
    result = h.sweep(msgs)
    # motion from another cycle contributes nothing: coasts on held speed, unmeasured
    assert len(result.points) == 1
    assert not result.points[0].measured

  def test_high_sigma_speed_coasts_instead_of_synthesizing(self):
    h = ScanHarness()
    h.object_sweep()
    h.object_sweep()
    cycle = (h.cycle + 1) & 0xF
    h._age = (h._age + 2) & 0xFFF
    msgs = quartet(0, cycle, age=h._age, handle=5) + \
      [motion_frame(0, cycle, sigma_raw=CLOSING_SPEED_SIGMA_TRUST_MAX + 1)]
    result = h.sweep(msgs)
    assert len(result.points) == 1
    assert not result.points[0].measured
    assert result.points[0].vRel == pytest.approx(0.0)  # the held speed, not a derivative

  def test_ratio_field_supplies_speed_when_native_missing(self):
    h = ScanHarness()
    dist_raw = 1000
    cycle = (h.cycle + 1) & 0xF
    h.sweep(quartet(0, cycle, dist_raw=dist_raw, age=100, handle=5) +
            [motion_frame(0, cycle, speed_raw=CLOSING_SPEED_RAW_INVALID, ratio_raw=490)])
    cycle = (h.cycle + 1) & 0xF
    result = h.sweep(quartet(0, cycle, dist_raw=dist_raw, age=102, handle=5) +
                     [motion_frame(0, cycle, speed_raw=CLOSING_SPEED_RAW_INVALID, ratio_raw=490)])
    assert len(result.points) == 1
    dist = DIST_LSB_M * dist_raw + DIST_BIAS_M
    dt = SWEEP_DT_NS * 1e-9
    assert result.points[0].vRel == pytest.approx(dist * (1.0 - 0.99) / dt)
    assert result.points[0].measured

  def test_fast_clean_range_rate_without_sources_is_withheld(self):
    h = ScanHarness()
    h.object_sweep(with_motion=False, dist_raw=1000)
    # large clean jump with no motion evidence: raw-rate limit rejects the range outright
    result = h.object_sweep(with_motion=False, dist_raw=3000)
    assert len(result.points) == 0


class TestRangeAcceptance:
  def test_discontinuity_is_rejected_and_never_becomes_baseline(self):
    h = ScanHarness()
    h.object_sweep(dist_raw=1000)
    h.object_sweep(dist_raw=1002)
    # jump far beyond the hard innovation gate while claiming zero closing speed
    result = h.object_sweep(dist_raw=3000)
    assert len(result.points) == 1
    assert not result.points[0].measured
    # the rejected range did not become the derivative baseline: returning to the
    # consistent range publishes measured again
    result = h.object_sweep(dist_raw=1004)
    assert result.points[0].measured

  def test_small_innovation_accepted(self):
    h = ScanHarness()
    h.object_sweep(dist_raw=1000)
    result = h.object_sweep(dist_raw=1005)
    assert result.points[0].measured


class TestSlotsAndIdentity:
  def test_slot_migration_preserves_identity(self):
    h = ScanHarness()
    h.object_sweep(slot=2)
    h.object_sweep(slot=2)
    result = h.object_sweep(slot=9)
    assert len(result.points) == 1
    assert result.points[0].trackId == 5

  def test_duplicate_identity_prefers_bound_slot(self):
    h = ScanHarness()
    h.object_sweep(slot=2, dist_raw=1000)
    h.object_sweep(slot=2, dist_raw=1002)
    cycle = (h.cycle + 1) & 0xF
    h._age = (h._age + 2) & 0xFFF
    msgs = quartet(2, cycle, dist_raw=1004, age=h._age, handle=5) + [motion_frame(2, cycle)] + \
      quartet(9, cycle, dist_raw=2000, age=h._age, handle=5) + [motion_frame(9, cycle)]
    result = h.sweep(msgs)
    assert len(result.points) == 1
    assert result.points[0].dRel == pytest.approx(DIST_LSB_M * 1004 + DIST_BIAS_M)

  def test_slot_replacement_hides_old_occupant(self):
    h = ScanHarness()
    h.object_sweep(slot=3, handle=7)
    h.object_sweep(slot=3, handle=7)
    # a different identity takes the slot; the old one is hidden but not destroyed
    result = h.object_sweep(slot=3, handle=9, age_step=333)
    assert all(p.trackId != 7 for p in result.points)

  def test_one_identity_never_two_points(self):
    h = ScanHarness()
    cycle = (h.cycle + 1) & 0xF
    msgs = quartet(1, cycle, age=100, handle=5) + [motion_frame(1, cycle)] + \
      quartet(6, cycle, age=100, handle=5) + [motion_frame(6, cycle)]
    h.sweep(msgs)
    cycle = (h.cycle + 1) & 0xF
    msgs = quartet(1, cycle, age=102, handle=5) + [motion_frame(1, cycle)] + \
      quartet(6, cycle, age=102, handle=5) + [motion_frame(6, cycle)]
    result = h.sweep(msgs)
    assert len(result.points) == 1


class TestBusSilence:
  def test_quiet_bus_publishes_empty_not_none(self):
    h = ScanHarness()
    h.object_sweep()
    h.object_sweep()
    result = None
    for _ in range(30):
      result = h.feed([], dt_ns=10_000_000)
      if result is not None:
        break
    assert result is not None
    assert result.errors.radarUnavailableTemporary
    assert len(result.points) == 0

  def test_recovery_after_silence_starts_fresh(self):
    h = ScanHarness()
    h.object_sweep()
    h.object_sweep()
    for _ in range(30):
      if h.feed([], dt_ns=10_000_000) is not None:
        break
    result = h.object_sweep()
    assert len(result.points) == 0
    result = h.object_sweep()
    assert len(result.points) == 1

  def test_no_stale_publication_before_first_sweep(self):
    h = ScanHarness()
    for _ in range(50):
      assert h.feed([], dt_ns=10_000_000) is None


class TestQuietTimeoutValue:
  def test_timeout_is_about_three_sweeps(self):
    assert QUIET_TIMEOUT_S == pytest.approx(3 / 15, abs=0.01)


class TestScanInterfaceGating:
  def build(self, candidate, alpha_long=False, docs=False):
    from iqdbc.car import gen_empty_fingerprint
    from iqdbc.car.honda.interface import CarInterface
    CP = CarInterface.get_params(candidate, gen_empty_fingerprint(), [], alpha_long, False, docs)
    return CP

  def test_verified_platform_has_radar(self):
    from iqdbc.car.honda.values import CAR
    for car in (CAR.HONDA_CIVIC_BOSCH, CAR.HONDA_ACCORD, CAR.HONDA_CRV_5G):
      assert not self.build(car).radarUnavailable

  def test_radar_survives_openpilot_longitudinal(self):
    from iqdbc.car.honda.values import CAR
    CP = self.build(CAR.HONDA_CIVIC_BOSCH, alpha_long=True)
    assert CP.openpilotLongitudinalControl
    assert not CP.radarUnavailable

  def test_unverified_family_platform_stays_off(self):
    from iqdbc.car.honda.values import CAR
    for car in (CAR.HONDA_E, CAR.HONDA_INSIGHT, CAR.HONDA_NBOX_2G, CAR.ACURA_RDX_3G, CAR.HONDA_CRV_HYBRID):
      assert self.build(car).radarUnavailable

  def test_radarless_and_canfd_stay_off(self):
    from iqdbc.car.honda.values import CAR
    assert self.build(CAR.HONDA_CIVIC_2022).radarUnavailable
    assert self.build(CAR.HONDA_CRV_6G).radarUnavailable

  def test_docs_never_claim_radar(self):
    from iqdbc.car.honda.values import CAR
    assert self.build(CAR.HONDA_CIVIC_BOSCH, docs=True).radarUnavailable

  def test_radar_interface_routes_scanner(self):
    from iqdbc.car import gen_empty_fingerprint
    from iqdbc.car.honda.interface import CarInterface
    from iqdbc.car.honda.values import CAR
    CP = self.build(CAR.HONDA_CIVIC_BOSCH)
    CP_IQ = CarInterface.get_params_iq(CP, CAR.HONDA_CIVIC_BOSCH, gen_empty_fingerprint(), [], False, False, False)
    ri = CarInterface.RadarInterface(CP, CP_IQ)
    assert ri.scanner is not None
    assert ri.trigger_msg == SWEEP_TRIGGER_ADDR

  def test_radar_interface_keeps_nidec_path(self):
    from iqdbc.car import gen_empty_fingerprint
    from iqdbc.car.honda.interface import CarInterface
    from iqdbc.car.honda.values import CAR
    CP = self.build(CAR.HONDA_CIVIC)
    CP_IQ = CarInterface.get_params_iq(CP, CAR.HONDA_CIVIC, gen_empty_fingerprint(), [], False, False, False)
    ri = CarInterface.RadarInterface(CP, CP_IQ)
    assert ri.scanner is None
    assert ri.trigger_msg == 0x445

  def test_radar_interface_sleeps_when_unavailable(self):
    from iqdbc.car import gen_empty_fingerprint
    from iqdbc.car.honda.interface import CarInterface
    from iqdbc.car.honda.values import CAR
    CP = self.build(CAR.HONDA_E)
    CP_IQ = CarInterface.get_params_iq(CP, CAR.HONDA_E, gen_empty_fingerprint(), [], False, False, False)
    ri = CarInterface.RadarInterface(CP, CP_IQ)
    assert ri.scanner is None and ri.rcp is None
