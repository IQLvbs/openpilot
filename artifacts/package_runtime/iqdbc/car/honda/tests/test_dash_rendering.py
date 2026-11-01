import math
from types import SimpleNamespace

import numpy as np

from iqdbc.can import CANPacker
from iqdbc.car.honda import dash_lane, dash_objects

V_EGO = 30.0


def model_at(center_y):
  x = list(np.linspace(0.0, 110.0, 23))

  def line(y):
    return SimpleNamespace(x=x, y=[y] * len(x))
  return SimpleNamespace(laneLines=[line(center_y + 3.3), line(center_y + 1.65), line(center_y - 1.65), line(center_y - 3.3)],
                         laneLineProbs=[0.0, 1.0, 1.0, 0.0],
                         leadsV3=[])


def lane_xy(center_y):
  m = model_at(center_y)
  return m.laneLines[1].x, [(a + b) / 2.0 for a, b in zip(m.laneLines[1].y, m.laneLines[2].y, strict=True)]


class TestLanePathSlew:
  def test_first_fit_shown_unslewed(self):
    renderer = dash_lane.LanePathRenderer()
    lane = renderer.update(model_at(-2.0), V_EGO, 0.0)
    assert lane.offsets == dash_lane.encode_lane_path(*lane_xy(-2.0))

  def test_step_is_rate_limited(self):
    renderer = dash_lane.LanePathRenderer()
    prev = renderer.update(model_at(0.0), V_EGO, 0.0).offsets
    assert all(o == 0 for o in prev)

    target = dash_lane.encode_lane_path(*lane_xy(-2.0))
    max_step = math.ceil(dash_lane.SLEW_MAX_STEP)
    for _ in range(10):
      cur = renderer.update(model_at(-2.0), V_EGO, 0.0).offsets
      for p, c, t in zip(prev, cur, target, strict=True):
        assert abs(c - p) <= max_step
        assert abs(t - c) <= abs(t - p)
      prev = cur
    assert prev == target

  def test_full_scale_takes_two_seconds(self):
    renderer = dash_lane.LanePathRenderer()
    renderer.update(model_at(0.0), V_EGO, 0.0)
    target = dash_lane.encode_lane_path(*lane_xy(-100.0))
    assert all(t == dash_lane.OFFSET_VALID_MAX for t in target)

    n_updates = round(dash_lane.SLEW_FULL_SCALE_S * dash_lane.SLEW_RATE_HZ)
    for i in range(n_updates):
      lane = renderer.update(model_at(-100.0), V_EGO, 0.0)
      if i < n_updates - 1:
        assert lane.offsets != target
    assert lane.offsets == target

  def test_blank_resets_slew(self):
    renderer = dash_lane.LanePathRenderer()
    renderer.update(model_at(0.0), V_EGO, 0.0)
    lane = renderer.update(None, V_EGO, 0.0)
    assert lane.offsets == [dash_lane.OFFSET_UNAVAILABLE] * dash_lane.POINT_COUNT
    lane = renderer.update(model_at(-2.0), V_EGO, 0.0)
    assert lane.offsets == dash_lane.encode_lane_path(*lane_xy(-2.0))

  def test_short_path_passthrough_and_reset(self):
    renderer = dash_lane.LanePathRenderer()
    renderer.update(model_at(0.0), V_EGO, 0.0)

    short = model_at(-2.0)
    for ll in short.laneLines:
      ll.x = ll.x[:10]
      ll.y = ll.y[:10]
    lane = renderer.update(short, V_EGO, 0.0)
    assert lane.offsets == [dash_lane.OFFSET_UNAVAILABLE] * dash_lane.POINT_COUNT

    lane = renderer.update(model_at(-2.0), V_EGO, 0.0)
    assert lane.offsets == dash_lane.encode_lane_path(*lane_xy(-2.0))


class TestLaneLineHysteresis:
  def test_single_line_offset_and_hysteresis(self):
    renderer = dash_lane.LanePathRenderer()
    m = model_at(0.0)
    m.laneLineProbs = [0.0, 0.0, 1.0, 0.0]
    lane = renderer.update(m, V_EGO, 0.0)
    assert not lane.left_line and lane.right_line
    assert lane.offsets == dash_lane.encode_lane_path(m.laneLines[2].x, [y - dash_lane.HALF_LANE_M for y in m.laneLines[2].y])

    # a left prob between OFF and ON must not switch the left line on
    m.laneLineProbs = [0.0, (dash_lane.LINE_PROB_OFF + dash_lane.LINE_PROB_ON) / 2, 1.0, 0.0]
    lane = renderer.update(m, V_EGO, 0.0)
    assert not lane.left_line

    # once on, the same mid prob keeps it on
    m.laneLineProbs = [0.0, dash_lane.LINE_PROB_ON, 1.0, 0.0]
    assert renderer.update(m, V_EGO, 0.0).left_line
    m.laneLineProbs = [0.0, (dash_lane.LINE_PROB_OFF + dash_lane.LINE_PROB_ON) / 2, 1.0, 0.0]
    assert renderer.update(m, V_EGO, 0.0).left_line


class TestCanfdReshape:
  def test_idle_pattern_when_blank(self):
    assert dash_lane.canfd_lane_offsets(dash_lane.RenderedLane()) == dash_lane.CANFD_IDLE_OFFSETS
    assert dash_lane.canfd_lane_length(dash_lane.RenderedLane()) == dash_lane.CANFD_MIN_VALID_PTS

  def test_terminated_prefix_matches_length_law(self):
    for v_ego, expected in ((0.0, 7), (10.0, 15), (19.0, 23), (38.0, 23)):
      lane = dash_lane.RenderedLane(offsets=[5] * dash_lane.POINT_COUNT, reach=1.0, v_ego=v_ego)
      n = dash_lane.canfd_lane_length(lane)
      assert n == expected
      offs = dash_lane.canfd_lane_offsets(lane)
      assert offs[:n] == [5] * n
      assert offs[n:] == [dash_lane.OFFSET_UNAVAILABLE] * (dash_lane.POINT_COUNT - n)


class TestMuxMapping:
  def test_mux_cycle_covers_all_banks(self):
    assert len(dash_lane.MUX_CYCLE) == 40
    assert set(dash_lane.MUX_CYCLE) == set(range(1, 11)) | set(range(17, 27)) | set(range(33, 43)) | set(range(49, 59))

  def test_lane_path_frame_selects_offsets_by_mux(self):
    packer = CANPacker("honda_bosch_radarless_generated")
    offsets = list(range(40))
    for mux in dash_lane.MUX_CYCLE:
      addr, dat, bus = dash_lane.create_lane_path(packer, 0, offsets, mux)
      base = ((mux - 1) % 16) * 4
      raw_mux = dat[0] >> 2
      assert raw_mux == mux
      assert base < 40


class TestDashObjectAuthor:
  def make_lead(self, prob=0.9, d=30.0, y=0.0, v=0.0):
    status = prob >= dash_objects.LEAD_PROB_ON
    return dash_objects.ModelLead(status, d, y, v, prob=prob)

  def payload(self, msg):
    return msg[1]

  def test_inactive_slot_bytes_match_stock_sentinel(self):
    packer = CANPacker("honda_common_canfd_generated")
    author = dash_objects.DashObjectAuthor()
    msg = author.create(packer, 0, self.make_lead(prob=0.0), None, 2, 0.0)
    parsed_long = ((self.payload(msg)[4] << 2) | (self.payload(msg)[5] >> 6)) & 0x3FF
    assert parsed_long == 1023

  def test_lead_rendered_in_slot0_only(self):
    packer = CANPacker("honda_common_canfd_generated")
    author = dash_objects.DashObjectAuthor()
    lead = self.make_lead()
    slot0 = author.create(packer, 0, lead, None, 1, 0.0)
    slot3 = author.create(packer, 0, lead, None, 4, 0.02)
    assert self.payload(slot0)[1] != 0
    assert self.payload(slot3)[1] & 0xF8 == 0

  def test_lead_prob_hysteresis_and_hold(self):
    packer = CANPacker("honda_common_canfd_generated")
    author = dash_objects.DashObjectAuthor()
    now = 0.0

    def object_id(prob):
      nonlocal now
      now += 0.02
      msg = author.create(packer, 0, self.make_lead(prob=prob), None, 1, now)
      return self.payload(msg)[1] >> 3

    assert object_id(0.6) != 0
    # dips below ON but above OFF keep rendering
    assert object_id(0.4) != 0
    # a full drop is bridged for LEAD_HOLD_S
    assert object_id(0.0) != 0
    now += dash_objects.LEAD_HOLD_S
    assert object_id(0.0) == 0

  def test_reid_on_range_discontinuity(self):
    ident = dash_objects.LeadIdentity()
    now = 0.0
    first = ident.update(True, 30.0, 0.0, now)
    # stay steady past the re-id refractory window
    for _ in range(int(dash_objects.REID_REFRACTORY / 0.02) + 10):
      now += 0.02
      same = ident.update(True, 30.0, 0.0, now)
      assert same == first
    now += 0.02
    assert ident.update(True, 60.0, 0.0, now) != first

  def test_camera_lead_never_forwarded(self):
    packer = CANPacker("honda_bosch_radarless_generated")
    author = dash_objects.DashObjectAuthor()
    tracks = [dash_objects.CameraObject(slot=i, object_id=0, d_rel=0.0, y_rel=0.0, is_lead_car=False, valid=False)
              for i in range(dash_objects.NUM_SLOTS)]
    tracks[0] = dash_objects.CameraObject(slot=0, object_id=9, d_rel=40.0, y_rel=0.0, is_lead_car=True, valid=True,
                                          car_type=7, rotation=0)
    msg = author.create(packer, 0, self.make_lead(prob=0.0), tracks, 1, 0.0)
    assert self.payload(msg)[1] >> 3 == 0

  def test_adjacent_car_forwarded_with_own_mux(self):
    packer = CANPacker("honda_bosch_radarless_generated")
    tracks = [dash_objects.CameraObject(slot=i, object_id=0, d_rel=0.0, y_rel=0.0, is_lead_car=False, valid=False)
              for i in range(dash_objects.NUM_SLOTS)]
    tracks[3] = dash_objects.CameraObject(slot=3, object_id=12, d_rel=25.0, y_rel=3.0, is_lead_car=False, valid=True,
                                          car_type=7, rotation=1)
    msg = dash_objects.forward_hud_object(packer, 0, 20, tracks)
    assert msg[1][0] >> 2 == 20
    assert msg[1][1] >> 3 == 12


class TestCameraObjectTracker:
  def test_tracks_persist_across_banks(self):
    tracker = dash_objects.CameraObjectTracker()

    class FakeParser:
      vl_all = {"HUD_OBJECTS": {
        "MUX": [2, 18], "OBJECT_ID": [5, 5], "LONG_DIST": [30.0, 31.0], "LAT_DIST": [1.0, 1.1],
        "IS_LEAD_CAR": [0, 0], "CAR_TYPE": [7, 7], "ROTATION": [0, 0],
      }}
    tracker.update(FakeParser())
    snap = tracker.snapshot()
    assert snap[1].valid and snap[1].object_id == 5
    assert snap[1].d_rel == 31.0

  def test_empty_sentinel_invalid(self):
    tracker = dash_objects.CameraObjectTracker()

    class FakeParser:
      vl_all = {"HUD_OBJECTS": {
        "MUX": [1], "OBJECT_ID": [0], "LONG_DIST": [196.9], "LAT_DIST": [204.7],
        "IS_LEAD_CAR": [0], "CAR_TYPE": [-1], "ROTATION": [-128],
      }}
    tracker.update(FakeParser())
    assert not tracker.snapshot()[0].valid
