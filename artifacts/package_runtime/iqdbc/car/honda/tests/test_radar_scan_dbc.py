from iqdbc.can.dbc import DBC as DbcFile
from iqdbc.car import Bus
from iqdbc.car.honda.values import CAR, DBC, HONDA_RADAR_SCAN_CAPABLE, HONDA_RADAR_SCAN_VERIFIED
from iqdbc.dbc.generator.honda.honda_radar_scan import (FRAME_SIGNALS, QUARTET_KINDS, SCAN_SLOTS,
                                                        frame_address, motion_address, quartet_base_address)

SCAN_DBC_NAME = 'honda_radar_scan_generated'


class TestScanAddressing:
  def test_quartet_bases(self):
    assert [quartet_base_address(s) for s in range(SCAN_SLOTS)] == \
      [0x280, 0x284, 0x288, 0x28C, 0x2D0, 0x2D4, 0x2D8, 0x2DC, 0x2E0, 0x2E4, 0x2E8, 0x2EC, 0x2F0, 0x2F4, 0x2F8, 0x2FC]

  def test_motion_addresses(self):
    assert [motion_address(s) for s in range(SCAN_SLOTS)] == \
      [0x2C8, 0x2C9, 0x2CA, 0x2CB, 0x2CC, 0x2CD, 0x2CE, 0x2CF, 0x290, 0x291, 0x292, 0x293, 0x294, 0x295, 0x296, 0x297]

  def test_eighty_unique_addresses(self):
    addrs = [frame_address(s, k) for s in range(SCAN_SLOTS) for k in (*QUARTET_KINDS, "MOTION")]
    assert len(addrs) == 80
    assert len(set(addrs)) == 80

  def test_quartet_kind_order(self):
    for slot in range(SCAN_SLOTS):
      base = quartet_base_address(slot)
      assert [frame_address(slot, k) for k in QUARTET_KINDS] == [base, base + 1, base + 2, base + 3]


class TestScanDbcGeometry:
  def setup_method(self):
    self.dbc = DbcFile(SCAN_DBC_NAME)

  def geometry(self, addr):
    msg = self.dbc.addr_to_msg[addr]
    return {sig.name: (sig.start_bit, sig.size) for sig in msg.sigs.values()}

  def test_every_frame_present_with_size_8(self):
    for slot in range(SCAN_SLOTS):
      for kind in (*QUARTET_KINDS, "MOTION"):
        msg = self.dbc.addr_to_msg[frame_address(slot, kind)]
        assert msg.name == f"RADAR_SCAN_{slot:02d}_{kind}"
        assert msg.size == 8

  def test_bit_geometry_matches_spec(self):
    expected = {kind: {name: (start, size) for name, start, size in sigs} for kind, sigs in FRAME_SIGNALS.items()}
    for slot in range(SCAN_SLOTS):
      for kind in (*QUARTET_KINDS, "MOTION"):
        assert self.geometry(frame_address(slot, kind)) == expected[kind], (slot, kind)

  def test_pos_frame_field_widths(self):
    geo = self.geometry(frame_address(0, "POS"))
    assert geo["DIST_RAW"] == (23, 12)
    assert geo["BEARING_RAW"] == (39, 11)
    assert geo["SCAN_STATE"] == (15, 4)
    assert geo["DIST_SIGMA_RAW"] == (7, 7)

  def test_ident_handle_is_byte_six(self):
    geo = self.geometry(frame_address(0, "IDENT"))
    assert geo["OBJECT_HANDLE"] == (55, 8)

  def test_motion_field_widths(self):
    geo = self.geometry(frame_address(0, "MOTION"))
    assert geo["CLOSING_SPEED_RAW"] == (7, 11)
    assert geo["CLOSING_SPEED_SIGMA_RAW"] == (23, 10)
    assert geo["DIST_RATIO_RAW"] == (55, 10)

  def test_cycle_positions_per_kind(self):
    positions = {"POS": (27, 4), "SHAPE": (28, 4), "LIFE": (11, 4), "IDENT": (12, 4), "MOTION": (12, 4)}
    for kind, expected in positions.items():
      assert self.geometry(frame_address(3, kind))["CYCLE"] == expected


class TestScanPlatformWiring:
  def test_scan_dbc_on_exactly_the_capable_family(self):
    for car in CAR:
      has_scan_dbc = DBC[car].get(Bus.radar) == SCAN_DBC_NAME
      assert has_scan_dbc == (car in HONDA_RADAR_SCAN_CAPABLE), car

  def test_verified_platforms_are_capable(self):
    assert HONDA_RADAR_SCAN_VERIFIED <= HONDA_RADAR_SCAN_CAPABLE

  def test_verified_set(self):
    assert HONDA_RADAR_SCAN_VERIFIED == {CAR.HONDA_ACCORD, CAR.HONDA_CIVIC_BOSCH, CAR.HONDA_CRV_5G}
