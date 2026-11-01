from iqdbc.car import DT_CTRL, gen_empty_fingerprint, structs
from iqdbc.car.honda.interface import CarInterface
from iqdbc.car.honda.values import CAR

CANFD_CAR = CAR.HONDA_CRV_6G

RADAR_DIAG_ADDR = 0x18DAB0F1
ACC_CONTROL_ADDR = 0x1DF
ACC_HUD_ADDR = 0x30C
SCM_BUTTONS_ADDR = 0x296
RADAR_HUD_ADDR = 0x310
LANE_PATH_ADDR = 0x6CD5558
HUD_OBJECTS_ADDR = 0x6CD5559
RADAR_LEAD_ADDR = 0xF31AA5C
RADAR_LEAD2_ADDR = 0xF31AA52
SUPPLEMENTAL_ADDR = 0x1A45AA4E
LOOKALIKE_ADDRS = (RADAR_HUD_ADDR, LANE_PATH_ADDR, HUD_OBJECTS_ADDR, RADAR_LEAD_ADDR, RADAR_LEAD2_ADDR, SUPPLEMENTAL_ADDR)

EXT_DIAG_SESSION = b'\x02\x10\x03\x00\x00\x00\x00\x00'
COMM_CONTROL_DISABLE = b'\x03\x28\x83\x03\x00\x00\x00\x00'


def build_long_interface():
  fingerprint = gen_empty_fingerprint()
  CP = CarInterface.get_params(CANFD_CAR, fingerprint, [], False, False, False)
  CP.openpilotLongitudinalControl = True
  CP.pcmCruise = False
  CP_IQ = CarInterface.get_params_iq(CP, CANFD_CAR, fingerprint, [], False, False, False)
  return CarInterface(CP, CP_IQ)


def make_cc(enabled=True):
  CC = structs.CarControl()
  CC.enabled = enabled
  CC.latActive = enabled
  CC.longActive = enabled
  return CC.as_reader()


class CanfdControllerHarness:
  def __init__(self):
    self.ci = build_long_interface()
    self.cs = self.ci.CS
    self.ci.update([])
    self.now_nanos = 0
    self.set_radar(alive=True, relay_open=False)
    self.set_ticks()

  def set_radar(self, alive, relay_open):
    self.cs.stock_acc_alive = alive
    self.cs.canfd_relay_open = relay_open

  def set_ticks(self, hud=False, supp=False, five=False, fifty=False):
    self.cs.hud_tick = hud
    self.cs.supp_tick = supp
    self.cs.radar_5hz_tick = five
    self.cs.radar_50hz_tick = fifty

  def step(self, CC=None, model=None):
    self.now_nanos += int(DT_CTRL * 1e9)
    _, can_sends = self.ci.apply(CC or make_cc(), structs.IQCarControl(), self.now_nanos, model)
    return can_sends

  @staticmethod
  def by_addr(can_sends, addr):
    return [m for m in can_sends if m[0] == addr]


class TestCanfdDeferredRadarDisable:
  def setup_method(self):
    self.h = CanfdControllerHarness()

  def test_no_disable_requests_before_relay_open(self):
    for _ in range(20):
      sends = self.h.step()
      assert not self.h.by_addr(sends, RADAR_DIAG_ADDR)
      assert not self.h.by_addr(sends, ACC_CONTROL_ADDR)
      assert not any(self.h.by_addr(sends, a) for a in LOOKALIKE_ADDRS)

  def test_disable_handshake_after_relay_open(self):
    self.h.set_radar(alive=True, relay_open=True)
    payloads = []
    for _ in range(101):
      for msg in self.h.by_addr(self.h.step(), RADAR_DIAG_ADDR):
        payloads.append(msg[1])
    assert payloads == [EXT_DIAG_SESSION, COMM_CONTROL_DISABLE, EXT_DIAG_SESSION, COMM_CONTROL_DISABLE, EXT_DIAG_SESSION]

  def test_tester_present_keeps_radar_down_once_silent(self):
    self.h.set_radar(alive=False, relay_open=True)
    payloads = []
    for _ in range(60):
      payloads += [m[1] for m in self.h.by_addr(self.h.step(), RADAR_DIAG_ADDR)]
    assert payloads == [b'\x02\x3E\x80\x00\x00\x00\x00\x00'] * 6


class TestCanfdReplacementStream:
  def setup_method(self):
    self.h = CanfdControllerHarness()
    self.h.set_radar(alive=False, relay_open=True)

  def test_acc_control_every_second_frame(self):
    seen = [bool(self.h.by_addr(self.h.step(), ACC_CONTROL_ADDR)) for _ in range(10)]
    assert sum(seen) == 5

  def test_no_acc_control_while_stock_alive(self):
    self.h.set_radar(alive=True, relay_open=True)
    for _ in range(10):
      assert not self.h.by_addr(self.h.step(), ACC_CONTROL_ADDR)

  def test_lookalikes_mirrored_byte_identical_on_both_buses(self):
    self.h.set_ticks(hud=True, supp=True, five=True, fifty=True)
    sends = self.h.step()
    for addr in LOOKALIKE_ADDRS:
      msgs = self.h.by_addr(sends, addr)
      assert len(msgs) == 2, hex(addr)
      buses = sorted(m[2] for m in msgs)
      assert buses == [0, 2], hex(addr)
      assert msgs[0][1] == msgs[1][1], hex(addr)

  def test_no_lookalikes_without_ticks(self):
    sends = self.h.step()
    for addr in (RADAR_HUD_ADDR, RADAR_LEAD_ADDR, RADAR_LEAD2_ADDR, SUPPLEMENTAL_ADDR, LANE_PATH_ADDR, HUD_OBJECTS_ADDR):
      assert not self.h.by_addr(sends, addr)

  def test_mux_sweep_contiguous_across_banks(self):
    self.h.set_ticks(fifty=True)
    muxes = []
    for _ in range(45):
      msgs = self.h.by_addr(self.h.step(), LANE_PATH_ADDR)
      muxes.append(msgs[0][1][0] >> 2)
    sweep = list(range(1, 11)) + list(range(17, 27)) + list(range(33, 43)) + list(range(49, 59))
    assert muxes == (sweep + sweep)[:45]

  def test_acc_hud_rides_hud_tick(self):
    assert not self.h.by_addr(self.h.step(), ACC_HUD_ADDR)
    self.h.set_ticks(hud=True)
    assert self.h.by_addr(self.h.step(), ACC_HUD_ADDR)
    self.h.set_ticks()
    assert not self.h.by_addr(self.h.step(), ACC_HUD_ADDR)


class TestCanfdButtonTakeover:
  def setup_method(self):
    self.h = CanfdControllerHarness()
    self.h.set_radar(alive=False, relay_open=True)

  def test_buttons_streamed_to_camera_while_engaged(self):
    seen = 0
    for _ in range(20):
      for msg in self.h.by_addr(self.h.step(), SCM_BUTTONS_ADDR):
        assert msg[2] == 2
        seen += 1
    assert seen == 5

  def test_no_button_stream_when_disengaged(self):
    for _ in range(20):
      assert not self.h.by_addr(self.h.step(make_cc(enabled=False)), SCM_BUTTONS_ADDR)

  def test_ambient_light_echoed(self):
    self.h.cs.scm_ambient_light = 0x77
    for _ in range(4):
      msgs = self.h.by_addr(self.h.step(), SCM_BUTTONS_ADDR)
      if msgs:
        assert msgs[0][1][2] == 0x77
        return
    raise AssertionError("no SCM_BUTTONS takeover frame seen")
