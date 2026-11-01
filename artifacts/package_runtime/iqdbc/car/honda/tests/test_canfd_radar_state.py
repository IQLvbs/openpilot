import pytest

from iqdbc.can import CANPacker
from iqdbc.car import Bus, DT_CTRL, gen_empty_fingerprint
from iqdbc.car.honda.interface import CarInterface
from iqdbc.car.honda.values import CAR, DBC
from iqdbc.car.common.conversions import Conversions as CV

CANFD_CAR = CAR.HONDA_CRV_6G
RADARLESS_CAR = CAR.HONDA_CIVIC_2022
CAMERA_MESSAGES_ADDR = 0x35E


def build_car(candidate, extra_pt_addrs=()):
  fingerprint = gen_empty_fingerprint()
  for addr in extra_pt_addrs:
    fingerprint[0][addr] = 8
  CP = CarInterface.get_params(candidate, fingerprint, [], False, False, False)
  CP_IQ = CarInterface.get_params_iq(CP, candidate, fingerprint, [], False, False, False)
  return CarInterface(CP, CP_IQ)


class CanFeed:
  def __init__(self, ci, dbc_name):
    self.ci = ci
    self.packer = CANPacker(dbc_name)
    self.nanos = 0
    # the first CarState.update lazily subscribes vl-read messages, so run one empty
    # cycle before feeding data or the first fed frame of those messages is dropped
    self.step()
    self.ci.CS.update(self.ci.can_parsers)

  def step(self, msgs=()):
    self.nanos += int(DT_CTRL * 1e9)
    packed = [self.packer.make_can_msg(name, bus, values) for name, bus, values in msgs]
    for parser in self.ci.can_parsers.values():
      parser.update([self.nanos, packed])


class TestHondaCanfdRadarState:
  def setup_method(self):
    self.ci = build_car(CANFD_CAR)
    self.cs = self.ci.CS
    self.feed = CanFeed(self.ci, DBC[CANFD_CAR][Bus.pt])

  def update(self, msgs=()):
    self.feed.step(msgs)
    return self.cs.update(self.ci.can_parsers)

  def test_parsers_include_radar_bus(self):
    assert Bus.radar in self.ci.can_parsers
    assert self.ci.can_parsers[Bus.radar].bus == 1

  def test_50hz_tick_fires_one_frame_before_next_tick(self):
    ticks = []
    for frame in range(20):
      msgs = [("RADAR_50HZ_TICK_REFERENCE", 1, {})] if frame % 2 == 0 else []
      self.update(msgs)
      ticks.append(self.cs.radar_50hz_tick)
    assert ticks[2:] == [frame % 2 == 1 for frame in range(2, 20)]

  def test_hud_tick_fires_one_frame_before_next_tick(self):
    fired = []
    for frame in range(40):
      msgs = [("RADAR_HUD_TICK_REFERENCE", 1, {})] if frame % 10 == 0 else []
      self.update(msgs)
      if self.cs.hud_tick:
        fired.append(frame)
    assert fired == [9, 19, 29, 39]

  def test_5hz_tick_fires_at_stock_radar_lead_offset(self):
    fired = []
    for frame in range(60):
      msgs = [("RADAR_REFERENCE", 0, {})] if frame % 20 == 0 else []
      self.update(msgs)
      if self.cs.radar_5hz_tick:
        fired.append(frame)
    assert fired == [11, 31, 51]

  def test_stock_acc_alive_until_four_silent_frames(self):
    for frame in range(11):
      msgs = [("ACC_CONTROL", 0, {})] if frame % 2 == 0 else []
      self.update(msgs)
      assert self.cs.stock_acc_alive

    silent_state = []
    for _ in range(6):
      self.update()
      silent_state.append(self.cs.stock_acc_alive)
    assert silent_state == [True, True, True, False, False, False]

    self.update([("ACC_CONTROL", 0, {})])
    assert self.cs.stock_acc_alive

  def test_relay_open_when_camera_steering_disappears(self):
    for _ in range(10):
      self.update([("STEERING_CONTROL", 0, {})])
      assert not self.cs.canfd_relay_open
    assert self.cs.camera_steer_seen

    open_state = []
    for _ in range(7):
      self.update()
      open_state.append(self.cs.canfd_relay_open)
    assert open_state == [False, False, False, False, True, True, True]

  def test_relay_open_fallback_without_camera(self):
    primed_frames = self.cs.canfd_frames
    for frame in range(510):
      self.update()
      assert self.cs.canfd_relay_open == (primed_frames + frame + 1 >= 500)

  def test_ambient_light_echoed_from_scm_buttons(self):
    self.update([("SCM_BUTTONS", 0, {"AMBIENT_LIGHT_MAYBE": 0x5A})])
    assert self.cs.scm_ambient_light == 0x5A


class TestHondaNonCanfdRadarState:
  def test_no_radar_parser_and_ticks_stay_low(self):
    ci = build_car(RADARLESS_CAR)
    feed = CanFeed(ci, DBC[RADARLESS_CAR][Bus.pt])
    assert Bus.radar not in ci.can_parsers
    for _ in range(5):
      feed.step()
      ci.CS.update(ci.can_parsers)
      assert not ci.CS.radar_50hz_tick
      assert not ci.CS.hud_tick
      assert not ci.CS.supp_tick
      assert not ci.CS.radar_5hz_tick


class TestCanfdLongInterface:
  def test_alpha_long_available_on_canfd(self):
    CP = CarInterface.get_params(CANFD_CAR, gen_empty_fingerprint(), [], False, False, False)
    assert CP.alphaLongitudinalAvailable
    assert not CP.openpilotLongitudinalControl
    assert CP.pcmCruise

  def test_alpha_long_enabled_on_canfd(self):
    CP = CarInterface.get_params(CANFD_CAR, gen_empty_fingerprint(), [], True, False, False)
    assert CP.openpilotLongitudinalControl
    assert not CP.pcmCruise
    assert CP.longitudinalActuatorDelay == pytest.approx(0.05)

  def test_canfd_long_init_clears_dtcs_without_disabling_radar(self, mocker):
    clear_all = mocker.patch("iqdbc.car.honda.interface.clear_all_dtcs")
    clear_ecu = mocker.patch("iqdbc.car.honda.interface.clear_ecu_dtcs")
    disable = mocker.patch("iqdbc.car.honda.interface.disable_ecu")

    CP = CarInterface.get_params(CANFD_CAR, gen_empty_fingerprint(), [], True, False, False)
    CarInterface.init(CP, None, None, None)
    assert clear_all.call_count == 1
    assert clear_all.call_args.args[1] == [0, 2]
    assert clear_ecu.call_count == 1
    assert disable.call_count == 0

  def test_canfd_deinit_reenables_radar(self, mocker):
    clear_all = mocker.patch("iqdbc.car.honda.interface.clear_all_dtcs")
    disable = mocker.patch("iqdbc.car.honda.interface.disable_ecu")

    CP = CarInterface.get_params(CANFD_CAR, gen_empty_fingerprint(), [], True, False, False)
    CarInterface.deinit(CP, None, None)
    assert clear_all.call_count == 0
    assert disable.call_count == 1

  def test_bosch_a_long_init_still_disables_radar(self, mocker):
    clear_all = mocker.patch("iqdbc.car.honda.interface.clear_all_dtcs")
    disable = mocker.patch("iqdbc.car.honda.interface.disable_ecu")

    CP = CarInterface.get_params(CAR.HONDA_ACCORD, gen_empty_fingerprint(), [], True, False, False)
    CarInterface.init(CP, None, None, None)
    assert clear_all.call_count == 0
    assert disable.call_count == 1


class TestHondaDashboardSpeedLimit:
  def build(self, candidate, with_camera_messages):
    extra = (CAMERA_MESSAGES_ADDR,) if with_camera_messages else ()
    return build_car(candidate, extra_pt_addrs=extra)

  @pytest.mark.parametrize("sign_value,expected_mph", [(101, 25), (97, 5), (113, 85)])
  def test_speed_limit_sign_reported(self, sign_value, expected_mph):
    ci = self.build(RADARLESS_CAR, True)
    feed = CanFeed(ci, DBC[RADARLESS_CAR][Bus.pt])
    feed.step([("CAMERA_MESSAGES", 2, {"SPEED_LIMIT_SIGN": sign_value})])
    _, ret_iq = ci.CS.update(ci.can_parsers)
    assert ret_iq.speedLimit == pytest.approx(expected_mph * CV.MPH_TO_MS)

  @pytest.mark.parametrize("sign_value", [125, 0, 32])
  def test_invalid_sign_reports_no_limit(self, sign_value):
    ci = self.build(RADARLESS_CAR, True)
    feed = CanFeed(ci, DBC[RADARLESS_CAR][Bus.pt])
    feed.step([("CAMERA_MESSAGES", 2, {"SPEED_LIMIT_SIGN": sign_value})])
    _, ret_iq = ci.CS.update(ci.can_parsers)
    assert ret_iq.speedLimit == 0.0

  def test_without_camera_messages_flag_no_limit(self):
    ci = self.build(RADARLESS_CAR, False)
    feed = CanFeed(ci, DBC[RADARLESS_CAR][Bus.pt])
    feed.step([("CAMERA_MESSAGES", 2, {"SPEED_LIMIT_SIGN": 101})])
    _, ret_iq = ci.CS.update(ci.can_parsers)
    assert ret_iq.speedLimit == 0.0
