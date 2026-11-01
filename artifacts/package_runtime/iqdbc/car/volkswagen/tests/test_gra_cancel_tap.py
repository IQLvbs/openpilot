from iqdbc.car.volkswagen.carcontroller import CarController
from iqdbc.car.volkswagen.values import CarControllerParams


class _Tapper:
  CCP = CarControllerParams

  def __init__(self):
    self.gra_cancel_ticks = 0

  def tap(self, cancel_req, gra_send_ready=True):
    return CarController._tap_gra_cancel(self, cancel_req, gra_send_ready)


def _pattern(tapper, ticks, cancel_req=True):
  return "".join("1" if tapper.tap(cancel_req) else "0" for _ in range(ticks))


def test_held_cancel_is_bounded():
  held = _pattern(_Tapper(), 2000).count("1")
  assert held == CarControllerParams.GRA_CANCEL_TAP_ON * CarControllerParams.GRA_CANCEL_MAX_TAPS
  assert held < 50


def test_cancel_is_tapped_not_held():
  on, off = CarControllerParams.GRA_CANCEL_TAP_ON, CarControllerParams.GRA_CANCEL_TAP_OFF
  expected = ("1" * on + "0" * off) * CarControllerParams.GRA_CANCEL_MAX_TAPS
  assert _pattern(_Tapper(), len(expected)) == expected


def test_taps_rearm_after_request_clears():
  tapper = _Tapper()
  _pattern(tapper, 2000)
  assert not tapper.tap(False)
  assert _pattern(tapper, CarControllerParams.GRA_CANCEL_TAP_ON) == "1" * CarControllerParams.GRA_CANCEL_TAP_ON


def test_ticks_only_advance_on_stock_counter_change():
  tapper = _Tapper()
  for _ in range(500):
    assert tapper.tap(True, gra_send_ready=False)
  assert tapper.gra_cancel_ticks == 0
