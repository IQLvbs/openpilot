import numpy as np

from iqpilot.system.hardware.fan_controller import FanController


class TestFanController:
  def test_ramp_anchors(self):
    c = FanController()
    assert c.update(60, True) == 0
    assert c.update(70, True) == 0
    assert c.update(85, True) == 80
    assert c.update(90, True) == 100
    assert c.update(100, True) == 100

  def test_ramp_is_monotonic_and_continuous(self):
    c = FanController()
    temps = np.arange(50.0, 105.0, 0.25)
    outs = [c.update(t, True) for t in temps]
    assert all(b >= a for a, b in zip(outs, outs[1:]))
    # no step may exceed the steepest segment's slope (4 %/deg) over a 0.25 deg move
    assert max(b - a for a, b in zip(outs, outs[1:])) <= 2

  def test_hot_onroad(self):
    assert FanController().update(100, True) >= 70

  def test_offroad_capped(self):
    c = FanController()
    for t in (60, 75, 85, 100):
      assert c.update(t, False) <= 30

  def test_no_fan_wear(self):
    assert FanController().update(10, False) == 0

  def test_max_cool(self):
    c = FanController()
    assert c.update(80, True, True) == 100
    assert c.update(80, False, True) == 100

  def test_target_band_has_airflow(self):
    # the design centers on 75 C; the curve must actually move air there
    assert 20 <= FanController().update(75, True) <= 40
