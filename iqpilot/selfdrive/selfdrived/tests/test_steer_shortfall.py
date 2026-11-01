"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos
"""
from iqpilot.common.realtime import DT_CTRL
from iqpilot.selfdrive.selfdrived.helpers import STEER_SHORTFALL_TIME, SteerShortfallCheck

V_EGO = 12.0
DELAY = 0.3
DELAY_FRAMES = int(round(DELAY / DT_CTRL))
CURVE = 2.0 / V_EGO ** 2


def run(desired, actual, v_ego=V_EGO, eligible=True):
  check = SteerShortfallCheck()
  for _ in range(int(1.0 / DT_CTRL)):
    check.update(0.0, v_ego, 0.0, DELAY, eligible)
  return [check.update(k, v_ego, a, DELAY, eligible) for k, a in zip(desired, actual, strict=True)]


def lagged_tracking(desired, gain=1.0):
  return [gain * desired[max(i - DELAY_FRAMES, 0)] * V_EGO ** 2 for i in range(len(desired))]


def curve_entry(frames=600, ramp=150):
  return [CURVE * min(i / ramp, 1.0) for i in range(frames)]


def test_lagging_car_that_follows_the_curve_never_alerts():
  desired = curve_entry()

  assert not any(run(desired, lagged_tracking(desired)))


def test_sustained_shortfall_alerts_after_hold_time():
  desired = [CURVE] * 300
  alerts = run(desired, lagged_tracking(desired, gain=0.4))

  elapsed = (alerts.index(True) + 1) * DT_CTRL
  assert elapsed >= STEER_SHORTFALL_TIME + DELAY - DT_CTRL / 2
  assert elapsed <= STEER_SHORTFALL_TIME + DELAY + 2 * DT_CTRL


def test_brief_shortfall_does_not_alert():
  frames = int((STEER_SHORTFALL_TIME - 0.2) / DT_CTRL)
  desired = [CURVE] * frames + [0.0] * 200
  actual = [0.0] * len(desired)

  assert not any(run(desired, actual))


def test_turning_the_wrong_way_counts_as_shortfall():
  desired = [CURVE] * 300
  actual = [-CURVE * V_EGO ** 2] * 300

  assert any(run(desired, actual))


def test_driver_steering_suppresses_alert():
  desired = [CURVE] * 300

  assert not any(run(desired, [0.0] * 300, eligible=False))


def test_low_speed_never_alerts():
  v_ego = 4.0
  desired = [3.0 / v_ego ** 2] * 300

  assert not any(run(desired, [0.0] * 300, v_ego=v_ego))


def test_gentle_curve_never_alerts():
  desired = [0.8 / V_EGO ** 2] * 300

  assert not any(run(desired, [0.0] * 300))
