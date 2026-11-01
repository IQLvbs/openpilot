"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""

from iqdbc.car.volkswagen.mebutils import LongStopHold

DT = 0.02


def build():
  return LongStopHold(dt=DT)


def settle(hold, frames=40, esp_hold=True):
  out = None
  for _ in range(frames):
    out = hold.update(True, True, False, -0.56, esp_hold, 0.0, False)
  return out


def test_hold_survives_single_frame_stop_request_dropout():
  hold = build()
  settle(hold)

  stopping, starting, esp_hold, accel = hold.update(True, False, True, 0.4, True, 0.0, False)

  assert stopping
  assert not starting
  assert esp_hold
  assert accel <= 0.0


def test_hold_survives_esp_confirmation_bounce():
  hold = build()
  settle(hold)

  stopping, _, esp_hold, _ = hold.update(True, True, False, -0.56, False, 0.0, False)

  assert stopping
  assert esp_hold


def test_esp_confirmation_clears_after_sustained_loss():
  hold = build()
  settle(hold)

  for _ in range(int(LongStopHold.HOLD_CLEAR_TIME / DT) + 1):
    _, _, esp_hold, _ = hold.update(True, True, False, -0.56, False, 0.0, False)

  assert not esp_hold


def test_sustained_drive_off_releases_hold():
  hold = build()
  settle(hold)

  released_after = None
  for i in range(int(LongStopHold.DRIVE_OFF_CONFIRM_TIME / DT) + 5):
    stopping, starting, _, accel = hold.update(True, False, True, 0.4, True, 0.0, False)
    if not stopping:
      released_after = (i + 1) * DT
      break

  assert released_after is not None
  assert released_after >= LongStopHold.DRIVE_OFF_CONFIRM_TIME
  assert starting
  assert accel == 0.4


def test_chatter_never_reaches_the_car():
  hold = build()
  settle(hold)

  for i in range(200):
    stop_request = i % 2 == 0
    accel = -0.56 if stop_request else 0.3
    stopping, starting, _, sent = hold.update(True, stop_request, not stop_request, accel, True, 0.0, False)
    assert stopping
    assert not starting
    assert sent <= 0.0


def test_driver_override_passes_through():
  hold = build()
  settle(hold)

  stopping, starting, esp_hold, accel = hold.update(True, False, True, 0.6, True, 0.0, True)

  assert not stopping
  assert starting
  assert esp_hold
  assert accel == 0.6


def test_rolling_car_is_never_held():
  hold = build()
  settle(hold)

  stopping, _, _, _ = hold.update(True, False, True, 0.4, False, 1.5, False)

  assert not stopping
