# Copyright (c) 2026 IQ.Lvbs. All rights reserved.

import errno
import os
import threading
import time
from types import SimpleNamespace

from iqpilot.system.sensord import sensord


class FakeSensor:
  def __init__(self):
    self.resets = 0
    self.inits = 0

  def reset(self):
    self.resets += 1

  def init(self):
    self.inits += 1


def busy_then_free(failures: int):
  calls = []

  def request(label, chip, pin):
    calls.append((label, chip, pin))
    if len(calls) <= failures:
      raise OSError(errno.EBUSY, "Device or resource busy")
    return 99

  return request, calls


def test_interrupt_line_is_retried_while_another_process_holds_it(monkeypatch):
  request, calls = busy_then_free(failures=3)
  monkeypatch.setattr(sensord, "gpiochip_get_ro_value_fd", request)
  monkeypatch.setattr(sensord, "GPIO_RETRY_S", 0.001)
  assert sensord.acquire_interrupt_fd(threading.Event()) == 99
  assert calls == [("sensord", 0, 84)] * 4


def test_interrupt_line_wait_ends_when_asked_to_stop(monkeypatch):
  request, _ = busy_then_free(failures=10**9)
  monkeypatch.setattr(sensord, "gpiochip_get_ro_value_fd", request)
  monkeypatch.setattr(sensord, "GPIO_RETRY_S", 0.001)
  event = threading.Event()
  threading.Timer(0.05, event.set).start()
  assert sensord.acquire_interrupt_fd(event) is None


def test_interrupt_thread_survives_a_failing_loop(monkeypatch):
  event = threading.Event()
  attempts = []

  def failing_loop(sensors, ev):
    attempts.append(1)
    if len(attempts) >= 3:
      ev.set()
    raise RuntimeError("boom")

  monkeypatch.setattr(sensord, "interrupt_loop", failing_loop)
  monkeypatch.setattr(sensord, "THREAD_RESTART_S", 0.001)
  sensord.interrupt_thread([], event)
  assert len(attempts) == 3


def test_silent_imu_is_reinitialized(monkeypatch):
  monkeypatch.setattr(sensord, "SILENCE_REINIT_S", 0.05)
  monkeypatch.setattr(sensord, "REINIT_MIN_INTERVAL_S", 0.05)
  monkeypatch.setattr(sensord, "sudo_write", lambda *a: None)
  accel, gyro = FakeSensor(), FakeSensor()
  sensors = [(accel, "accelerometer", True), (gyro, "gyroscope", True)]
  read_fd, write_fd = os.pipe()
  event = threading.Event()
  worker = threading.Thread(
    target=sensord.run_interrupt_loop,
    args=(sensors, event, SimpleNamespace(send=lambda *a: None), read_fd),
  )
  worker.start()
  time.sleep(0.8)
  event.set()
  worker.join(timeout=2)
  os.close(read_fd)
  os.close(write_fd)
  assert not worker.is_alive()
  assert accel.inits >= 3 and gyro.inits >= 3
  assert accel.resets == accel.inits


def test_irq_affinity_failure_does_not_kill_the_loop(monkeypatch):
  def fail(*a):
    raise OSError(errno.EIO, "io")

  monkeypatch.setattr(sensord, "sudo_write", fail)
  monkeypatch.setattr(os.path, "exists", lambda p: p.endswith("336/smp_affinity_list"))
  event = threading.Event()
  event.set()
  read_fd, write_fd = os.pipe()
  sensord.run_interrupt_loop([], event, SimpleNamespace(send=lambda *a: None), read_fd)
  os.close(read_fd)
  os.close(write_fd)
