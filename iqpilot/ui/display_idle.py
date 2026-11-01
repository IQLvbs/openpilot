# Copyright (c) 2026 IQ.Lvbs. All rights reserved.
import time
from collections.abc import Callable


class DisplayIdleTimer:
  def __init__(self, clock: Callable[[], float] = time.monotonic):
    self._clock = clock
    self._delay: float | None = None
    self._deadline: float | None = None
    self._visible = True
    self._alert = False

  def configure(self, delay: float | None) -> None:
    if delay != self._delay:
      self._delay = delay
      self.wake()

  def set_visible(self, visible: bool) -> None:
    self._visible = visible
    if not visible:
      self._alert = False
    self.wake()

  def set_alert(self, active: bool) -> None:
    changed = self._alert != active
    self._alert = active
    if active or changed:
      self.wake()

  def wake(self) -> None:
    self._deadline = self._clock() + self._delay if self._visible and self._delay is not None else None

  @property
  def expired(self) -> bool:
    return self._visible and not self._alert and self._deadline is not None and self._clock() >= self._deadline
