import math


class PT2Filter:
  def __init__(self, w0: float, zeta: float, dt: float):
    self.w0 = w0
    self.zeta = zeta
    self.dt = dt
    self.a1, self.a2, self.b0, self.b1, self.b2 = self._design(w0, zeta, dt)
    self.y1 = 0.0
    self.y2 = 0.0
    self.u1 = 0.0
    self.u2 = 0.0

  @staticmethod
  def _design(w0: float, zeta: float, dt: float):
    # bilinear transform of H(s) = w0^2 / (s^2 + 2*zeta*w0*s + w0^2)
    alpha = 2.0 / dt
    a2_den = alpha**2 + (2.0 * zeta * w0 * alpha) + w0**2
    a1_den = (-2.0 * alpha**2) + (2.0 * w0**2)
    a0_den = alpha**2 - (2.0 * zeta * w0 * alpha) + w0**2
    return (a1_den / a2_den, a0_den / a2_den,
            w0**2 / a2_den, 2.0 * w0**2 / a2_den, w0**2 / a2_den)

  def reset(self, value: float = 0.0) -> None:
    self.y1 = value
    self.y2 = value
    self.u1 = value
    self.u2 = value

  def update(self, u: float) -> float:
    y = (-self.a1 * self.y1) - (self.a2 * self.y2) + (self.b0 * u) + (self.b1 * self.u1) + (self.b2 * self.u2)
    self.y2 = self.y1
    self.y1 = y
    self.u2 = self.u1
    self.u1 = u
    return y

  def steady_state_steps(self) -> int:
    return math.ceil((4.0 / (self.zeta * self.w0)) / self.dt)
