"""Stopping preferences in path coordinates; perception remains a measurement.

Sources supply *unadjusted* stop targets and own permission to release them. A
numerical model horizon endpoint alone does not satisfy that contract.
"""
from dataclasses import dataclass, replace
import math

CONTROLLER_VERSION = 1
VALIDATION_MAX_OFFSET = 10.0
TARGET_GRACE_TIME = 0.35
STATIONARY_SPEED = 0.1
LEAD_RESUME_SPEED = 0.5
LEAD_RESUME_ACCEL = 0.3
MIN_STOPPING_DECEL = 0.3
QUALIFICATION_TIME = 0.15
BASE_LEAD_GAP = 3.0
HOLD_TARGET_RADIUS = 0.15


@dataclass(frozen=True)
class StopFrame:
  time: float
  position: float
  speed: float
  acceleration: float
  active: bool = True
  override: bool = False
  commanded_acceleration: float = 0.0
  commanded_speed: float | None = None


@dataclass(frozen=True)
class StopTrajectory:
  acceleration: float
  speeds: tuple[float, ...]
  accelerations: tuple[float, ...]
  jerks: tuple[float, ...]
  should_stop: bool
  shortfall: float


def stopping_trajectory(frame, distance, times, delay, decel, jerk, dt):
  """A delay-compensated S-curve ending at zero speed and acceleration.

  Solve for the peak deceleration, then integrate its jerk-down, constant
  deceleration and jerk-up segments exactly. An unreachable target retains
  its signed error and uses bounded comfort braking; physical braking remains
  an independent planner candidate.
  """
  initial_a = frame.commanded_acceleration
  # The prefix approximates already-issued commands and actuator lag. The
  # returned trajectory starts at their predicted actuation time.
  prefix_a = (frame.acceleration + initial_a) * 0.5
  prefix_time = min(delay, frame.speed / -prefix_a) if prefix_a < 0.0 else delay
  prefix_distance = frame.speed * prefix_time + 0.5 * prefix_a * prefix_time ** 2
  predicted_speed = max(0.0, frame.speed + prefix_a * prefix_time)
  # Controls tracks the current commanded speed, as in the physical MPC. A
  # future physical speed here would make a speed-error PID integrate the
  # actuator-delay prediction as a persistent tracking error.
  v = predicted_speed if frame.commanded_speed is None else max(frame.commanded_speed, 0.0)
  remaining = distance - prefix_distance

  def integrate(segments, sample):
    x, speed, acceleration = 0.0, v, initial_a
    for duration, rate in segments:
      t = min(sample, duration)
      if acceleration < 0.0 or rate < 0.0:
        if rate == 0.0 and acceleration < 0.0:
          t = min(t, speed / -acceleration)
        elif rate != 0.0:
          discriminant = acceleration ** 2 - 2.0 * rate * speed
          if discriminant >= 0.0:
            roots = [(-acceleration + sign * math.sqrt(discriminant)) / rate for sign in (-1.0, 1.0)]
            positive = [root for root in roots if root >= 0.0]
            if positive:
              t = min(t, min(positive))
      x += speed * t + 0.5 * acceleration * t ** 2 + rate * t ** 3 / 6.0
      speed = max(0.0, speed + acceleration * t + 0.5 * rate * t ** 2)
      acceleration += rate * t
      sample -= t
      if sample <= 1e-9:
        return x, speed, acceleration, rate
      if t < duration - 1e-9 or speed <= 1e-8:
        return x, 0.0, 0.0, 0.0
    return x, speed, acceleration, 0.0

  def segments_for(peak):
    ramp = abs(initial_a + peak) / jerk
    rate = -jerk if initial_a > -peak else jerk
    v_after = v + initial_a * ramp + 0.5 * rate * ramp ** 2
    plateau = max(0.0, (v_after - peak ** 2 / (2.0 * jerk)) / peak)
    return ((ramp, rate), (plateau, 0.0), (peak / jerk, jerk))

  if v <= 1e-8:
    hold_accel = max(min(initial_a, -0.25), initial_a - jerk * dt)
    return StopTrajectory(hold_accel, (0.0,) * len(times), (0.0,) * len(times), (0.0,) * len(times),
                          frame.speed <= 0.25, max(0.0, prefix_distance - distance))
  if initial_a < 0.0 and v < initial_a ** 2 / (2.0 * jerk):
    # Already braking too hard to taper to zero acceleration before stopping.
    segments = ((max(-initial_a / jerk, dt), jerk),)
  else:
    upper = min(decel, math.sqrt(jerk * v + initial_a ** 2 / 2.0))
    lower = 1e-5
    for _ in range(35):
      peak = (lower + upper) * 0.5
      candidate = segments_for(peak)
      stop_distance = integrate(candidate, sum(segment[0] for segment in candidate))[0]
      if stop_distance > remaining:
        lower = peak
      else:
        upper = peak
    segments = segments_for(upper)
  stop_distance = integrate(segments, sum(segment[0] for segment in segments))[0]
  samples = [integrate(segments, t) for t in times]
  command = integrate(segments, dt)[2]
  should_stop = frame.speed <= 0.05 or (frame.speed <= 0.25 and distance <= HOLD_TARGET_RADIUS)
  if should_stop:
    command = max(min(command, -0.25), initial_a - jerk * dt)
  return StopTrajectory(command, tuple(sample[1] for sample in samples), tuple(sample[2] for sample in samples),
                        tuple(sample[3] for sample in samples), should_stop, max(0.0, prefix_distance + stop_distance - distance))


@dataclass(frozen=True)
class StopTarget:
  event_id: int
  source: str
  position: float
  observed_at: float
  stop_time: float = 0.0
  stationary: bool = True
  uncertainty: float = 0.5
  released: bool = False


@dataclass(frozen=True)
class StopOffsetPlan:
  offset: float = 0.0
  state: str = 'disabled'
  target: StopTarget | None = None
  requested_position: float = 0.0
  remaining: float = 0.0
  shortfall: float = 0.0
  trajectory: StopTrajectory | None = None

  @property
  def active(self):
    return self.target is not None


def minimum_stopping_distance(speed, acceleration, delay, decel, jerk):
  """Reachable bound including pending acceleration and a bounded jerk ramp.

  Used for diagnostics, never to silently reduce the requested offset.
  """
  v, a = max(speed, 0.0), max(acceleration, -decel)
  if v == 0.0:
    return 0.0
  prefix = min(delay, v / -a) if a < 0.0 else delay
  distance = v * prefix + 0.5 * a * prefix ** 2
  v = max(0.0, v + a * prefix)
  ramp = min(max((a + decel) / jerk, 0.0), (a + math.sqrt(a * a + 2.0 * jerk * v)) / jerk)
  distance += v * ramp + 0.5 * a * ramp ** 2 - jerk * ramp ** 3 / 6.0
  v = max(0.0, v + a * ramp - 0.5 * jerk * ramp ** 2)
  return max(0.0, distance + v * v / (2.0 * decel))


class StopOffset:
  def __init__(self, offset, prediction_times, *, delay=0.0, decel=2.5, jerk=1.5, dt=0.05):
    if not math.isfinite(offset) or not 0.0 <= offset <= VALIDATION_MAX_OFFSET:
      raise ValueError('unsupported stopping offset')
    if not all(math.isfinite(x) for x in (delay, decel, jerk, dt)) or delay < 0.0 or decel <= 0.0 or jerk <= 0.0 or dt <= 0.0:
      raise ValueError('invalid longitudinal limits')
    self.offset = float(offset)
    self.times = tuple(prediction_times)
    if (len(self.times) < 2 or self.times[0] != 0.0 or not all(math.isfinite(t) for t in self.times)
        or any(a >= b for a, b in zip(self.times, self.times[1:], strict=False))):
      raise ValueError('invalid prediction times')
    self.delay, self.decel, self.jerk = delay, decel, jerk
    self.dt = dt
    self.targets = {}
    self.anchors = {}
    self.committed = set()

  def reset(self):
    self.targets.clear()
    self.anchors.clear()
    self.committed.clear()

  def update(self, frame: StopFrame, baseline_stop_targets) -> StopOffsetPlan:
    empty = StopOffsetPlan(offset=self.offset, state='unavailable')
    if not frame.active or frame.override:
      self.reset()
      return replace(empty, state='inactive')
    if not all(math.isfinite(x) for x in (frame.time, frame.position, frame.speed, frame.acceleration, frame.commanded_acceleration)):
      self.reset()
      return empty
    if frame.commanded_speed is not None and not math.isfinite(frame.commanded_speed):
      self.reset()
      return empty

    for target in baseline_stop_targets:
      if target.released:
        self.targets.pop(target.event_id, None)
        self.anchors.pop(target.event_id, None)
        self.committed.discard(target.event_id)
        continue
      if (not all(math.isfinite(x) for x in (target.position, target.observed_at, target.stop_time, target.uncertainty))
          or not 0.0 <= frame.time - target.observed_at <= TARGET_GRACE_TIME or target.uncertainty < 0.0 or target.stop_time < 0.0):
        continue
      previous = self.targets.get(target.event_id)
      if target.stationary and target.source not in ('lead0', 'lead1'):
        if previous is None or not previous.stationary:
          self.anchors[target.event_id] = (target.position, target.uncertainty)
        # Refine within the original observation's uncertainty, never ratchet
        # the target by accepting an unbounded succession of small shifts.
        anchor, uncertainty = self.anchors[target.event_id]
        uncertainty = min(uncertainty, target.uncertainty)
        target = replace(target, position=min(max(target.position, anchor - uncertainty), anchor + uncertainty))
      self.targets[target.event_id] = target

    for event_id, target in tuple(self.targets.items()):
      age = frame.time - target.observed_at
      if age < 0.0 or (age > TARGET_GRACE_TIME and event_id not in self.committed):
        del self.targets[event_id]
        self.anchors.pop(event_id, None)
        self.committed.discard(event_id)
    if not self.targets:
      return empty

    selected = min(self.targets.values(), key=lambda target: target.position)
    requested = selected.position - self.offset
    remaining = requested - frame.position
    shortfall = max(0.0, minimum_stopping_distance(frame.speed, frame.acceleration, self.delay, self.decel, self.jerk) - remaining)
    trigger_distance = minimum_stopping_distance(frame.speed, frame.acceleration, self.delay, min(self.decel, 1.2), self.jerk) + frame.speed + 2.0
    if remaining <= trigger_distance and (frame.speed > 0.25 or remaining <= 0.5):
      self.committed.add(selected.event_id)
    trajectory = None
    if selected.event_id in self.committed:
      trajectory = stopping_trajectory(frame, remaining, self.times, self.delay, self.decel, self.jerk, self.dt)
      shortfall = max(shortfall, trajectory.shortfall)
    state = 'qualified' if trajectory is None else 'holding' if trajectory.should_stop else 'approach'
    if frame.time - selected.observed_at > TARGET_GRACE_TIME:
      # Finish the already committed stop, then hold for explicit reacquisition,
      # release or driver override. Loss of perception is not permission to go.
      state = 'degraded'
    elif frame.time - selected.observed_at > 0.1:
      state = 'dropout'
    return StopOffsetPlan(self.offset, state, selected, requested, remaining, shortfall, trajectory)


class LeadStopTargets:
  """Qualify stopping maneuvers, rather than fading on current lead speed.

  Forecasts use measured braking, separately from MPC's acceleration-decay
  extrapolation, which need not predict a standstill at all.
  """
  def __init__(self, dt=0.05):
    self.dt = dt
    self.next_id = 1
    self.next_vision_id = 1
    self.tracks = {}

  def reset(self):
    self.tracks.clear()

  def update(self, frame: StopFrame, leads):
    targets = []
    observed = []
    observed_indices = set()
    used = set()
    for index, lead in enumerate(leads):
      if not lead.status or not all(math.isfinite(x) for x in (lead.dRel, lead.vLead, lead.aLeadK)):
        continue
      observed_indices.add(index)
      if lead.radarTrackId >= 0:
        key = ('radar', lead.radarTrackId)
      else:
        if any(abs(lead.dRel - old.dRel) < 0.5 and abs(lead.vLead - old.vLead) < 0.3 and abs(lead.aLeadK - old.aLeadK) < 0.3
               for _, old, _ in observed):
          continue
        matches = []
        for old_key, previous in self.tracks.items():
          age = frame.time - previous['time']
          error = abs(frame.position + lead.dRel - previous['position'] - previous['speed'] * age)
          if (old_key[0] == 'vision' and old_key not in used and age >= 0.0 and error < 1.0
              and abs(lead.vLead - previous['speed']) < 2.0
              and (age <= TARGET_GRACE_TIME or (previous['speed'] <= STATIONARY_SPEED and lead.vLead <= STATIONARY_SPEED))):
            matches.append((error, old_key))
        if matches:
          key = min(matches)[1]
        else:
          key = ('vision', self.next_vision_id)
          self.next_vision_id += 1
      observed.append((index, lead, key))
      used.add(key)
    keys = {key for _, _, key in observed}
    for key, previous in tuple(self.tracks.items()):
      if key not in keys and previous['index'] in observed_indices:
        targets.append(replace(previous['target'], released=True))
        del self.tracks[key]
    seen = set()
    for index, lead, key in observed:
      if key in seen:
        continue
      seen.add(key)
      previous = self.tracks.get(key)
      position = frame.position + lead.dRel
      age = frame.time - previous['time'] if previous is not None else 0.0
      continuous = (previous is not None and previous['track'] == lead.radarTrackId
                    and age >= 0.0
                    and (age <= TARGET_GRACE_TIME or (previous['speed'] <= STATIONARY_SPEED and lead.vLead <= STATIONARY_SPEED))
                    and abs(position - previous['position'] - previous['speed'] * age) < 3.0)
      if not continuous:
        if previous is not None:
          targets.append(replace(previous['target'], released=True))
        event_id = self.next_id
        self.next_id += 1
        qualified_time = 0.0
      else:
        event_id, qualified_time = previous['id'], previous['qualified']

      speed = max(lead.vLead, 0.0)
      stationary = speed <= STATIONARY_SPEED and lead.aLeadK <= 0.1
      braking = lead.aLeadK <= -MIN_STOPPING_DECEL
      stop_time = speed / -lead.aLeadK if braking else 0.0
      qualified_time = qualified_time + self.dt if braking and stop_time <= 10.0 else 0.0
      # A qualified stop needs positive evidence of departure. Vision speed
      # noise around 0.1 m/s must not cancel braking on the final approach.
      retaining_stop = (continuous and not previous['target'].released
                        and speed <= LEAD_RESUME_SPEED and lead.aLeadK <= LEAD_RESUME_ACCEL)
      qualifies = stationary or qualified_time >= QUALIFICATION_TIME or retaining_stop
      stop_position = position + 0.5 * speed * stop_time if braking and not stationary else position
      target = StopTarget(event_id, f'lead{index}', stop_position - BASE_LEAD_GAP, frame.time, stop_time,
                          stationary, released=not qualifies)
      targets.append(target)
      self.tracks[key] = {'id': event_id, 'track': lead.radarTrackId, 'position': position, 'speed': speed,
                          'time': frame.time, 'qualified': qualified_time, 'target': target, 'index': index}
    return tuple(targets)
