#!/usr/bin/env python3
"""Blog-style lateral actuator response plot.

Renders the single-panel "requested vs actual lateral acceleration + 50% response time" figure
comma publishes for lateral maneuver comparisons, for one or more routes on the same axes.

    ./iqpilot/tools/maneuvers/lateral_response_plot.py 1ce1b50dd82993a1'|'00000011--6cb007b200/0:4 \
        --maneuver 'sine 0.5Hz 30mph' --label 'IQ.Lvbs angle (Golf MK7)'

The 50% response time is only comparable between runs of the SAME maneuver at the SAME speed;
a step and a sine of equal amplitude do not produce comparable numbers.
"""
import argparse
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np

from iqpilot.tools.maneuvers.lateral_report import lat_accel, open_route, steering_overridden

SERIES_COLORS = ('#2ca02c', '#ff7f0e', '#1f77b4', '#d62728')
REQUESTED_COLOR = '#999999'


def completed_runs(msgs, maneuver):
  runs, active_prev, desc_prev = [], False, None
  for m in msgs:
    if m.which() == 'alertDebug':
      active = 'Active' in m.alertDebug.alertText1 or m.alertDebug.alertText1 == 'Complete'
      if active and not active_prev:
        if m.alertDebug.alertText2 == desc_prev:
          runs[-1][1].append([])
        else:
          runs.append((m.alertDebug.alertText2, [[]]))
        desc_prev = runs[-1][0]
      active_prev = active
    if active_prev:
      runs[-1][1][-1].append(m)

  out = []
  for description, windows in runs:
    if maneuver is not None and description != maneuver:
      continue
    for w in windows:
      if any(m.alertDebug.alertText1 == 'Complete' for m in w if m.which() == 'alertDebug'):
        out.append((description, w))
  return out


class Run(NamedTuple):
  t_requested: np.ndarray
  requested: np.ndarray
  t_actual: np.ndarray
  actual: np.ndarray
  t_wheel: np.ndarray
  angle: np.ndarray
  rate: np.ndarray
  t_angle_cmd: np.ndarray
  angle_cmd: np.ndarray | None
  v_mean: float
  valid: bool


def extract(msgs, raw_angle: bool = False) -> Run:
  """requested/actual lateral accel and wheel angle on a common relative timebase, baseline removed"""
  t_cs, carState = zip(*[(m.logMonoTime, m.carState) for m in msgs if m.which() == 'carState'], strict=True)
  t_ct, controlsState = zip(*[(m.logMonoTime, m.controlsState) for m in msgs if m.which() == 'controlsState'], strict=True)
  t_cc, carControl = zip(*[(m.logMonoTime, m.carControl) for m in msgs if m.which() == 'carControl'], strict=True)
  t_lp, lateralPlan = zip(*[(m.logMonoTime, m.lateralManeuverPlan) for m in msgs
                            if m.which() == 'lateralManeuverPlan' and m.valid], strict=True)

  t0 = t_lp[0]

  def rel(ts):
    return np.array([(t - t0) / 1e9 for t in ts])

  t_cs_s, t_ct_s, t_cc_s, t_lp_s = rel(t_cs), rel(t_ct), rel(t_cc), rel(t_lp)

  v_ego = np.array([m.vEgo for m in carState])
  v_at_lp = np.interp(t_lp_s, t_cs_s, v_ego)
  v_at_ct = np.interp(t_ct_s, t_cs_s, v_ego)

  baseline = lat_accel(controlsState[0].curvature, carState[0].vEgo)
  requested = np.array([lat_accel(m.desiredCurvature, v) for m, v in zip(lateralPlan, v_at_lp, strict=True)]) - baseline
  actual = np.array([lat_accel(m.curvature, v) for m, v in zip(controlsState, v_at_ct, strict=True)]) - baseline

  # controlsd derives curvature as -calc_curvature(steeringAngleDeg), so raw wheel angle always reads
  # opposite to lateral accel; flip it unless the caller wants the raw signal
  sign = 1.0 if raw_angle else -1.0
  angle = sign * (np.array([m.steeringAngleDeg for m in carState]) - carState[0].steeringAngleDeg)
  rate = sign * np.array([m.steeringRateDeg for m in carState])
  if not np.any(rate):  # not all brands populate steeringRateDeg
    rate = np.gradient(angle, t_cs_s)

  # angle command only exists on angle-control cars
  cmd = sign * (np.array([m.actuators.steeringAngleDeg for m in carControl]) - carControl[0].actuators.steeringAngleDeg)
  angle_cmd = cmd if np.any(cmd) else None

  window = lambda t: (t >= 0) & (t <= t_lp_s[-1])  # noqa: E731
  k_ct, k_cs, k_cc = window(t_ct_s), window(t_cs_s), window(t_cc_s)

  lat_active = all(m.latActive for m in carControl)
  # steering_overridden takes seconds, not raw logMonoTime
  overridden = steering_overridden(t_cs_s.tolist(), carState)
  return Run(t_lp_s, requested, t_ct_s[k_ct], actual[k_ct],
             t_cs_s[k_cs], angle[k_cs], rate[k_cs],
             t_cc_s[k_cc], angle_cmd[k_cc] if angle_cmd is not None else None,
             float(np.mean(v_ego)), lat_active and not overridden)


def response_time(t_actual, actual, requested):
  """time to first reach 50% of the requested peak, comma's metric"""
  amplitude = float(np.max(np.abs(requested)))
  if amplitude < 1e-3:
    return None, amplitude
  threshold = 0.5 * amplitude
  crossed = np.flatnonzero(np.abs(actual) > threshold)
  if not len(crossed):
    return None, amplitude
  return float(t_actual[crossed[0]]), amplitude


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('routes', nargs='+', help='route, local rlog path, or directory (one per series)')
  parser.add_argument('--label', action='append', default=[], help='series label, repeat to match routes')
  parser.add_argument('--maneuver', default='sine 0.5Hz 30mph', help='maneuver description to plot')
  parser.add_argument('--run', type=int, default=0, help='which completed run to plot (default first)')
  parser.add_argument('--out', type=Path, default=Path('lateral_response.png'))
  parser.add_argument('--title', default=None)
  parser.add_argument('--accel-only', action='store_true',
                      help="just comma's single lateral-accel panel, without wheel angle and rate")
  parser.add_argument('--raw-angle', action='store_true',
                      help='plot wheel angle in the raw log sign instead of aligned to lateral accel')
  args = parser.parse_args()

  if args.accel_only:
    fig, ax0 = plt.subplots(figsize=(9, 5.5), dpi=200)
    ax_angle = ax_rate = None
    axes = [ax0]
  else:
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), dpi=200, sharex=True,
                             gridspec_kw={'height_ratios': [3, 2, 2]})
    ax0, ax_angle, ax_rate = axes
  ax = ax0
  annotations = []
  plotted_requested = False
  plotted_cmd = False

  for i, route in enumerate(args.routes):
    label = args.label[i] if i < len(args.label) else route
    color = SERIES_COLORS[i % len(SERIES_COLORS)]

    msgs = list(open_route(route))
    runs = completed_runs(msgs, args.maneuver)
    if not runs:
      have = sorted({d for d, _ in completed_runs(msgs, None)})
      raise SystemExit(f"{route}: no completed '{args.maneuver}' runs. completed maneuvers in this route: {have or 'none'}")
    if args.run >= len(runs):
      raise SystemExit(f"{route}: only {len(runs)} completed '{args.maneuver}' run(s), --run {args.run} out of range")

    description, msgs_run = runs[args.run]
    r = extract(msgs_run, args.raw_angle)
    t_req, requested, t_act, actual, v_mean, valid = r.t_requested, r.requested, r.t_actual, r.actual, r.v_mean, r.valid
    cross, amplitude = response_time(t_act, actual, requested)

    if not plotted_requested:
      ax.plot(t_req, requested, color=REQUESTED_COLOR, linestyle=':', linewidth=2.5, label='requested', zorder=1)
      plotted_requested = True
    ax.plot(t_act, actual, color=color, linewidth=2.5, label=label, zorder=3)

    if ax_angle is not None:
      if r.angle_cmd is not None and not plotted_cmd:
        ax_angle.plot(r.t_angle_cmd, r.angle_cmd, color=REQUESTED_COLOR, linestyle=':', linewidth=2.5,
                      label='commanded', zorder=1)
        plotted_cmd = True
      ax_angle.plot(r.t_wheel, r.angle, color=color, linewidth=2.5, zorder=3)
      ax_rate.plot(r.t_wheel, r.rate, color=color, linewidth=2.5, zorder=3)

    if cross is not None:
      y = float(np.interp(cross, t_act, actual))
      ax.axvline(cross, color=color, linestyle='--', linewidth=1.5, ymax=0.92, zorder=2)
      ax.plot(cross, y, marker='o', markersize=9, markeredgewidth=2,
              markeredgecolor=color, markerfacecolor='none', zorder=4)
      annotations.append((f'50% response in {cross:.3f} s', color))
    else:
      annotations.append(('50% response not reached', color))

    flag = '' if valid else '   (INVALID: lat not active or steering overridden)'
    cross_str = f'{cross:.3f} s' if cross else 'n/a'
    print(', '.join([
      f"{label}: {description}",
      f"run {args.run}",
      f"{v_mean * 2.23694:.1f} mph",
      f"peak requested {amplitude:.2f} m/s^2",
      f"50% in {cross_str}",
      f"peak wheel {np.abs(r.angle).max():.1f} deg",
      f"peak rate {np.abs(r.rate).max():.0f} deg/s{flag}",
    ]))

  # headroom so the annotation block never sits on the trace
  lo, hi = ax.get_ylim()
  ax.set_ylim(lo, hi + (hi - lo) * 0.10 * max(len(annotations), 1))
  for j, (text, color) in enumerate(annotations):
    ax.text(0.03, 0.965 - j * 0.06, text, transform=ax.transAxes, color=color,
            fontsize=11, fontweight='bold', va='top',
            bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.75, 'pad': 2})

  ax.set_ylabel('Lateral acceleration (m/s²)')
  if ax_angle is not None:
    ax_angle.set_ylabel('Steering wheel angle (deg)')
    ax_rate.set_ylabel('Steering wheel rate (deg/s)')
    if not args.raw_angle:
      fig.text(0.5, 0.005, 'wheel angle sign aligned to lateral accel (raw log sign is inverted; --raw-angle to keep it)',
               ha='center', fontsize=8, color='#777777')
    if plotted_cmd:
      ax_angle.legend(loc='upper right', frameon=False, fontsize=9)
  axes[-1].set_xlabel('Time (s)')

  for a in axes:
    a.grid(True, color='#dddddd', linewidth=0.8)
    a.set_axisbelow(True)
    for side in ('top', 'right'):
      a.spines[side].set_visible(False)

  ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12 if args.accel_only else 1.10),
            ncol=4, frameon=False, fontsize=10)
  if args.title:
    ax.set_title(args.title, pad=28)

  fig.tight_layout()
  fig.savefig(args.out, bbox_inches='tight')
  print(f"\nwrote {args.out}")


if __name__ == '__main__':
  main()
