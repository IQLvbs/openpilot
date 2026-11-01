import json
import math
import os
import tomllib
from collections import deque
from difflib import SequenceMatcher
from importlib.resources import files

import numpy as np

from iqpilot.cereal import log
from iqdbc.car import structs
from iqdbc.car.hyundai.values import HyundaiFlags
from iqdbc.car.lateral import FRICTION_THRESHOLD, get_friction
from iqdbc.car.toyota.values import ToyotaFlags
from iqdbc.lvbs.car.interfaces import LatControlInputs
from iqdbc.lvbs.car.iq_lateral import get_friction as get_friction_in_torque_space
from iqpilot.common.basedir import BASEDIR
from iqpilot.common.constants import ACCELERATION_DUE_TO_GRAVITY
from iqpilot.common.filter_simple import FirstOrderFilter
from iqpilot.common.params import Params
from iqpilot.common.pid import PIDController
from iqpilot.common.swaglog import cloudlog
from iqpilot.common.turn_guard_settings import get_turn_guards_enabled
from iqpilot.selfdrive.controls.lib.actuator_feedback import SteeringActuatorFeedback
from iqpilot.selfdrive.controls.lib.hkg_tuning import uses_hkg_tuning
from iqpilot.selfdrive.controls.lib.lateral_acceleration_slew_limiter import LateralAccelerationSlewLimiter
from iqpilot.selfdrive.controls.lib.latcontrol import LatControl
from iqpilot.selfdrive.controls.lib.turn_guards import apply_turn_guards
from iqpilot.selfdrive.iqmodeld.config import ModelConstants
from iqpilot.selfdrive.iqmodeld.parser import safe_exp
from iqpilot.selfdrive.controls.lib.helpers.nav_torque_pulse import NavTorquePulseBrain



TORQUE_NN_MODEL_PATH = os.path.join(BASEDIR, "iqpilot", "iqpilot_iq_nnff_models", "neural_network_lateral_control")
TORQUE_NN_MODEL_SUBSTITUTE_PATH = files("iqdbc").joinpath("car", "torque_data", "substitute.toml")
MOCK_MODEL_PATH = os.path.join(TORQUE_NN_MODEL_PATH, "MOCK.json")

_EXACT_THRESHOLD = 0.99
_ACCEPT_THRESHOLD = 0.9


def _score(a: str, b: str) -> float:
  return SequenceMatcher(None, a, b).ratio()


def _best_model_for(candidate: str, fingerprint: str) -> tuple[str | None, float]:
  best_path, best_score = None, -1.0
  if not os.path.isdir(TORQUE_NN_MODEL_PATH):
    return best_path, best_score
  for entry in sorted(os.listdir(TORQUE_NN_MODEL_PATH)):
    if not entry.endswith(".json"):
      continue
    stem = os.path.splitext(entry)[0]
    if stem != fingerprint and not stem.startswith(fingerprint + " "):
      continue
    score = _score(stem, candidate)
    if score > best_score:
      best_path, best_score = os.path.join(TORQUE_NN_MODEL_PATH, entry), score
  return best_path, best_score


def _substitute_for(fingerprint: str) -> str:
  with open(TORQUE_NN_MODEL_SUBSTITUTE_PATH, 'rb') as f:
    table = tomllib.load(f)
  return table.get(fingerprint, fingerprint)


def _eps_suffix(CP: structs.CarParams) -> str:
  eps_fw = str(next((fw.fwVersion for fw in CP.carFw if fw.ecu == "eps"), ""))
  return eps_fw.replace("\\", "") if len(eps_fw) > 3 else ""


def get_nn_model_path(CP: structs.CarParams) -> tuple[str, str, bool]:
  if CP.steerControlType != structs.CarParams.SteerControlType.torque:
    return MOCK_MODEL_PATH, "MOCK", False

  fingerprint = CP.carFingerprint
  suffix = _eps_suffix(CP)
  if suffix:
    path, score = _best_model_for(f"{fingerprint} {suffix}", fingerprint)
    if path is not None and fingerprint in path and score >= _ACCEPT_THRESHOLD:
      name = os.path.splitext(os.path.basename(path))[0]
      return path, name, score >= _EXACT_THRESHOLD
  path, score = _best_model_for(fingerprint, fingerprint)
  if path is not None and fingerprint in path and score >= _ACCEPT_THRESHOLD:
    name = os.path.splitext(os.path.basename(path))[0]
    return path, name, score >= _EXACT_THRESHOLD
  substitute = _substitute_for(fingerprint)
  path, _ = _best_model_for(substitute, substitute) if substitute != fingerprint else (None, -1.0)
  name = os.path.splitext(os.path.basename(path))[0] if path else "MOCK"
  return (path or MOCK_MODEL_PATH), name, False

_MIN_INPUT_LEN = 2
_FRICTION_PROBE = (10.0, 0.0, 0.2)
_FRICTION_THRESHOLD = 0.1

_ACTIVATION_ALIASES = {"σ": "sigmoid"}


def _sigmoid(x):
  return 1.0 / (1.0 + safe_exp(-x))


def _identity(x):
  return x


_ACTIVATIONS = {"sigmoid": _sigmoid, "identity": _identity}


def _resolve_activation(name: str):
  if not isinstance(name, str):
    raise ValueError("NNFF activation must be a string")
  for symbol, canonical in _ACTIVATION_ALIASES.items():
    name = name.replace(symbol, canonical)
  fn = _ACTIVATIONS.get(name)
  if fn is None:
    raise ValueError(f"Unknown activation: {name}")
  return fn


def _pick(layer: dict, suffix: str):
  key = next(k for k in layer if k.endswith(suffix))
  return np.array(layer[key], dtype=np.float32).T


class NNTorqueModel:
  def __init__(self, params_file, zero_bias=False):
    with open(params_file) as f:
      params = json.load(f)

    self.input_size = params["input_size"]
    self.output_size = params["output_size"]
    self.input_mean = np.array(params["input_mean"], dtype=np.float32).T
    self.input_std = np.array(params["input_std"], dtype=np.float32).T
    if (self.input_mean.shape != (1, self.input_size) or self.input_std.shape != self.input_mean.shape or
        not np.isfinite(self.input_mean).all() or not np.isfinite(self.input_std).all() or np.any(self.input_std <= 0)):
      raise ValueError("Invalid NNFF input normalization")

    self._weights = []
    self._biases = []
    self._activations = []
    width = self.input_size
    for layer in params["layers"]:
      weight = _pick(layer, "_W")
      bias = np.zeros_like(_pick(layer, "_b")) if zero_bias else _pick(layer, "_b")
      if (weight.ndim != 2 or weight.shape[0] != width or bias.shape != (1, weight.shape[1]) or
          not np.isfinite(weight).all() or not np.isfinite(bias).all()):
        raise ValueError("Invalid NNFF layer dimensions or weights")
      self._weights.append(weight)
      self._biases.append(bias)
      self._activations.append(_resolve_activation(layer["activation"]))
      width = weight.shape[1]
    if not self._weights or width != self.output_size:
      raise ValueError("Invalid NNFF output dimensions")

    self.friction_override = self.evaluate(list(_FRICTION_PROBE)) < _FRICTION_THRESHOLD

  def forward(self, x):
    for weight, bias, activation in zip(self._weights, self._biases, self._activations, strict=True):
      x = activation(x.dot(weight) + bias)
    return x

  def evaluate(self, input_array):
    if len(input_array) != self.input_size:
      if len(input_array) < _MIN_INPUT_LEN:
        raise ValueError(f"Input array length {len(input_array)} must be length 2 or greater")
      input_array = input_array + [0] * (self.input_size - len(input_array))
    x = (np.array(input_array, dtype=np.float32) - self.input_mean) / self.input_std
    return float(self.forward(x)[0, 0])

  @staticmethod
  def sigmoid(x):
    return _sigmoid(x)

  @staticmethod
  def identity(x):
    return _identity(x)

PLAN_SAMPLE_START = 5
LAG_EXTRA_S = 0.0
JERK_AHEAD_TAU_S = 0.15
JERK_PARAM_REFRESH = 100
STEERING_RATE_FILTER_SECONDS = 0.05

BASE_P = 0.8
BASE_I = 0.15
PID_SPEED_BP = [1, 1.5, 2.0, 3.0, 5, 7.5, 10, 15, 30]
PID_P_GAIN = [250, 120, 65, 30, 11.5, 5.5, 3.5, 2.0, BASE_P]

_JERK_FALLBACK_IDX = 16


def sign(value: float) -> float:
  if value > 0.0:
    return 1.0
  if value < 0.0:
    return -1.0
  return 0.0


polarity = sign


def _pointwise_jerk(accel_trace, dt_trace) -> list:
  delta = np.diff(accel_trace)
  span = min(len(delta), len(dt_trace))
  if span <= 0:
    return []
  return (delta[:span] / np.array(dt_trace)[:span]).tolist()


def sign_locked_min(future_vals, seed_val):
  if not future_vals:
    return seed_val
  agreeing = [v for v in future_vals if sign(v) == sign(seed_val)]
  if len(agreeing) < len(future_vals):
    return 0.0
  return min(agreeing + [seed_val], key=abs)


class PilotLateralBrain:

  def __init__(self, torque_ctrl, cp, cp_iq, car_if):
    del cp_iq
    self.lac_torque = torque_ctrl
    self.torque_from_lateral_accel_in_torque_space = car_if.torque_from_lateral_accel_in_torque_space()

    self.model_v2 = None
    self.model_valid = False

    self.jerk_now = 0.0
    self.jerk_goal = 0.0
    self.jerk_obs = 0.0
    self.jerk_ahead = 0.0
    self._ff = 0.0
    self._hkg_fix = uses_hkg_tuning(cp)
    if self._hkg_fix:
      self._pid = PIDController(BASE_P, BASE_I, rate=1 / torque_ctrl.dt)
    else:
      self._pid = PIDController([PID_SPEED_BP, PID_P_GAIN], BASE_I)
    self._pid_log = None
    self._accel_goal = 0.0
    self._accel_obs = 0.0
    self._roll_g = 0.0
    self._deadband = 0.0
    self._want_la = 0.0
    self._have_la = 0.0
    self._want_cv = 0.0
    self._have_cv = 0.0
    self._grav_la = 0.0
    self._capped = False
    self._out_tq = 0.0
    self.friction_look_ahead_v = [1.4, 2.0]
    self.friction_look_ahead_bp = [9.0, 30.0]
    self.lat_jerk_friction_factor = 0.4
    self.lat_accel_friction_factor = 0.7
    self._jerk_lp = FirstOrderFilter(0.0, JERK_AHEAD_TAU_S, torque_ctrl.dt if self._hkg_fix else 0.01)
    # classic Hyundai SAS_Speed is an unsigned magnitude, so the signed rate comes from the raw angle derivative
    self._steering_rate_from_angle = self._hkg_fix and not (cp.flags & HyundaiFlags.CANFD)
    self._last_steering_angle = None
    self._steering_rate_filter = FirstOrderFilter(0.0, STEERING_RATE_FILTER_SECONDS, torque_ctrl.dt)
    self._jerk_gain = 1.0
    self._jerk_param_frame = 0
    self._jerk_param_ok = True
    self._params = Params()

    self.t_diffs = np.diff(ModelConstants.T_IDXS)
    self.desired_lat_jerk_time = cp.steerActuatorDelay + LAG_EXTRA_S

  def update_model_v2(self, model_packet):
    self.model_valid = False
    if model_packet is not None:
      traces = (model_packet.orientation.x, model_packet.orientation.y, model_packet.acceleration.y)
      self.model_valid = all(len(trace) == len(ModelConstants.T_IDXS) and np.isfinite(trace).all() for trace in traces)
    self.model_v2 = model_packet if self.model_valid else None

  def update_lateral_lag(self, lag):
    self.desired_lat_jerk_time = max(0.01, lag) + LAG_EXTRA_S

  def update_friction_input(self, target_val, measured_val):
    error = target_val - measured_val
    return self.lat_accel_friction_factor * error + self.lat_jerk_friction_factor * self.jerk_ahead

  def _measured_jerk(self, car_state, vehicle_model) -> float:
    steering_rate = car_state.steeringRateDeg
    if self._steering_rate_from_angle:
      angle = car_state.steeringAngleDeg
      if not math.isfinite(angle):
        self._last_steering_angle = None
        self._steering_rate_filter.x = 0.0
        raise ValueError("Nonfinite steering angle for NNFF jerk")
      if self._last_steering_angle is None:
        steering_rate = 0.0
      else:
        steering_rate = (angle - self._last_steering_angle) / self.lac_torque.dt
      self._last_steering_angle = angle
      steering_rate = self._steering_rate_filter.update(steering_rate)
    curvature_rate = -vehicle_model.calc_curvature(math.radians(steering_rate), car_state.vEgo, 0.0)
    return curvature_rate * car_state.vEgo ** 2

  def _horizon_index(self, speed_mps: float) -> int:
    lookahead = np.interp(speed_mps, self.friction_look_ahead_bp, self.friction_look_ahead_v)
    return next((i for i, t in enumerate(ModelConstants.T_IDXS) if t > lookahead), _JERK_FALLBACK_IDX)

  def _reset_jerk_estimates(self, car_state, vehicle_model):
    self.jerk_now = self._measured_jerk(car_state, vehicle_model)
    self.jerk_goal = 0.0
    self.jerk_obs = 0.0
    self.jerk_ahead = 0.0

  def update_calculations(self, car_state, vehicle_model, desired_lat_accel):
    self._jerk_param_frame += 1
    if self._jerk_param_ok and self._jerk_param_frame % JERK_PARAM_REFRESH == 0:
      try:
        raw = self._params.get("IQLatJerkGain")
        self._jerk_gain = float(raw) if raw not in (None, b"", "") else 1.0
      except (ValueError, TypeError):
        self._jerk_gain = 1.0
      except Exception:
        self._jerk_param_ok = False
        self._jerk_gain = 1.0
    self._reset_jerk_estimates(car_state, vehicle_model)
    if not self.model_valid:
      return

    accel_y = self.model_v2.acceleration.y
    horizon_accel = np.interp(self.desired_lat_jerk_time, ModelConstants.T_IDXS, accel_y)
    desired_jerk = (horizon_accel - desired_lat_accel) / self.desired_lat_jerk_time

    forecast = _pointwise_jerk(accel_y, self.t_diffs)
    window = forecast[PLAN_SAMPLE_START:self._horizon_index(car_state.vEgo)]
    self.jerk_ahead = sign_locked_min(window, desired_jerk)
    self.jerk_ahead = self._jerk_lp.update(self.jerk_ahead) * self._jerk_gain

    if self._hkg_fix:
      self.lat_accel_friction_factor = 0.7
    if self.jerk_ahead == 0.0:
      self.jerk_now = 0.0
      self.lat_accel_friction_factor = 1.0

    self.jerk_goal = self.lat_jerk_friction_factor * self.jerk_ahead
    self.jerk_obs = self.lat_jerk_friction_factor * self.jerk_now


TorqueBrainCore = PilotLateralBrain
LOW_SPEED_X = [0, 10, 20, 30]
LOW_SPEED_Y = [12, 3, 1, 0]
_ERROR_BLEND_BP = [1.0, 2.0]
_ERROR_BLEND_V = [0.0, 1.0]


def roll_pitch_adjust(roll, pitch):
  return roll * math.cos(pitch)


class _HistoryWindow:
  def __init__(self, past_times, future_times, jerk_time):
    self.past_times = past_times
    self.future_times = future_times
    self.jerk_time = jerk_time
    self.nn_future_times = [t + jerk_time for t in future_times]

    check_frames = [int(abs(t) * 100) for t in past_times]
    self.frame_offsets = [check_frames[0] - f for f in check_frames]
    maxlen = check_frames[0]
    self.roll = deque(maxlen=maxlen)
    self.lat_accel_desired = deque(maxlen=maxlen)
    self.past_future_len = len(past_times) + len(self.nn_future_times)

  def refresh_lag(self, jerk_time):
    self.jerk_time = jerk_time
    self.nn_future_times = [t + jerk_time for t in self.future_times]

  def push(self, roll, lat_accel_desired):
    self.roll.append(roll)
    self.lat_accel_desired.append(lat_accel_desired)

  def _sample(self, buf):
    return [buf[min(len(buf) - 1, i)] for i in self.frame_offsets]

  def past_rolls(self):
    return self._sample(self.roll)

  def past_lat_accels(self):
    return self._sample(self.lat_accel_desired)


class NeuralNetworkFeedForward(PilotLateralBrain):
  def __init__(self, lac_torque, CP, CP_IQ, CI):
    super().__init__(lac_torque, CP, CP_IQ, CI)
    self.params = Params()
    self.enabled = self.params.get_bool("NeuralNetworkFeedForward")
    self.model = None
    model_path = CP_IQ.iqLateralNet.model.path
    if model_path != MOCK_MODEL_PATH and os.path.isfile(model_path):
      try:
        model = NNTorqueModel(model_path)
        if model.input_size != 18 or model.output_size != 1 or not np.isfinite(model.evaluate([0.0] * 18)):
          raise ValueError("NNFF requires a finite 18-input, 1-output model")
        self.model = model
      except (OSError, ValueError, TypeError, KeyError, IndexError, StopIteration):
        cloudlog.exception("Unable to load NNFF model; using standard torque control")
    self.has_nn_model = self.model is not None
    self.pitch = FirstOrderFilter(0.0, 0.5, 0.01)
    self.pitch_last = 0.0

    self.future_times = [0.3, 0.6, 1.0, 1.5]
    self._window = _HistoryWindow([-0.3, -0.2, -0.1], self.future_times, self.desired_lat_jerk_time)
    self.nav_torque_pulse = NavTorquePulseBrain(lac_torque)

  @property
  def nn_future_times(self):
    return self._window.nn_future_times

  @property
  def past_future_len(self):
    return self._window.past_future_len

  @property
  def _nnff_enabled(self):
    return self.enabled and self.model_valid and self.has_nn_model

  def update_limits(self):
    if not self._nnff_enabled:
      return
    self._pid.set_limits(self.lac_torque.steer_max, -self.lac_torque.steer_max)

  def update_lateral_lag(self, lag):
    super().update_lateral_lag(lag)
    self._window.refresh_lag(self.desired_lat_jerk_time)

  def reset_history(self):
    self._window.roll.clear()
    self._window.lat_accel_desired.clear()
    self._jerk_lp.x = 0.0
    self._last_steering_angle = None
    self._steering_rate_filter.x = 0.0
    self.pitch.x = 0.0
    self.pitch_last = 0.0
    self.lat_accel_friction_factor = 0.7

  def _torque_space(self, lateral_accel, CS, gravity_adjusted):
    return self.torque_from_lateral_accel_in_torque_space(
      LatControlInputs(lateral_accel, self._roll_g, CS.vEgo, CS.aEgo),
      self.lac_torque.torque_params, gravity_adjusted=gravity_adjusted)

  def update_feedforward_torque_space(self, CS):
    torque_from_setpoint = self._torque_space(self._accel_goal, CS, gravity_adjusted=False)
    torque_from_measurement = self._torque_space(self._accel_obs, CS, gravity_adjusted=False)
    self._pid_log.error = float(torque_from_setpoint - torque_from_measurement)
    self._ff = self._torque_space(self._grav_la, CS, gravity_adjusted=True)
    self._ff += get_friction_in_torque_space(self._want_la - self._have_la,
                                             self._deadband, FRICTION_THRESHOLD,
                                             self.lac_torque.torque_params)

  def update_output_torque(self, CS):
    if not np.isfinite((self._pid_log.error, self._ff)).all():
      raise ValueError("Nonfinite NNFF torque request")
    freeze_integrator = self._capped or CS.steeringPressed or CS.vEgo < 5
    self._out_tq = self._pid.update(self._pid_log.error, feedforward=self._ff,
                                           speed=CS.vEgo, freeze_integrator=freeze_integrator)

  def _effective_roll(self, params, calibrated_pose):
    roll = params.roll
    if calibrated_pose is not None:
      pitch = self.pitch.update(calibrated_pose.orientation.pitch)
      roll = roll_pitch_adjust(roll, pitch)
      self.pitch_last = pitch
    return roll

  def _future_rolls(self, roll, adjusted_future_times):
    return [roll_pitch_adjust(np.interp(t, ModelConstants.T_IDXS, self.model_v2.orientation.x) + roll,
                              np.interp(t, ModelConstants.T_IDXS, self.model_v2.orientation.y) + self.pitch_last)
            for t in adjusted_future_times]

  def _future_lat_accels(self, adjusted_future_times):
    return [np.interp(t, ModelConstants.T_IDXS, self.model_v2.acceleration.y) for t in adjusted_future_times]

  def _query(self, lead_scalar, jerk_scalar, tail):
    head = [self._v, lead_scalar, jerk_scalar, self._roll]
    return self._evaluate(head + tail)

  def _evaluate(self, inputs):
    output = self.model.evaluate(inputs)
    if not np.isfinite(output):
      raise ValueError("Nonfinite NNFF model output")
    return output

  def update_neural_network_feedforward(self, CS, params, calibrated_pose) -> None:
    if not self._nnff_enabled:
      return

    self.update_feedforward_torque_space(CS)
    creep = float(np.interp(CS.vEgo, LOW_SPEED_X, LOW_SPEED_Y)) ** 2
    self._accel_goal = self._want_la + creep * self._want_cv
    self._accel_obs = self._have_la + creep * self._have_cv

    self._v = CS.vEgo
    self._roll = self._effective_roll(params, calibrated_pose)
    self._window.push(self._roll, self._want_la)

    horizon = [t + 0.5 * CS.aEgo * (t / max(CS.vEgo, 1.0)) for t in self.nn_future_times]
    roll_ctx = self._window.past_rolls() + self._future_rolls(self._roll, horizon)
    accel_ctx = self._window.past_lat_accels() + self._future_lat_accels(horizon)

    goal_torque = self._query(self._accel_goal, self.jerk_goal, [self._accel_goal] * self.past_future_len + roll_ctx)
    obs_torque = self._query(self._accel_obs, self.jerk_obs, [self._accel_obs] * self.past_future_len + roll_ctx)
    self._pid_log.error = goal_torque - obs_torque
    self._apply_error_blend()

    friction_input = self.update_friction_input(self._accel_goal, self._accel_obs)
    self._ff = self._query(self._want_la, friction_input, accel_ctx + roll_ctx)
    if self.model.friction_override:
      friction_fn = get_friction_in_torque_space if self._hkg_fix else get_friction
      self._pid_log.error += friction_fn(friction_input, self._deadband,
                                         FRICTION_THRESHOLD, self.lac_torque.torque_params)

    self.update_output_torque(CS)

  def _apply_error_blend(self):
    blend = float(np.interp(abs(self._want_la), _ERROR_BLEND_BP, _ERROR_BLEND_V))
    if blend <= 0.0:
      return
    from_error = self._evaluate([self._v, self._accel_goal - self._accel_obs,
                                      self.jerk_goal - self.jerk_obs, 0.0])
    live = self._pid_log.error
    if sign(live) == sign(from_error) and abs(live) < abs(from_error):
      self._pid_log.error = live * (1.0 - blend) + from_error * blend

  def _snapshot_cycle(self, feedforward_seed, pid_core, pid_trace, torque_goal, torque_actual, roll_bias,
                      deadzone, lat_accel_goal, lat_accel_actual, curvature_goal, curvature_actual,
                      gravity_lat_accel, safety_limited, torque_output) -> None:
    self._ff = feedforward_seed
    self._pid = pid_core
    self._pid_log = pid_trace
    self._accel_goal = torque_goal
    self._accel_obs = torque_actual
    self._roll_g = roll_bias
    self._deadband = deadzone
    self._want_la = lat_accel_goal
    self._have_la = lat_accel_actual
    self._want_cv = curvature_goal
    self._have_cv = curvature_actual
    self._grav_la = gravity_lat_accel
    self._capped = safety_limited
    self._out_tq = torque_output

  def update(self, car_state, vehicle_model, pid_core, calibrator, feedforward_seed, pid_trace,
             torque_goal, torque_actual, calibrated_pose, roll_bias, lat_accel_goal, lat_accel_actual,
             deadzone, gravity_lat_accel, curvature_goal, curvature_actual, safety_limited, torque_output):
    self._snapshot_cycle(feedforward_seed, pid_core, pid_trace, torque_goal, torque_actual, roll_bias,
                         deadzone, lat_accel_goal, lat_accel_actual, curvature_goal, curvature_actual,
                         gravity_lat_accel, safety_limited, torque_output)
    if self._nnff_enabled or not self._hkg_fix:
      self.update_calculations(car_state, vehicle_model, lat_accel_goal)
      self.update_neural_network_feedforward(car_state, calibrator, calibrated_pose)
    self._out_tq = self.nav_torque_pulse.nudge_output_torque(True, car_state, self._out_tq)
    return self._pid_log, self._out_tq


KP = 0.8
KI = 0.15

INTERP_SPEEDS = [1, 1.5, 2.0, 3.0, 5, 7.5, 10, 15, 30]
KP_INTERP = [250, 120, 65, 30, 11.5, 5.5, 3.5, 2.0, KP]
HYUNDAI_INTERP_SPEEDS = [0, 7.5, 10, 15, 30]
HYUNDAI_KP_INTERP = [5.5, 5.5, 3.5, 2.0, KP]

LP_FILTER_CUTOFF_HZ = 1.2
JERK_LOOKAHEAD_SECONDS = 0.19
JERK_GAIN = 0.3
LAT_ACCEL_REQUEST_BUFFER_SECONDS = 1.0
VERSION = 1

TUCSON_FINGERPRINT = "HYUNDAI_TUCSON_4TH_GEN"
# Measured on one owner's Tucson drives; every table is scoped to this fingerprint.
LAT_ACCEL_FACTOR_SPEED_TABLE = {
  TUCSON_FINGERPRINT: ([8.0, 12.0, 15.0, 25.0], [2.95, 3.45, 4.05, 4.05]),
}
LAT_TUNE_V6 = {
  TUCSON_FINGERPRINT: {
    "split_ff_fb": True,
    "fb_lat_accel_factor_speed": ([8.0, 12.0, 15.0, 25.0], [2.95, 3.45, 4.05, 4.05]),
    "handback": {"blend_s": 0.25, "fric_reset_on_reversal": True},
    "timing": {"lat_delay_tau": 0.0, "fractional_delay": False},
    "fric_err_lp_tau": 0.3,
    "kp_scale_speed": ([3.0, 5.0, 7.0, 10.0, 13.0, 17.0], [0.7, 0.45, 0.45, 0.7, 0.7, 0.6]),
  },
}
LAT_TUNE_V7 = {
  TUCSON_FINGERPRINT: {
    "friction_speed": ([15.0, 20.0, 27.0], [0.12, 0.09, 0.05]),
    "friction_threshold": 0.5,
    "jerk_gain_speed": ([15.0, 25.0], [1.0, 0.5]),
  },
}
LAT_TUNE_V8 = {
  TUCSON_FINGERPRINT: {
    "ff_lat_accel_factor_speed": ([8.0, 12.0, 15.0, 20.0, 25.0, 30.0], [2.95, 3.45, 4.05, 7.0, 9.0, 10.5]),
    "ff_torque_scale_speed": ([8.5, 10.0, 11.0, 15.0, 16.0, 18.0, 20.0, 24.0, 27.0, 32.0, 36.0],
                              [1.0, 0.86, 0.80, 0.80, 0.88, 0.94, 1.0, 1.0, 1.08, 1.08, 1.12]),
  },
}

def friction_error_with_preview(error: float, measurement: float, desired_jerk: float) -> float:
  return float(np.clip(error + JERK_GAIN * desired_jerk,
                       min(error, -measurement, 0.0), max(error, -measurement, 0.0)))


class LatControlTorque(LatControl):
  supports_legacy_curvature_lookahead = True

  def __init__(self, CP, CP_IQ, CI, dt):
    super().__init__(CP, CP_IQ, CI, dt)
    self.torque_params = CP.lateralTuning.torque.as_builder()
    self.torque_from_lateral_accel = CI.torque_from_lateral_accel()
    self.lateral_accel_from_torque = CI.lateral_accel_from_torque()
    params = Params()
    self.hkg_fix = uses_hkg_tuning(CP)
    self.turn_guards_enabled = self.hkg_fix and get_turn_guards_enabled(params)
    if self.hkg_fix:
      cloudlog.event("steering_turn_guards", enabled=self.turn_guards_enabled, vehicle=CP.carFingerprint)
    self.hkg_reduced_torque_feedback = (CP.brand == "hyundai" and
                                        params.get_bool("IQHkgReducedTorqueFeedback") and
                                        not params.get_bool("NeuralNetworkFeedForward"))
    self.standard_speeds = HYUNDAI_INTERP_SPEEDS if self.hkg_fix else INTERP_SPEEDS
    base_gains = HYUNDAI_KP_INTERP if self.hkg_fix else KP_INTERP
    self.standard_p_gains = [gain * 0.8 for gain in base_gains] if self.hkg_reduced_torque_feedback else base_gains
    self.friction_scale = 0.7 if self.hkg_reduced_torque_feedback else 1.0
    self.pid = PIDController([self.standard_speeds, self.standard_p_gains], KI, rate=1/self.dt)
    self._kp_base = [self.pid._k_p[0], list(self.pid._k_p[1])]
    self._fric_err_lp = 0.0
    self._nnff_active = False
    self._actuator_feedback = None
    self.update_limits()
    self.steering_angle_deadzone_deg = self.torque_params.steeringAngleDeadzoneDeg
    self.lat_accel_request_buffer_len = int(LAT_ACCEL_REQUEST_BUFFER_SECONDS / self.dt)
    self.lat_accel_request_buffer = deque([0.] * self.lat_accel_request_buffer_len , maxlen=self.lat_accel_request_buffer_len)
    self.lookahead_frames = int(JERK_LOOKAHEAD_SECONDS / self.dt)
    self.jerk_filter = FirstOrderFilter(0.0, 1 / (2 * np.pi * LP_FILTER_CUTOFF_HZ), self.dt)
    self.setpoint_lead_enabled = CP.brand == "toyota" and bool(CP.flags & ToyotaFlags.TSS2)
    self.setpoint_lead_filter = FirstOrderFilter(0.0, 1 / (2 * np.pi * LP_FILTER_CUTOFF_HZ), self.dt)
    self.lateral_acceleration_slew_limiter = LateralAccelerationSlewLimiter(Params().get_bool("IQLateralAccelSlew"))
    self.lat_accel_factor_speed_table = LAT_ACCEL_FACTOR_SPEED_TABLE.get(str(CP.carFingerprint))

    self.v6 = LAT_TUNE_V6.get(str(CP.carFingerprint))
    if self.v6 is not None:
      self._fb_torque_params = CP.lateralTuning.torque.as_builder()
      self._fb_torque_params.latAccelFactor = self.torque_params.latAccelFactor
      self._lat_delay_lp = FirstOrderFilter(0.0, self.v6["timing"].get("lat_delay_tau", 0.0), self.dt)
      self._hb_elapsed = None
      self._sp_sign = 0.0

    self.v7 = LAT_TUNE_V7.get(str(CP.carFingerprint))
    if self.v7 is not None and self.v7.get("friction_speed") is not None:
      self._fric_torque_params = CP.lateralTuning.torque.as_builder()
      self._fric_torque_params.friction = self.torque_params.friction

    self.v8 = LAT_TUNE_V8.get(str(CP.carFingerprint))
    if self.v8 is not None:
      self._ff_torque_params = CP.lateralTuning.torque.as_builder()
    self.reference_slew_limiter = LateralAccelerationSlewLimiter(self.lateral_acceleration_slew_limiter.enabled)
    self.turn_guard_reference_buffer = deque(maxlen=self.lat_accel_request_buffer_len)
    self.friction_accel_request_buffer = deque(maxlen=self.lat_accel_request_buffer_len)

    self.nnff_assist = NeuralNetworkFeedForward(self, CP, CP_IQ, CI)

  def update_live_torque_params(self, latAccelFactor, latAccelOffset, friction):
    self.torque_params.latAccelFactor = latAccelFactor
    self.torque_params.latAccelOffset = latAccelOffset
    self.torque_params.friction = friction
    self.update_limits()

  def update_limits(self):
    if self._nnff_active:
      if self.hkg_fix:
        self.pid._k_p = [[0.0], [BASE_P]]
      self.pid.set_limits(self.steer_max, -self.steer_max)
    else:
      if self.hkg_fix:
        self.pid._k_p = [self.standard_speeds, self.standard_p_gains]
      self.pid.set_limits(self.lateral_accel_from_torque(self.steer_max, self.torque_params),
                          self.lateral_accel_from_torque(-self.steer_max, self.torque_params))

  def _friction_lateral_accel(self, CS, error, setpoint, desired_lateral_jerk, lateral_accel_deadzone):
    if self.hkg_fix:
      friction_error = friction_error_with_preview(error, setpoint - error, desired_lateral_jerk)
      return self.friction_scale * get_friction(friction_error, lateral_accel_deadzone, FRICTION_THRESHOLD, self.torque_params)
    fric_gain = 1.0
    hb = self.v6.get("handback") if self.v6 is not None else None
    if hb:
      if CS.steeringPressed:
        self._fric_err_lp = 0.0
        self._hb_elapsed = 0.0
        fric_gain = 0.0
      elif self._hb_elapsed is not None:
        self._hb_elapsed += self.dt
        fric_gain = min(1.0, self._hb_elapsed / hb["blend_s"])
        if fric_gain >= 1.0:
          self._hb_elapsed = None
      if hb.get("fric_reset_on_reversal") and abs(setpoint) > lateral_accel_deadzone:
        sign_now = float(np.sign(setpoint))
        if self._sp_sign != 0.0 and sign_now != self._sp_sign:
          self._fric_err_lp = 0.0
        self._sp_sign = sign_now
    fric_err = error
    if self.v6 is not None and self.v6.get("fric_err_lp_tau"):
      self._fric_err_lp += (error - self._fric_err_lp) * self.dt / self.v6["fric_err_lp_tau"]
      fric_err = self._fric_err_lp
    jerk_gain = JERK_GAIN
    fric_threshold = FRICTION_THRESHOLD
    fric_params = self.torque_params
    if self.v7 is not None:
      if self.v7.get("jerk_gain_speed") is not None:
        jerk_gain = JERK_GAIN * float(np.interp(CS.vEgo, *self.v7["jerk_gain_speed"]))
      if self.v7.get("friction_threshold") is not None:
        fric_threshold = self.v7["friction_threshold"]
      if self.v7.get("friction_speed") is not None:
        self._fric_torque_params.latAccelFactor = self.torque_params.latAccelFactor
        self._fric_torque_params.latAccelOffset = self.torque_params.latAccelOffset
        self._fric_torque_params.friction = float(np.interp(CS.vEgo, *self.v7["friction_speed"]))
        fric_params = self._fric_torque_params
    return fric_gain * self.friction_scale * get_friction(fric_err + jerk_gain * desired_lateral_jerk, lateral_accel_deadzone,
                                                          fric_threshold, fric_params)

  def _standard_output_torque(self, CS, error, ff, ff_core, fric_la, steer_limited_by_safety):
    freeze_integrator = self._freeze_integrator(CS, steer_limited_by_safety, error, ff)
    if self.v6 is not None and self.v6.get("kp_scale_speed") is not None:
      kp_scale = float(np.interp(CS.vEgo, *self.v6["kp_scale_speed"]))
      self.pid._k_p = [self._kp_base[0], [k * kp_scale for k in self._kp_base[1]]]
    output_lataccel = self.pid.update(error, speed=CS.vEgo, feedforward=ff, freeze_integrator=freeze_integrator)
    if self.v6 is None or not self.v6.get("split_ff_fb"):
      return self.torque_from_lateral_accel(output_lataccel, self.torque_params)
    fb_lataccel = output_lataccel - ff
    if "fb_lat_accel_factor_speed" in self.v6:
      self._fb_torque_params.latAccelFactor = float(np.interp(CS.vEgo, *self.v6["fb_lat_accel_factor_speed"]))
    if self.v8 is None:
      return (self.torque_from_lateral_accel(ff, self.torque_params)
              + self.torque_from_lateral_accel(fb_lataccel, self._fb_torque_params))
    self._ff_torque_params.latAccelFactor = float(np.interp(CS.vEgo, *self.v8["ff_lat_accel_factor_speed"]))
    ff_scale = float(np.interp(CS.vEgo, *self.v8.get("ff_torque_scale_speed", ([0.0], [1.0]))))
    return (ff_scale * self.torque_from_lateral_accel(ff_core, self._ff_torque_params)
            + self.torque_from_lateral_accel(fric_la, self._fb_torque_params)
            + self.torque_from_lateral_accel(fb_lataccel, self._fb_torque_params))

  def reset(self):
    super().reset()
    self._actuator_feedback = None
    if self.hkg_fix:
      self.pid.reset()
      self.nnff_assist.reset_history()
      self.friction_accel_request_buffer.clear()
      self.jerk_filter.x = 0.0

  def update_actuator_feedback(self, feedback: SteeringActuatorFeedback) -> None:
    self._actuator_feedback = feedback

  def _freeze_integrator(self, CS, steer_limited_by_safety, error, feedforward):
    if not self.hkg_fix:
      return steer_limited_by_safety or CS.steeringPressed or CS.vEgo < 5
    if CS.steeringPressed or CS.vEgo < 5:
      return True
    if not steer_limited_by_safety:
      return False
    feedback = self._actuator_feedback
    if (feedback is None or not feedback.limited or feedback.torque_difference is None or
        feedback.applied_torque is None or self.torque_params.latAccelFactor <= 0.0):
      return True

    self.pid.speed = CS.vEgo
    current_accel = self.pid.k_p * error + self.pid.i + feedforward
    proposed_accel = current_accel + self.pid.k_i * self.pid.i_dt * error
    current = -self.torque_from_lateral_accel(current_accel, self.torque_params)
    proposed = -self.torque_from_lateral_accel(proposed_accel, self.torque_params)
    applied, difference = feedback.applied_torque, feedback.torque_difference
    if not all(math.isfinite(value) for value in (current, proposed, applied, difference)):
      return True
    # after a reversal, wait for a fresh pair instead of letting the previous direction authorize integration
    recovering = (difference * (current - applied) > 0.0 and
                  difference * (proposed - applied) >= 0.0 and
                  abs(proposed - applied) < abs(current - applied))
    return not recovering

  def update(self, active, CS, VM, params, steer_limited_by_safety, desired_curvature, calibrated_pose, curvature_limited, lat_delay):
    nnff_active = active and self.nnff_assist._nnff_enabled
    if nnff_active != self._nnff_active:
      # The shared PID operates in torque units for NNFF and acceleration units otherwise.
      self.pid.reset()
      self.nnff_assist.reset_history()
      self._nnff_active = nnff_active
      self.update_limits()
    pid_log = log.ControlsState.LateralTorqueState.new_message()
    pid_log.version = VERSION
    measured_curvature = -VM.calc_curvature(math.radians(CS.steeringAngleDeg - params.angleOffsetDeg), CS.vEgo, params.roll)
    measurement = measured_curvature * CS.vEgo ** 2
    if self.lat_accel_factor_speed_table is not None:
      scheduled_factor = float(np.interp(CS.vEgo, *self.lat_accel_factor_speed_table))
      if scheduled_factor != self.torque_params.latAccelFactor:
        self.torque_params.latAccelFactor = scheduled_factor
        self.update_limits()
    target_curvature = desired_curvature
    if self.hkg_fix:
      delay_frames = int(np.clip(lat_delay / self.dt + 1, 1, self.lat_accel_request_buffer_len))
      reference_limiter = self.reference_slew_limiter
      reference_limiter.enabled = self.lateral_acceleration_slew_limiter.enabled
      requested_accel = desired_curvature * CS.vEgo ** 2
      if not active and reference_limiter.enabled:
        reference_limiter.reset(requested_accel)
      reference_curvature = reference_limiter.update(desired_curvature, CS.vEgo, self.dt)
      reference_accel = reference_curvature * CS.vEgo ** 2
      if active:
        if not self.friction_accel_request_buffer:
          self.friction_accel_request_buffer.extend([reference_accel] * self.lat_accel_request_buffer_len)
        else:
          self.friction_accel_request_buffer.append(reference_accel)
      if self.turn_guards_enabled:
        if not active:
          self.turn_guard_reference_buffer.clear()
        else:
          self.turn_guard_reference_buffer.append(reference_accel)
          if CS.vEgo != 0.0 and len(self.turn_guard_reference_buffer) >= delay_frames:
            guarded_accel = apply_turn_guards(requested_accel, measurement, self.turn_guard_reference_buffer[-delay_frames])
            if guarded_accel != requested_accel:
              target_curvature = guarded_accel / CS.vEgo ** 2
    if not active and self.lateral_acceleration_slew_limiter.enabled:
      self.lateral_acceleration_slew_limiter.reset(target_curvature * CS.vEgo ** 2)
    limited_curvature = self.lateral_acceleration_slew_limiter.update(target_curvature, CS.vEgo, self.dt)
    future_desired_lateral_accel = limited_curvature * CS.vEgo ** 2
    self.lat_accel_request_buffer.append(future_desired_lateral_accel)

    roll_compensation = params.roll * ACCELERATION_DUE_TO_GRAVITY
    curvature_deadzone = abs(VM.calc_curvature(math.radians(self.steering_angle_deadzone_deg), CS.vEgo, 0.0))
    lateral_accel_deadzone = curvature_deadzone * CS.vEgo ** 2

    if self.v6 is not None and self.v6["timing"].get("lat_delay_tau"):
      lat_delay = self._lat_delay_lp.update(lat_delay)
    delay_f = np.clip(lat_delay / self.dt + 1, 1, self.lat_accel_request_buffer_len)
    delay_frames = int(delay_f)
    if self.v6 is not None and self.v6["timing"].get("fractional_delay"):
      older = self.lat_accel_request_buffer[-delay_frames]
      newer = self.lat_accel_request_buffer[min(-delay_frames + 1, -1)]
      expected_lateral_accel = older + (delay_f - delay_frames) * (newer - older)
    else:
      expected_lateral_accel = self.lat_accel_request_buffer[-delay_frames]
    setpoint = expected_lateral_accel
    if self.setpoint_lead_enabled:
      request_ramp_rate = (future_desired_lateral_accel - expected_lateral_accel) / max(lat_delay, self.dt)
      setpoint += self.setpoint_lead_filter.update(request_ramp_rate) * lat_delay
    if self.hkg_fix and not nnff_active:
      setpoint = float(np.clip(setpoint, min(0.0, future_desired_lateral_accel), max(0.0, future_desired_lateral_accel)))
    error = setpoint - measurement

    lookahead_idx = int(np.clip(-delay_frames + self.lookahead_frames, -self.lat_accel_request_buffer_len+1, -2))
    jerk_reference = self.friction_accel_request_buffer if self.hkg_fix else self.lat_accel_request_buffer
    raw_lateral_jerk = ((jerk_reference[lookahead_idx+1] - jerk_reference[lookahead_idx-1]) / (2 * self.dt)
                        if active or not self.hkg_fix else 0.0)
    desired_lateral_jerk = self.jerk_filter.update(raw_lateral_jerk)
    gravity_adjusted_future_lateral_accel = future_desired_lateral_accel - roll_compensation
    ff = gravity_adjusted_future_lateral_accel
    ff -= self.torque_params.latAccelOffset
    ff_core, fric_la = ff, self._friction_lateral_accel(CS, error, setpoint, desired_lateral_jerk, lateral_accel_deadzone)
    ff = ff_core + fric_la

    if not active:
      if self.hkg_fix:
        self.reset()
      if self.v6 is not None and self.v6.get("handback"):
        self._fric_err_lp = 0.0
      output_torque = 0.0
      pid_log.active = False
    else:
      pid_log.error = float(error)

      output_torque = 0.0
      if not nnff_active:
        output_torque = self._standard_output_torque(CS, pid_log.error, ff, ff_core, fric_la, steer_limited_by_safety)

      try:
        pid_log, output_torque = self.nnff_assist.update(
          CS, VM, self.pid, params, ff, pid_log, setpoint, measurement, calibrated_pose, roll_compensation,
          future_desired_lateral_accel, measurement, lateral_accel_deadzone, gravity_adjusted_future_lateral_accel,
          limited_curvature, measured_curvature, steer_limited_by_safety, output_torque)
      except (ValueError, FloatingPointError):
        if not nnff_active:
          raise
        cloudlog.exception("NNFF inference failed; using standard torque control for this drive")
        self.nnff_assist.has_nn_model = False
        self.nnff_assist.reset_history()
        self._nnff_active = False
        self.pid.reset()
        self.update_limits()
        pid_log.error = float(error)
        output_torque = self._standard_output_torque(CS, pid_log.error, ff, ff_core, fric_la, steer_limited_by_safety)
        output_torque = self.nnff_assist.nav_torque_pulse.nudge_output_torque(True, CS, output_torque)

      pid_log.active = True
      pid_log.p = float(self.pid.p)
      pid_log.i = float(self.pid.i)
      pid_log.d = float(self.pid.d)
      pid_log.f = float(self.pid.f)
      pid_log.output = float(-output_torque)
      pid_log.actualLateralAccel = float(measurement)
      pid_log.desiredLateralAccel = float(setpoint)
      pid_log.desiredLateralJerk = float(desired_lateral_jerk)
      pid_log.saturated = bool(self._check_saturation(self.steer_max - abs(output_torque) < 1e-3, CS, steer_limited_by_safety, curvature_limited))

    return -output_torque, 0.0, pid_log
