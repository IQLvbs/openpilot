import numpy as np
import pytest

from iqpilot.selfdrive.controls.tests.lateral_control_scenarios import build_controls, load_controls_module, set_frame


@pytest.fixture
def controls_module(monkeypatch):
  return load_controls_module(monkeypatch)


@pytest.mark.parametrize("case", ["torque", "tss2", "gm"])
def test_controls_feeds_nnff_and_updates_pid_once(controls_module, monkeypatch, case):
  controls = build_controls(controls_module, case, nnff=True)
  nnff = controls.LaC.nnff_assist
  assert not nnff.model_valid
  calls = []
  lag_updates = []
  controller_delays = []
  original_update = controls.LaC.pid.update
  original_lag_update = nnff.update_lateral_lag
  original_control_update = controls.LaC.update

  def record_update(*args, **kwargs):
    calls.append((args, kwargs))
    return original_update(*args, **kwargs)

  def record_lag_update(lag):
    lag_updates.append(lag)
    original_lag_update(lag)

  def record_control_update(*args):
    controller_delays.append(args[-1])
    assert nnff.desired_lat_jerk_time == pytest.approx(max(0.01, args[-1]))
    return original_control_update(*args)

  monkeypatch.setattr(controls.LaC.pid, "update", record_update)
  monkeypatch.setattr(nnff, "update_lateral_lag", record_lag_update)
  monkeypatch.setattr(controls.LaC, "update", record_control_update)
  for frame in range(120, 130):
    set_frame(controls, frame)
    controls.sm["lateralDelay"].lateralDelay = 0.25 if frame < 125 else 0.45
    if frame == 120:
      assert not nnff.model_valid  # Only state_control may feed the live model.
    previous_queries = nnff.model.queries
    cc, _ = controls.state_control()
    assert cc.latActive
    assert nnff.model_valid
    assert nnff.model.queries > previous_queries
    assert len(calls) == frame - 119
    assert abs(cc.actuators.torque) <= controls.LaC.steer_max
  assert lag_updates == controller_delays == [0.25] * 5 + [0.45] * 5
  assert nnff.nn_future_times == pytest.approx([t + nnff.desired_lat_jerk_time for t in nnff.future_times])


@pytest.mark.parametrize("unavailable", [
  "disabled", "missing_nn_model", "invalid_message", "dead_message", "missing_orientation_x",
  "short_orientation_y", "missing_acceleration", "nonfinite_orientation", "nonfinite_acceleration",
])
def test_unavailable_nnff_uses_standard_controller(controls_module, unavailable):
  controls = build_controls(controls_module, "torque", nnff=True)
  baseline = build_controls(controls_module, "torque")
  nnff = controls.LaC.nnff_assist
  network = nnff.model
  if unavailable == "disabled":
    nnff.enabled = False
  elif unavailable == "missing_nn_model":
    nnff.has_nn_model = False
    nnff.model = None

  for frame in range(120, 130):
    set_frame(controls, frame)
    set_frame(baseline, frame)
    model = controls.sm["modelV2"]
    if unavailable == "invalid_message":
      controls.sm.valid["modelV2"] = False
    elif unavailable == "dead_message":
      controls.sm.alive["modelV2"] = False
    elif unavailable == "missing_orientation_x":
      model.orientation.x = []
    elif unavailable == "short_orientation_y":
      model.orientation.y = [0.0]
    elif unavailable == "missing_acceleration":
      model.acceleration.y = []
    elif unavailable == "nonfinite_orientation":
      model.orientation.x = [float("nan")] * len(model.orientation.x)
    elif unavailable == "nonfinite_acceleration":
      model.acceleration.y = [float("inf")] * len(model.acceleration.y)
    actual, trace = controls.state_control()
    expected, expected_trace = baseline.state_control()
    assert actual.actuators.torque == expected.actuators.torque
    for field in ("p", "i", "d", "f", "error", "output"):
      assert getattr(trace, field) == getattr(expected_trace, field)
  assert network.queries == 0


@pytest.mark.parametrize("lost_service", ["valid", "alive"])
def test_lost_live_model_stops_inference_and_restores_limits(controls_module, lost_service):
  controls = build_controls(controls_module, "torque", nnff=True)
  nnff = controls.LaC.nnff_assist
  set_frame(controls, 120)
  controls.state_control()
  queries = nnff.model.queries
  assert queries > 0
  assert controls.LaC.pid.pos_limit == controls.LaC.steer_max
  assert controls.LaC.pid.neg_limit == -controls.LaC.steer_max

  set_frame(controls, 121)
  getattr(controls.sm, lost_service)["modelV2"] = False
  cc, _ = controls.state_control()
  assert not nnff.model_valid
  assert nnff.model.queries == queries
  assert np.isfinite(cc.actuators.torque)
  assert abs(cc.actuators.torque) <= controls.LaC.steer_max
  assert controls.LaC.pid.pos_limit == controls.LaC.lateral_accel_from_torque(controls.LaC.steer_max, controls.LaC.torque_params)
  assert controls.LaC.pid.neg_limit == controls.LaC.lateral_accel_from_torque(-controls.LaC.steer_max, controls.LaC.torque_params)


def test_first_live_torque_update_uses_nnff_limits(controls_module):
  controls = build_controls(controls_module, "torque", nnff=True)
  set_frame(controls, 120)
  controls.sm.valid["lateralTorqueParameters"] = True
  controls.sm.alive["lateralTorqueParameters"] = True
  live_torque = controls.sm["lateralTorqueParameters"]
  live_torque.useParams = True
  live_torque.valid = True
  live_torque.latAccelFactorFiltered = 4.0
  live_torque.latAccelOffsetFiltered = 0.05
  live_torque.frictionCoefficientFiltered = 0.12
  cc, _ = controls.state_control()
  assert controls.LaC.torque_params.latAccelFactor == 4.0
  assert controls.LaC.nnff_assist.model.queries > 0
  assert controls.LaC.pid.pos_limit == controls.LaC.steer_max
  assert controls.LaC.pid.neg_limit == -controls.LaC.steer_max
  assert abs(cc.actuators.torque) <= controls.LaC.steer_max


@pytest.mark.parametrize("output", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_inference_falls_back_before_pid_update(controls_module, monkeypatch, output):
  controls = build_controls(controls_module, "torque", nnff=True)
  baseline = build_controls(controls_module, "torque")
  monkeypatch.setattr(controls.LaC.nnff_assist.model, "evaluate", lambda inputs: output)
  calls = []
  original_update = controls.LaC.pid.update

  def record_update(*args, **kwargs):
    calls.append((args, kwargs))
    return original_update(*args, **kwargs)

  monkeypatch.setattr(controls.LaC.pid, "update", record_update)
  for frame in range(120, 125):
    set_frame(controls, frame)
    set_frame(baseline, frame)
    actual, _ = controls.state_control()
    expected, _ = baseline.state_control()
    assert actual.actuators.torque == expected.actuators.torque
    assert np.isfinite(controls.LaC.pid.i)
    assert len(calls) == frame - 119
  assert not controls.LaC.nnff_assist.has_nn_model


def test_nnff_recovers_after_invalid_model_without_reusing_integral_or_history(controls_module, monkeypatch):
  controls = build_controls(controls_module, "torque", nnff=True)
  nnff = controls.LaC.nnff_assist
  pid = controls.LaC.pid
  integral_before_update = []
  original_update = pid.update

  def record_update(*args, **kwargs):
    integral_before_update.append(pid.i)
    return original_update(*args, **kwargs)

  monkeypatch.setattr(pid, "update", record_update)
  previous_queries = 0
  for valid, frames, factor in ((True, 20, 2.7), (False, 30, 4.0), (True, 5, 3.2)):
    for frame in range(frames):
      set_frame(controls, 200)
      state = controls.sm["carState"]
      state.vEgo = 20.0
      state.steeringAngleDeg = 0.15
      state.steeringRateDeg = 0.0
      state.steeringPressed = False
      controls.sm["vehicleParameters"].roll = 0.0
      controls.sm["lateralDelay"].lateralDelay = 0.12
      controls.sm.valid["modelV2"] = valid
      model = controls.sm["modelV2"]
      model.action.desiredCurvature = 0.0001
      model.orientation.x = [0.0] * len(model.orientation.x)
      model.orientation.y = [0.0] * len(model.orientation.y)
      model.acceleration.y = [0.04] * len(model.acceleration.y)
      controls.sm.valid["lateralTorqueParameters"] = True
      controls.sm.alive["lateralTorqueParameters"] = True
      live_torque = controls.sm["lateralTorqueParameters"]
      live_torque.useParams = True
      live_torque.valid = True
      live_torque.latAccelFactorFiltered = factor
      live_torque.latAccelOffsetFiltered = 0.0
      live_torque.frictionCoefficientFiltered = 0.12

      cc, _ = controls.state_control()
      assert cc.latActive
      assert nnff.model_valid == valid
      assert controls.LaC.torque_params.latAccelFactor == pytest.approx(factor)
      assert np.isfinite(cc.actuators.torque)
      assert abs(cc.actuators.torque) <= controls.LaC.steer_max
      if frame == 0:
        assert integral_before_update[-1] == 0.0
      if valid:
        assert nnff.model.queries > previous_queries
        assert pid.pos_limit == controls.LaC.steer_max
        assert pid.neg_limit == -controls.LaC.steer_max
        assert len(nnff._window.roll) == frame + 1
        assert len(nnff._window.lat_accel_desired) == frame + 1
      else:
        assert nnff.model.queries == previous_queries
        assert pid.pos_limit == controls.LaC.lateral_accel_from_torque(controls.LaC.steer_max, controls.LaC.torque_params)
        assert pid.neg_limit == controls.LaC.lateral_accel_from_torque(-controls.LaC.steer_max, controls.LaC.torque_params)
        assert len(nnff._window.roll) == 0
        assert len(nnff._window.lat_accel_desired) == 0
      previous_queries = nnff.model.queries
    assert pid.i != 0.0  # Each transition must discard an integral accumulated in the previous mode.


@pytest.mark.parametrize("case", ["pq", "mqb_v0", "pid", "angle", "curvature"])
def test_non_nnff_controllers_accept_live_model_updates(controls_module, case):
  controls = build_controls(controls_module, case)
  assert not hasattr(controls.LaC, "nnff_assist")
  set_frame(controls, 120)
  cc, _ = controls.state_control()
  assert cc.latActive
  assert np.isfinite(cc.actuators.torque)
  assert np.isfinite(cc.actuators.steeringAngleDeg)
