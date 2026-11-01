import random
import numpy as np
import time
import pytest

from iqpilot.cereal import messaging, log, car
from iqpilot.selfdrive.locationd.lagd import LateralLagEstimator, retrieve_initial_lag, masked_normalized_cross_correlation, \
                                               BLOCK_NUM_NEEDED, BLOCK_SIZE, MIN_OKAY_WINDOW_SEC, VERSION, MIN_LAG, MAX_LAG
from iqpilot.selfdrive.test.process_replay.migration import migrate, migrate_carParams
from iqpilot.selfdrive.locationd.test.test_locationd_scenarios import TEST_ROUTE
from iqpilot.common.params import Params
from iqpilot.tools.lib.logreader import LogReader

MAX_ERR_FRAMES = 1
DT = 0.05
LAGD_MIN_LAG_FRAMES, LAGD_MAX_LAG_FRAMES = int(round(MIN_LAG / DT)), int(round(MAX_LAG / DT))


def process_messages(estimator, lag_frames, n_frames, vego=25.0, rejection_threshold=0.0):
  for i in range(n_frames):
    t = i * estimator.dt
    desired_la = np.cos(10 * t) * 0.3
    actual_la = np.cos(10 * (t - lag_frames * estimator.dt)) * 0.3

    # if sample is masked out, set it to desired value (no lag)
    rejected = random.uniform(0, 1) < rejection_threshold
    if rejected:
      actual_la = desired_la

    desired_cuvature = float(desired_la / (vego ** 2))
    actual_yr = float(actual_la / vego)
    msgs = [
      (t, "carControl", car.CarControl(latActive=not rejected)),
      (t, "carState", car.CarState(vEgo=vego, steeringPressed=False)),
      (t, "controlsState", log.ControlsState(desiredCurvature=desired_cuvature)),
      (t, "deviceMotion", log.DeviceMotion(angularVelocityDevice=log.DeviceMotion.XYZMeasurement(z=actual_yr, valid=True),
                                   posenetOK=True, inputsOK=True)),
      (t, "extrinsicsCalibration", log.ExtrinsicsCalibration(rpyCalib=[0, 0, 0], calStatus=log.ExtrinsicsCalibration.Status.calibrated)),
    ]
    for t, w, m in msgs:
      estimator.handle_log(t, w, m)
    estimator.update_points()
    estimator.update_estimate()


class TestLagd:
  def test_read_saved_params(self):
    params = Params()

    lr = migrate(LogReader(TEST_ROUTE), [migrate_carParams])
    CP = next(m for m in lr if m.which() == "carParams").carParams

    msg = messaging.new_message('lateralDelay')
    msg.lateralDelay.lateralDelayEstimate = random.random()
    msg.lateralDelay.validBlocks = random.randint(1, 10)
    msg.lateralDelay.version = VERSION
    params.put("LiveDelay", msg.to_bytes())
    params.put("CarParamsPrevRoute", CP.as_builder().to_bytes())

    saved_lag_params = retrieve_initial_lag(params, CP)
    assert saved_lag_params is not None

    lag, valid_blocks = saved_lag_params
    assert lag == msg.lateralDelay.lateralDelayEstimate
    assert valid_blocks == msg.lateralDelay.validBlocks

  def test_ncc(self):
    lag_frames = random.randint(1, 19)

    desired_sig = np.sin(np.arange(0.0, 10.0, 0.1))
    actual_sig = np.sin(np.arange(0.0, 10.0, 0.1) - lag_frames * 0.1)
    mask = np.ones(len(desired_sig), dtype=bool)

    corr = masked_normalized_cross_correlation(desired_sig, actual_sig, mask, 200)[len(desired_sig) - 1:len(desired_sig) + 20]
    assert np.argmax(corr) == lag_frames

    # add some noise
    desired_sig += np.random.normal(0, 0.05, len(desired_sig))
    actual_sig += np.random.normal(0, 0.05, len(actual_sig))
    corr = masked_normalized_cross_correlation(desired_sig, actual_sig, mask, 200)[len(desired_sig) - 1:len(desired_sig) + 20]
    assert np.argmax(corr)  in range(lag_frames - MAX_ERR_FRAMES, lag_frames + MAX_ERR_FRAMES + 1)

    # mask out 40% of the values, and make them noise
    mask = np.random.choice([True, False], size=len(desired_sig), p=[0.6, 0.4])
    desired_sig[~mask] = np.random.normal(0, 1, size=np.sum(~mask))
    actual_sig[~mask] = np.random.normal(0, 1, size=np.sum(~mask))
    corr = masked_normalized_cross_correlation(desired_sig, actual_sig, mask, 200)[len(desired_sig) - 1:len(desired_sig) + 20]
    assert np.argmax(corr) in range(lag_frames - MAX_ERR_FRAMES, lag_frames + MAX_ERR_FRAMES + 1)

  def test_empty_estimator(self):
    mocked_CP = car.CarParams(steerActuatorDelay=0.5)
    estimator = LateralLagEstimator(mocked_CP, DT)
    msg = estimator.get_msg(True)
    assert msg.lateralDelay.status == 'unestimated'
    assert np.allclose(msg.lateralDelay.lateralDelay, estimator.initial_lag)
    assert np.allclose(msg.lateralDelay.lateralDelayEstimate, estimator.initial_lag)
    assert msg.lateralDelay.validBlocks == 0
    assert msg.lateralDelay.calPerc == 0

  @pytest.mark.parametrize("fingerprint", ["HYUNDAI_TUCSON_4TH_GEN", "HYUNDAI_SONATA", "TOYOTA_RAV4"])
  @pytest.mark.parametrize("speed", [0.0, 15.0, 20.0, 25.0, 36.0])
  @pytest.mark.parametrize("learned,blocks", [(0.3, 0), (0.2, BLOCK_NUM_NEEDED),
                                             (0.55, BLOCK_NUM_NEEDED), (0.7, BLOCK_NUM_NEEDED)])
  def test_tucson_model_delay_schedule(self, fingerprint, speed, learned, blocks):
    estimator = LateralLagEstimator(car.CarParams(carFingerprint=fingerprint, steerActuatorDelay=0.1), DT)
    estimator.reset(learned, blocks)
    estimator.v_ego = speed
    delay = estimator.get_msg(True).lateralDelay
    raw_delay = learned if blocks else estimator.initial_lag
    if fingerprint == "HYUNDAI_TUCSON_4TH_GEN":
      highway_delay = min(0.6, max(0.35, learned)) if blocks else 0.5
      expected = {0.0: 0.3, 15.0: 0.3, 20.0: (0.3 + highway_delay) / 2,
                  25.0: highway_delay, 36.0: highway_delay}[speed]
    else:
      expected = min(MAX_LAG, max(MIN_LAG, raw_delay)) if blocks else raw_delay
    assert delay.lateralDelay == pytest.approx(expected)
    assert delay.lateralDelayEstimate == pytest.approx(raw_delay)
    assert delay.validBlocks == blocks
    assert delay.status == ("estimated" if blocks else "unestimated")

  @pytest.mark.parametrize("fingerprint", ["HYUNDAI_TUCSON_4TH_GEN", "HYUNDAI_SONATA"])
  @pytest.mark.parametrize("angular_valid,posenet_ok,inputs_ok", [
    (True, True, True), (True, True, False), (False, True, True), (True, False, True),
  ])
  def test_tucson_pose_gate_keeps_yaw_and_posenet_checks(self, fingerprint, angular_valid, posenet_ok, inputs_ok):
    estimator = LateralLagEstimator(car.CarParams(carFingerprint=fingerprint, steerActuatorDelay=0.1), DT)
    msg = log.DeviceMotion(angularVelocityDevice=log.DeviceMotion.XYZMeasurement(valid=angular_valid),
                          posenetOK=posenet_ok, inputsOK=inputs_ok)
    estimator.handle_log(1.0, "deviceMotion", msg)
    expected = angular_valid and posenet_ok and (inputs_ok or fingerprint == "HYUNDAI_TUCSON_4TH_GEN")
    assert estimator.pose_valid == expected

  def test_estimator_basics(self, subtests):
    for lag_frames in range(LAGD_MIN_LAG_FRAMES, LAGD_MAX_LAG_FRAMES - 1):
      with subtests.test(msg=f"lag_frames={lag_frames}"):
        mocked_CP = car.CarParams(steerActuatorDelay=0.5)
        estimator = LateralLagEstimator(mocked_CP, DT, min_recovery_buffer_sec=0.0, min_yr=0.0)
        process_messages(estimator, lag_frames, int(MIN_OKAY_WINDOW_SEC / DT) + BLOCK_NUM_NEEDED * BLOCK_SIZE)
        msg = estimator.get_msg(True)
        assert msg.lateralDelay.status == 'estimated'
        assert np.allclose(msg.lateralDelay.lateralDelay, lag_frames * DT, atol=0.01)
        assert np.allclose(msg.lateralDelay.lateralDelayEstimate, lag_frames * DT, atol=0.01)
        assert np.allclose(msg.lateralDelay.lateralDelayEstimateStd, 0.0, atol=0.01)
        assert msg.lateralDelay.validBlocks == BLOCK_NUM_NEEDED
        assert msg.lateralDelay.calPerc == 100

  def test_estimator_masking(self):
    mocked_CP, lag_frames = car.CarParams(steerActuatorDelay=0.5), random.randint(LAGD_MIN_LAG_FRAMES, LAGD_MAX_LAG_FRAMES - 1)
    estimator = LateralLagEstimator(mocked_CP, DT, min_recovery_buffer_sec=0.0, min_yr=0.0, min_valid_block_count=1)
    process_messages(estimator, lag_frames, (int(MIN_OKAY_WINDOW_SEC / DT) + BLOCK_SIZE) * 2, rejection_threshold=0.4)
    msg = estimator.get_msg(True)
    assert np.allclose(msg.lateralDelay.lateralDelayEstimate, lag_frames * DT, atol=0.01)
    assert np.allclose(msg.lateralDelay.lateralDelayEstimateStd, 0.0, atol=0.01)
    assert msg.lateralDelay.calPerc == 100

  @pytest.mark.timeout(60)
  def test_estimator_performance(self):
    mocked_CP = car.CarParams(steerActuatorDelay=0.5)
    estimator = LateralLagEstimator(mocked_CP, DT)

    ds = []
    for _ in range(1000):
      st = time.perf_counter()
      estimator.update_points()
      estimator.update_estimate()
      d = time.perf_counter() - st
      ds.append(d)

    assert np.mean(ds) < DT
