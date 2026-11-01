import pytest
import json
import os
import random
import subprocess
import time
import capnp
from pathlib import Path

import iqpilot.cereal.messaging as messaging
from iqpilot.cereal.services import SERVICE_LIST
from iqpilot.common.params import Params
from iqpilot.common.transformations.coordinates import ecef2geodetic
from iqpilot.common.basedir import BASEDIR


@pytest.mark.linux
class TestIQLocdProc:
  LLD_MSGS = ['gpsLocationExternal', 'cameraOdometry', 'carState', 'extrinsicsCalibration',
              'accelerometer', 'gyroscope']

  @pytest.fixture(autouse=True)
  def setup_iqlocd(self, openpilot_function_fixture):
    self.pm = messaging.PubMaster(self.LLD_MSGS)
    self.sm = messaging.SubMaster(['iqLiveLocation'])
    self.params = Params()
    assert self.params.get_param_path().endswith(os.environ['OPENPILOT_PREFIX'])
    self.params.put_bool("UbloxAvailable", True)
    iqlocd_dir = Path(BASEDIR) / 'iqpilot/selfdrive/iqlocd'
    self.proc = subprocess.Popen(['./iqlocd'], cwd=iqlocd_dir, env=os.environ.copy())
    yield
    self.proc.terminate()
    self.proc.wait(timeout=5)

  def get_msg(self, name, t):
    try:
      msg = messaging.new_message(name)
    except capnp.lib.capnp.KjException:
      msg = messaging.new_message(name, 0)

    if name == "gpsLocationExternal":
      gps = getattr(msg, name)
      gps.flags = 1
      gps.hasFix = True
      gps.source = 'ublox'
      gps.horizontalAccuracy = 1.0
      gps.verticalAccuracy = 1.0
      gps.speedAccuracy = 1.0
      gps.bearingAccuracyDeg = 1.0
      gps.vNED = [0.0, 0.0, 0.0]
      gps.latitude = float(self.lat)
      gps.longitude = float(self.lon)
      gps.unixTimestampMillis = t // 1_000_000
      gps.altitude = float(self.alt)
    elif name == 'cameraOdometry':
      msg.cameraOdometry.rot = [0.0, 0.0, 0.0]
      msg.cameraOdometry.rotStd = [0.01, 0.01, 0.01]
      msg.cameraOdometry.trans = [0.0, 0.0, 0.0]
      msg.cameraOdometry.transStd = [0.01, 0.01, 0.01]
    elif name == 'extrinsicsCalibration':
      msg.extrinsicsCalibration.calStatus = 'calibrated'
      msg.extrinsicsCalibration.rpyCalib = [0.0, 0.0, 0.0]
    elif name == 'accelerometer':
      msg.accelerometer.sensor = 1
      msg.accelerometer.type = 1
      msg.accelerometer.timestamp = t
      msg.accelerometer.init('acceleration').v = [0.0, 0.0, 9.81]
    elif name == 'gyroscope':
      msg.gyroscope.sensor = 5
      msg.gyroscope.type = 16
      msg.gyroscope.timestamp = t
      msg.gyroscope.init('gyroUncalibrated').v = [0.0, 0.0, 0.0]
    msg.logMonoTime = t
    msg.valid = True
    return msg

  def test_params_gps(self):
    random.seed(123489234)
    self.params.remove('LastGPSPositionIQLoc')

    self.x = -2710700 + (random.random() * 1e5)
    self.y = -4280600 + (random.random() * 1e5)
    self.z = 3850300 + (random.random() * 1e5)
    self.lat, self.lon, self.alt = ecef2geodetic([self.x, self.y, self.z])
    msgs = []
    for sec in range(1, 4):
      for name in self.LLD_MSGS:
        for j in range(int(SERVICE_LIST[name].frequency)):
          msgs.append(self.get_msg(name, int((sec + j / SERVICE_LIST[name].frequency) * 1e9)))

    for msg in sorted(msgs, key=lambda x: x.logMonoTime):
      self.pm.send(msg.which(), msg)
      if msg.which() == "cameraOdometry":
        self.pm.wait_for_readers_to_update(msg.which(), 0.1, dt=0.005)
        self.sm.update(0)
      time.sleep(0.001)
    deadline = time.monotonic() + 5.0
    last_gps_raw = None
    while time.monotonic() < deadline and last_gps_raw is None:
      last_gps_raw = self.params.get('LastGPSPositionIQLoc')
      self.sm.update(0)
      time.sleep(0.05)
    assert self.proc.poll() is None
    location = self.sm['iqLiveLocation']
    assert last_gps_raw is not None, {
      'gpsHealthy': location.gpsHealthy,
      'inputsHealthy': location.inputsHealthy,
      'sensorsHealthy': location.sensorsHealthy,
      'isolatedPath': self.params.get_param_path(),
    }
    lastGPS = json.loads(last_gps_raw)
    assert lastGPS['latitude'] == pytest.approx(self.lat, abs=0.001)
    assert lastGPS['longitude'] == pytest.approx(self.lon, abs=0.001)
    assert lastGPS['altitude'] == pytest.approx(self.alt, abs=0.2)

  def test_delayed_gps_fixes_keep_the_position_converged(self):
    self.x, self.y, self.z = -2710700.0, -4280600.0, 3850300.0
    self.lat, self.lon, self.alt = ecef2geodetic([self.x, self.y, self.z])
    msgs = []
    for sec in range(1, 31):
      for name in self.LLD_MSGS:
        for j in range(int(SERVICE_LIST[name].frequency)):
          msgs.append(self.get_msg(name, int((sec + j / SERVICE_LIST[name].frequency) * 1e9)))
    deviations = []
    for msg in sorted(msgs, key=lambda x: x.logMonoTime):
      self.pm.send(msg.which(), msg)
      if msg.which() == "cameraOdometry":
        self.pm.wait_for_readers_to_update(msg.which(), 0.1, dt=0.005)
        self.sm.update(0)
        if self.sm.updated["iqLiveLocation"] and msg.logMonoTime > 20e9:
          deviations.append(float(sum(v * v for v in self.sm["iqLiveLocation"].ecefPosition.deviations) ** 0.5))
    assert self.proc.poll() is None
    assert deviations, "iqlocd published no location after 20 s of inputs"
    assert max(deviations) < 10.0, f"GPS fixes stamped behind the filter were dropped: position std grew to {max(deviations):.1f} m"

  def _well_formed_burst(self, t0, frames=60):
    published = 0
    for i in range(frames):
      t = t0 + i * 50_000_000
      for name in self.LLD_MSGS:
        self.pm.send(name, self.get_msg(name, t))
      self.pm.wait_for_readers_to_update("cameraOdometry", 0.1, dt=0.005)
      self.sm.update(0)
      published += int(self.sm.updated["iqLiveLocation"])
    return t0 + frames * 50_000_000, published

  def _malformed(self, case, t):
    if case in ("odometry_short", "odometry_empty", "odometry_nan", "odometry_nan_std"):
      msg = messaging.new_message("cameraOdometry")
      odo = msg.cameraOdometry
      shapes = {
        "odometry_short": ([0.0] * 6, [0.0] * 6, [0.01] * 6, [0.01] * 6),
        "odometry_nan": ([float("nan"), 0.0, 0.0], [0.0] * 3, [0.01] * 3, [0.01] * 3),
        "odometry_nan_std": ([0.0] * 3, [0.0] * 3, [float("nan"), 0.01, 0.01], [0.01] * 3),
      }
      if case in shapes:
        odo.rot, odo.trans, odo.rotStd, odo.transStd = shapes[case]
    elif case in ("calibration_short", "calibration_nan"):
      msg = messaging.new_message("extrinsicsCalibration")
      msg.extrinsicsCalibration.calStatus = "calibrated"
      msg.extrinsicsCalibration.rpyCalib = [0.0, 0.0] if case == "calibration_short" else [float("nan"), 0.0, 0.0]
    elif case in ("gps_short", "gps_nan"):
      msg = self.get_msg("gpsLocationExternal", t)
      msg.gpsLocationExternal.vNED = [0.0, 0.0] if case == "gps_short" else [float("nan"), 0.0, 0.0]
    elif case == "gyro_nan":
      msg = self.get_msg("gyroscope", t)
      msg.gyroscope.gyroUncalibrated.v = [float("nan"), 0.0, 0.0]
    elif case == "accel_short":
      msg = self.get_msg("accelerometer", t)
      msg.accelerometer.acceleration.v = [0.0, 0.0]
    msg.logMonoTime = t
    msg.valid = True
    return msg

  @pytest.mark.parametrize("case", (
    "odometry_short", "odometry_empty", "odometry_nan", "odometry_nan_std",
    "calibration_short", "calibration_nan", "gps_short", "gps_nan", "gyro_nan", "accel_short",
  ))
  def test_malformed_input_does_not_kill_the_process(self, case):
    random.seed(1)
    self.x, self.y, self.z = -2710700.0, -4280600.0, 3850300.0
    self.lat, self.lon, self.alt = ecef2geodetic([self.x, self.y, self.z])

    t, _ = self._well_formed_burst(int(1e9))
    assert self.proc.poll() is None

    msg = self._malformed(case, t)
    self.pm.send(msg.which(), msg)
    time.sleep(0.05)
    _, published = self._well_formed_burst(t + 50_000_000)

    assert self.proc.poll() is None, f"iqlocd died on {case} with {self.proc.returncode}"
    assert published >= 30
