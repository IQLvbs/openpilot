import os
import time

import pytest
import numpy as np
from msgq.visionipc import VisionIpcClient

import iqpilot.cereal.messaging as messaging
from iqpilot.cereal.services import SERVICE_LIST
from iqpilot.selfdrive.test.helpers import processes_context
from iqpilot.system.camerad.snapshot import VISION_STREAMS
from iqpilot.system.manager.process_config import managed_processes
from iqpilot.tools.lib.logreader import msgs_to_time_series

TEST_TIMESPAN = 10
CAMERAS = ('roadCameraState', 'driverCameraState', 'wideRoadCameraState')
TEST_PATTERN_FRAMES = 200
TEST_PATTERN_MIN_CONFIDENCE = 10
TEST_PATTERN_CONNECT_TIMEOUT = 15
TEST_PATTERN_CONFIGS = {
  'ox03c10': (41, 4),
  'os04c10': (97, 4),
}


def _pattern_sample(client):
  buf = client.recv(1000)
  if buf is None:
    return None

  y = np.asarray(buf.data[:buf.uv_offset], dtype=np.uint8).reshape((-1, buf.stride))[:buf.height, :buf.width]
  profile = y[:, ::8].mean(axis=1)
  padded = np.pad(profile, (4, 4), mode='edge')
  neighbors = [padded[i:i + len(profile)] for i in range(9) if i != 4]
  residual = profile - np.median(neighbors, axis=0)
  position = int(np.argmax(residual))
  return client.frame_id, client.timestamp_sof, position, residual[position], buf.height


def _test_pattern_session():
  samples = {camera: [] for camera in CAMERAS}
  sockets = {camera: messaging.sub_sock(camera, conflate=False, timeout=100) for camera in CAMERAS}
  logs = []
  with pytest.MonkeyPatch.context() as monkeypatch:
    monkeypatch.setenv('SPECTRA_TEST_PATTERN', '1')
    monkeypatch.setenv('SPECTRA_ERROR_PROB', '-1')
    with processes_context(['camerad']) as processes:
      clients = {camera: VisionIpcClient('camerad', VISION_STREAMS[camera], False) for camera in CAMERAS}
      pending = set(clients)
      deadline = time.monotonic() + TEST_PATTERN_CONNECT_TIMEOUT
      while pending and time.monotonic() < deadline:
        assert processes[0].proc is not None and processes[0].proc.exitcode is None
        pending = {camera for camera in pending if not clients[camera].connect(False)}
        if pending:
          time.sleep(0.1)
      assert not pending, f'VisionIPC connection timeout: {sorted(pending)}'

      for _ in range(TEST_PATTERN_FRAMES):
        for camera, client in clients.items():
          sample = _pattern_sample(client)
          if sample is not None:
            samples[camera].append(sample)
        for sock in sockets.values():
          logs.extend(messaging.drain_sock(sock))

  return msgs_to_time_series(logs), samples


def run_and_log(procs, services, duration):
  logs = []

  try:
    for p in procs:
      managed_processes[p].start()
    socks = [messaging.sub_sock(s, conflate=False, timeout=100) for s in services]

    start_time = time.monotonic()
    while time.monotonic() - start_time < duration:
      for s in socks:
        logs.extend(messaging.drain_sock(s))
    for p in procs:
      assert managed_processes[p].proc.is_alive()
  finally:
    for p in procs:
      managed_processes[p].stop()

  return logs

@pytest.fixture(scope="module")
def logs():
  logs = run_and_log(["camerad", ], CAMERAS, TEST_TIMESPAN)
  ts = msgs_to_time_series(logs)

  for cam in CAMERAS:
    expected_frames = SERVICE_LIST[cam].frequency * TEST_TIMESPAN
    cnt = len(ts[cam]['t'])
    assert expected_frames*0.8 < cnt < expected_frames*1.2, f"unexpected frame count {cam}: {expected_frames=}, got {cnt}"

    dts = np.abs(np.diff([ts[cam]['timestampSof']/1e6]) - 1000/SERVICE_LIST[cam].frequency)
    assert (dts < 1.0).all(), f"{cam} dts(ms) out of spec: max diff {dts.max()}, 99 percentile {np.percentile(dts, 99)}"
  return ts

@pytest.mark.tici
class TestCamerad:
  def test_frame_skips(self, logs):
    for c in CAMERAS:
      assert set(np.diff(logs[c]['frameId'])) == {1, }, f"{c} has frame skips"

  def test_frame_sync(self, logs):
    n = range(len(logs['roadCameraState']['t'][:-10]))

    frame_ids = {i: [logs[cam]['frameId'][i] for cam in CAMERAS] for i in n}
    assert all(len(set(v)) == 1 for v in frame_ids.values()), "frame IDs not aligned"

    frame_times = {i: [logs[cam]['timestampSof'][i] for cam in CAMERAS] for i in n}
    diffs = {i: (max(ts) - min(ts))/1e6 for i, ts in frame_times.items()}

    laggy_frames = {k: v for k, v in diffs.items() if v > 1.1}
    assert len(laggy_frames) == 0, f"Frames not synced properly: {laggy_frames=}"

  def test_sanity_checks(self, logs):
    self._sanity_checks(logs)

  def _sanity_checks(self, ts):
    for c in CAMERAS:
      assert c in ts
      assert len(ts[c]['t']) > 20

      # not a valid request id
      assert 0 not in ts[c]['requestId']

      # should monotonically increase
      assert np.all(np.diff(ts[c]['frameId']) >= 1)
      assert np.all(np.diff(ts[c]['requestId']) >= 1)

      # EOF > SOF
      assert np.all((ts[c]['timestampEof'] - ts[c]['timestampSof']) > 0)

      # logMonoTime > SOF
      assert np.all((ts[c]['t'] - ts[c]['timestampSof']/1e9) > 1e-7)

      # logMonoTime > EOF, needs some tolerance since EOF is (SOF + readout time) but there is noise in the SOF timestamping (done via IRQ)
      assert np.mean((ts[c]['t'] - ts[c]['timestampEof']/1e9) > 1e-7) > 0.7  # should be mostly logMonoTime > EOF
      assert np.all((ts[c]['t'] - ts[c]['timestampEof']/1e9) > -0.10)        # when EOF > logMonoTime, it should never be more than two frames

  def test_stress_test(self):
    os.environ['SPECTRA_ERROR_PROB'] = '0.008'
    logs = run_and_log(["camerad", ], CAMERAS, 10)
    ts = msgs_to_time_series(logs)

    # we should see some jumps from introduced errors
    assert np.max([ np.max(np.diff(ts[c]['frameId'])) for c in CAMERAS ]) > 1
    assert np.max([ np.max(np.diff(ts[c]['requestId'])) for c in CAMERAS ]) > 1

    self._sanity_checks(ts)


@pytest.fixture(scope="module")
def test_pattern_data():
  return _test_pattern_session()


@pytest.mark.tici
@pytest.mark.xdist_group("camerad_test_pattern")
class TestCameradTestPattern:
  def test_frame_delivery(self, test_pattern_data):
    logs, samples_by_camera = test_pattern_data
    for camera in CAMERAS:
      assert camera in logs
      samples = samples_by_camera[camera]
      assert len(samples) > TEST_PATTERN_FRAMES * 0.9

      state_frame_ids = logs[camera]['frameId']
      state_request_ids = logs[camera]['requestId']
      vipc_frame_ids = np.array([sample[0] for sample in samples])
      for source, frame_ids in (('camera state', state_frame_ids), ('VisionIPC', vipc_frame_ids)):
        frame_steps = np.diff(frame_ids)
        skipped = frame_ids[1:][frame_steps != 1]
        assert len(skipped) == 0, f'{camera} {source} skipped frames before {skipped}'

      expected_sof_step = 1e9 / SERVICE_LIST[camera].frequency
      sof_step_errors = np.diff(logs[camera]['timestampSof']) - expected_sof_step
      assert np.all(np.abs(sof_step_errors) < 1e6), f'{camera} SOF cadence errors: {sof_step_errors[np.abs(sof_step_errors) >= 1e6]}'

      request_steps = np.diff(state_request_ids)
      skipped_requests = state_request_ids[1:][request_steps != 1]
      assert len(skipped_requests) == 0, f'{camera} skipped requests before {skipped_requests}'

      state_sofs = dict(zip(state_frame_ids, logs[camera]['timestampSof'], strict=True))
      matched_samples = [sample for sample in samples if sample[0] in state_sofs]
      assert len(matched_samples) > len(samples) * 0.8
      mismatched_sofs = {
        frame_id: (timestamp_sof, state_sofs[frame_id]) for frame_id, timestamp_sof, *_ in matched_samples if timestamp_sof != state_sofs[frame_id]
      }
      assert not mismatched_sofs, f'{camera} VisionIPC/camera state SOFs disagree: {mismatched_sofs}'

  def test_pattern(self, test_pattern_data):
    logs, samples_by_camera = test_pattern_data
    for camera in CAMERAS:
      sensors = set(logs[camera]['sensor'])
      assert len(sensors) == 1
      sensor = sensors.pop()
      assert sensor in TEST_PATTERN_CONFIGS, f'unsupported test pattern sensor: {sensor}'
      cycle_frames, position_tolerance = TEST_PATTERN_CONFIGS[sensor]

      samples = samples_by_camera[camera]
      confident = [sample for sample in samples if sample[3] > TEST_PATTERN_MIN_CONFIDENCE]
      positions = np.array([sample[2] for sample in confident])
      assert len(confident) > len(samples) * 0.7, f'{camera} test pattern confidence too low'
      assert len(np.unique(positions)) > 20, f'{camera} test pattern is not moving'
      assert np.ptp(positions) > confident[0][4] * 0.75, f'{camera} test pattern does not span the frame'

      samples_by_frame = {sample[0]: sample for sample in confident}
      repeating_pairs = [(sample, samples_by_frame[sample[0] + cycle_frames]) for sample in confident if sample[0] + cycle_frames in samples_by_frame]
      assert len(repeating_pairs) > 20
      unexpected = [(first[0], first[2], second[2]) for first, second in repeating_pairs if abs(second[2] - first[2]) > position_tolerance]
      assert len(unexpected) < len(repeating_pairs) * 0.3, f'{camera} test pattern cycle mismatches: {unexpected}'
