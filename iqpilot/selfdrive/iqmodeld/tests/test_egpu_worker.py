"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""
from __future__ import annotations

import io
import json
import time
import urllib.request

import numpy as np
import pytest

from iqpilot.selfdrive.iqmodeld.egpu_helpers import (
  resolve_backend, resolve_download_url, usbgpu_present,
)
from iqpilot.selfdrive.iqmodeld.egpu_pipeline import (
  EgpuPipeline, EgpuPipelineError, make_big_channel_payload,
)
from iqpilot.selfdrive.iqmodeld.egpu_model import EGPU_MODELS, get_egpu_model
from iqpilot.selfdrive.iqmodeld.temporal_state import MODEL_INPUT_SPEC as INPUT_SPEC


class FakeParams:
  def __init__(self, **flags):
    self._flags = {k: bool(v) for k, v in flags.items()}

  def get_bool(self, key: str) -> bool:
    return self._flags.get(key, False)


def _fake_usb_device(root, vid: str, pid: str, name: str = "1-1", product: str | None = None):
  from iqpilot.system.hardware.usb import EGPU_DOCK_FW_PRODUCT
  d = root / name
  d.mkdir()
  (d / "idVendor").write_text(vid + "\n")
  (d / "idProduct").write_text(pid + "\n")
  (d / "product").write_text((product if product is not None else EGPU_DOCK_FW_PRODUCT) + "\n")


class TestPresence:
  def test_present(self, tmp_path):
    _fake_usb_device(tmp_path, "add1", "0001")
    assert usbgpu_present(str(tmp_path))

  def test_foreign_firmware_absent(self, tmp_path):
    _fake_usb_device(tmp_path, "add1", "0001", product="custom deadbeef-CLEAN")
    assert not usbgpu_present(str(tmp_path))

  def test_wrong_ids_absent(self, tmp_path):
    _fake_usb_device(tmp_path, "05ac", "12a8")
    assert not usbgpu_present(str(tmp_path))

  def test_empty_bus_absent(self, tmp_path):
    assert not usbgpu_present(str(tmp_path))

  def test_unreadable_entries_skipped(self, tmp_path):
    (tmp_path / "usb1").mkdir()
    _fake_usb_device(tmp_path, "add1", "0001", name="1-2")
    assert usbgpu_present(str(tmp_path))


class TestBackendResolution:
  def test_none(self):
    assert resolve_backend(False, False) is None

  def test_emac_only(self):
    assert resolve_backend(True, False) == "emac"

  def test_egpu_only(self):
    assert resolve_backend(False, True) == "egpu"

  def test_force_param_yields_to_emac_without_hardware(self):
    assert resolve_backend(True, True) == "emac"

  def test_present_dock_wins_over_emac(self):
    assert resolve_backend(True, True, True) == "egpu"


class TestManagerGating:
  @pytest.fixture
  def pc(self):
    return pytest.importorskip("iqpilot.system.manager.process_config")

  def test_egpu_needs_presence(self, pc, monkeypatch):
    monkeypatch.setattr(pc, "usbgpu_present", lambda: True)
    assert pc.egpu_enabled(True, FakeParams(IQEgpuEnabled=True), None)
    assert pc.egpu_enabled(True, FakeParams(), None)
    monkeypatch.setattr(pc, "usbgpu_present", lambda: False)
    assert not pc.egpu_enabled(True, FakeParams(IQEgpuEnabled=True), None)
    assert not pc.egpu_enabled(True, FakeParams(), None)

  def test_present_dock_wins_over_left_on_emac(self, pc, monkeypatch):
    monkeypatch.setattr(pc, "usbgpu_present", lambda: True)
    both = FakeParams(IQEmacEnabled=True, IQEgpuEnabled=True)
    assert not pc.emac_enabled(True, both, None)
    assert pc.egpu_enabled(True, both, None)

  def test_emac_runs_when_no_dock(self, pc, monkeypatch):
    monkeypatch.setattr(pc, "usbgpu_present", lambda: False)
    assert pc.emac_enabled(True, FakeParams(IQEmacEnabled=True), None)

  def test_disabled_dock_yields_to_emac(self, pc, monkeypatch):
    monkeypatch.setattr(pc, "usbgpu_present", lambda: True)
    both = FakeParams(IQEmacEnabled=True, IQEgpuDisabled=True)
    assert pc.emac_enabled(True, both, None)
    assert not pc.egpu_enabled(True, both, None)

  def test_disabled_dock_runs_no_backend_when_no_emac(self, pc, monkeypatch):
    monkeypatch.setattr(pc, "usbgpu_present", lambda: True)
    off = FakeParams(IQEgpuDisabled=True)
    assert not pc.egpu_enabled(True, off, None)
    assert not pc.emac_enabled(True, off, None)

  def test_selector_runs_for_either_backend(self, pc):
    assert pc.big_model_enabled(True, FakeParams(IQEmacEnabled=True), None)
    assert pc.big_model_enabled(True, FakeParams(IQEgpuEnabled=True), None)
    assert not pc.big_model_enabled(True, FakeParams(), None)

  def test_iqegpumodeld_registered(self, pc):
    assert "iqegpumodeld" in pc.managed_processes
    assert "maciqmodeld" in pc.managed_processes


class TestDownloadResolve:
  def test_direct_url_passthrough(self):
    assert resolve_download_url("https://x/y.onnx", "0" * 64, 5) == "https://x/y.onnx"

  def test_commalfs_batch(self, monkeypatch):
    seen = {}

    def fake_urlopen(req, timeout=0):
      seen["url"] = req.full_url
      seen["body"] = json.loads(req.data)
      return io.BytesIO(json.dumps(
        {"objects": [{"actions": {"download": {"href": "https://signed/url"}}}]}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    sha = "a5" * 32
    url = resolve_download_url(f"commalfs:{sha}", sha, 1234)
    assert url == "https://signed/url"
    assert seen["body"]["objects"] == [{"oid": sha, "size": 1234}]
    assert seen["url"].endswith("/info/lfs/objects/batch")


def _zero_infer(output_len: int, fill=None):
  calls = []

  def infer(inputs):
    for name, (shape, dtype) in INPUT_SPEC.items():
      assert tuple(inputs[name].shape) == shape, name
      assert inputs[name].dtype == np.dtype(dtype), name
    calls.append({k: v.copy() for k, v in inputs.items()})
    out = np.zeros(output_len, dtype=np.float32)
    if fill is not None:
      out[:] = fill
    return out

  infer.calls = calls
  return infer


def _frame_inputs(seed=0):
  rng = np.random.default_rng(seed)
  warped = rng.integers(0, 256, (2, 6, 128, 256)).astype(np.uint8)
  desire = np.zeros(8, dtype=np.float32)
  traffic = np.array([1.0, 0.0], dtype=np.float32)
  action_t = np.array([0.25, 0.55], dtype=np.float32)
  return warped, desire, traffic, action_t


class TestEgpuPipeline:
  def setup_method(self):
    self.meta = get_egpu_model()

  def test_split_model_rejected(self):
    split_meta = {**get_egpu_model(), "key": "some_split", "split": True}
    with pytest.raises(EgpuPipelineError, match="split"):
      EgpuPipeline(split_meta, _zero_infer(split_meta["output_len"]))

  def test_registry_is_fused_only(self):
    assert not any(m.get("split") for m in EGPU_MODELS.values())

  def test_run_shapes_and_output(self):
    infer = _zero_infer(self.meta["output_len"])
    pipe = EgpuPipeline(self.meta, infer)
    out = pipe.run(*_frame_inputs())
    assert out.shape == (self.meta["output_len"],)
    assert len(infer.calls) == 1

  def test_hidden_state_feeds_next_features_buffer(self):
    output_len = self.meta["output_len"]
    hidden = self.meta["output_slices"]["hidden_state"]

    def infer(inputs):
      out = np.zeros(output_len, dtype=np.float32)
      out[hidden] = np.arange(hidden.stop - hidden.start, dtype=np.float32)
      return out

    pipe = EgpuPipeline(self.meta, infer)
    pipe.run(*_frame_inputs(1))
    np.testing.assert_array_equal(
      pipe.state.prev_feat.reshape(-1), np.arange(hidden.stop - hidden.start, dtype=np.float32))
    pipe.run(*_frame_inputs(2))
    np.testing.assert_array_equal(
      pipe.state.feat_q[-1].reshape(-1), np.arange(hidden.stop - hidden.start, dtype=np.float32))

  def test_desire_rising_edge_pulse(self):
    infer = _zero_infer(self.meta["output_len"])
    pipe = EgpuPipeline(self.meta, infer)
    warped, _, traffic, action_t = _frame_inputs()
    desire_on = np.zeros(8, dtype=np.float32)
    desire_on[3] = 1.0
    pipe.run(warped, desire_on, traffic, action_t)
    assert infer.calls[-1]["desire_pulse"][0, -1, 3] == 1.0
    for _ in range(5):
      pipe.run(warped, desire_on, traffic, action_t)
    assert infer.calls[-1]["desire_pulse"][0, :, 3].sum() == 1.0

  def test_wrong_output_len_raises(self):
    pipe = EgpuPipeline(self.meta, _zero_infer(self.meta["output_len"] - 1))
    with pytest.raises(EgpuPipelineError, match="length"):
      pipe.run(*_frame_inputs())

  def test_non_finite_output_raises(self):
    pipe = EgpuPipeline(self.meta, _zero_infer(self.meta["output_len"], fill=np.nan))
    with pytest.raises(EgpuPipelineError, match="finite"):
      pipe.run(*_frame_inputs())


class TestChannelContract:
  def _real_msgs(self):
    import iqpilot.cereal.messaging as messaging
    msgs = {}
    for svc in ("modelV2", "drivingModelData", "cameraOdometry", "iqDriveModelData"):
      m = messaging.new_message(svc)
      msgs[svc] = m.to_bytes()
    return msgs

  def test_payload_keys_match_selector_contract(self):
    payload = make_big_channel_payload(7, True, 0.031, 24.0, {"modelV2": b"x"})
    assert payload["source"] == "egpu_big"
    for key in ("frame_id", "live_calib_seen", "model_execution_time", "msgs"):
      assert key in payload

  def test_selector_consumes_egpu_payload(self, tmp_path):
    from iqpilot.selfdrive.iqmodeld.model_channel import ModelChannel
    from iqpilot.selfdrive.iqmodeld.modeld_selector import wait_for_big

    chan = ModelChannel(str(tmp_path / "big"), create=True)
    payload = make_big_channel_payload(100, True, 0.03, 25.0, self._real_msgs())
    chan.write(100, payload)

    got, peek = wait_for_big(chan, 100, time.perf_counter() + 0.01)
    assert peek == 100
    assert got is not None
    assert got["source"] == "egpu_big"
    assert got["frame_id"] == 100

  def test_selector_patch_and_send_parses_egpu_msgs(self):
    from iqpilot.selfdrive.iqmodeld.modeld_selector import _patch_and_send

    sent = {}

    class PM:
      def send(self, service, msg):
        sent[service] = msg

    payload = make_big_channel_payload(42, True, 0.03, 25.0, self._real_msgs())
    _patch_and_send(PM(), payload, frame_drop_perc=0.0, selector_dropped=0, target=42, source_lag=0)
    assert set(sent) == {"modelV2", "drivingModelData", "cameraOdometry", "iqDriveModelData"}
    assert sent["modelV2"].modelV2.frameDropPerc == 0.0
    assert sent["cameraOdometry"].valid

  def test_selector_big_flag_follows_payload_then_source(self):
    from iqpilot.selfdrive.iqmodeld.modeld_selector import _patch_and_send

    sent = {}

    class PM:
      def send(self, service, msg):
        sent[service] = msg

    payload = make_big_channel_payload(42, True, 0.03, 25.0, self._real_msgs())
    _patch_and_send(PM(), payload, frame_drop_perc=0.0, selector_dropped=0, target=42, source_lag=0)
    assert sent["modelV2"].modelV2.big and sent["drivingModelData"].drivingModelData.big

    small_on_mac = {**make_big_channel_payload(43, True, 0.03, 25.0, self._real_msgs()), "source": "mac_big", "big": False}
    _patch_and_send(PM(), small_on_mac, frame_drop_perc=0.0, selector_dropped=0, target=43, source_lag=0)
    assert not sent["modelV2"].modelV2.big and not sent["drivingModelData"].drivingModelData.big

  def _small_payload(self, position_y):
    import iqpilot.cereal.messaging as messaging
    msgs = self._real_msgs()
    small = messaging.new_message("modelV2")
    small.modelV2.position.x = [0.0, 10.0, 20.0]
    small.modelV2.position.y = position_y
    small.modelV2.position.z = [0.0, 0.0, 0.0]
    msgs["modelV2"] = small.to_bytes()
    return {"source": "small", "frame_id": 42, "live_calib_seen": True, "model_execution_time": 0.02, "msgs": msgs}

  @pytest.mark.parametrize("source", ["egpu_big", "mac_big"])
  def test_selector_publishes_backup_path_while_big_drives(self, source):
    from iqpilot.selfdrive.iqmodeld.modeld_selector import _patch_and_send

    sent = {}

    class PM:
      def send(self, service, msg):
        sent[service] = msg

    payload = {**make_big_channel_payload(42, True, 0.03, 25.0, self._real_msgs()), "source": source, "big": True}
    _patch_and_send(PM(), payload, frame_drop_perc=0.0, selector_dropped=0, target=42, source_lag=0,
                    backup_payload=self._small_payload([0.0, 0.5, 1.5]))
    path = sent["iqDriveModelData"].iqDriveModelData.backupPath
    assert list(path.x) == [0.0, 10.0, 20.0]
    assert list(path.y) == [0.0, 0.5, 1.5]
    assert list(path.z) == [0.0, 0.0, 0.0]

  def test_selector_has_no_backup_path_without_a_big_model(self):
    from iqpilot.cereal.messaging import log_from_bytes
    from iqpilot.selfdrive.iqmodeld.modeld_selector import _patch_and_send

    sent = {}

    class PM:
      def send(self, service, msg):
        sent[service] = msg

    def backup_len():
      msg = sent["iqDriveModelData"]
      msg = log_from_bytes(msg) if isinstance(msg, bytes) else msg
      return len(msg.iqDriveModelData.backupPath.x)

    small = self._small_payload([0.0, 0.5, 1.5])
    _patch_and_send(PM(), small, frame_drop_perc=0.0, selector_dropped=0, target=42, source_lag=0)
    assert backup_len() == 0

    small_on_mac = {**make_big_channel_payload(42, True, 0.03, 25.0, self._real_msgs()), "source": "mac_big", "big": False}
    _patch_and_send(PM(), small_on_mac, frame_drop_perc=0.0, selector_dropped=0, target=42, source_lag=0, backup_payload=small)
    assert backup_len() == 0

  def test_selector_lag_patches_frame_id(self):
    from iqpilot.selfdrive.iqmodeld.modeld_selector import _patch_and_send

    sent = {}

    class PM:
      def send(self, service, msg):
        sent[service] = msg

    payload = make_big_channel_payload(40, True, 0.03, 25.0, self._real_msgs())
    _patch_and_send(PM(), payload, frame_drop_perc=0.0, selector_dropped=0, target=42, source_lag=2)
    assert sent["modelV2"].modelV2.frameId == 42
    assert not sent["cameraOdometry"].valid


def _import_worker():
  try:
    import iqpilot.selfdrive.iqmodeld.iqegpumodeld as w
    return w
  except ImportError as e:
    if any(tag in str(e) for tag in ("pyx", "visionipc", "proprietary_runtime")):
      pytest.skip(f"device-only import chain unavailable on this host: {e}")
    raise


class TestWorkerModule:
  def test_module_imports_off_device(self):
    w = _import_worker()
    assert w.PROCESS_NAME.endswith("iqegpumodeld")
    assert callable(w.main)

  def test_warmup_validates_output(self):
    w = _import_worker()
    spec = {name: (shape, dtype) for name, (shape, dtype) in INPUT_SPEC.items()}
    def good(inputs):
      return np.zeros(10, dtype=np.float32)
    assert w._warmup(good, spec, 10) >= 0.0
    with pytest.raises(RuntimeError, match="invalid"):
      w._warmup(good, spec, 11)


class TestSelectorBackendKeys:
  def test_emac_default(self):
    from iqpilot.selfdrive.iqmodeld.modeld_selector import EMAC_STATUS_KEYS, backend_status_keys
    assert backend_status_keys(False, False) is EMAC_STATUS_KEYS
    assert backend_status_keys(True, False) is EMAC_STATUS_KEYS

  def test_egpu_selected(self):
    from iqpilot.selfdrive.iqmodeld.modeld_selector import EGPU_STATUS_KEYS, backend_status_keys
    assert backend_status_keys(False, True) is EGPU_STATUS_KEYS
    assert backend_status_keys(False, True)["active"] == "UsbGpuActive"
    assert backend_status_keys(False, True)["failed"] == "UsbGpuFailed"

  def test_emac_wins_when_both(self):
    from iqpilot.selfdrive.iqmodeld.modeld_selector import EMAC_STATUS_KEYS, backend_status_keys
    assert backend_status_keys(True, True) is EMAC_STATUS_KEYS

  def test_key_maps_cover_same_roles(self):
    from iqpilot.selfdrive.iqmodeld.modeld_selector import EGPU_STATUS_KEYS, EMAC_STATUS_KEYS
    assert set(EGPU_STATUS_KEYS) == set(EMAC_STATUS_KEYS)


class TestBackendSeparation:
  EGPU_SOURCES = (
    "egpu_helpers.py", "egpu_pipeline.py", "egpu_model.py", "iqegpumodeld.py",
    "big_catalog.py", "tools/compile_egpu_model.py",
  )
  BANNED_IMPORTS = ("emac_input_state", "emac_model_meta", "maciqmodeld", "mac_protocol", "mac_client")

  def _sources(self):
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    return {name: (root / name).read_text() for name in self.EGPU_SOURCES}

  def test_no_emac_module_imports(self):
    for name, src in self._sources().items():
      for banned in self.BANNED_IMPORTS:
        assert f"import {banned}" not in src and f"iqmodeld.{banned}" not in src, f"{name} imports {banned}"

  def test_no_macmodel_params(self):
    for name, src in self._sources().items():
      assert "MacModel" not in src, f"{name} references MacModel* params"

  def test_emac_shim_reexports_temporal_state(self):
    from iqpilot.selfdrive.iqmodeld import emac_input_state, temporal_state
    assert emac_input_state.EmacInputState is temporal_state.TemporalInputState
    assert emac_input_state.SplitInputState is temporal_state.SplitTemporalState

  def test_emac_modules_are_not_in_the_public_tree(self):
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    for gone in ("mac_protocol.py", "mac_client.py", "maciqmodeld.py", "bulk_transport.py"):
      assert not (root / gone).exists(), f"{gone} must live only in konn3kt_private"


class TestMetaDrivenInputSpec:

  def _run_one(self, meta):
    seen = {}
    def infer(inputs):
      seen.update({k: v.shape for k, v in inputs.items()})
      return np.zeros(meta["output_len"], dtype=np.float32)
    pipe = EgpuPipeline(meta, infer)
    pipe.run(np.zeros((2, 6, 128, 256), np.uint8), np.zeros(8, np.float32),
             np.array([1, 0], np.float32), np.zeros(2, np.float32))
    return seen

  def test_default_contract_unchanged(self):
    meta = get_egpu_model()
    seen = self._run_one(meta)
    assert seen["features_buffer"] == (1, 24, 512)
    assert seen["desire_pulse"] == (1, 25, 8)

  def test_registry_shapes_drive_the_state(self):
    meta = dict(get_egpu_model())
    meta["output_len"] = 18452
    meta["output_slices"] = dict(meta["output_slices"], hidden_state=slice(2066, 18450))
    meta["input_shapes"] = {
      "img": (1, 12, 128, 256), "big_img": (1, 12, 128, 256),
      "desire_pulse": (1, 33, 8), "traffic_convention": (1, 2),
      "action_t": (1, 2), "features_buffer": (1, 32, 32, 512),
    }
    seen = self._run_one(meta)
    assert seen["features_buffer"] == (1, 32, 32, 512)
    assert seen["desire_pulse"] == (1, 33, 8)


class TestCatalogResolution:

  def _params(self, model, doc=None):
    class P:
      def get(self, k):
        if k == "IQEmacModel":
          return model
        if k == "IQEmacCatalogCache":
          return json.dumps(doc) if doc else None
        return None
    return P()

  def _doc(self):
    return {"schema": 1, "bundles": [{
      "short_name": "ttx", "display_name": "TTx", "index": 1,
      "model_name": "big_driving_supercombo",
      "wire": {"output_len": 2580, "frame_skip": 4, "pipeline": True,
               "output_slices": {"plan": [917, 1907], "hidden_state": [2066, 2578], "pad": [-2, None]},
               "input_shapes": {"img": [1, 12, 128, 256], "big_img": [1, 12, 128, 256],
                                "desire_pulse": [1, 33, 8], "traffic_convention": [1, 2],
                                "action_t": [1, 2], "features_buffer": [1, 32, 512]},
               "lat_smooth_seconds": 0.1},
      "source": {"kind": "comma_lfs", "sha256": "c" * 64, "size": 1},
    }]}

  def test_unset_selection_is_the_builtin_default(self):
    from iqpilot.selfdrive.iqmodeld.egpu_model import resolve_egpu_model
    m = resolve_egpu_model(self._params(None))
    assert m["key"] == "lebrowski" and m["sha256"].startswith("a501760a")

  def test_catalog_selection_resolves_with_shapes_and_smoothing(self):
    from iqpilot.selfdrive.iqmodeld.egpu_model import resolve_egpu_model
    m = resolve_egpu_model(self._params("ttx", self._doc()))
    assert m["key"] == "ttx"
    assert m["input_shapes"]["features_buffer"] == (1, 32, 512)
    assert m["input_shapes"]["desire_pulse"] == (1, 33, 8)
    assert m["lat_smooth_seconds"] == 0.1
    assert m["output_slices"]["pad"] == slice(-2, None)

  def test_unknown_selection_is_a_park_not_a_silent_default(self):
    from iqpilot.selfdrive.iqmodeld.egpu_model import resolve_egpu_model
    assert resolve_egpu_model(self._params("ghost", self._doc()), allow_refresh=False) is None

  def test_bench_model_is_not_selectable(self):
    from iqpilot.selfdrive.iqmodeld.egpu_model import resolve_egpu_model
    assert resolve_egpu_model(self._params("comma_small", self._doc()), allow_refresh=False) is None

  def test_registry_carries_no_model_list(self):
    from iqpilot.selfdrive.iqmodeld.egpu_model import EGPU_MODELS
    assert set(EGPU_MODELS) == {"lebrowski", "comma_small"}


class TestConsentAndIntegrity:
  def test_disabled_param_denies_present_dock(self, monkeypatch):
    from iqpilot.selfdrive.iqmodeld import egpu_helpers
    monkeypatch.setattr(egpu_helpers, "usbgpu_present", lambda sysfs_root=egpu_helpers.USB_SYSFS_ROOT: True)
    assert egpu_helpers.egpu_present_consented(FakeParams()) is True
    assert egpu_helpers.egpu_present_consented(FakeParams(IQEgpuDisabled=True)) is False

  def test_local_onnx_quarantines_bad_content(self, tmp_path, monkeypatch):
    import hashlib
    from iqpilot.selfdrive.iqmodeld import egpu_helpers
    onnx = tmp_path / "m.onnx"
    onnx.write_bytes(b"good")
    meta = {"sha256": hashlib.sha256(b"good").hexdigest(), "download": {"size": 4}}
    monkeypatch.setattr(egpu_helpers, "onnx_cache_path", lambda m: str(onnx))
    assert egpu_helpers.local_onnx(meta) == str(onnx)
    onnx.write_bytes(b"bad!")
    assert egpu_helpers.local_onnx(meta) is None
    assert not onnx.exists()
    assert (tmp_path / "m.onnx.unusable").exists()


class TestEgpuDockStatus:
  def _run(self, seq):
    from iqpilot.system.hardware.egpu_dock.status import EgpuDockStatus
    st = EgpuDockStatus()
    fired = {}
    def set_alert(name, cond, extra=None):
      fired[name] = (bool(cond), extra)
    for args in seq:
      st.update(*args, set_alert)
    return {k: v for k, v in fired.items() if v[0]}

  def _dock(self, speed=10000, product="custom ed4e39b7-CLEAN"):
    return [{"vendorId": 0xADD1, "productId": 0x0001, "product": product, "speedMbps": speed}]

  def test_no_dock_no_alerts(self):
    assert self._run([(True, [], False, False, None, True, None)]) == {}

  def test_usb2_dock_warns_slow(self):
    fired = self._run([(True, self._dock(speed=480), False, False, None, True, None)])
    assert fired.get("Offroad_EgpuUsbSlow") == (True, "480 Mbps")

  def test_power_fault_reports_pcie_unavailable(self):
    class St:
      supplyFault = True
      supplyVoltage = 0
      pcieLtssm = 0x78
      tempC = memoryTempC = 40.0
      fanSpeedRpm = 1500
    d = self._dock()
    fired = self._run([
      (True, d, False, False, None, True, None),
      (False, d, False, True, None, True, None),
      (False, d, False, False, b"1", True, St()),
    ])
    assert "Offroad_EgpuPcieUnavailable" in fired
