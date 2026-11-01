"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""
import hashlib
from types import SimpleNamespace
import http.server
import os
import threading

import pytest

from iqpilot.selfdrive.iqmodeld import model_bundle_downloader as dl


class _RangeHandler(http.server.BaseHTTPRequestHandler):
  store: dict[str, bytes] = {}
  cut_first: dict[str, int] = {}
  hits: list[tuple[str, str | None]] = []

  def log_message(self, *a):
    pass

  ignore_ranges = False

  def do_GET(self):
    oid = self.path.rsplit("/", 1)[-1]
    data = self.store[oid]
    rng = None if self.ignore_ranges else self.headers.get("Range")
    self.hits.append((oid, self.headers.get("Range")))
    start, end = 0, len(data) - 1
    if rng:
      a, _, b = rng.split("=")[1].partition("-")
      start = int(a)
      end = int(b) if b else len(data) - 1
    body = data[start:end + 1]
    cut = self.cut_first.pop(oid, None)
    if cut is not None:
      body = body[:cut]
    self.send_response(206 if rng else 200)
    self.send_header("Content-Length", str(len(body)))
    if rng:
      self.send_header("Content-Range", f"bytes {start}-{start + len(body) - 1}/{len(data)}")
    self.end_headers()
    self.wfile.write(body)


@pytest.fixture
def server():
  _RangeHandler.ignore_ranges = False
  srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _RangeHandler)
  t = threading.Thread(target=srv.serve_forever, daemon=True)
  t.start()
  yield srv
  srv.shutdown()
  srv.server_close()


def _fake_broker(port):
  return SimpleNamespace(broker_lfs_url=lambda oid, size: f"http://127.0.0.1:{port}/o/{oid}",
                         broker_headers=lambda: {"Authorization": "JWT test"})


def _objects(parts):
  return [{"oid": hashlib.sha256(p).hexdigest(), "size": len(p)} for p in parts]


def test_resume_continues_a_cut_part_and_reuses_finished_parts(server, tmp_path, monkeypatch):
  parts = [os.urandom(300_000), os.urandom(300_000), os.urandom(120_000)]
  objs = _objects(parts)
  _RangeHandler.store = {o["oid"]: p for o, p in zip(objs, parts, strict=True)}
  _RangeHandler.hits = []
  _RangeHandler.cut_first = {objs[1]["oid"]: 100_000}
  port = server.server_address[1]
  monkeypatch.setattr(dl, "_git_auth", lambda: _fake_broker(port))
  monkeypatch.setattr(dl, "STREAM_RETRIES", 3)
  monkeypatch.setattr(dl, "CHUNK", 64 * 1024)
  whole = b"".join(parts)
  dst = str(tmp_path / "model.pkl")
  out = dl.download_lfs_bundle(objs, dst, hashlib.sha256(whole).hexdigest(), len(whole))
  with open(dst, "rb") as f:
    assert out == dst and f.read() == whole
  assert not os.path.exists(dst + ".parts")
  ranges = [r for o, r in _RangeHandler.hits if o == objs[1]["oid"]]
  assert ranges[0] is None and ranges[1] == "bytes=100000-"
  assert sum(1 for o, _ in _RangeHandler.hits if o == objs[0]["oid"]) == 1


def test_corrupt_finished_part_is_refetched(server, tmp_path, monkeypatch):
  parts = [os.urandom(200_000), os.urandom(50_000)]
  objs = _objects(parts)
  _RangeHandler.store = {o["oid"]: p for o, p in zip(objs, parts, strict=True)}
  _RangeHandler.hits = []
  _RangeHandler.cut_first = {}
  port = server.server_address[1]
  monkeypatch.setattr(dl, "_git_auth", lambda: _fake_broker(port))
  dst = str(tmp_path / "model.pkl")
  os.makedirs(dst + ".parts")
  with open(dl._part_path(dst, objs[0]["oid"]), "wb") as f:
    f.write(os.urandom(200_000))
  whole = b"".join(parts)
  dl.download_lfs_bundle(objs, dst, hashlib.sha256(whole).hexdigest(), len(whole))
  with open(dst, "rb") as f:
    assert f.read() == whole


def test_hf_single_file_resumes_after_cut(server, tmp_path, monkeypatch):
  data = os.urandom(700_000)
  oid = hashlib.sha256(data).hexdigest()
  _RangeHandler.store = {oid: data}
  _RangeHandler.hits = []
  _RangeHandler.cut_first = {oid: 250_000}
  port = server.server_address[1]
  monkeypatch.setattr(dl, "_resolve_hf", lambda p: (f"http://127.0.0.1:{port}/o/{oid}", {"Authorization": "Bearer test"}))
  monkeypatch.setattr(dl, "STREAM_RETRIES", 3)
  monkeypatch.setattr(dl, "CHUNK", 64 * 1024)
  dst = str(tmp_path / "policy.pkl")
  out = dl.download_hf_file("egpu/policy/x.pkl", dst, oid, len(data))
  with open(dst, "rb") as f:
    assert out == dst and f.read() == data
  ranges = [r for o, r in _RangeHandler.hits if o == oid]
  assert ranges[0] is None and ranges[1] == "bytes=250000-"
  assert not os.path.exists(dst + ".hfpart")


def test_download_onnx_prefers_hf_then_falls_back(tmp_path, monkeypatch):
  from iqpilot.selfdrive.iqmodeld import egpu_helpers as eh
  meta = {"key": "m", "sha256": "ab" * 32, "download": {"kind": "comma_lfs", "size": 5}}
  monkeypatch.setattr(eh, "onnx_cache_path", lambda m: str(tmp_path / "m.onnx"))
  monkeypatch.setattr("iqpilot.selfdrive.iqmodeld.egpu_model.download_descriptor", lambda m: ("commalfs:" + m["sha256"], 5), raising=False)
  calls = []
  import iqpilot.selfdrive.iqmodeld.model_bundle_downloader as dlm
  monkeypatch.setattr(dlm, "download_hf_file", lambda path, dst, sha, size, progress_cb=None: (calls.append(("hf", path)), open(dst, "wb").close(), dst)[2])
  monkeypatch.setattr(eh, "resolve_download_url", lambda *a, **k: (calls.append(("lfs",)), "http://unused")[1])
  out = eh.download_onnx(meta)
  assert calls == [("hf", "onnx/" + "ab" * 32 + ".onnx")] and out == str(tmp_path / "m.onnx")
  calls.clear()
  def boom(*a, **k):
    calls.append(("hf-fail",)); raise RuntimeError("hf down")
  monkeypatch.setattr(dlm, "download_hf_file", boom)
  with pytest.raises(Exception):
    eh.download_onnx(meta)
  assert calls[:2] == [("hf-fail",), ("lfs",)]


def _parallel(monkeypatch, port, oid):
  monkeypatch.setattr(dl, "_resolve_hf", lambda p: (f"http://127.0.0.1:{port}/o/{oid}", {"Authorization": "Bearer test"}))
  monkeypatch.setattr(dl, "STREAM_RETRIES", 2)
  monkeypatch.setattr(dl, "CHUNK", 64 * 1024)
  monkeypatch.setattr(dl, "PARALLEL_MIN_BYTES", 100_000)
  monkeypatch.setattr(dl, "RANGE_BYTES", 200_000)
  monkeypatch.setattr(dl, "PARALLEL_STREAMS", 4)


def test_hf_big_file_uses_parallel_ranges(server, tmp_path, monkeypatch):
  data = os.urandom(1_050_000)
  oid = hashlib.sha256(data).hexdigest()
  _RangeHandler.store = {oid: data}
  _RangeHandler.hits = []
  port = server.server_address[1]
  _parallel(monkeypatch, port, oid)
  seen = []
  dst = str(tmp_path / "model.onnx")
  out = dl.download_hf_file("onnx/x.onnx", dst, oid, len(data), progress_cb=seen.append)
  with open(dst, "rb") as f:
    assert out == dst and f.read() == data
  ranges = sorted(r for o, r in _RangeHandler.hits if o == oid)
  assert ranges == ["bytes=0-199999", "bytes=1000000-1049999", "bytes=200000-399999", "bytes=400000-599999",
                    "bytes=600000-799999", "bytes=800000-999999"]
  assert seen[-1] == 1.0 and not os.path.exists(dst + ".hfranges") and not os.path.exists(dst + ".hfranges.ranges")


def test_hf_parallel_resumes_finished_ranges(server, tmp_path, monkeypatch):
  data = os.urandom(1_050_000)
  oid = hashlib.sha256(data).hexdigest()
  _RangeHandler.store = {oid: data}
  _RangeHandler.hits = []
  port = server.server_address[1]
  _parallel(monkeypatch, port, oid)
  dst = str(tmp_path / "model.onnx")
  rtmp = dst + ".hfranges"
  with open(rtmp, "wb") as f:
    f.truncate(len(data))
    f.seek(0)
    f.write(data[:400_000])
  with open(rtmp + ".ranges", "w") as f:
    f.write(f'{{"size": {len(data)}, "span": 200000, "done": [0, 1]}}')
  dl.download_hf_file("onnx/x.onnx", dst, oid, len(data))
  with open(dst, "rb") as f:
    assert f.read() == data
  ranges = sorted(r for o, r in _RangeHandler.hits if o == oid)
  assert ranges == ["bytes=1000000-1049999", "bytes=400000-599999", "bytes=600000-799999", "bytes=800000-999999"]


def test_hf_falls_back_to_one_stream_when_ranges_are_ignored(server, tmp_path, monkeypatch):
  data = os.urandom(1_050_000)
  oid = hashlib.sha256(data).hexdigest()
  _RangeHandler.store = {oid: data}
  _RangeHandler.hits = []
  _RangeHandler.ignore_ranges = True
  port = server.server_address[1]
  _parallel(monkeypatch, port, oid)
  dst = str(tmp_path / "model.onnx")
  dl.download_hf_file("onnx/x.onnx", dst, oid, len(data))
  with open(dst, "rb") as f:
    assert f.read() == data
  assert not os.path.exists(dst + ".hfranges") and not os.path.exists(dst + ".hfranges.ranges")
  assert (oid, None) in _RangeHandler.hits


def test_hf_parallel_sha_mismatch_is_rejected(server, tmp_path, monkeypatch):
  data = os.urandom(1_050_000)
  oid = hashlib.sha256(data).hexdigest()
  _RangeHandler.store = {oid: data}
  _RangeHandler.hits = []
  port = server.server_address[1]
  _parallel(monkeypatch, port, oid)
  dst = str(tmp_path / "model.onnx")
  with pytest.raises(RuntimeError, match="HF download failed"):
    dl.download_hf_file("onnx/x.onnx", dst, "0" * 64, len(data))
  assert not os.path.exists(dst)
