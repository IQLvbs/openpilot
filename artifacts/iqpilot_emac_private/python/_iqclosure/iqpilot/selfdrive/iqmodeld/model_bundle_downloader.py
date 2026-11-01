"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor

MODELS_BASE_URLS = (
  "https://git.konn3kt.com/teal/IQModels/raw/branch/main",
  "https://gitlvb.teallvbs.xyz/teal/IQModels/raw/branch/main",
)
CHUNK = 4 * 1024 * 1024
HTTP_TIMEOUT_S = 60.0
STREAM_RETRIES = 6
PARALLEL_STREAMS = max(1, int(os.environ.get("IQ_DOWNLOAD_STREAMS", "8")))
PARALLEL_MIN_BYTES = 32 * 1024 * 1024
RANGE_BYTES = 16 * 1024 * 1024
RANGE_RETRIES = 4
PARALLEL_PARTS = max(1, int(os.environ.get("IQ_DOWNLOAD_PARTS", "3")))


def _git_auth():
  import importlib
  for mod in ("iqpilot_private.models.git_auth", "iqpilot.models_private_src.git_auth",
              "iqpilot.selfdrive.iqmodeld.models.git_auth"):
    try:
      return importlib.import_module(mod)
    except Exception:
      continue
  return None


def _resolve_hf(hf_path: str) -> tuple[str | None, dict]:
  m = _git_auth()
  if m is None:
    return None, {}
  return m.resolve_hf(hf_path)


def nutcracker_challenge() -> str:
  m = _git_auth()
  return m.nutcracker_challenge() if m is not None and hasattr(m, "nutcracker_challenge") else ""


def nutcracker_link_nonce(challenge: str) -> str:
  return _git_auth().nutcracker_link_nonce(challenge)


def request_nutcracker_grant(**proof) -> bool:
  m = _git_auth()
  return m.request_nutcracker_grant(**proof) if m is not None and hasattr(m, "request_nutcracker_grant") else False


def requests_get(url: str, **kwargs):
  m = _git_auth()
  if m is None:
    raise RuntimeError("no asset broker available")
  return m.requests_get(url, **kwargs)


class RangesNotHonoured(Exception):
  pass


def _range_plan(size: int, span: int | None = None) -> list[tuple[int, int]]:
  span = RANGE_BYTES if span is None else span
  return [(a, min(a + span, size) - 1) for a in range(0, size, span)]


def _ranges_state_path(tmp: str) -> str:
  return tmp + ".ranges"


def _load_done_ranges(tmp: str, size: int, plan: list[tuple[int, int]]) -> set[int]:
  state = _ranges_state_path(tmp)
  if not (os.path.isfile(tmp) and os.path.getsize(tmp) == size and os.path.isfile(state)):
    return set()
  try:
    with open(state) as f:
      doc = json.load(f)
    if doc.get("size") == size and doc.get("span") == RANGE_BYTES:
      return {int(i) for i in doc.get("done", []) if 0 <= int(i) < len(plan)}
  except (OSError, ValueError):
    pass
  return set()


def _fetch_range(get, url: str, headers: dict, fd: int, start: int, end: int, on_bytes) -> None:
  with get(url, headers={**headers, "Range": f"bytes={start}-{end}"}, stream=True, timeout=HTTP_TIMEOUT_S,
           allow_redirects=True) as r:
    r.raise_for_status()
    if r.status_code != 206:
      raise RangesNotHonoured(f"server answered {r.status_code} to a Range request")
    off = start
    for chunk in r.iter_content(CHUNK):
      if off + len(chunk) > end + 1:
        raise RuntimeError(f"range {start}-{end} overflowed")
      os.pwrite(fd, chunk, off)
      off += len(chunk)
      on_bytes(len(chunk))
    if off != end + 1:
      raise RuntimeError(f"range {start}-{end} short: {off - start} bytes")


def ranged_download(get, url: str, headers: dict, tmp: str, size: int, progress_cb=None,
                    streams: int = PARALLEL_STREAMS) -> str:
  """Fill tmp with size bytes through concurrent Range requests, resuming any finished ranges from
  a previous attempt, and return the file's sha256 hashed in order while ranges land. Raises
  RangesNotHonoured when the server ignores Range so the caller can fall back to a single stream."""
  plan = _range_plan(size)
  done = _load_done_ranges(tmp, size, plan)
  if not done:
    with open(tmp, "wb") as f:
      f.truncate(size)
  todo = [i for i in range(len(plan)) if i not in done]
  lock = threading.Lock()
  landed = threading.Condition(lock)
  got = [sum(plan[i][1] - plan[i][0] + 1 for i in done)]
  state = _ranges_state_path(tmp)
  fd = os.open(tmp, os.O_RDWR)
  failed = [False]
  digest = hashlib.sha256()

  def hasher() -> None:
    for i, (start, end) in enumerate(plan):
      with landed:
        while i not in done and not failed[0]:
          landed.wait()
        if failed[0]:
          return
      off = start
      while off <= end:
        block = os.pread(fd, min(CHUNK, end + 1 - off), off)
        if not block:
          return
        digest.update(block)
        off += len(block)

  def on_bytes(n: int) -> None:
    with lock:
      got[0] += n
      if progress_cb is not None:
        progress_cb(min(1.0, got[0] / size))

  def finish(i: int) -> None:
    with landed:
      done.add(i)
      with open(state + ".tmp", "w") as f:
        json.dump({"size": size, "span": RANGE_BYTES, "done": sorted(done)}, f)
      os.replace(state + ".tmp", state)
      landed.notify_all()

  def work(i: int) -> None:
    start, end = plan[i]
    last: Exception | None = None
    for _attempt in range(RANGE_RETRIES):
      before = got[0]
      try:
        _fetch_range(get, url, headers, fd, start, end, on_bytes)
        finish(i)
        return
      except RangesNotHonoured:
        raise
      except Exception as e:
        last = e
        with lock:
          got[0] = before
    raise RuntimeError(f"range {start}-{end} failed: {last}")

  hash_thread = threading.Thread(target=hasher, name="ranged-download-hash", daemon=True)
  hash_thread.start()
  try:
    with ThreadPoolExecutor(max_workers=max(1, min(streams, len(todo) or 1))) as pool:
      for _ in pool.map(work, todo):
        pass
  except BaseException:
    with landed:
      failed[0] = True
      landed.notify_all()
    hash_thread.join()
    os.close(fd)
    raise
  hash_thread.join()
  os.close(fd)
  if os.path.isfile(state):
    os.remove(state)
  return digest.hexdigest()


def _verify_file(tmp: str, sha256: str, size: int) -> None:
  digest = hashlib.sha256()
  with open(tmp, "rb") as f:
    for chunk in iter(lambda: f.read(CHUNK), b""):
      digest.update(chunk)
  if size and os.path.getsize(tmp) != size:
    raise RuntimeError(f"size mismatch: {os.path.getsize(tmp)}/{size} bytes")
  if sha256 and digest.hexdigest() != sha256:
    os.remove(tmp)
    raise RuntimeError("sha256 mismatch")


def download_hf_file(hf_path: str, dst: str, sha256: str, size: int, progress_cb=None) -> str:
  import requests
  url, headers = _resolve_hf(hf_path)
  if url is None:
    raise RuntimeError("no HF credentials available")
  os.makedirs(os.path.dirname(dst), exist_ok=True)
  tmp = dst + ".hfpart"
  last_error: Exception | None = None
  if size and size >= PARALLEL_MIN_BYTES and PARALLEL_STREAMS > 1:
    rtmp = dst + ".hfranges"
    try:
      digest = ranged_download(requests.get, url, headers, rtmp, size, progress_cb)
      if os.path.getsize(rtmp) != size:
        raise RuntimeError(f"size mismatch: {os.path.getsize(rtmp)}/{size} bytes")
      if sha256 and digest != sha256:
        os.remove(rtmp)
        raise RuntimeError("sha256 mismatch")
      os.replace(rtmp, dst)
      return dst
    except RangesNotHonoured:
      for leftover in (rtmp, _ranges_state_path(rtmp)):
        if os.path.exists(leftover):
          os.remove(leftover)
    except Exception as e:
      last_error = e
  for attempt in range(STREAM_RETRIES):
    try:
      if attempt:
        url, headers = _resolve_hf(hf_path)
      have = os.path.getsize(tmp) if os.path.isfile(tmp) else 0
      if size and have > size:
        os.remove(tmp)
        have = 0
      if not size or have < size:
        req_headers = dict(headers)
        if have:
          req_headers["Range"] = f"bytes={have}-"
        with requests.get(url, headers=req_headers, stream=True, timeout=HTTP_TIMEOUT_S, allow_redirects=True) as r:
          r.raise_for_status()
          if have and r.status_code != 206:
            have = 0
          with open(tmp, "ab" if have else "wb") as f:
            got = have
            for chunk in r.iter_content(CHUNK):
              f.write(chunk)
              got += len(chunk)
              if progress_cb is not None and size:
                progress_cb(min(1.0, got / size))
      _verify_file(tmp, sha256, size)
      os.replace(tmp, dst)
      return dst
    except Exception as e:
      last_error = e
  raise RuntimeError(f"HF download failed: {last_error}")


def _part_path(dst: str, oid: str) -> str:
  return os.path.join(dst + ".parts", oid)


def _part_complete(path: str, oid: str, size: int) -> bool:
  if not os.path.isfile(path) or os.path.getsize(path) != size:
    return False
  digest = hashlib.sha256()
  with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(CHUNK), b""):
      digest.update(chunk)
  return digest.hexdigest() == oid


def _fetch_part(session, obj: dict, path: str, progress) -> None:
  size = int(obj["size"])
  have = os.path.getsize(path) if os.path.isfile(path) else 0
  if have > size:
    os.remove(path)
    have = 0
  m = _git_auth()
  if m is None:
    raise RuntimeError("no asset broker available for LFS parts")
  target = m.broker_lfs_url(obj["oid"], size)
  headers = m.broker_headers() if target else {}
  if not (target and headers):
    raise RuntimeError("no device identity for the konn3kt asset broker")
  _stream_part(session, target, headers, path, have, progress)


def _stream_part(session, href: str, headers: dict, path: str, have: int, progress) -> None:
  # LFS parts are content-addressed (oid == sha256), so a half-written part can be resumed with a
  # Range request and verified afterwards instead of being thrown away on every restart.
  if have:
    headers = {**headers, "Range": f"bytes={have}-"}
  with session.get(href, headers=headers, stream=True, timeout=HTTP_TIMEOUT_S) as r:
    r.raise_for_status()
    if have and r.status_code != 206:
      have = 0
    with open(path, "ab" if have else "wb") as f:
      for chunk in r.iter_content(CHUNK):
        f.write(chunk)
        progress(len(chunk))


def download_lfs_bundle(objects: list, dst: str, sha256: str, size: int, progress_cb=None) -> str:
  import requests
  os.makedirs(dst + ".parts", exist_ok=True)
  total = int(size) or sum(int(o["size"]) for o in objects)
  done_bytes = sum(int(o["size"]) for o in objects if _part_complete(_part_path(dst, o["oid"]), o["oid"], int(o["size"])))
  got = [done_bytes]

  def progress(n: int) -> None:
    got[0] += n
    if progress_cb is not None and total:
      progress_cb(min(1.0, got[0] / total))

  last_error: Exception | None = None
  lock = threading.Lock()

  def locked_progress(n: int) -> None:
    with lock:
      progress(n)

  def fetch(obj: dict) -> None:
    path = _part_path(dst, obj["oid"])
    if _part_complete(path, obj["oid"], int(obj["size"])):
      return
    _fetch_part(requests.Session(), obj, path, locked_progress)
    if not _part_complete(path, obj["oid"], int(obj["size"])):
      if os.path.getsize(path) >= int(obj["size"]):
        os.remove(path)
      raise RuntimeError(f"part {obj['oid'][:12]} incomplete or failed verification")

  for _attempt in range(STREAM_RETRIES):
    try:
      pending = [o for o in objects if not _part_complete(_part_path(dst, o["oid"]), o["oid"], int(o["size"]))]
      got[0] = total - sum(int(o["size"]) for o in pending)
      with ThreadPoolExecutor(max_workers=max(1, min(PARALLEL_PARTS, len(pending) or 1))) as pool:
        for _ in pool.map(fetch, pending):
          pass
      break
    except Exception as e:
      last_error = e
  else:
    raise RuntimeError(f"model bundle download failed: {last_error}")

  tmp = dst + ".part"
  digest = hashlib.sha256()
  with open(tmp, "wb") as out:
    for obj in objects:
      with open(_part_path(dst, obj["oid"]), "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
          out.write(chunk)
          digest.update(chunk)
  if total and os.path.getsize(tmp) != total:
    os.remove(tmp)
    raise RuntimeError(f"size mismatch: {os.path.getsize(tmp) if os.path.exists(tmp) else 0}/{total} bytes")
  if sha256 and digest.hexdigest() != sha256:
    os.remove(tmp)
    for obj in objects:
      try:
        os.remove(_part_path(dst, obj["oid"]))
      except OSError:
        pass
    raise RuntimeError("sha256 mismatch")
  os.replace(tmp, dst)
  for obj in objects:
    try:
      os.remove(_part_path(dst, obj["oid"]))
    except OSError:
      pass
  try:
    os.rmdir(dst + ".parts")
  except OSError:
    pass
  return dst
