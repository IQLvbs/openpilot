#!/usr/bin/env python3
import bz2
from functools import partial
import multiprocessing
import capnp
import enum
import os
import pathlib
import sys
import tqdm
import urllib.parse
import warnings
import zstandard as zstd
import numpy as np

from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from iqpilot.cereal import log as capnp_log, messaging
from iqpilot.cereal.services import SERVICE_LIST
from iqpilot.common.swaglog import cloudlog
from iqpilot.tools.lib.filereader import DATA_ENDPOINT, FileReader, file_exists, internal_source_available
from iqpilot.tools.lib.route import Route, SegmentRange, FileName

LogMessage = type[capnp._DynamicStructReader]
LogIterable = Iterable[LogMessage]
RawLogIterable = Iterable[bytes]
FileNames = tuple[str, ...]
Source = Callable[[SegmentRange, list[int], FileNames], dict[int, str]]
InternalUnavailableException = Exception("Internal source not available")
OPENPILOT_CI_BASE_URL = "https://commadataci.blob.core.windows.net/openpilotci/"
OPENPILOT_CI_ACCOUNT_URL = "https://commadataci.blob.core.windows.net"


def get_url(route_name: str, segment_num: str | int, filename: str) -> str:
  return f"{OPENPILOT_CI_BASE_URL}{route_name.replace('|', '/')}/{segment_num}/{filename}"


def upload_file(path: str, blob_name: str, overwrite=False) -> str:
  from azure.identity import AzureCliCredential
  from azure.storage.blob import BlobClient
  token_path = Path("/data/azure_token")
  credential = os.environ.get("AZURE_TOKEN") or (token_path.read_text().strip() if token_path.is_file() else AzureCliCredential())
  client = BlobClient(OPENPILOT_CI_ACCOUNT_URL, container_name="openpilotci", blob_name=blob_name, credential=credential)
  with open(path, "rb") as f:
    client.upload_blob(f, overwrite=overwrite)
  return OPENPILOT_CI_BASE_URL + blob_name


def comma_api_source(sr: SegmentRange, seg_idxs: list[int], fns: FileNames) -> dict[int, str]:
  route = Route(sr.route_name)
  if fns == FileName.RLOG:
    return {seg: route.log_paths()[seg] for seg in seg_idxs if route.log_paths()[seg] is not None}
  return {seg: route.qlog_paths()[seg] for seg in seg_idxs if route.qlog_paths()[seg] is not None}


def internal_source(sr: SegmentRange, seg_idxs: list[int], fns: FileNames, endpoint_url: str = DATA_ENDPOINT) -> dict[int, str]:
  if not internal_source_available(endpoint_url):
    raise InternalUnavailableException

  def internal_url(seg, file):
    return f"{endpoint_url.rstrip('/')}/{sr.dongle_id}/{sr.log_id}/{seg}/{file}"

  return eval_source({seg: [internal_url(seg, fn) for fn in fns] for seg in seg_idxs})


def openpilotci_source(sr: SegmentRange, seg_idxs: list[int], fns: FileNames) -> dict[int, str]:
  return eval_source({seg: [get_url(sr.route_name, seg, fn) for fn in fns] for seg in seg_idxs})


def eval_source(files: dict[int, list[str] | str]) -> dict[int, str]:
  valid_files: dict[int, str] = {}
  for seg_idx, urls in files.items():
    if isinstance(urls, str):
      urls = [urls]
    for url in urls:
      if file_exists(url):
        valid_files[seg_idx] = url
        break
  return valid_files


ALL_SERVICES = list(SERVICE_LIST.keys())


def raw_live_logreader(services: list[str] = ALL_SERVICES, addr: str = '127.0.0.1') -> RawLogIterable:
  if addr != "127.0.0.1":
    os.environ["ZMQ"] = "1"
    messaging.reset_context()
  poller = messaging.Poller()
  for service in services:
    messaging.sub_sock(service, poller, addr=addr)
  while True:
    for sock in poller.poll(100):
      yield sock.receive()


def live_logreader(services: list[str] = ALL_SERVICES, addr: str = '127.0.0.1') -> LogIterable:
  for msg in raw_live_logreader(services, addr):
    with capnp_log.Event.from_bytes(msg) as evt:
      yield evt


def flatten_type_dict(data, sep="/", prefix=None):
  result = {}
  if isinstance(data, dict):
    for key, value in data.items():
      result.update(flatten_type_dict(value, sep, key if prefix is None else prefix + sep + key))
    return result
  if isinstance(data, list):
    return {prefix: np.array(data)}
  return {prefix: data}


def get_message_dict(message, typ):
  valid = message.valid
  message = message._get(typ)
  if not hasattr(message, 'to_dict') or typ in ('qcomGnss', 'ubloxGnss'):
    return None
  result = flatten_type_dict(message.to_dict(verbose=True))
  result['_valid'] = valid
  return result


def potentially_ragged_array(values, dtype=None, **kwargs):
  try:
    return np.array(values, dtype=dtype, **kwargs)
  except ValueError:
    return np.array(values, dtype=object, **kwargs)


def msgs_to_time_series(msgs):
  values = {}
  for msg in msgs:
    typ = msg.which()
    msg_dict = get_message_dict(msg, typ)
    if msg_dict is None:
      continue
    group = values.setdefault(typ, {"t": [], **{key: [] for key in msg_dict}})
    group["t"].append(msg.logMonoTime / 1.0e9)
    for key, value in msg_dict.items():
      group[key].append(value)
  for group in values.values():
    order = np.argsort(group["t"])
    for name, group_values in group.items():
      group[name] = potentially_ragged_array(group_values)[order]
  return values


def save_log(dest, log_msgs, compress=True):
  dat = b"".join(msg.as_builder().to_bytes() for msg in log_msgs)

  if compress and dest.endswith(".bz2"):
    dat = bz2.compress(dat)
  elif compress and dest.endswith(".zst"):
    dat = zstd.compress(dat, 10)

  with open(dest, "wb") as f:
    f.write(dat)


def decompress_stream(data: bytes):
  dctx = zstd.ZstdDecompressor()
  decompressed_data = b""

  with dctx.stream_reader(data) as reader:
    decompressed_data = reader.read()

  return decompressed_data


class CachedEventReader:
  __slots__ = ('_evt', '_enum')

  def __init__(self, evt: capnp._DynamicStructReader, _enum: str | None = None):
    """All capnp attribute accesses are expensive, and which() is often called multiple times"""
    self._evt = evt
    self._enum: str | None = _enum

  # fast pickle support
  def __reduce__(self):
    return CachedEventReader._reducer, (self._evt.as_builder().to_bytes(), self._enum)

  @staticmethod
  def _reducer(data: bytes, _enum: str | None = None):
    with capnp_log.Event.from_bytes(data) as evt:
      return CachedEventReader(evt, _enum)

  def __repr__(self):
    return self._evt.__repr__()

  def __str__(self):
    return self._evt.__str__()

  def __dir__(self):
    return dir(self._evt)

  def which(self) -> str:
    if self._enum is None:
      self._enum = self._evt.which()
    return self._enum

  def __getattr__(self, name: str):
    if name.startswith("__") and name.endswith("__"):
      return getattr(self, name)
    return getattr(self._evt, name)


class _LogFileReader:
  def __init__(self, fn, only_union_types=False, sort_by_time=False, dat=None):
    self.data_version = None
    self._only_union_types = only_union_types

    ext = None
    if not dat:
      _, ext = os.path.splitext(urllib.parse.urlparse(fn).path)
      if ext not in ('', '.bz2', '.zst'):
        # old rlogs weren't compressed
        raise ValueError(f"unknown extension {ext}")

      with FileReader(fn) as f:
        dat = f.read()

    if ext == ".bz2" or dat.startswith(b'BZh9'):
      dat = bz2.decompress(dat)
    elif ext == ".zst" or dat.startswith(b'\x28\xB5\x2F\xFD'):
      # https://github.com/facebook/zstd/blob/dev/doc/zstd_compression_format.md#zstandard-frames
      dat = decompress_stream(dat)

    ents = capnp_log.Event.read_multiple_bytes(dat)

    self._ents = []
    try:
      for e in ents:
        self._ents.append(CachedEventReader(e))
    except capnp.KjException:
      warnings.warn("Corrupted events detected", RuntimeWarning, stacklevel=1)

    if sort_by_time:
      self._ents.sort(key=lambda x: x.logMonoTime)

  def __iter__(self) -> Iterator[capnp._DynamicStructReader]:
    for ent in self._ents:
      if self._only_union_types:
        try:
          ent.which()
          yield ent
        except (capnp.lib.capnp.KjException, RuntimeError):
          pass
      else:
        yield ent


class ReadMode(enum.StrEnum):
  RLOG = "r"  # only read rlogs
  QLOG = "q"  # only read qlogs
  AUTO = "a"  # default to rlogs, fallback to qlogs
  AUTO_INTERACTIVE = "i"  # default to rlogs, fallback to qlogs with a prompt from the user


class LogsUnavailable(Exception):
  pass


def direct_source(file_or_url: str) -> list[str]:
  return [file_or_url]


# TODO this should apply to camera files as well
def auto_source(identifier: str, sources: list[Source], default_mode: ReadMode) -> list[str]:
  exceptions = {}

  sr = SegmentRange(identifier)
  needed_seg_idxs = sr.seg_idxs

  mode = default_mode if sr.selector is None else ReadMode(sr.selector)
  if mode == ReadMode.QLOG:
    try_fns = [FileName.QLOG]
  else:
    try_fns = [FileName.RLOG]

  # If selector allows it, fallback to qlogs
  if mode in (ReadMode.AUTO, ReadMode.AUTO_INTERACTIVE):
    try_fns.append(FileName.QLOG)

  # Build a dict of valid files as we evaluate each source. May contain mix of rlogs, qlogs, and None.
  # This function only returns when we've sourced all files, or throws an exception
  valid_files: dict[int, str] = {}
  for fn in try_fns:
    for source in sources:
      try:
        files = source(sr, needed_seg_idxs, fn)

        # Build a dict of valid files
        valid_files |= files

        # Don't check for segment files that have already been found
        needed_seg_idxs = [idx for idx in needed_seg_idxs if idx not in valid_files]

        # We've found all files, return them
        if len(needed_seg_idxs) == 0:
          return list(valid_files.values())
        else:
          raise FileNotFoundError(f"Did not find {fn} for seg idxs {needed_seg_idxs} of {sr.route_name}")

      except Exception as e:
        exceptions[source.__name__] = e

    if fn == try_fns[0]:
      missing_logs = len(needed_seg_idxs)
      if mode == ReadMode.AUTO:
        cloudlog.warning(f"{missing_logs}/{len(sr.seg_idxs)} rlogs were not found, falling back to qlogs for those segments...")
      elif mode == ReadMode.AUTO_INTERACTIVE:
        if input(f"{missing_logs}/{len(sr.seg_idxs)} rlogs were not found, would you like to fallback to qlogs for those segments? (y/N) ").lower() != "y":
          break

  missing_logs = len(needed_seg_idxs)
  raise LogsUnavailable(f"{missing_logs}/{len(sr.seg_idxs)} logs were not found, please ensure all logs " +
                        "are uploaded. You can fall back to qlogs with '/a' selector at the end of the route name.\n\n" +
                        "Exceptions for sources:\n  - " + "\n  - ".join([f"{k}: {repr(v)}" for k, v in exceptions.items()]))


def parse_indirect(identifier: str) -> str:
  if "useradmin.comma.ai" in identifier:
    query = parse_qs(urlparse(identifier).query)
    identifier = query["onebox"][0]
  elif "konn3kt.com" in identifier:
    path = urlparse(identifier).path.strip("/").split("/")
    if path and path[0] == "connectdata":
      # signed data URL from the API host (api-*.konn3kt.com/connectdata/...), not a share link
      return identifier
    path = ['/'.join(path[:2]), *path[2:]]  # recombine log id

    identifier = path[0]
    if len(path) > 2:
      # convert url with seconds to segments
      start, end = int(path[1]) // 60, int(path[2]) // 60 + 1
      identifier = f"{identifier}/{start}:{end}"

      # add selector if it exists
      if len(path) > 3:
        identifier += f"/{path[3]}"
    else:
      # add selector if it exists
      identifier = "/".join(path)

  return identifier


def parse_direct(identifier: str):
  if identifier.startswith(("http://", "https://", "cd:/")) or pathlib.Path(identifier).exists():
    return identifier
  return None


class LogReader:
  def _parse_identifier(self, identifier: str) -> list[str]:
    # useradmin, etc.
    identifier = parse_indirect(identifier)

    # direct url or file
    direct_parsed = parse_direct(identifier)
    if direct_parsed is not None:
      return direct_source(identifier)

    identifiers = auto_source(identifier, self.sources, self.default_mode)
    return identifiers

  def __init__(self, identifier: str | list[str], default_mode: ReadMode = ReadMode.RLOG,
               sources: list[Source] | None = None, sort_by_time=False, only_union_types=False):
    if sources is None:
      sources = [internal_source, comma_api_source, openpilotci_source]

    self.default_mode = default_mode
    self.sources = sources
    self.identifier = identifier
    if isinstance(identifier, str):
      self.identifier = [identifier]

    self.sort_by_time = sort_by_time
    self.only_union_types = only_union_types

    self.__lrs: dict[int, _LogFileReader] = {}
    self.reset()

  def _get_lr(self, i):
    if i not in self.__lrs:
      self.__lrs[i] = _LogFileReader(self.logreader_identifiers[i], sort_by_time=self.sort_by_time, only_union_types=self.only_union_types)
    return self.__lrs[i]

  def __iter__(self):
    for i in range(len(self.logreader_identifiers)):
      yield from self._get_lr(i)

  def _run_on_segment(self, func, i):
    return func(self._get_lr(i))

  def run_across_segments(self, num_processes, func, disable_tqdm=False, desc=None):
    with multiprocessing.Pool(num_processes) as pool:
      ret = []
      num_segs = len(self.logreader_identifiers)
      for p in tqdm.tqdm(pool.imap(partial(self._run_on_segment, func), range(num_segs)), total=num_segs, disable=disable_tqdm, desc=desc):
        ret.extend(p)
      return ret

  def reset(self):
    self.logreader_identifiers = []
    for identifier in self.identifier:
      self.logreader_identifiers.extend(self._parse_identifier(identifier))

  @staticmethod
  def from_bytes(dat):
    return _LogFileReader("", dat=dat)

  def filter(self, msg_type: str):
    return (getattr(m, m.which()) for m in filter(lambda m: m.which() == msg_type, self))

  def first(self, msg_type: str):
    return next(self.filter(msg_type), None)

if __name__ == "__main__":
  import codecs

  # capnproto <= 0.8.0 throws errors converting byte data to string
  # below line catches those errors and replaces the bytes with \x__
  codecs.register_error("strict", codecs.backslashreplace_errors)
  log_path = sys.argv[1]
  lr = LogReader(log_path, sort_by_time=True)
  for msg in lr:
    print(msg)
