import capnp
import contextlib
import shutil
import tempfile
import os
import pytest
import requests

from parameterized import parameterized

from iqpilot.cereal import log as capnp_log
from iqpilot.tools.lib.logreader import InternalUnavailableException, LogReader, parse_indirect
from iqpilot.tools.lib.route import SegmentRange
from iqpilot.tools.lib.url_file import URLFileException

NUM_SEGS = 17  # number of segments in the test route
ALL_SEGS = list(range(NUM_SEGS))
TEST_ROUTE = "344c5c15b34f2d8a/2024-01-03--09-37-12"
QLOG_FILE = "https://commadataci.blob.core.windows.net/openpilotci/0375fdf7b1ce594d/2019-06-13--08-32-25/3/qlog.bz2"


@contextlib.contextmanager
def setup_source_scenario(mocker, is_internal=False):
  internal_source_mock = mocker.patch("iqpilot.tools.lib.logreader.internal_source")
  internal_source_mock.__name__ = internal_source_mock._mock_name

  openpilotci_source_mock = mocker.patch("iqpilot.tools.lib.logreader.openpilotci_source")
  openpilotci_source_mock.__name__ = openpilotci_source_mock._mock_name

  comma_api_source_mock = mocker.patch("iqpilot.tools.lib.logreader.comma_api_source")
  comma_api_source_mock.__name__ = comma_api_source_mock._mock_name

  if is_internal:
    internal_source_mock.return_value = {3: QLOG_FILE}
  else:
    internal_source_mock.side_effect = InternalUnavailableException

  openpilotci_source_mock.return_value = {}
  comma_api_source_mock.return_value = {3: QLOG_FILE}

  yield


class TestLogReader:
  @pytest.mark.parametrize(("identifier", "expected"), [
    (f"{TEST_ROUTE}", ALL_SEGS),
    (f"{TEST_ROUTE.replace('/', '|')}", ALL_SEGS),
    (f"{TEST_ROUTE}--0", [0]),
    (f"{TEST_ROUTE}--5", [5]),
    (f"{TEST_ROUTE}/0", [0]),
    (f"{TEST_ROUTE}/5", [5]),
    (f"{TEST_ROUTE}/0:10", ALL_SEGS[0:10]),
    (f"{TEST_ROUTE}/0:0", []),
    (f"{TEST_ROUTE}/4:6", ALL_SEGS[4:6]),
    (f"{TEST_ROUTE}/0:-1", ALL_SEGS[0:-1]),
    (f"{TEST_ROUTE}/:5", ALL_SEGS[:5]),
    (f"{TEST_ROUTE}/2:", ALL_SEGS[2:]),
    (f"{TEST_ROUTE}/2:-1", ALL_SEGS[2:-1]),
    (f"{TEST_ROUTE}/-1", [ALL_SEGS[-1]]),
    (f"{TEST_ROUTE}/-2", [ALL_SEGS[-2]]),
    (f"{TEST_ROUTE}/-2:-1", ALL_SEGS[-2:-1]),
    (f"{TEST_ROUTE}/-4:-2", ALL_SEGS[-4:-2]),
    (f"{TEST_ROUTE}/:10:2", ALL_SEGS[:10:2]),
    (f"{TEST_ROUTE}/5::2", ALL_SEGS[5::2]),
    (f"https://konn3kt.com/{TEST_ROUTE}", ALL_SEGS),
    (f"https://konn3kt.com/{TEST_ROUTE}/60/180", ALL_SEGS[1:4]),
    (f"https://useradmin.comma.ai/?onebox={TEST_ROUTE}", ALL_SEGS),
    (f"https://useradmin.comma.ai/?onebox={TEST_ROUTE.replace('/', '|')}", ALL_SEGS),
    (f"https://useradmin.comma.ai/?onebox={TEST_ROUTE.replace('/', '%7C')}", ALL_SEGS),
  ])
  def test_indirect_parsing(self, identifier, expected, mocker):
    mocker.patch("iqpilot.tools.lib.route.get_max_seg_number_cached", return_value=NUM_SEGS - 1)
    parsed = parse_indirect(identifier)
    sr = SegmentRange(parsed)
    assert list(sr.seg_idxs) == expected, identifier

  @parameterized.expand([
    (f"{TEST_ROUTE}", f"{TEST_ROUTE}"),
    (f"{TEST_ROUTE.replace('/', '|')}", f"{TEST_ROUTE}"),
    (f"{TEST_ROUTE}--5", f"{TEST_ROUTE}/5"),
    (f"{TEST_ROUTE}/0/q", f"{TEST_ROUTE}/0/q"),
    (f"{TEST_ROUTE}/5:6/r", f"{TEST_ROUTE}/5:6/r"),
    (f"{TEST_ROUTE}/5", f"{TEST_ROUTE}/5"),
  ])
  def test_canonical_name(self, identifier, expected):
    sr = SegmentRange(identifier)
    assert str(sr) == expected

  @pytest.mark.parametrize("cache_enabled", [True, False])
  def test_direct_parsing(self, mocker, cache_enabled):
    file_exists_mock = mocker.patch("iqpilot.tools.lib.filereader.file_exists")
    if cache_enabled:
      os.environ.pop("DISABLE_FILEREADER_CACHE", None)
    else:
      os.environ["DISABLE_FILEREADER_CACHE"] = "1"
    qlog = tempfile.NamedTemporaryFile(mode='wb', delete=False)

    with requests.get(QLOG_FILE, stream=True) as r:
      with qlog as f:
        shutil.copyfileobj(r.raw, f)

    for f in [QLOG_FILE, qlog.name]:
      l = len(list(LogReader(f)))
      assert l > 100

    with pytest.raises(URLFileException) if not cache_enabled else pytest.raises(AssertionError):
      l = len(list(LogReader(QLOG_FILE.replace("/3/", "/200/"))))

    # file_exists should not be called for direct files
    assert file_exists_mock.call_count == 0

  @parameterized.expand([
    (f"{TEST_ROUTE}///",),
    (f"{TEST_ROUTE}---",),
    (f"{TEST_ROUTE}/-4:--2",),
    (f"{TEST_ROUTE}/-a",),
    (f"{TEST_ROUTE}/j",),
    (f"{TEST_ROUTE}/0:1:2:3",),
    (f"{TEST_ROUTE}/:::3",),
    (f"{TEST_ROUTE}3",),
    (f"{TEST_ROUTE}-3",),
    (f"{TEST_ROUTE}--3a",),
  ])
  def test_bad_ranges(self, segment_range):
    with pytest.raises(AssertionError):
      _ = SegmentRange(segment_range).seg_idxs

  @pytest.mark.parametrize("segment_range, api_call", [
    (f"{TEST_ROUTE}/0", False),
    (f"{TEST_ROUTE}/:2", False),
    (f"{TEST_ROUTE}/0:", True),
    (f"{TEST_ROUTE}/-1", True),
    (f"{TEST_ROUTE}", True),
  ])
  def test_slicing_api_call(self, mocker, segment_range, api_call):
    max_seg_mock = mocker.patch("iqpilot.tools.lib.route.get_max_seg_number_cached")
    max_seg_mock.return_value = NUM_SEGS
    _ = SegmentRange(segment_range).seg_idxs
    assert api_call == max_seg_mock.called

  @pytest.mark.parametrize("is_internal", [True, False])
  def test_auto_source_scenarios(self, mocker, is_internal):
    lr = LogReader(QLOG_FILE)
    qlog_len = len(list(lr))

    with setup_source_scenario(mocker, is_internal=is_internal):
      lr = LogReader(f"{TEST_ROUTE}/3/q")
      log_len = len(list(lr))
      assert qlog_len == log_len

  def test_only_union_types(self):
    with tempfile.NamedTemporaryFile() as qlog:
      # write valid Event messages
      num_msgs = 100
      with open(qlog.name, "wb") as f:
        f.write(b"".join(capnp_log.Event.new_message().to_bytes() for _ in range(num_msgs)))

      msgs = list(LogReader(qlog.name))
      assert len(msgs) == num_msgs
      [m.which() for m in msgs]

      # append non-union Event message
      event_msg = capnp_log.Event.new_message()
      non_union_bytes = bytearray(event_msg.to_bytes())
      non_union_bytes[event_msg.total_size.word_count * 8] = 0xff  # set discriminant value out of range using Event word offset
      with open(qlog.name, "ab") as f:
        f.write(non_union_bytes)

      # ensure new message is added, but is not a union type
      msgs = list(LogReader(qlog.name))
      assert len(msgs) == num_msgs + 1
      with pytest.raises((capnp.KjException, RuntimeError)):
        [m.which() for m in msgs]

      # should not be added when only_union_types=True
      msgs = list(LogReader(qlog.name, only_union_types=True))
      assert len(msgs) == num_msgs
      [m.which() for m in msgs]
