# Copyright (c) 2026 IQ.Lvbs. All rights reserved.
from iqpilot.system.tombstoned import read_apport_metadata


def test_excluded_sections_do_not_hide_later_metadata():
  lines = [
    'ExecutablePath: /data/openpilot/selfdrive/pandad/pandad\n',
    'ProcMaps:\n',
    ' 0000-1000 memory mapping\n',
    'Signal: 11\n',
    'CoreDump: base64\n',
    ' compressed data\n',
    'ProcStatus:\n',
    ' Name: pandad\n',
  ]
  contents, executable, signal = read_apport_metadata(lines)
  assert contents == ''.join([lines[0], lines[3], lines[6], lines[7]])
  assert executable == '/data/openpilot/selfdrive/pandad/pandad'
  assert signal == '11'


def test_field_names_inside_values_do_not_change_section():
  lines = ['Title: Signal and CoreDump appear in this description\n',
           ' ProcMaps: a quoted field\n', 'Signal: 6\n']
  assert read_apport_metadata(lines) == (''.join(lines), '', '6')


def test_empty_report():
  assert read_apport_metadata([]) == ('', '', '')


def test_metadata_requires_an_exact_field_name():
  lines = ['OtherSignal: 11\n', 'OtherExecutablePath: /tmp/file\n']
  assert read_apport_metadata(lines) == (''.join(lines), '', '')


def test_missing_proc_status_does_not_discard_other_sections():
  lines = ['ProcMaps:\n', ' mapping\n', 'Architecture: aarch64\n', 'Package: iqpilot\n']
  assert read_apport_metadata(lines) == (''.join(lines[2:]), '', '')


def test_report_retains_full_local_artifact_and_filters_remote_body(tmp_path, monkeypatch):
  from types import SimpleNamespace
  from iqpilot.system import tombstoned

  original = 'ExecutablePath: /data/openpilot/selfdrive/pandad/pandad\nSignal: 11\nProcMaps:\n map\nCoreDump: base64\n binary\nArchitecture: aarch64\n'
  crash = tmp_path / 'input.crash'
  crash.write_text(original)
  reports = []
  monkeypatch.setattr(tombstoned, 'get_apport_stacktrace', lambda _: 'trace')
  monkeypatch.setattr(tombstoned.sentry, 'report_tombstone', lambda *args: reports.append(args))
  monkeypatch.setattr(tombstoned.Paths, 'log_root', lambda: str(tmp_path / 'logs'))
  monkeypatch.setattr(tombstoned, 'get_build_metadata', lambda: SimpleNamespace(openpilot=SimpleNamespace(git_commit='12345678')))
  tombstoned.report_tombstone_apport(str(crash))
  assert len(reports) == 1
  assert reports[0][1] == 'selfdrive/pandad/pandad - Signal: 11 (SIGSEGV) - No stacktrace'
  assert reports[0][2] == 'trace\n\nExecutablePath: /data/openpilot/selfdrive/pandad/pandad\nSignal: 11\nArchitecture: aarch64\n'
  artifacts = list((tmp_path / 'logs' / 'crash').iterdir())
  assert len(artifacts) == 1
  assert artifacts[0].read_text() == original
  assert not crash.exists()


def test_local_collection_runs_when_telemetry_is_disabled(monkeypatch):
  import pytest
  from iqpilot.system import tombstoned

  collected = []
  monkeypatch.setattr(tombstoned.sentry, 'init', lambda _: False)
  monkeypatch.setattr(tombstoned, 'get_tombstones', lambda: [('already-present.crash', 1)])
  monkeypatch.setattr(tombstoned, 'report_tombstone_apport', collected.append)

  def stop(_):
    raise InterruptedError('one scan complete')

  monkeypatch.setattr(tombstoned.time, 'sleep', stop)
  with pytest.raises(InterruptedError, match='one scan complete'):
    tombstoned.main()
  assert collected == ['already-present.crash']


def test_failed_collection_is_retried_and_completed_reports_are_not_duplicated(monkeypatch):
  from iqpilot.system import tombstoned

  calls = []
  current = [('retry.crash', 1), ('ready.crash', 1)]
  monkeypatch.setattr(tombstoned, 'get_tombstones', lambda: current)

  def collect(filename):
    calls.append(filename)
    if filename == 'retry.crash' and calls.count(filename) == 1:
      raise OSError('archive temporarily unavailable')

  monkeypatch.setattr(tombstoned, 'report_tombstone_apport', collect)
  completed = set()
  tombstoned.collect_crash_reports(completed)
  assert completed == {('ready.crash', 1)}
  tombstoned.collect_crash_reports(completed)
  tombstoned.collect_crash_reports(completed)
  assert calls == ['ready.crash', 'retry.crash', 'retry.crash']
  current[:] = [('ready.crash', 2)]
  tombstoned.collect_crash_reports(completed)
  assert completed == {('ready.crash', 2)}
  assert calls[-1] == 'ready.crash'


def test_unsupported_reports_are_retained_without_repeated_processing(monkeypatch):
  from iqpilot.system import tombstoned

  errors = []
  monkeypatch.setattr(tombstoned, 'get_tombstones', lambda: [('tombstone-unknown', 1)])
  monkeypatch.setattr(tombstoned.cloudlog, 'error', errors.append)
  completed = set()
  tombstoned.collect_crash_reports(completed)
  tombstoned.collect_crash_reports(completed)
  assert completed == {('tombstone-unknown', 1)}
  assert errors == ['unsupported crash report format: tombstone-unknown']
