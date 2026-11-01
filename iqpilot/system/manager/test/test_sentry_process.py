# Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
import itertools
from types import SimpleNamespace

import pytest

from iqpilot.selfdrive.selfdrived.selfdrived import NON_BLOCKING_PROCESSES
from iqpilot.system.manager.process_config import managed_processes, sentry_enabled, sentry_recording


def params(**values):
  return SimpleNamespace(get_bool=lambda key: values.get(key, False))


def test_sentry_is_off_by_default():
  assert not sentry_enabled(False, params(), None)
  assert not sentry_enabled(True, params(), None)
  assert not sentry_recording(False, params(), None)


@pytest.mark.parametrize("recording,keep", list(itertools.product([False, True], repeat=2)))
def test_sentry_runs_onroad_only_to_finish_an_event_or_keep_startups(recording, keep):
  p = params(IQSentry=True, IQSentryRecording=recording, IQSentryKeepStartups=keep)
  assert sentry_enabled(False, p, None)
  assert sentry_enabled(True, p, None) == (recording or keep)
  assert sentry_recording(True, p, None) == recording
  assert sentry_recording(False, p, None) == recording


def test_sentry_never_raises_an_onroad_alert():
  assert "iqsentryd" in NON_BLOCKING_PROCESSES
  assert managed_processes["iqsentryd"].should_run is sentry_enabled
