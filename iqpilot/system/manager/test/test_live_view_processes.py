# Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
import pytest

from iqpilot.selfdrive.selfdrived.selfdrived import NON_BLOCKING_PROCESSES
from iqpilot.system.manager.process_config import managed_processes


@pytest.mark.parametrize("name", ["webrtcd", "stream_encoderd"])
def test_live_view_processes_never_block_engagement(name):
  assert name in managed_processes
  assert name in NON_BLOCKING_PROCESSES


@pytest.mark.parametrize("name", ["webrtcd", "stream_encoderd"])
def test_live_view_processes_restart_after_a_crash(name):
  assert managed_processes[name].restart_if_crash
