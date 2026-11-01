import pytest

from iqpilot.cereal import log
from iqpilot.selfdrive.selfdrived.selfdrived import rx_checks_failed_while_powered


def panda_state(rx_invalid, ignition_line, ignition_can=False):
  ps = log.PandaState.new_message()
  ps.safetyRxChecksInvalid = rx_invalid
  ps.ignitionLine = ignition_line
  ps.ignitionCan = ignition_can
  return ps


@pytest.mark.parametrize("rx_invalid,line,can,expected", [
  (True, True, False, True),
  (True, False, True, True),
  (True, False, False, False),
  (False, True, False, False),
  (False, False, False, False),
])
def test_rx_check_failure_only_counts_while_the_car_is_powered(rx_invalid, line, can, expected):
  assert rx_checks_failed_while_powered(panda_state(rx_invalid, line, can)) is expected
