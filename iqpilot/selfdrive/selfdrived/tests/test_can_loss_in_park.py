import pytest

from iqpilot.cereal import log
from iqpilot.selfdrive.selfdrived.selfdrived import can_bus_loss_expected_in_park


def panda_state(ignition_can):
  ps = log.PandaState.new_message()
  ps.ignitionCan = ignition_can
  return ps


@pytest.mark.parametrize("in_park,states,expected", [
  (True, [panda_state(False)], True),
  (True, [panda_state(True)], False),
  (False, [panda_state(False)], False),
  (True, [], False),
])
def test_can_loss_is_expected_only_in_park_with_the_car_bus_asleep(in_park, states, expected):
  assert can_bus_loss_expected_in_park(in_park, states) is expected
