import pytest

from iqpilot.common.params import Params, ParamKeyFlag
from iqpilot.common.ui_defaults import apply_ui_defaults


@pytest.mark.parametrize("enabled", [False, True])
def test_nnff_selection_survives_manager_startup(tmp_path, enabled):
  params = Params(str(tmp_path))
  params.put_bool("NeuralNetworkFeedForward", enabled)
  params.clear_all(ParamKeyFlag.CLEAR_ON_MANAGER_START)
  apply_ui_defaults(params)
  assert params.get("NeuralNetworkFeedForward") is not None
  assert params.get_bool("NeuralNetworkFeedForward") == enabled


def test_nnff_remains_opt_in(tmp_path):
  params = Params(str(tmp_path))
  apply_ui_defaults(params)
  assert not params.get_bool("NeuralNetworkFeedForward")
