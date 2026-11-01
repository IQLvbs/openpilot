"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""

from iqpilot.common.params import ParamKeyFlag, ParamKeyType, Params
from iqpilot.common.ui_defaults import apply_ui_defaults


def test_reasoning_ui_default_and_persistence(tmp_path):
  params = Params(str(tmp_path))
  params.remove("IQShowReasoning")
  assert params.get("IQShowReasoning", return_default=True) is False
  assert params.get_type("IQShowReasoning") == ParamKeyType.BOOL
  params.put("IQShowReasoning", params.get_default_value("IQShowReasoning"))
  apply_ui_defaults(params)
  assert not params.get_bool("IQShowReasoning")
  params.put_bool("IQShowReasoning", True)
  params.clear_all(ParamKeyFlag.CLEAR_ON_MANAGER_START)
  apply_ui_defaults(params)
  assert params.get_bool("IQShowReasoning")
