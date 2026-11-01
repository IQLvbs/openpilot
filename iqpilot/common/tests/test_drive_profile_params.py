"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""

from iqpilot.common.params import ParamKeyFlag, ParamKeyType, Params
from iqpilot.common.ui_defaults import apply_ui_defaults


def test_drive_profile_param_defaults_and_types():
  params = Params()
  params.remove("IQDriveProfile")
  params.remove("IQDriveDesiredArrival")
  assert params.get("IQDriveProfile") is None
  assert params.get("IQDriveProfile", return_default=True) == "neutral"
  assert params.get_type("IQDriveProfile") == ParamKeyType.STRING
  assert params.get_default_value("IQDriveDesiredArrival") is None
  assert params.get("IQDriveDesiredArrival", return_default=True) is None
  assert params.get_type("IQDriveDesiredArrival") == ParamKeyType.STRING


def test_drive_profile_params_survive_manager_clear_and_ui_defaults():
  params = Params()
  params.put("IQDriveProfile", "urgent")
  params.put("IQDriveDesiredArrival", "2000000000")
  params.clear_all(ParamKeyFlag.CLEAR_ON_MANAGER_START)
  apply_ui_defaults(params)
  assert params.get("IQDriveProfile") == "urgent"
  assert params.get("IQDriveDesiredArrival") == "2000000000"
