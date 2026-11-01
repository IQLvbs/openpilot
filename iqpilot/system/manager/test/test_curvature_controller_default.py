"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos
"""
from iqpilot.common.params import Params
from iqpilot.system.manager.manager import apply_curvature_controller_default


def fresh_params():
  params = Params()
  params.remove("EnableCurvatureController")
  params.remove("EnableCurvatureControllerDefaultApplied")
  return params


def test_existing_device_with_controller_off_is_switched_on_once():
  params = fresh_params()
  params.put_bool("EnableCurvatureController", False)

  apply_curvature_controller_default(params)

  assert params.get_bool("EnableCurvatureController")
  assert params.get_bool("EnableCurvatureControllerDefaultApplied")


def test_user_choice_after_migration_is_kept():
  params = fresh_params()
  apply_curvature_controller_default(params)
  params.put_bool("EnableCurvatureController", False)

  apply_curvature_controller_default(params)

  assert not params.get_bool("EnableCurvatureController")


def test_new_install_defaults_on():
  assert Params().get_default_value("EnableCurvatureController") is True
