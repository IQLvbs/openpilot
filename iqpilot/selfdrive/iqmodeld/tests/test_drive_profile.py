"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""

import iqpilot.cereal.messaging as messaging
from iqpilot.selfdrive.iqmodeld import drive_profile as dp


def test_profile_not_declared_is_absent_and_neutral():
  inputs, state = dp.compose_drive_profile_inputs({}, None, {}, 0, 0.0)
  assert inputs == {}
  assert state == dp.DriveProfileState()


def test_live_profile_message_fields():
  message = messaging.new_message("iqDriveModelData")
  dp.populate_drive_profile_message(message.iqDriveModelData, dp.DriveProfileState(5, 0.75))
  assert message.iqDriveModelData.driveProfile == 5
  assert message.iqDriveModelData.arrivalUrgency == 0.75


def test_undeclared_profile_message_bytes_unchanged():
  pristine = messaging.new_message("iqDriveModelData")
  pristine.logMonoTime = 0
  before = pristine.to_bytes()
  message = messaging.new_message("iqDriveModelData")
  message.logMonoTime = 0
  dp.populate_drive_profile_message(message.iqDriveModelData, dp.DriveProfileState())
  assert message.to_bytes() == before
