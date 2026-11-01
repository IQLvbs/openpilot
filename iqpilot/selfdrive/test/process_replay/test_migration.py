"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos
"""
from iqpilot.cereal import messaging
from iqpilot.selfdrive.test.process_replay.migration import migrate_drivingModelData


def test_driving_model_migration_ignores_incomplete_lane_metadata():
  msg = messaging.new_message("modelV2")
  msg.modelV2.init("laneLines", 4)
  for lane_line in msg.modelV2.laneLines:
    lane_line.y = [1.0]
  msg.modelV2.laneLineProbs = [0.5]

  _, added, _ = migrate_drivingModelData([(0, msg.as_reader())])

  assert len(added) == 1
  assert added[0].drivingModelData.laneLineMeta.leftProb == 0.0
  assert added[0].drivingModelData.laneLineMeta.rightProb == 0.0
