# Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos
from iqpilot._proprietary_loader import ProprietaryModuleMissing, load_private_module

try:
  load_private_module(__name__, "iqpilot_private.navd.action_engine")
except ProprietaryModuleMissing:
  from iqpilot.cereal import custom, log

  class ActionEngine:
    def __init__(self):
      self.desire = log.Desire.none
      self.lane_change_state = log.LaneChangeState.off
      self.lane_change_direction = log.LaneChangeDirection.none
      self.lane_turn_direction = custom.IQTurnSignalDirection.none
      self.lateral_edge_block = custom.IQLateralEdgeBlock.none

    def update(self, carstate, lateral_active, lane_change_prob, nav_state=None, modeldata=None, radar_state=None):
      return None
