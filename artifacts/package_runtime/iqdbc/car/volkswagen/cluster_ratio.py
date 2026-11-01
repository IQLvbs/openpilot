"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos
"""
from iqdbc.car import DT_CTRL
from iqdbc.car.common.filter_simple import FirstOrderFilter


class ClusterRatioEstimator:
  TAU = 4.0
  MIN_V_EGO = 8.0
  MAX_A_EGO = 0.4
  MIN_RATIO = 1.0
  MAX_RATIO = 1.08

  def __init__(self):
    self._filter = FirstOrderFilter(self.MIN_RATIO, self.TAU, DT_CTRL)

  @property
  def ratio(self) -> float:
    return self._filter.x

  def update(self, v_ego: float, a_ego: float, v_ego_cluster: float) -> float:
    if v_ego >= self.MIN_V_EGO and v_ego_cluster > 0.0 and abs(a_ego) <= self.MAX_A_EGO:
      self._filter.update(min(max(v_ego_cluster / v_ego, self.MIN_RATIO), self.MAX_RATIO))
    return self._filter.x
