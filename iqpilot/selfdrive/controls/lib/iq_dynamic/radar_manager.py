"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""
from iqpilot.common.constants import CV

CANCEL_CEIL_MS = 1.0 * CV.KPH_TO_MS

class RadarManager:
  def __init__(self, CP, params):
    self.CP = CP
    self.params = params
    self.is_pq = self._detect_pq(CP)
    self.enabled_param = False

  @staticmethod
  def _detect_pq(CP) -> bool:
    if getattr(CP, "brand", "") != "volkswagen":
      return False
    try:
      from iqdbc.car.volkswagen.values import VolkswagenFlags
      return bool(CP.flags & VolkswagenFlags.PQ)
    except Exception:
      return False

  def read_params(self) -> None:
    if self.is_pq:
      self.enabled_param = self.params.get_bool("IQDynamicBlendStockRadar")

  def update(self, CC_IQ, sm, set_speed_kph: float) -> None:
    blend = bool(self.is_pq and self.enabled_param and self.CP.openpilotLongitudinalControl)
    CC_IQ.radarBlendActive = blend
    if not blend:
      return

    ss = sm['selfdriveState']
    cs = sm['carState']
    iq = sm['iqPlan'].iqDynamic

    long_engaged = bool(ss.enabled) and self.CP.openpilotLongitudinalControl
    iq_engaged = long_engaged and bool(iq.enabled)
    chill = iq_engaged and bool(iq.active) and (iq.state == 'acc')
    brake = bool(cs.brakePressed)
    v_ego = float(cs.vEgo)
    CC_IQ.radarEngageReq = iq_engaged and not brake
    CC_IQ.radarCancelReq = (not iq_engaged) or brake or (v_ego <= CANCEL_CEIL_MS)
    CC_IQ.useRadarAccel = chill
    CC_IQ.radarSetSpeedKph = float(max(set_speed_kph, 0.0))
    CC_IQ.radarGapBars = int(min(3, max(1, ss.personality.raw + 1)))
