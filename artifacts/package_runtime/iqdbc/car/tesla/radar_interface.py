from iqdbc.can import CANParser
from iqdbc.car import Bus, structs
from iqdbc.car.interfaces import RadarInterfaceBase
from iqdbc.car.tesla.values import DBC, LEGACY_BOSCH_RADAR_CARS, LEGACY_CARS, get_legacy_canbus

RADAR_START_ADDR = 0x410
RADAR_MSG_COUNT = 80  # 40 points * 2 messages each

BOSCH_NUM_POINTS = 32
BOSCH_RADAR_POINT_FREQ = 8
BOSCH_TRIGGER_MSG = 878
BOSCH_MAX_RANGE = 250.0
BOSCH_MIN_PROB_EXIST = 50.0


def _is_bosch_radar(CP):
  return CP.carFingerprint in LEGACY_BOSCH_RADAR_CARS


def _radar_bus(CP):
  if CP.carFingerprint in LEGACY_CARS:
    return get_legacy_canbus(CP.carFingerprint).radar
  return 1


def get_radar_can_parser(CP):
  if Bus.radar not in DBC[CP.carFingerprint]:
    return None

  if _is_bosch_radar(CP):
    messages = [('TeslaRadarSguInfo', 8)]
    num_points = BOSCH_NUM_POINTS
    freq = BOSCH_RADAR_POINT_FREQ
  else:
    messages = [('RadarStatus', 16)]
    num_points = RADAR_MSG_COUNT // 2
    freq = 16

  for i in range(num_points):
    messages.extend([
      (f'RadarPoint{i}_A', freq),
      (f'RadarPoint{i}_B', freq),
    ])

  return CANParser(DBC[CP.carFingerprint][Bus.radar], messages, _radar_bus(CP))


class RadarInterface(RadarInterfaceBase):
  def __init__(self, CP, CP_IQ):
    super().__init__(CP, CP_IQ)
    self.updated_messages = set()
    self.track_id = 0

    self.bosch_radar = _is_bosch_radar(CP)
    if self.bosch_radar:
      self.trigger_msg = BOSCH_TRIGGER_MSG
      self.num_points = BOSCH_NUM_POINTS
    else:
      self.trigger_msg = RADAR_START_ADDR + RADAR_MSG_COUNT - 1
      self.num_points = RADAR_MSG_COUNT // 2

    self.radar_off_can = CP.radarUnavailable
    self.rcp = get_radar_can_parser(CP)
    self.last_radar_data: structs.RadarData | None = None

  def update(self, can_strings):
    if self.radar_off_can or self.rcp is None:
      return super().update(None)

    vls = self.rcp.update(can_strings)
    self.updated_messages.update(vls)

    if self.trigger_msg not in self.updated_messages:
      if self.bosch_radar:
        return self.last_radar_data
      return None

    rr = self._update(self.updated_messages)
    self.updated_messages.clear()
    self.last_radar_data = rr

    return rr

  def _update(self, updated_messages):
    ret = structs.RadarData()
    if self.rcp is None:
      return ret

    if not self.rcp.can_valid:
      ret.errors.canError = True

    if self.bosch_radar:
      if self.rcp.vl['TeslaRadarSguInfo']['RADC_HWFail']:
        ret.errors.radarFault = True
    else:
      radar_status = self.rcp.vl['RadarStatus']
      if radar_status['shortTermUnavailable']:
        ret.errors.radarUnavailableTemporary = True
      if radar_status['sensorBlocked'] or radar_status['vehDynamicsError']:
        ret.errors.radarFault = True

    for i in range(self.num_points):
      msg_a = self.rcp.vl[f'RadarPoint{i}_A']
      msg_b = self.rcp.vl[f'RadarPoint{i}_B']

      # Make sure msg A and B are together
      if msg_a['Index'] != msg_b['Index2']:
        continue

      if not msg_a['Tracked']:
        if i in self.pts:
          del self.pts[i]
        continue

      if self.bosch_radar and (msg_a['LongDist'] > BOSCH_MAX_RANGE or msg_a['LongDist'] <= 0 or msg_a['ProbExist'] < BOSCH_MIN_PROB_EXIST):
        if i in self.pts:
          del self.pts[i]
        continue

      if i not in self.pts:
        self.pts[i] = structs.RadarData.RadarPoint()
        self.pts[i].trackId = self.track_id
        self.track_id += 1

      self.pts[i].dRel = msg_a['LongDist']
      self.pts[i].yRel = msg_a['LatDist']
      self.pts[i].vRel = msg_a['LongSpeed']
      self.pts[i].aRel = msg_a['LongAccel']
      self.pts[i].yvRel = msg_b['LatSpeed']
      self.pts[i].measured = bool(msg_a['Meas'])

    ret.points = list(self.pts.values())
    return ret
