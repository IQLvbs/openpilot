#!/usr/bin/env python3
from iqpilot.cereal import car, custom
from iqpilot.common.gps import get_gps_location_service
from iqpilot.common.params import Params
from iqpilot.common.realtime import CycleCollector, Priority, config_realtime_process
from iqpilot.common.swaglog import cloudlog
from iqpilot.selfdrive.controls.lib.ldw import LaneDepartureWarning
from iqpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner
import iqpilot.cereal.messaging as messaging


def main():
  config_realtime_process(5, Priority.CTRL_LOW)

  cloudlog.info("plannerd is waiting for CarParams")
  params = Params()
  CP = messaging.log_from_bytes(params.get("CarParams", block=True), car.CarParams)
  cloudlog.info("plannerd got CarParams: %s", CP.brand)

  cloudlog.info("plannerd is waiting for IQCarParams")
  CP_IQ = messaging.log_from_bytes(params.get("IQCarParams", block=True), custom.IQCarParams)
  cloudlog.info("plannerd got IQCarParams")

  gps_location_service = get_gps_location_service(params)

  ldw = LaneDepartureWarning()
  longitudinal_planner = LongitudinalPlanner(CP, CP_IQ)
  pm = messaging.PubMaster(['longitudinalPlan', 'driverAssistance', 'iqPlan'])
  # poll modelV2 (stock): the whole loop body is gated on sm.updated['modelV2'], so a
  # carState poll only makes carState the polled service with strict 100Hz receive bounds —
  # each 20Hz planner iteration (>10ms) then swallows conflated carState messages, freq_ok
  # drops below 80Hz, and every published plan goes event-invalid (commIssue, long degraded).
  sm = messaging.SubMaster(['carControl', 'carState', 'controlsState', 'vehicleParameters', 'radarState', 'modelV2', 'selfdriveState',
                            'iqLiveLocation', 'iqLiveData', 'iqNavState', 'iqCarState', 'iqConstructionZone', gps_location_service],
                           poll='modelV2')
  collector = CycleCollector()

  while True:
    sm.update()
    collector.tick()
    if sm.updated['modelV2']:
      longitudinal_planner.update(sm)
      longitudinal_planner.publish(sm, pm)

      ldw.update(sm.frame, sm['modelV2'], sm['carState'], sm['carControl'])
      msg = messaging.new_message('driverAssistance')
      msg.valid = sm.all_checks(['carState', 'carControl', 'modelV2', 'vehicleParameters'])
      msg.driverAssistance.leftLaneDeparture = ldw.left
      msg.driverAssistance.rightLaneDeparture = ldw.right
      pm.send('driverAssistance', msg)


if __name__ == "__main__":
  main()
