#!/usr/bin/env python3
import iqpilot.cereal.messaging as messaging
from iqpilot.common.params import Params
from iqpilot.common.realtime import config_realtime_process
from iqpilot.selfdrive.monitoring.helpers import DriverMonitoring


def dmonitoringd_thread():
  config_realtime_process([0, 1, 2, 3], 5)

  params = Params()
  pm = messaging.PubMaster(['driverMonitoringState'])
  sm = messaging.SubMaster(['driverStateV2', 'extrinsicsCalibration', 'carState', 'selfdriveState', 'modelV2',
                            'carControl'], poll='driverStateV2')

  rhd_saved = params.get_bool("IsRhdDetected")
  DM = DriverMonitoring(rhd_saved=rhd_saved, always_on=params.get_bool("AlwaysOnDM"),
                        force_rhd=params.get_bool("ForceRHDForBSM"))
  demo_mode=False

  # 20Hz <- dmonitoringmodeld
  while True:
    sm.update()
    if not sm.updated['driverStateV2']:
      # iterate when model has new output
      continue

    valid = sm.all_checks()
    if demo_mode and sm.valid['driverStateV2']:
      DM.run_step(sm, demo=demo_mode)
    elif valid:
      DM.run_step(sm, demo=demo_mode)

    # publish
    dat = DM.get_state_packet(valid=valid)
    pm.send('driverMonitoringState', dat)

    # load live always-on toggle
    if sm['driverStateV2'].frameId % 40 == 1:
      DM.always_on = params.get_bool("AlwaysOnDM")
      DM.force_rhd = params.get_bool("ForceRHDForBSM")
      demo_mode = params.get_bool("IsDriverViewEnabled")

    # persist as soon as the estimate converges: the next drive starts on the right seat, and the
    # driving model gets the right traffic convention from the first frame instead of after a minute
    if (not demo_mode and not DM.force_rhd and DM.wheel_on_right != rhd_saved and
     DM.wheelpos.prob_offseter.filtered_stat.n > DM.settings._WHEELPOS_FILTER_MIN_COUNT and
     DM.wheel_on_right == (DM.wheelpos.prob_offseter.filtered_stat.M > DM.settings._WHEELPOS_THRESHOLD)):
      params.put_bool_nonblocking("IsRhdDetected", DM.wheel_on_right)
      rhd_saved = DM.wheel_on_right

def main():
  dmonitoringd_thread()


if __name__ == '__main__':
  main()
