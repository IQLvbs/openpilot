#!/usr/bin/env python3
import os
import sys
import time
import json

from iqpilot.common.basedir import BASEDIR
from iqpilot.common.params import Params
from iqpilot.selfdrive.selfdrived.alertmanager import set_offroad_alert
from iqpilot.system.updated.updated import parse_release_notes

if __name__ == "__main__":
  params = Params()

  with open(os.path.join(BASEDIR, "iqpilot/selfdrive/selfdrived/alerts_offroad.json")) as f:
    offroad_alerts = json.load(f)

  t = 10 if len(sys.argv) < 2 else int(sys.argv[1])
  while True:
    print("setting alert update")
    params.put_bool("UpdateAvailable", True)
    params.put("UpdaterNewReleaseNotes", parse_release_notes(BASEDIR))

    time.sleep(t)
    params.put_bool("UpdateAvailable", False)

    # cycle through normal alerts
    for a in offroad_alerts:
      print("setting alert:", a)
      set_offroad_alert(a, True)
      time.sleep(t)
      set_offroad_alert(a, False)

    print("no alert")
    time.sleep(t)
