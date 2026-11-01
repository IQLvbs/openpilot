# Copyright (c) 2026 IQ.Lvbs. All rights reserved.
import os
import unicodedata

from iqpilot.cereal import car
from iqpilot.selfdrive.controls.lib.latcontrol_torque import get_nn_model_path, MOCK_MODEL_PATH
from iqpilot.selfdrive.controls.lib.latcontrol_torque_v0 import is_vw_mqb_torque
from iqpilot.system.ui.lib.multilang import tr


def nnff_availability(cp, params) -> tuple[bool, str]:
  if cp is None or cp.carFingerprint in ("", "MOCK"):
    return False, tr("connect vehicle to check availability")
  bundle = params.get("CarPlatformBundle")
  if bundle and bundle.get("platform", cp.carFingerprint) != cp.carFingerprint:
    return False, tr("connect selected vehicle")
  if (cp.steerControlType != car.CarParams.SteerControlType.torque or
      cp.lateralTuning.which() != "torque"):
    return False, tr("unavailable for this steering system")
  if cp.brand == "volkswagen":
    from iqdbc.car.volkswagen.values import VolkswagenFlags
    if cp.flags & VolkswagenFlags.PQ or is_vw_mqb_torque(cp):
      return False, tr("unavailable for this steering system")
  path, name, _ = get_nn_model_path(cp)
  if path == MOCK_MODEL_PATH or not os.path.isfile(path):
    return False, tr("no matching model installed")
  label = unicodedata.normalize("NFD", name).encode("ascii", "ignore").decode("ascii").replace("_", " ")
  return True, f"{label}: {tr('changes apply next drive')}"
