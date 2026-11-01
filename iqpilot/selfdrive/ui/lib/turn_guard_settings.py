# Copyright (c) 2026 IQ.Lvbs. All rights reserved.
from iqpilot.cereal import car
from iqpilot.selfdrive.controls.lib.hkg_tuning import uses_hkg_tuning
from iqpilot.system.ui.lib.multilang import tr


TURN_GUARDS_DESCRIPTION = (
  "Apply cut and apex corrections when estimated turning exceeds the request. " +
  "Changes apply next drive. Can alter turn transitions; does not enforce lane boundaries."
)


def turn_guards_availability(cp, params) -> tuple[bool, str]:
  if cp is None or cp.carFingerprint in ("", "MOCK"):
    return False, tr("Connect a vehicle to check availability.")
  bundle = params.get("CarPlatformBundle")
  if bundle and bundle.get("platform", cp.carFingerprint) != cp.carFingerprint:
    return False, tr("Connect the selected vehicle.")
  if cp.brand != "hyundai":
    return False, tr("Available on Hyundai, Kia and Genesis only.")
  if not uses_hkg_tuning(cp):
    return False, tr("Unavailable for this vehicle.")
  if (cp.steerControlType != car.CarParams.SteerControlType.torque or
      cp.lateralTuning.which() != "torque"):
    return False, tr("Unavailable for this steering system.")
  return True, ""
