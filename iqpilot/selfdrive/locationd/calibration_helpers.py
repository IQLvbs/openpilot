import math

import numpy as np
from iqpilot.cereal import log

from iqpilot.selfdrive.locationd.calibrationd import HEIGHT_INIT, HEIGHT_SANE_MIN, HEIGHT_SANE_MAX


def get_calibrated_rpy(live_calib: log.ExtrinsicsCalibration) -> np.ndarray | None:
  if live_calib.calStatus != log.ExtrinsicsCalibration.Status.calibrated:
    return None

  if len(live_calib.rpyCalib) != 3:
    return None

  calib_rpy = np.asarray(live_calib.rpyCalib, dtype=np.float32)
  return calib_rpy if np.isfinite(calib_rpy).all() else None


def get_render_path_height(live_calib: log.ExtrinsicsCalibration) -> float:
  if live_calib.calStatus != log.ExtrinsicsCalibration.Status.calibrated:
    return float(HEIGHT_INIT[0])

  if len(live_calib.height) != 1:
    return float(HEIGHT_INIT[0])

  height = float(live_calib.height[0])
  if not math.isfinite(height):
    return float(HEIGHT_INIT[0])
  if not (HEIGHT_SANE_MIN <= height <= HEIGHT_SANE_MAX):
    return float(HEIGHT_INIT[0])
  return height
