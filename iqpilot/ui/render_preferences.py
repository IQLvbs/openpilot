# Copyright (c) 2026 IQ.Lvbs. All rights reserved.
import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RenderPreferences:
  camera_offset: float = 0.0
  ambient_tracks: bool = False

  @classmethod
  def read(cls, params, *, has_active_model: bool):
    offset = 0.0
    if has_active_model:
      try:
        offset = float(params.get('CameraOffset', return_default=True))
      except (TypeError, ValueError):
        pass
    return cls(camera_offset=offset if math.isfinite(offset) else 0.0,
               ambient_tracks=bool(params.get('AmbientTrackDots', return_default=True)))
