"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from iqpilot.selfdrive.iqmodeld.egpu_helpers import egpu_model_oob_pkl_path, egpu_oob_pkl_path, egpu_pkl_path, egpu_policy_pkl_path

PRECOMPILED_FIELDS = ("egpu_model_oob_artifact", "egpu_oob_artifact", "egpu_policy_artifact")
LOCAL_ARTIFACTS = (egpu_model_oob_pkl_path, egpu_oob_pkl_path, egpu_policy_pkl_path, egpu_pkl_path)


@dataclass(frozen=True)
class EgpuArtifactStatus:
  on_device: bool
  precompiled: bool
  download_bytes: int


def egpu_artifact_status(meta: dict) -> EgpuArtifactStatus:
  on_device = any(os.path.isfile(path(meta)) for path in LOCAL_ARTIFACTS)
  for field in PRECOMPILED_FIELDS:
    artifact = meta.get(field)
    if artifact:
      size = artifact.get("size", 0) if isinstance(artifact, dict) else 0
      return EgpuArtifactStatus(on_device, True, int(size or 0))
  return EgpuArtifactStatus(on_device, False, 0)
