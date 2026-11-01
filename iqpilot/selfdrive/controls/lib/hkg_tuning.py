# Copyright (c) 2026 IQ.Lvbs. All rights reserved.
DEDICATED_TUNE_FINGERPRINTS = frozenset({"HYUNDAI_TUCSON_4TH_GEN"})


def uses_hkg_tuning(cp) -> bool:
  return cp.brand == "hyundai" and str(cp.carFingerprint) not in DEDICATED_TUNE_FINGERPRINTS
