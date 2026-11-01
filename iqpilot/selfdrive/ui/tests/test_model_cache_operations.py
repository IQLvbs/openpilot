# Copyright (c) 2026 IQ.Lvbs. All rights reserved.
from types import SimpleNamespace as NS

import pytest

from iqpilot.selfdrive.iqmodeld.model_cache import bundle_matches, remove_bundle_files


@pytest.mark.parametrize("left,right,expected", [
  (None, NS(index=1), False), (NS(index=1), NS(index=1), True),
  (NS(index=1, ref="a"), NS(index=2, ref="a"), True),
  (NS(index="bad", internalName="a"), NS(index=2, internalName="a"), True),
  (NS(index=1, displayName="a"), NS(index=2, displayName="a"), True),
  (NS(index=1), NS(index=2), False),
])
def test_bundle_identity_fallbacks(left, right, expected):
  assert bundle_matches(left, right) == expected


def test_bundle_removal_preserves_unrelated_and_external_files(tmp_path):
  root = tmp_path / "models"
  root.mkdir()
  outside = tmp_path / "outside"
  outside.write_bytes(b"keep")
  (root / "escape").symlink_to(outside)
  for name in ("model", "model.download", "metadata", "metadata.download", "unrelated"):
    (root / name).write_bytes(b"data")
  models = [NS(artifact=NS(fileName=name), metadata=None) for name in ("model", "metadata", "escape", "../outside", str(outside), "")]
  remove_bundle_files(str(root), NS(models=models))
  assert sorted(p.name for p in root.iterdir()) == ["escape", "unrelated"]
  assert outside.read_bytes() == b"keep"
