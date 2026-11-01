"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/

The eMac bundles ship `minimum_selector_version = 17`, and the version
gate lives in the COMPILED private selector bundle, not in this repo. If that
bundle is rebuilt from stale source the gate still reads 16, every eMac bundle
is silently dropped as "too new", and the selector simply shows no eMac models
— with no error anywhere. Assert the effective gate instead, so a stale
private bundle fails here rather than on a device.
"""
from iqpilot.selfdrive.iqmodeld.emac_model_meta import EMAC_BUNDLE_MIN_SELECTOR_VERSION
from iqpilot.selfdrive.iqmodeld.models.helpers import is_bundle_version_compatible


def test_gate_accepts_the_version_our_emac_bundles_ship():
  assert is_bundle_version_compatible({"minimumSelectorVersion": EMAC_BUNDLE_MIN_SELECTOR_VERSION}), (
    f"the effective selector gate rejects minimumSelectorVersion="
    f"{EMAC_BUNDLE_MIN_SELECTOR_VERSION}; the private selector bundle is stale. "
    f"Rebuild it from BOTH iqpilot/models_private_src/helpers.py "
    f"(CURRENT_SELECTOR_VERSION) and fetcher.py (MANIFEST_VERSION)."
  )


def test_gate_still_accepts_older_bundles():
  # the window is a range, not a floor: bumping it must not orphan the existing catalogue
  assert is_bundle_version_compatible({"minimumSelectorVersion": 12})
  assert is_bundle_version_compatible({"minimumSelectorVersion": 16})


def test_gate_rejects_a_bundle_from_the_future():
  assert not is_bundle_version_compatible({"minimumSelectorVersion": EMAC_BUNDLE_MIN_SELECTOR_VERSION + 5})


def test_gate_accepts_encrypted_iqc_bundles():
  assert is_bundle_version_compatible({"minimumSelectorVersion": 20})


def test_bundles_kept_only_for_older_builds_are_hidden():
  assert not is_bundle_version_compatible({"minimumSelectorVersion": 18, "maximum_selector_version": "19"})
  assert not is_bundle_version_compatible({"minimumSelectorVersion": 18, "maximumSelectorVersion": 19})
  assert is_bundle_version_compatible({"minimumSelectorVersion": 18, "maximum_selector_version": 20})
