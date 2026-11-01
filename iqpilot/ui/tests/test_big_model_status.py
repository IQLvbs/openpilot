"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/

The BIG (tici/tizi) and SMALL (mici) onroad source indicators share one
resolver (comma PR #38492 states, backend-aware label). Pin its truth table.
"""
from iqpilot.ui.onroad.big_model_status import SourceState, resolve_source


class _FakeParams:
  def __init__(self, **flags):
    self._flags = flags

  def get_bool(self, key: str) -> bool:
    return bool(self._flags.get(key, False))


def _resolve(engaged=False, **flags):
  return resolve_source(_FakeParams(**flags), engaged)


def test_no_backend_enabled_is_hidden():
  assert _resolve() == ("", SourceState.HIDDEN)


def test_emac_label_is_mac():
  label, _ = _resolve(IQEmacEnabled=True, MacModelReachable=True, MacModelActive=True)
  assert label == "MAC"


def test_egpu_label_is_gpu():
  label, _ = _resolve(IQEgpuEnabled=True, UsbGpuPresent=True, UsbGpuActive=True)
  assert label == "GPU"


def test_present_dock_wins_over_emac():
  assert _resolve(IQEmacEnabled=True, MacModelReachable=True,
                  UsbGpuPresent=True, UsbGpuActive=True) == ("GPU", SourceState.ACTIVE)
  assert _resolve(IQEmacEnabled=True, UsbGpuPresent=True, UsbGpuActive=True) == ("GPU", SourceState.ACTIVE)


def test_emac_wins_without_consented_dock():
  label, _ = _resolve(IQEmacEnabled=True, IQEgpuEnabled=True, MacModelReachable=True)
  assert label == "MAC"
  label, _ = _resolve(IQEmacEnabled=True, MacModelReachable=True, UsbGpuPresent=True, IQEgpuDisabled=True)
  assert label == "MAC"


def test_egpu_states_mirror_emac():
  assert _resolve(IQEgpuEnabled=True)[1] == SourceState.HIDDEN                        # not present
  assert _resolve(IQEgpuEnabled=True, UsbGpuPresent=True)[1] == SourceState.LOADING
  assert _resolve(IQEgpuEnabled=True, UsbGpuPresent=True, UsbGpuActive=True)[1] == SourceState.ACTIVE
  assert _resolve(IQEgpuEnabled=True, UsbGpuPresent=True, UsbGpuFailed=True)[1] == SourceState.FAILED
  assert _resolve(engaged=True, IQEgpuEnabled=True, UsbGpuPresent=True,
                  UsbGpuFailed=True)[1] == SourceState.CROSSED


def test_emac_states():
  assert _resolve(IQEmacEnabled=True)[1] == SourceState.HIDDEN                       # unreachable
  assert _resolve(IQEmacEnabled=True, MacModelReachable=True)[1] == SourceState.LOADING
  assert _resolve(IQEmacEnabled=True, MacModelReachable=True, MacModelActive=True)[1] == SourceState.ACTIVE
  assert _resolve(IQEmacEnabled=True, MacModelReachable=True, MacModelFailed=True)[1] == SourceState.FAILED
  # engaged on small because big failed -> crossed
  assert _resolve(engaged=True, IQEmacEnabled=True, MacModelReachable=True,
                  MacModelFailed=True)[1] == SourceState.CROSSED
  # active always wins over a stale failed flag
  assert _resolve(IQEmacEnabled=True, MacModelReachable=True,
                  MacModelActive=True, MacModelFailed=True)[1] == SourceState.ACTIVE


def test_failed_but_disconnected_is_hidden():
  # unplugged mid-drive: nothing to show, not a red/orange "failed"
  assert _resolve(IQEmacEnabled=True, MacModelReachable=False, MacModelFailed=True)[1] == SourceState.HIDDEN
