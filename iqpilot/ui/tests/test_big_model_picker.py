# Copyright (c) 2026 IQ.Lvbs. All rights reserved.
from types import SimpleNamespace

import pytest

from iqpilot.common.params import Params
from iqpilot.selfdrive.iqmodeld.egpu_artifact_status import EgpuArtifactStatus


@pytest.fixture
def panels(monkeypatch):
  import iqpilot.selfdrive.iqmodeld.egpu_artifact_status as artifacts
  import iqpilot.selfdrive.iqmodeld.egpu_model as egpu_model
  import iqpilot.ui.layouts.settings.iq_panels as module
  params = Params()
  for key in ("IQEmacEnabled", "IQEmacSmallModel", "UsbGpuLoading", "UsbGpuCompiled", "MacModelReady"):
    params.put_bool(key, False)
  state = SimpleNamespace(params=params, sm={"deviceState": SimpleNamespace(egpuDockPresent=True)})
  monkeypatch.setattr(module, "ui_state", state)
  monkeypatch.setattr(module, "_big_model_options", lambda: [("cinque", "Cinque 2")])
  statuses = {}
  monkeypatch.setattr(egpu_model, "resolve_egpu_model", lambda _p, key, allow_refresh=True: None if key == "gone" else {"key": key})
  monkeypatch.setattr(artifacts, "egpu_artifact_status", lambda meta: statuses[meta["key"]])
  return SimpleNamespace(module=module, params=params, state=state, statuses=statuses)


@pytest.mark.parametrize("status,label,detail", [
  (EgpuArtifactStatus(True, True, 3_200_000_000), "on device", ""),
  (EgpuArtifactStatus(False, True, 3_200_000_000), "precompiled", "3.2 GB"),
  (EgpuArtifactStatus(False, True, 812_000_000), "precompiled", "812 MB"),
  (EgpuArtifactStatus(False, False, 0), "compile on dock", ""),
])
def test_picker_row_shows_what_the_dock_will_do(panels, status, label, detail):
  panels.statuses["cinque"] = status
  pill, text = panels.module._big_model_decoration("cinque")
  assert pill.label == label
  assert text == detail


def test_unknown_model_has_no_pill(panels):
  assert panels.module._big_model_decoration("gone") == (None, "")


def test_big_model_value_follows_the_dock_and_setup_progress(panels):
  m, p = panels.module, panels.params
  p.put("IQEmacModel", "cinque")
  panels.state.sm["deviceState"] = SimpleNamespace(egpuDockPresent=False)
  assert m.ModelsLayout._big_model_value() == "Off"

  panels.state.sm["deviceState"] = SimpleNamespace(egpuDockPresent=True)
  assert m.ModelsLayout._big_model_value() == "Cinque 2"

  p.put_bool("UsbGpuLoading", True)
  p.put("UsbGpuSetupProgress", "0.37")
  assert m.ModelsLayout._big_model_value() == "Cinque 2 37%"

  p.put_bool("UsbGpuCompiled", True)
  assert m.ModelsLayout._big_model_value() == "Cinque 2"
