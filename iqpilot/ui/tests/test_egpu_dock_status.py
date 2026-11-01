# Copyright (c) 2026 IQ.Lvbs. All rights reserved.
from types import SimpleNamespace

import pytest

from iqpilot.common.params import Params
from iqpilot.selfdrive.selfdrived.alertmanager import set_offroad_alert
from iqpilot.ui.onroad.big_model_status import EGPU_DEGRADED_ALERTS, EGPU_FAULT_ALERTS, DockStatus, egpu_dock_status

DOCKED = SimpleNamespace(egpuDockPresent=True)


@pytest.fixture
def params():
  p = Params()
  for key in ("UsbGpuFailed", "UsbGpuLoading", "UsbGpuCompiled"):
    p.put_bool(key, False)
  p.remove("UsbGpuSetupProgress")
  for key in EGPU_FAULT_ALERTS + EGPU_DEGRADED_ALERTS:
    p.remove(key)
  return p


def test_no_dock_hides_the_indicator(params):
  params.put_bool("UsbGpuFailed", True)
  assert egpu_dock_status(params, SimpleNamespace(egpuDockPresent=False)) == (DockStatus.HIDDEN, 0.0)
  assert egpu_dock_status(params, SimpleNamespace()) == (DockStatus.HIDDEN, 0.0)


def test_idle_dock_is_ready(params):
  assert egpu_dock_status(params, DOCKED) == (DockStatus.READY, 0.0)


@pytest.mark.parametrize("alert", EGPU_FAULT_ALERTS)
def test_raised_fault_alert_is_a_fault(params, alert):
  set_offroad_alert(alert, True)
  assert not params.get_bool(alert)
  assert egpu_dock_status(params, DOCKED)[0] == DockStatus.FAULT


def test_stalled_big_model_is_a_fault(params):
  params.put_bool("UsbGpuFailed", True)
  assert egpu_dock_status(params, DOCKED)[0] == DockStatus.FAULT


def test_setup_reports_clamped_progress(params):
  params.put_bool("UsbGpuLoading", True)
  params.put("UsbGpuSetupProgress", "0.42")
  assert egpu_dock_status(params, DOCKED) == (DockStatus.SETUP, pytest.approx(0.42))
  params.put("UsbGpuSetupProgress", "7")
  assert egpu_dock_status(params, DOCKED) == (DockStatus.SETUP, 1.0)
  params.put_bool("UsbGpuCompiled", True)
  assert egpu_dock_status(params, DOCKED)[0] == DockStatus.READY


@pytest.mark.parametrize("alert", EGPU_DEGRADED_ALERTS)
def test_degraded_alert_greys_the_indicator(params, alert):
  set_offroad_alert(alert, True)
  assert egpu_dock_status(params, DOCKED)[0] == DockStatus.DEGRADED


def test_fault_outranks_setup_and_degraded(params):
  params.put_bool("UsbGpuLoading", True)
  set_offroad_alert(EGPU_DEGRADED_ALERTS[0], True)
  set_offroad_alert(EGPU_FAULT_ALERTS[0], True)
  assert egpu_dock_status(params, DOCKED)[0] == DockStatus.FAULT
