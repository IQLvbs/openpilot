from types import SimpleNamespace

import pytest

from iqpilot.cereal import car
from iqpilot.common.params import Params
from iqpilot.ui.layouts.settings import iq_panels


@pytest.fixture
def panel(monkeypatch, mocker, tmp_path):
  cp = SimpleNamespace(brand="hyundai", flags=0, carFingerprint="HYUNDAI_KONA_EV",
                       steerControlType=car.CarParams.SteerControlType.torque,
                       lateralTuning=SimpleNamespace(which=lambda: "torque"))
  state = SimpleNamespace(CP=cp, params=Params(str(tmp_path / "params")), offroad=True)
  state.is_offroad = lambda: state.offroad
  monkeypatch.setattr(iq_panels, "ui_state", state)
  monkeypatch.setattr(iq_panels.gui_app, "font", lambda weight: iq_panels.rl.Font())
  monkeypatch.setattr(iq_panels, "load_catalog", dict)
  available = mocker.Mock(return_value=(True, "HYUNDAI KONA EV: changes apply next drive"))
  monkeypatch.setattr(iq_panels, "nnff_availability", available)
  layout = iq_panels.VehicleLayout()
  return SimpleNamespace(layout=layout, state=state, available=available)


@pytest.mark.parametrize("saved,available,offroad,expected", [
  (False, True, True, True),
  (True, True, True, False),
  (True, False, True, False),
  (False, False, True, False),
  (False, True, False, False),
  (True, True, False, True),
  (True, False, False, True),
])
def test_real_iq_toggle_validates_before_saving_and_requesting_cycle(panel, mocker, saved, available, offroad, expected):
  state, layout = panel.state, panel.layout
  state.params.put_bool("NeuralNetworkFeedForward", saved)
  state.params.put_bool("IQHkgReducedTorqueFeedback", True)
  panel.available.return_value = available, "model status"
  layout._refresh_nnff(force=True)
  state.offroad = offroad
  put_bool = state.params.put_bool
  writes = []

  def record_put(key, value):
    if key == "OnroadCycleRequested":
      assert state.params.get_bool("NeuralNetworkFeedForward") is expected
    writes.append((key, value))
    put_bool(key, value)

  state.params = mocker.Mock(wraps=state.params)
  state.params.put_bool.side_effect = record_put
  toggle = layout._nnff.action_item.toggle
  assert toggle.param_key is None  # No widget autosave may bypass the guarded callback.
  toggle._handle_mouse_release(None)
  changed = saved != expected
  assert toggle.get_state() is expected
  assert state.params.get_bool("NeuralNetworkFeedForward") is expected
  assert state.params.get_bool("OnroadCycleRequested") is changed
  assert state.params.get_bool("IQHkgReducedTorqueFeedback") is not (changed and expected)
  if changed:
    assert writes[-1] == ("OnroadCycleRequested", True)
  else:
    assert writes == []


@pytest.mark.parametrize("available,saved,offroad,enabled", [
  (True, False, True, True),
  (True, True, False, False),
  (False, False, True, False),
  (False, True, True, True),
  (False, True, False, False),
])
def test_frame_syncs_saved_state_and_offroad_availability(panel, available, saved, offroad, enabled):
  panel.available.return_value = available, "model status"
  panel.layout._refresh_nnff(force=True)
  panel.state.params.put_bool("NeuralNetworkFeedForward", saved)
  panel.state.offroad = offroad
  panel.layout._update_state()
  assert panel.layout._nnff.action_item.get_state() is saved
  assert panel.layout._nnff.action_item.enabled is enabled


def test_model_availability_is_cached_until_vehicle_changes_or_panel_opens(panel):
  layout, state = panel.layout, panel.state
  assert panel.available.call_count == 1
  for _ in range(20):
    layout._update_state()
  assert panel.available.call_count == 1

  state.CP.carFingerprint = "HYUNDAI_IONIQ_5"
  panel.available.return_value = False, "no matching model installed"
  layout._update_state()
  assert panel.available.call_count == 2
  assert not layout._nnff.action_item.enabled
  assert "no matching model installed" in layout._nnff.description

  state.params.put("CarPlatformBundle", {"platform": "HYUNDAI_KONA_EV", "brand": "hyundai"})
  layout._on_vehicle_changed()
  assert panel.available.call_count == 3
  layout._update_state()
  assert panel.available.call_count == 3

  state.CP.flags = 1
  layout._update_state()
  assert panel.available.call_count == 4
  panel.available.return_value = True, "HYUNDAI IONIQ 5: changes apply next drive"
  layout.show_event()
  assert panel.available.call_count == 5
  assert layout._nnff.action_item.enabled
  assert "HYUNDAI IONIQ 5" in layout._nnff.description


def test_nnff_row_survives_brand_panel_rebuilds(panel, monkeypatch, mocker):
  layout = panel.layout
  brand_row = iq_panels.toggle_item("Brand setting")
  brand_panel = SimpleNamespace(items=[brand_row], update_settings=mocker.Mock())
  monkeypatch.setattr(iq_panels, "brand_settings_for", lambda brand: brand_panel if brand == "toyota" else None)
  assert layout.items == [layout._vehicle_item, layout._legend_widget, layout._nnff]
  assert layout._nnff.title == "NNFF Steering"
  assert "HYUNDAI KONA EV" in layout._nnff.description
  panel.state.CP.brand = "toyota"
  layout._update_state()
  assert layout.items == [layout._vehicle_item, layout._legend_widget, layout._nnff, brand_row]
  assert layout._scroller._items == layout.items
  panel.state.CP.brand = "hyundai"
  layout._update_state()
  assert layout.items == [layout._vehicle_item, layout._legend_widget, layout._nnff]
