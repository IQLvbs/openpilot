# Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
import re
from pathlib import Path

ROOT = Path(__file__).parents[3]


def test_param_is_off_by_default():
  keys = (ROOT / "common" / "params_keys.h").read_text()
  assert re.search(r'\{"DashcamMode", \{PERSISTENT, BOOL, "0"\}\}', keys)


def test_card_forces_passive_when_dashcam_mode_set():
  card = (ROOT / "selfdrive" / "car" / "card.py").read_text()
  assert 'and not self.params.get_bool("DashcamMode")' in card
  assert "self.CP.passive = not controller_available" in card


def test_toggle_present_in_both_uis():
  big = (ROOT / "selfdrive" / "ui" / "layouts" / "settings" / "toggles.py").read_text()
  mici = (ROOT / "selfdrive" / "ui" / "mici" / "layouts" / "settings" / "dashcam.py").read_text()
  assert '"DashcamMode"' in big
  assert '"DashcamMode"' in mici


def test_tesla_stock_autosteer_param_survives_onroad_cycle():
  keys = (ROOT / "common" / "params_keys.h").read_text()
  match = re.search(r'\{"TeslaStockAutosteerActive", \{([^}]*)\}\}', keys)
  assert match
  assert "CLEAR_ON_IGNITION_ON" in match.group(1)
  assert "CLEAR_ON_OFFROAD_TRANSITION" not in match.group(1)


def test_card_yields_to_stock_autosteer_on_tesla():
  card = (ROOT / "selfdrive" / "car" / "card.py").read_text()
  assert 'not self.stock_autosteer_standby' in card
  assert 'self.params.put_bool("TeslaStockAutosteerActive", True)' in card
  assert "self._yield_to_stock_autosteer(CS)" in card


def test_selfdrived_shows_autosteer_active_when_standing_by():
  selfdrived = (ROOT / "selfdrive" / "selfdrived" / "selfdrived.py").read_text()
  assert 'self.params.get_bool("TeslaStockAutosteerActive")' in selfdrived
