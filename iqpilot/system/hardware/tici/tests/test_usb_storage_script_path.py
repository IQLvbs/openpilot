import importlib.util
import shutil
from pathlib import Path

SOURCE = Path(__file__).parents[1] / "usb_storage.py"


def load_from(directory: Path):
  target = directory / "usb_storage.py"
  shutil.copy(SOURCE, target)
  spec = importlib.util.spec_from_file_location(f"usb_storage_{directory.name}", target)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def test_script_beside_the_module_is_used(tmp_path):
  (tmp_path / "set_usb_storage.sh").write_text("")
  assert load_from(tmp_path).SCRIPT_PATH == str(tmp_path / "set_usb_storage.sh")


def test_bundled_module_without_the_script_uses_the_install_tree(tmp_path, monkeypatch):
  monkeypatch.setenv("BASEDIR", "/data/iqpilot")
  assert load_from(tmp_path).SCRIPT_PATH == "/data/iqpilot/iqpilot/system/hardware/tici/set_usb_storage.sh"
