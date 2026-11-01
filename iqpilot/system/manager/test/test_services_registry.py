import os
import re

from iqpilot.cereal.services import REGISTRY_TAG_PREFIX, SERVICE_LIST, build_header, registry_hash, registry_tag
from iqpilot.system.manager.build import REGISTRY_ARTIFACTS, purge_registry_artifacts, stale_registry_artifacts


def test_registry_tag_is_derived_from_the_service_list():
  assert re.fullmatch(r"[0-9a-f]{16}", registry_hash())
  assert registry_tag() == REGISTRY_TAG_PREFIX + registry_hash()
  assert "extrinsicsCalibration" in SERVICE_LIST


def test_generated_header_embeds_the_tag():
  header = build_header()
  assert f'static const char SERVICES_REGISTRY_TAG[] = "{registry_tag()}";' in header
  assert '{ "extrinsicsCalibration", {"extrinsicsCalibration"' in header


def _write(tmp_path, rel, payload):
  path = tmp_path / rel
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_bytes(payload)
  return path


def test_stale_detection_flags_only_binaries_carrying_an_old_tag(tmp_path):
  current = registry_tag().encode()
  old = (REGISTRY_TAG_PREFIX + "0" * 16).encode()
  _write(tmp_path, "iqpilot/selfdrive/iqlocd/iqlocd", b"\x7fELF" + old)
  _write(tmp_path, "iqpilot/system/loggerd/loggerd", b"\x7fELF" + current)
  _write(tmp_path, "iqpilot/system/camerad/camerad", b"\x7fELF no registry linked")
  assert stale_registry_artifacts(str(tmp_path)) == ["iqpilot/selfdrive/iqlocd/iqlocd"]


def test_purge_removes_the_stale_binary_and_the_messaging_table(tmp_path):
  for rel in REGISTRY_ARTIFACTS:
    _write(tmp_path, rel, b"x")
  purge_registry_artifacts(["iqpilot/selfdrive/iqlocd/iqlocd"], str(tmp_path))
  remaining = [rel for rel in REGISTRY_ARTIFACTS if os.path.isfile(tmp_path / rel)]
  assert "iqpilot/selfdrive/iqlocd/iqlocd" not in remaining
  assert "iqpilot/cereal/services.h" not in remaining
  assert "iqpilot/cereal/messaging/socketmaster.o" not in remaining
  assert "iqpilot/cereal/libsocketmaster.a" not in remaining
  assert "iqpilot/system/loggerd/loggerd" in remaining


def test_generated_header_declares_the_stamp_for_the_compile_fallback():
  assert "#define SERVICES_REGISTRY_STAMPED 1" in build_header()


def test_pre_build_purge_drops_an_unstamped_or_old_header(tmp_path):
  from iqpilot.system.manager.build import purge_stale_registry_header
  _write(tmp_path, "iqpilot/cereal/services.h", b"static std::map<std::string, service> services = {};\n")
  _write(tmp_path, "iqpilot/cereal/messaging/socketmaster.o", b"o")
  _write(tmp_path, "iqpilot/cereal/libsocketmaster.a", b"a")
  assert purge_stale_registry_header(str(tmp_path))
  assert not (tmp_path / "iqpilot/cereal/services.h").exists()
  assert not (tmp_path / "iqpilot/cereal/messaging/socketmaster.o").exists()
  assert not (tmp_path / "iqpilot/cereal/libsocketmaster.a").exists()

  _write(tmp_path, "iqpilot/cereal/services.h", build_header().encode())
  assert not purge_stale_registry_header(str(tmp_path))
  assert (tmp_path / "iqpilot/cereal/services.h").exists()
  assert not purge_stale_registry_header(str(tmp_path / "nowhere"))
