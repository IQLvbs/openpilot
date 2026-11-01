import contextlib
import fnmatch
import gc
import os
import shutil
import uuid
import pytest

from iqpilot.common.prefix import OpenpilotPrefix
from iqpilot.system.manager import manager
from iqpilot.system.hardware import TICI, HARDWARE

COLLECT_IGNORE = (
  "iqpilot/selfdrive/test/process_replay/test_processes.py",
  "iqpilot/selfdrive/test/process_replay/test_regen.py",
)
COLLECT_IGNORE_GLOB = (
  "iqpilot/selfdrive/dmonitoringmodeld/*.py",
)


def pytest_ignore_collect(collection_path, config):
  try:
    rel = collection_path.relative_to(config.rootpath).as_posix()
  except ValueError:
    return None
  if rel in COLLECT_IGNORE or any(fnmatch.fnmatch(rel, g) for g in COLLECT_IGNORE_GLOB):
    return True
  return None


def _isolate_session_prefix() -> OpenpilotPrefix | None:
  if os.environ.get("OPENPILOT_PREFIX"):
    return None
  prefix = OpenpilotPrefix(prefix="pytest" + uuid.uuid4().hex[:12], clean_dirs_on_exit=False)
  prefix.__enter__()
  return prefix


# msgq only reclaims dead-reader slots on Linux (/proc check), so shared /tmp/msgq_* files on the macOS runner fill up and ENOBUFS every later subscriber.
SESSION_PREFIX = _isolate_session_prefix()


def pytest_sessionfinish(session, exitstatus):
  if SESSION_PREFIX is None or os.environ.get("PYTEST_XDIST_WORKER"):
    return
  SESSION_PREFIX.clean_dirs()
  shutil.rmtree(SESSION_PREFIX.msgq_path, ignore_errors=True)


def pytest_sessionstart(session):
  # TODO: fix tests and enable test order randomization
  if session.config.pluginmanager.hasplugin('randomly'):
    session.config.option.randomly_reorganize = False


@pytest.hookimpl(hookwrapper=True, trylast=True)
def pytest_runtest_call(item):
  # ensure we run as a hook after capturemanager's
  if item.get_closest_marker("nocapture") is not None:
    capmanager = item.config.pluginmanager.getplugin("capturemanager")
    with capmanager.global_and_fixture_disabled():
      yield
  else:
    yield


@contextlib.contextmanager
def clean_env():
  starting_env = dict(os.environ)
  yield
  os.environ.clear()
  os.environ.update(starting_env)


@pytest.fixture(scope="function", autouse=True)
def openpilot_function_fixture(request):
  with clean_env():
    log_root = os.environ.get("LOG_ROOT")
    worker = os.environ.get("PYTEST_XDIST_WORKER")
    if log_root is not None and worker is not None:
      os.environ["LOG_ROOT"] = os.path.join(log_root, worker)
    # setup a clean environment for each test
    with OpenpilotPrefix(shared_download_cache=request.node.get_closest_marker("shared_download_cache") is not None) as prefix:
      prefix = os.environ["OPENPILOT_PREFIX"]

      yield

      # ensure the test doesn't change the prefix
      assert "OPENPILOT_PREFIX" in os.environ and prefix == os.environ["OPENPILOT_PREFIX"]

    # cleanup any started processes
    manager.manager_cleanup()

    # some processes disable gc for performance, re-enable here
    if not gc.isenabled():
      gc.enable()
      gc.collect()

# If you use setUpClass, the environment variables won't be cleared properly,
# so we need to hook both the function and class pytest fixtures
@pytest.fixture(scope="class", autouse=True)
def openpilot_class_fixture():
  with clean_env():
    yield


@pytest.fixture(scope="function")
def tici_setup_fixture(request, openpilot_function_fixture):
  """Ensure a consistent state for tests on-device. Needs the openpilot function fixture to run first."""
  if 'skip_tici_setup' in request.keywords:
    return
  HARDWARE.initialize_hardware()
  HARDWARE.set_power_save(False)
  os.system("pkill -9 -f hephaestusd")


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
  inventory_tici = config.option.collectonly and os.environ.get("IQPILOT_TEST_INVENTORY") == "1"
  deselected = []
  selected = []
  for item in items:
    if item.get_closest_marker("tici") is not None and not TICI and not inventory_tici:
      deselected.append(item)
      continue
    selected.append(item)

    if item.get_closest_marker("tici") is not None and not inventory_tici:
      item.fixturenames.append('tici_setup_fixture')

    if "xdist_group_class_property" in item.keywords:
      class_property_name = item.get_closest_marker('xdist_group_class_property').args[0]
      class_property_value = getattr(item.cls, class_property_name)
      item.add_marker(pytest.mark.xdist_group(class_property_value))

  if deselected:
    config.hook.pytest_deselected(items=deselected)
    items[:] = selected


@pytest.hookimpl(trylast=True)
def pytest_configure(config):
  if os.environ.get("COVERAGE_PROCESS_START"):
    import coverage
    coverage.process_startup()

  config_line = "xdist_group_class_property: group tests by a property of the class that contains them"
  config.addinivalue_line("markers", config_line)

  config_line = "nocapture: don't capture test output"
  config.addinivalue_line("markers", config_line)

  config_line = "shared_download_cache: share download cache between tests"
  config.addinivalue_line("markers", config_line)
