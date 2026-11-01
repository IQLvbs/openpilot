from types import SimpleNamespace

import pytest

from iqpilot.system.manager.process_config import managed_processes, navigation


@pytest.mark.parametrize("started", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("route_request", [None, "NavigationDestination", "AthenaNavigationRoute"])
def test_navigation_process_runs_for_offroad_requests(started, enabled, route_request):
  params = SimpleNamespace(get_bool=lambda key: enabled, get=lambda key: {} if key != route_request else {"destination": True})
  assert navigation(started, params, None) == (enabled and (started or route_request is not None))
  assert managed_processes["navd"].should_run is navigation
