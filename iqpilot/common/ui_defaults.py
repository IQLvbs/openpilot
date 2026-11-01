"""Startup policy for settings that are no longer user configurable."""


def apply_ui_defaults(params) -> None:
  for key, value in {
    "IQAutoUnits": True,
    "IQDynamicMode": False,
    "IQExpandedStatus": True,
    "IQSteerEffortArc": True,
    "IQBlinkerIndicators": True,
    "IQAccelMeter": False,
    "IQEdgeGuard": False,
    "LaneChangeContinuous": False,
    "AolUnifiedEngagementMode": True,
    "IQE2ESetSpeedUseCurrent": True,
    "OnScreenNavigation": True,
    "OnlineOSMaps": True,
    "OfflineOSMaps": True,
    "NavigateOnIQPilot": True,
    "NavExitLaneChange": True,
    "NavOnlineTargets": True,
    "NavOfflineFallback": True,
    "NavPreferOfflineSources": False,
    "SLCFallbackPreviousSpeedLimit": True,
    "SLCOnlineFiller": True,
  }.items():
    if params.get_bool(key) != value:
      params.put_bool(key, value)
  bundle = params.get("CarPlatformBundle") or {}
  if bundle.get("brand") not in {"rivian", "tesla"}:
    params.put_bool("AolMainCruiseAllowed", True)
  for key, value in {
    "OnroadScreenOffBrightness": 0, "IQLaneTurnValue": 20.0,
    "IQE2ESetSpeedMode": 1, "SLCPolicy": 1,
  }.items():
    if params.get(key, return_default=True) != value:
      params.put(key, value)
  timer = int(params.get("IQLaneChangeTimer", return_default=True))
  if timer not in (0, 1):
    params.put("IQLaneChangeTimer", int(timer > 0))
  if params.get("IQSpeedAssistMode", return_default=True) not in (1, 3):
    params.put("IQSpeedAssistMode", 1)
