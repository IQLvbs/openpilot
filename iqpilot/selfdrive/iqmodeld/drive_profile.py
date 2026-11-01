# Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos
from iqpilot._proprietary_loader import ProprietaryModuleMissing, load_private_module

try:
  load_private_module(__name__, "iqpilot_private.context.drive_profile")
except ProprietaryModuleMissing:
  from dataclasses import dataclass

  DRIVE_PROFILE_PARAM_VALUES = ("neutral",)
  DRIVE_PROFILE_LABELS = ("Neutral",)

  @dataclass
  class DriveProfileState:
    drive_profile: int = 0
    arrival_urgency: float = 0.0

  def drive_profile_name(params) -> str:
    return "neutral"

  def profile_label(profile: str) -> str:
    return "Neutral"

  def arrival_time_choices(now_unix_s=None, interval_s: int = 900, count: int = 96):
    return []

  def desired_arrival_label(params, now_unix_s=None) -> str:
    return ""

  def active_model_declares_drive_profile(params) -> bool:
    return False

  def drive_profile_requests_speed_limit_compliance(input_shapes, params) -> bool:
    return False

  def compose_drive_profile_inputs(input_shapes, params, sub, frame_timestamp_eof_ns: int, vehicle_speed: float,
                                   composed_inputs=None, now_unix_s=None):
    import numpy as np
    target = input_shapes.get("drive_profile") if hasattr(input_shapes, "get") else None
    inputs = {} if target is None else {"drive_profile": np.zeros(getattr(target, "shape", target), dtype=np.float32)}
    return inputs, DriveProfileState()

  def populate_drive_profile_message(message, state) -> None:
    message.driveProfile = state.drive_profile
    message.arrivalUrgency = state.arrival_urgency
