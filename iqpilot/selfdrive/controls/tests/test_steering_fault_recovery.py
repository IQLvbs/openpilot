"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos
"""
from iqpilot.selfdrive.controls.steering_fault_recovery import STEER_FAULT_RECOVERY_FRAMES, SteeringFaultRecovery


def test_steering_fault_recovery_starts_ready():
  recovery = SteeringFaultRecovery()
  assert recovery.update(False, False)


def test_temporary_fault_requires_continuous_clear_interval():
  recovery = SteeringFaultRecovery()
  assert not recovery.update(True, False)
  for _ in range(STEER_FAULT_RECOVERY_FRAMES - 1):
    assert not recovery.update(False, False)
  assert recovery.update(False, False)


def test_repeated_fault_restarts_recovery_interval():
  recovery = SteeringFaultRecovery()
  assert not recovery.update(False, True)
  for _ in range(STEER_FAULT_RECOVERY_FRAMES - 1):
    assert not recovery.update(False, False)
  assert not recovery.update(True, False)
  for _ in range(STEER_FAULT_RECOVERY_FRAMES - 1):
    assert not recovery.update(False, False)
  assert recovery.update(False, False)
