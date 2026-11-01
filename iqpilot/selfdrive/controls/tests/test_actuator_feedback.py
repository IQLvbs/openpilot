# Copyright (c) 2026 IQ.Lvbs. All rights reserved.
import os
from pathlib import Path
from types import SimpleNamespace

import capnp
import pytest

from iqpilot.cereal import car
from iqpilot.selfdrive.controls.lib.actuator_feedback import (
  AppliedActuators, CAR_CONTROL_TIMEOUT_NS, PAIRING_FIELDS, car_output_supports_pairing, steering_feedback_from_output,
  steering_limited_from_feedback,
)


REQUEST_NS = 1_000_000_000
APPLY_NS = REQUEST_NS + 10_000_000
PUBLISH_NS = APPLY_NS + 10_000_000
NOW_NS = PUBLISH_NS + 10_000_000


def make_feedback(torque=0.2, applied=0.2, angle=5.0, applied_angle=5.0):
  command = car.CarControl.new_message(latActive=True, actuators={"torque": torque, "steeringAngleDeg": angle})
  output = car.CarOutput.new_message(actuatorsOutput={"torque": applied, "steeringAngleDeg": applied_angle})
  AppliedActuators.capture(command, REQUEST_NS, APPLY_NS).write_to(output)
  return output


def limited(output, **kwargs):
  args = dict(feedback_valid=True, output_mono_time=PUBLISH_NS, now_mono_time=NOW_NS,
              angle_control=False, angle_threshold=2.5)
  args.update(kwargs)
  return steering_limited_from_feedback(output, **args)


def feedback_detail(output, **kwargs):
  args = dict(feedback_valid=True, output_mono_time=PUBLISH_NS, now_mono_time=NOW_NS,
              angle_control=False, angle_threshold=2.5)
  args.update(kwargs)
  return steering_feedback_from_output(output, **args)


def test_input_snapshot_survives_mutating_the_command():
  command = car.CarControl.new_message(latActive=True, actuators={"torque": 0.2, "steeringAngleDeg": 3.0})
  feedback = AppliedActuators.capture(command, REQUEST_NS, APPLY_NS)
  command.actuators.torque = -0.6
  command.actuators.steeringAngleDeg = -9.0
  command.latActive = False
  output = car.CarOutput.new_message()
  feedback.write_to(output)
  assert output.requestedTorque == pytest.approx(0.2)
  assert output.requestedSteeringAngleDeg == pytest.approx(3.0)
  assert output.latActive


@pytest.mark.parametrize("torque,applied,expected", [(0.2, 0.2, False), (0.2, 0.195, False),
                                                  (0.2, 0.18, True), (-0.2, -0.18, True)])
def test_same_request_torque_limit(torque, applied, expected):
  assert limited(make_feedback(torque, applied)) is expected
  detail = feedback_detail(make_feedback(torque, applied))
  assert detail.limited is expected
  assert detail.torque_difference == pytest.approx(torque - applied, abs=1e-7)
  assert detail.applied_torque == pytest.approx(applied)


@pytest.mark.parametrize("applied,expected", [(5.0, False), (2.5, False), (2.49, True)])
def test_existing_angle_threshold(applied, expected):
  assert limited(make_feedback(applied_angle=applied), angle_control=True) is expected
  detail = feedback_detail(make_feedback(applied_angle=applied), angle_control=True)
  assert detail.limited is expected
  assert detail.torque_difference is None
  assert detail.applied_torque is None


@pytest.mark.parametrize("field,value", [("carControlMonoTime", 0), ("actuationMonoTime", 0), ("latActive", False),
                                        ("carControlMonoTime", APPLY_NS + 1), ("actuationMonoTime", PUBLISH_NS + 1)])
def test_missing_fallback_inactive_or_unordered_origin_freezes(field, value):
  output = make_feedback()
  setattr(output, field, value)
  assert limited(output)
  assert feedback_detail(output).torque_difference is None
  assert feedback_detail(output).applied_torque is None


def test_invalid_envelope_and_future_publication_freeze():
  assert limited(make_feedback(), feedback_valid=False)
  assert limited(make_feedback(), now_mono_time=PUBLISH_NS - 1)
  assert feedback_detail(make_feedback(), feedback_valid=False).torque_difference is None
  assert feedback_detail(make_feedback(), feedback_valid=False).applied_torque is None
  assert feedback_detail(make_feedback(), now_mono_time=PUBLISH_NS - 1).torque_difference is None


def test_freshness_uses_existing_watchdog_and_does_not_refresh_an_old_request():
  output = make_feedback()
  deadline = REQUEST_NS + CAR_CONTROL_TIMEOUT_NS
  assert not limited(output, now_mono_time=deadline)
  assert limited(output, now_mono_time=deadline + 1)
  assert feedback_detail(output, now_mono_time=deadline + 1).torque_difference is None
  assert feedback_detail(output, now_mono_time=deadline + 1).applied_torque is None
  output.actuationMonoTime = deadline + 1
  assert limited(output, output_mono_time=deadline + 2, now_mono_time=deadline + 3)


@pytest.mark.parametrize("input_field,output_field,angle_control", [("requestedTorque", "torque", False),
                                                                 ("requestedSteeringAngleDeg", "steeringAngleDeg", True)])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_requested_or_applied_value_freezes(input_field, output_field, angle_control, value):
  output = make_feedback()
  setattr(output, input_field, value)
  assert limited(output, angle_control=angle_control)
  assert feedback_detail(output, angle_control=angle_control).torque_difference is None
  assert feedback_detail(output, angle_control=angle_control).applied_torque is None
  output = make_feedback()
  setattr(output.actuatorsOutput, output_field, value)
  assert limited(output, angle_control=angle_control)
  assert feedback_detail(output, angle_control=angle_control).torque_difference is None
  assert feedback_detail(output, angle_control=angle_control).applied_torque is None


def test_capnp_round_trip_and_legacy_wire_compatibility(tmp_path):
  import iqdbc
  legacy_path = tmp_path / "legacy_car_output.capnp"
  legacy_path.write_text('''@0xef024e93fa827d4d;
using Car = import "/car.capnp";
struct LegacyCarOutput { actuatorsOutput @0 :Car.CarControl.Actuators; }
''')
  car_import_path = os.path.join(os.path.realpath(Path(iqdbc.__file__).parent), "car")
  legacy = capnp.load(str(legacy_path), imports=[car_import_path])
  old_output = legacy.LegacyCarOutput.new_message(actuatorsOutput={"torque": 0.2})
  with car.CarOutput.from_bytes(old_output.to_bytes()) as decoded:
    assert decoded.actuatorsOutput.torque == pytest.approx(0.2)
    assert decoded.carControlMonoTime == 0
    assert limited(decoded)

  payload = make_feedback().to_bytes()
  with car.CarOutput.from_bytes(payload) as decoded:
    assert decoded.carControlMonoTime == REQUEST_NS
    assert decoded.actuationMonoTime == APPLY_NS
    assert not limited(decoded)
  with legacy.LegacyCarOutput.from_bytes(payload) as decoded:
    assert decoded.actuatorsOutput.torque == pytest.approx(0.2)


def test_pairing_requires_every_field_in_the_loaded_schema():
  assert car_output_supports_pairing(SimpleNamespace(fieldnames=["actuatorsOutput", *PAIRING_FIELDS]))
  assert not car_output_supports_pairing(SimpleNamespace(fieldnames=["actuatorsOutput"]))
  for missing in PAIRING_FIELDS:
    fields = ["actuatorsOutput", *(name for name in PAIRING_FIELDS if name != missing)]
    assert not car_output_supports_pairing(SimpleNamespace(fieldnames=fields))


def test_pairing_support_follows_the_real_car_output_schema():
  assert car_output_supports_pairing() == (set(PAIRING_FIELDS) <= set(car.CarOutput.schema.fieldnames))
