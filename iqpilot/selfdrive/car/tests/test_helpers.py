from dataclasses import dataclass
from enum import Enum

import pytest

from iqdbc.car import structs
from iqdbc.car.hyundai.values import HyundaiFlagsIQ

from iqpilot.cereal import custom
from iqpilot.selfdrive.car.helpers import (asdictref, convert_to_capnp, convert_iq_car_control, convert_iq_car_control_compact,
                                          invalidate_stale_longitudinal_comfort)


class SampleEnum(Enum):
  value = 7


@dataclass
class SampleStruct:
  enum: SampleEnum
  values: tuple[int, ...]
  mapping: dict[str, list[SampleEnum]]


def test_convert_to_capnp_normalizes_enum_values():
  params = structs.IQCarParams(flags=HyundaiFlagsIQ.HAS_LFA_BUTTON)
  assert asdictref(params)["flags"] == HyundaiFlagsIQ.HAS_LFA_BUTTON.value
  assert convert_to_capnp(params).flags == HyundaiFlagsIQ.HAS_LFA_BUTTON.value


def test_asdictref_preserves_container_types_and_resolves_enums():
  source = SampleStruct(SampleEnum.value, (1, 2), {"items": [SampleEnum.value]})
  converted = asdictref(source)
  assert converted == {"enum": 7, "values": (1, 2), "mapping": {"items": [7]}}
  assert isinstance(converted["values"], tuple)
  assert isinstance(converted["mapping"]["items"], list)


def test_asdictref_rejects_non_dataclass_values():
  with pytest.raises(TypeError, match="dataclass instances"):
    asdictref(object())


def test_convert_to_capnp_supports_iq_car_state():
  state = convert_to_capnp(structs.IQCarState())
  assert state.speedLimit == 0
  assert not state.accelPressed


def test_convert_to_capnp_rejects_unknown_dataclass():
  with pytest.raises(ValueError, match="Unsupported struct type"):
    convert_to_capnp(SampleStruct(SampleEnum.value, (), {}))


def test_longitudinal_capabilities_round_trip_and_legacy_defaults_select_software():
  legacy = convert_to_capnp(structs.IQCarParams())
  assert legacy.longitudinalActuation.jerkControl == 'software'
  assert legacy.longitudinalActuation.comfortBand == 'none'
  params = structs.IQCarParams()
  params.longitudinalActuation = structs.IQCarParams.LongitudinalActuation(
    jerkControl='nativeAsymmetric', comfortBand='asymmetric', maxJerkUp=5.0, maxJerkDown=5.0)
  encoded = convert_to_capnp(params).to_bytes()
  with custom.IQCarParams.from_bytes(encoded) as decoded:
    assert decoded.longitudinalActuation.jerkControl == 'nativeAsymmetric'
    assert decoded.longitudinalActuation.comfortBand == 'asymmetric'
    assert decoded.longitudinalActuation.maxJerkUp == decoded.longitudinalActuation.maxJerkDown == 5.0


@pytest.mark.parametrize('compact', [None, False, True])
def test_both_iq_control_converters_preserve_the_complete_longitudinal_policy(compact):
  comfort = structs.IQCarControl.LongitudinalComfort(
    valid=True, jerkUp=0.8, jerkDown=5.0, accelDeviationAbove=0.1, accelDeviationBelow=0.02,
    comfortEnabled=True, bypassJerkLimit=True, bypassJerkDownLimit=True)
  wire = custom.IQCarControl.new_message(longitudinalComfort=asdictref(comfort)).to_bytes()
  with custom.IQCarControl.from_bytes(wire) as decoded:
    converted = convert_iq_car_control(decoded) if compact is None else convert_iq_car_control_compact(decoded, include_leads=compact)
  assert asdictref(converted.longitudinalComfort) == pytest.approx(asdictref(comfort))
  # An old producer with no comfort field must still select legacy behavior.
  empty = custom.IQCarControl.new_message()
  converted = convert_iq_car_control(empty) if compact is None else convert_iq_car_control_compact(empty, include_leads=compact)
  assert not converted.longitudinalComfort.valid


@pytest.mark.parametrize('age_ns,valid,expected', [
  (0, True, True), (100_000_000, True, True), (100_000_001, True, False), (0, False, False), (-1, True, False),
])
def test_comfort_freshness_requires_a_valid_message_no_older_than_100ms(age_ns, valid, expected):
  control = structs.IQCarControl(longitudinalComfort=structs.IQCarControl.LongitudinalComfort(valid=True))
  invalidate_stale_longitudinal_comfort(control, 1_000_000_000, 1_000_000_000 - age_ns, valid)
  assert control.longitudinalComfort.valid == expected
  control.longitudinalComfort.valid = False
  invalidate_stale_longitudinal_comfort(control, 1_000_000_000, 1_000_000_000, True)
  assert not control.longitudinalComfort.valid
