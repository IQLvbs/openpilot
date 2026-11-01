import os
import pytest
import hypothesis.strategies as st
from hypothesis import Phase, given, settings
from parameterized import parameterized
from types import SimpleNamespace

from iqpilot.cereal import car, custom
from iqdbc.car import DT_CTRL
from iqdbc.car.structs import CarParams
from iqdbc.car.car_helpers import interfaces
from iqdbc.car.tests.test_car_interfaces import get_fuzzy_car_interface, get_fuzzy_strategy
from iqdbc.car.mock.values import CAR as MOCK
from iqdbc.car.values import PLATFORMS
from iqpilot.selfdrive.car.card import run_optional_pre_init
from iqpilot.selfdrive.car.helpers import convert_iq_car_control, convert_iq_car_control_compact
from iqpilot.selfdrive.controls.lib.latcontrol_angle import LatControlAngle
from iqpilot.selfdrive.controls.lib.latcontrol_pid import LatControlPID
from iqpilot.selfdrive.controls.lib.latcontrol_torque import LatControlTorque
from iqpilot.selfdrive.controls.lib.longcontrol import LongControl
from iqpilot.selfdrive.test.fuzzy_generation import FuzzyGenerator

from iqpilot.selfdrive.car import interfaces as iqpilot_interfaces

MAX_EXAMPLES = int(os.environ.get('MAX_EXAMPLES', '60'))
VW_MLB_LONG_APPLY_EXCLUDE = {"AUDI_Q5_MK1", "PORSCHE_MACAN_MK1"}
SIGNED_RUNTIME_BRANDS = {"tesla", "volkswagen"}
SIGNED_RUNTIME_PLATFORMS = tuple(car_name for car_name in sorted(PLATFORMS)
                                 if interfaces[car_name].__module__.split(".")[-2] in SIGNED_RUNTIME_BRANDS)
PUBLIC_INTERFACE_PLATFORMS = tuple(car_name for car_name in sorted(PLATFORMS)
                                   if car_name not in SIGNED_RUNTIME_PLATFORMS)


def exercise_car_interface(car_name, draw):
  car_interface = get_fuzzy_car_interface(car_name, draw)
  if car_name in VW_MLB_LONG_APPLY_EXCLUDE:
    car_interface.CP.openpilotLongitudinalControl = False
  car_params = car_interface.CP.as_reader()
  car_params_iq = car_interface.CP_IQ
  iqpilot_interfaces.apply_iq_car_config(car_interface)

  cc_msg = FuzzyGenerator.get_random_msg(draw, car.CarControl, real_floats=True)
  cc_sp_msg = FuzzyGenerator.get_random_msg(draw, custom.IQCarControl, real_floats=True)
  now_nanos = 0
  CC = car.CarControl.new_message(**cc_msg).as_reader()
  CC_IQ = convert_iq_car_control(custom.IQCarControl.new_message(**cc_sp_msg).as_reader())
  for _ in range(10):
    car_interface.update([])
    car_interface.apply(CC, CC_IQ, now_nanos)
    now_nanos += DT_CTRL * 1e9

  CC = car.CarControl.new_message(**cc_msg)
  CC.enabled = True
  CC.latActive = True
  CC.longActive = True
  CC = CC.as_reader()
  for _ in range(10):
    car_interface.update([])
    car_interface.apply(CC, CC_IQ, now_nanos)
    now_nanos += DT_CTRL * 1e9

  LongControl(car_params, car_params_iq)
  if car_params.steerControlType == CarParams.SteerControlType.angle:
    LatControlAngle(car_params, car_params_iq, car_interface, DT_CTRL)
  elif car_params.lateralTuning.which() == 'pid':
    LatControlPID(car_params, car_params_iq, car_interface, DT_CTRL)
  elif car_params.lateralTuning.which() == 'torque':
    LatControlTorque(car_params, car_params_iq, car_interface, DT_CTRL)


@pytest.mark.car_ports
class TestCarInterfaces:
  # FIXME: Due to the lists used in carParams, Phase.target is very slow and will cause
  #  many generated examples to overrun when max_examples > ~20, don't use it
  @parameterized.expand([(car,) for car in PUBLIC_INTERFACE_PLATFORMS] + [MOCK.MOCK])
  @settings(max_examples=MAX_EXAMPLES, deadline=None,
            phases=(Phase.reuse, Phase.generate, Phase.shrink))
  @given(data=st.data())
  def test_car_interfaces(self, car_name, data):
    exercise_car_interface(car_name, data.draw)

  @parameterized.expand([(car,) for car in SIGNED_RUNTIME_PLATFORMS])
  @settings(max_examples=MAX_EXAMPLES, deadline=None,
            phases=(Phase.reuse, Phase.generate, Phase.shrink))
  @given(data=st.data())
  def test_public_car_params(self, car_name, data):
    params = data.draw(get_fuzzy_strategy())
    params['fingerprints'] |= {key + 1: params['fingerprints'][0] for key in range(6)}
    car_interface = interfaces[car_name]
    car_params = car_interface.get_params(car_name, params['fingerprints'], params['car_fw'],
                                           alpha_long=params['alpha_long'], is_release=False, docs=False)
    car_params_iq = car_interface.get_params_iq(car_params, car_name, params['fingerprints'], params['car_fw'],
                                                 alpha_long=params['alpha_long'], is_release_iq=False, docs=False)
    assert car_params.mass > 1
    assert car_params.wheelbase > 0
    assert car_params.maxLateralAccel > 0
    assert car_params_iq is not None


def test_convert_iq_car_control_compact_skips_leads():
  msg = custom.IQCarControl.new_message()
  msg.aol.enabled = True
  msg.aol.active = True
  msg.aol.available = True
  param = msg.init("params", 1)
  param[0].key = "enhancedStockLongitudinalControl.setSpeedKph"
  param[0].type = "float"
  param[0].value = b"42.0"
  msg.leadOne.dRel = 42.0
  msg.leadOne.status = True
  msg.leadTwo.dRel = 84.0
  msg.leadTwo.status = True

  compact = convert_iq_car_control_compact(msg.as_reader(), include_leads=False)
  assert compact.aol.enabled
  assert compact.aol.active
  assert compact.aol.available
  assert len(compact.params) == 1
  assert compact.params[0].key == "enhancedStockLongitudinalControl.setSpeedKph"
  assert compact.params[0].type == "float"
  assert compact.params[0].value == b"42.0"
  assert compact.leadOne.dRel == 0.0
  assert not compact.leadOne.status
  assert compact.leadTwo.dRel == 0.0
  assert not compact.leadTwo.status

  full = convert_iq_car_control_compact(msg.as_reader(), include_leads=True)
  assert full.leadOne.dRel == 42.0
  assert full.leadOne.status
  assert full.leadTwo.dRel == 84.0
  assert full.leadTwo.status


def test_run_optional_pre_init_skips_missing_hook():
  ci = SimpleNamespace()
  run_optional_pre_init(ci, car.CarParams(), custom.IQCarParams(), (lambda wait_for_one=False: [], lambda msgs: None))


def test_run_optional_pre_init_calls_hook():
  called = []

  class CI:
    @staticmethod
    def pre_init(CP, CP_IQ, can_recv, can_send):
      called.append((CP, CP_IQ, can_recv, can_send))

  can_callbacks = (lambda wait_for_one=False: [], lambda msgs: None)
  cp = car.CarParams()
  cp_iq = custom.IQCarParams()
  run_optional_pre_init(CI(), cp, cp_iq, can_callbacks)

  assert called == [(cp, cp_iq, *can_callbacks)]
