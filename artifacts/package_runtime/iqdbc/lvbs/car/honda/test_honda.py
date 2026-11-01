"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos
"""
from parameterized import parameterized
from types import SimpleNamespace

from iqdbc.can import CANParser
from iqdbc.car import Bus, gen_empty_fingerprint
from iqdbc.car.structs import CarParams, CarState, IQCarState as IQCarStateStruct
from iqdbc.car.car_helpers import interfaces
from iqdbc.car.honda.values import CAR
from iqdbc.lvbs.car.honda.iq_carstate import IQCarState

CarFw = CarParams.CarFw


class TestHondaEpsMod:

  @parameterized.expand([(CAR.HONDA_CIVIC, b'39990-TBA,A030\x00\x00'), (CAR.HONDA_CIVIC, b'39990-TBA-A030\x00\x00'),
                         (CAR.HONDA_CLARITY, b'39990-TRW-A020\x00\x00'), (CAR.HONDA_CLARITY, b'39990,TRW,A020\x00\x00')])
  def test_eps_mod_fingerprint(self, car_name, fw):
    fingerprint = gen_empty_fingerprint()
    car_fw = [CarFw(ecu="eps", fwVersion=fw)]

    CarInterface = interfaces[car_name]
    CP = CarInterface.get_params(car_name, fingerprint, car_fw, False, False, False)
    _ = CarInterface.get_params_iq(CP, car_name, fingerprint, car_fw, False, False, False)

    assert not CP.dashcamOnly


class TestHondaGasInterceptor:
  def test_gas_sensor_registration_and_threshold(self):
    CP = SimpleNamespace()
    CP_IQ = SimpleNamespace(enableGasInterceptor=True, flags=0)
    parser = CANParser("acura_ilx_2016_can_generated", [], 0)
    state = IQCarState(CP, CP_IQ)
    ret = CarState()
    ret_iq = IQCarStateStruct()

    state.update(ret, ret_iq, {Bus.pt: parser, Bus.cam: parser})
    assert "GAS_SENSOR" in parser.vl
    assert not ret.gasPressed

    parser.vl["GAS_SENSOR"]["INTERCEPTOR_GAS"] = 493
    parser.vl["GAS_SENSOR"]["INTERCEPTOR_GAS2"] = 493
    state.update(ret, ret_iq, {Bus.pt: parser, Bus.cam: parser})
    assert ret.gasPressed
