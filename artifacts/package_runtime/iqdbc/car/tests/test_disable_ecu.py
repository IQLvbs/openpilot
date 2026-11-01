from iqdbc.car.can_definitions import CanData
from iqdbc.car.disable_ecu import (CLEAR_DTC_ISOTP_SF, CLEAR_DTC_REQUEST, EXT_DIAG_REQUEST,
                                   FUNCTIONAL_ADDR_29BIT, clear_all_dtcs, clear_ecu_dtcs, disable_ecu)

RADAR_ADDR = 0x18DAB0F1
COM_CONT_REQUEST = b'\x28\x83\x03'


class QueryRecorder:
  def __init__(self):
    self.requests = []

  def make_fake_query(self):
    recorder = self

    class FakeIsoTpParallelQuery:
      def __init__(self, can_send, can_recv, bus, addrs, requests, responses, response_offset=0x8):
        self.bus = bus
        self.addrs = addrs
        self.request = requests[0]
        recorder.requests.append((bus, addrs[0][0], requests[0]))

      def get_data(self, timeout):
        return {(self.addrs[0][0], None): b''}

    return FakeIsoTpParallelQuery


def test_clear_all_dtcs_broadcasts_single_frame():
  sent = []
  clear_all_dtcs(lambda msgs: sent.extend(msgs), [0, 2])

  assert sent == [
    CanData(FUNCTIONAL_ADDR_29BIT, CLEAR_DTC_ISOTP_SF, 0),
    CanData(FUNCTIONAL_ADDR_29BIT, CLEAR_DTC_ISOTP_SF, 2),
  ]


def test_clear_dtc_isotp_framing():
  assert len(CLEAR_DTC_ISOTP_SF) == 8
  assert CLEAR_DTC_ISOTP_SF[0] == len(CLEAR_DTC_REQUEST)
  assert CLEAR_DTC_ISOTP_SF[1:1 + len(CLEAR_DTC_REQUEST)] == CLEAR_DTC_REQUEST
  assert CLEAR_DTC_REQUEST == b'\x14\xff\xff\xff'


def test_clear_ecu_dtcs_sequence(mocker):
  recorder = QueryRecorder()
  mocker.patch("iqdbc.car.disable_ecu.IsoTpParallelQuery", recorder.make_fake_query())

  assert clear_ecu_dtcs(None, None, bus=0, addr=RADAR_ADDR)
  assert recorder.requests == [
    (0, RADAR_ADDR, EXT_DIAG_REQUEST),
    (0, RADAR_ADDR, CLEAR_DTC_REQUEST),
  ]


def test_disable_ecu_sequence(mocker):
  recorder = QueryRecorder()
  mocker.patch("iqdbc.car.disable_ecu.IsoTpParallelQuery", recorder.make_fake_query())

  assert disable_ecu(None, None, bus=1, addr=RADAR_ADDR, com_cont_req=COM_CONT_REQUEST)
  assert recorder.requests == [
    (1, RADAR_ADDR, EXT_DIAG_REQUEST),
    (1, RADAR_ADDR, COM_CONT_REQUEST),
  ]


def test_disable_ecu_clears_dtcs_before_comm_control(mocker):
  recorder = QueryRecorder()
  mocker.patch("iqdbc.car.disable_ecu.IsoTpParallelQuery", recorder.make_fake_query())

  assert disable_ecu(None, None, bus=1, addr=RADAR_ADDR, com_cont_req=COM_CONT_REQUEST, clear_dtc=True)
  assert recorder.requests == [
    (1, RADAR_ADDR, EXT_DIAG_REQUEST),
    (1, RADAR_ADDR, CLEAR_DTC_REQUEST),
    (1, RADAR_ADDR, COM_CONT_REQUEST),
  ]


def test_disable_ecu_retries_then_fails(mocker):
  attempts = []

  class NoResponseQuery:
    def __init__(self, can_send, can_recv, bus, addrs, requests, responses, response_offset=0x8):
      attempts.append(requests[0])

    def get_data(self, timeout):
      return {}

  mocker.patch("iqdbc.car.disable_ecu.IsoTpParallelQuery", NoResponseQuery)

  assert not disable_ecu(None, None, addr=RADAR_ADDR, retry=3)
  assert attempts == [EXT_DIAG_REQUEST] * 3
