from iqdbc.car.can_definitions import CanData
from iqdbc.car.carlog import carlog
from iqdbc.car.isotp_parallel_query import IsoTpParallelQuery

EXT_DIAG_REQUEST = b'\x10\x03'
EXT_DIAG_RESPONSE = b'\x50\x03'

COM_CONT_RESPONSE = b''

CLEAR_DTC_REQUEST = b'\x14\xff\xff\xff'
CLEAR_DTC_RESPONSE = b'\x54'

FUNCTIONAL_ADDR_29BIT = 0x18DB33F1
CLEAR_DTC_ISOTP_SF = bytes([len(CLEAR_DTC_REQUEST)]) + CLEAR_DTC_REQUEST + b'\x00' * (7 - len(CLEAR_DTC_REQUEST))


def clear_all_dtcs(can_send, buses, functional_addr=FUNCTIONAL_ADDR_29BIT):
  # broadcast clears stored DTCs on every ECU on the bus, including safety-relevant modules
  for bus in buses:
    carlog.warning(f"clear all DTCs (functional) on bus {bus} ...")
    can_send([CanData(functional_addr, CLEAR_DTC_ISOTP_SF, bus)])


def clear_ecu_dtcs(can_recv, can_send, bus=0, addr=0x7d0, sub_addr=None, timeout=0.1, retry=10, response_offset: int = 0x8):
  carlog.warning(f"ecu clear DTCs {hex(addr), sub_addr} ...")

  for i in range(retry):
    try:
      query = IsoTpParallelQuery(can_send, can_recv, bus, [(addr, sub_addr)], [EXT_DIAG_REQUEST], [EXT_DIAG_RESPONSE], response_offset)

      for _, _ in query.get_data(timeout).items():
        carlog.warning("clear diagnostic information ...")
        query = IsoTpParallelQuery(can_send, can_recv, bus, [(addr, sub_addr)], [CLEAR_DTC_REQUEST], [CLEAR_DTC_RESPONSE], response_offset)
        query.get_data(timeout)

        carlog.warning("ecu DTCs cleared")
        return True

    except Exception:
      carlog.exception("ecu clear DTCs exception")

    carlog.error(f"ecu clear DTCs retry ({i + 1}) ...")
  carlog.error("ecu clear DTCs failed")
  return False


def disable_ecu(can_recv, can_send, bus=0, addr=0x7d0, sub_addr=None, com_cont_req=b'\x28\x83\x01',
                timeout=0.1, retry=10, response_offset: int = 0x8, clear_dtc=False):
  """Silence an ECU by disabling sending and receiving messages using UDS 0x28.
  The ECU will stay silent as long as openpilot keeps sending Tester Present.

  This is used to disable the radar in some cars. Openpilot will emulate the radar.
  WARNING: THIS DISABLES AEB!"""
  carlog.warning(f"ecu disable {hex(addr), sub_addr} ...")

  for i in range(retry):
    try:
      query = IsoTpParallelQuery(can_send, can_recv, bus, [(addr, sub_addr)], [EXT_DIAG_REQUEST], [EXT_DIAG_RESPONSE], response_offset)

      for _, _ in query.get_data(timeout).items():
        # a DTC clear can take the ECU several hundred ms, so it must complete before comms go down
        if clear_dtc:
          carlog.warning("clear diagnostic information ...")
          query = IsoTpParallelQuery(can_send, can_recv, bus, [(addr, sub_addr)], [CLEAR_DTC_REQUEST], [CLEAR_DTC_RESPONSE], response_offset)
          query.get_data(timeout)

        carlog.warning("communication control disable tx/rx ...")

        query = IsoTpParallelQuery(can_send, can_recv, bus, [(addr, sub_addr)], [com_cont_req], [COM_CONT_RESPONSE], response_offset)
        query.get_data(0)

        carlog.warning("ecu disabled")
        return True

    except Exception:
      carlog.exception("ecu disable exception")

    carlog.error(f"ecu disable retry ({i + 1}) ...")
  carlog.error("ecu disable failed")
  return False
