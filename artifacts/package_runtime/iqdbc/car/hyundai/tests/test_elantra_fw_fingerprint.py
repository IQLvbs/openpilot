from iqdbc.car import structs
from iqdbc.car.fw_versions import match_fw_to_car
from iqdbc.car.hyundai.values import CAR

Ecu = structs.CarParams.Ecu

CN7_ELANTRA_2021_FW = [
  (Ecu.abs, 0x7d1, b'\xf1\x00CN ESC \t 105 \x10\x03 58910-AA800'),
  (Ecu.fwdRadar, 0x7d0, b'\xf1\x00CN7_ SCC FNCUP      1.00 1.01 99110-AA000         '),
  (Ecu.eps, 0x7d4, b'\xf1\x00CN7 MDPS C 1.00 1.07 56310AA050\x00 4CNDC107'),
  (Ecu.fwdCamera, 0x7c4, b'\xf1\x00CN7 MFC  AT USA LHD 1.00 1.09 99210-AA000 221108'),
]


def live_fw(entries):
  return [structs.CarParams.CarFw(ecu=ecu, address=address, subAddress=0, fwVersion=version, brand="hyundai")
          for ecu, address, version in entries]


def test_elantra_2021_with_the_november_2022_camera_is_recognised():
  exact, matches = match_fw_to_car(live_fw(CN7_ELANTRA_2021_FW), "", allow_fuzzy=False)

  assert exact
  assert matches == {CAR.HYUNDAI_ELANTRA_2021}


def test_elantra_2021_is_not_confused_with_the_hybrid_sharing_that_camera_date():
  hybrid_camera = (Ecu.fwdCamera, 0x7c4, b'\xf1\x00CN7HMFC  AT USA LHD 1.00 1.09 99210-AA000 221108')
  _, matches = match_fw_to_car(live_fw(CN7_ELANTRA_2021_FW[:3] + [hybrid_camera]), "")

  assert CAR.HYUNDAI_ELANTRA_2021 not in matches
