"""The MQB/PQ cluster's lead marker and gap notches, as encoded on the wire."""
import pytest

from iqdbc.can import CANParser
from iqdbc.car import structs
from iqdbc.car.common.conversions import Conversions as CV
from iqdbc.car.volkswagen.carcontroller import (
  LEAD_HEADWAY_MIN_SPEED, LEAD_INDEX_RANGE_ANALOG, LEAD_INDEX_RANGE_DIGITAL, lead_headway_index,
)
from iqdbc.car.volkswagen.interface import CarInterface
from iqdbc.car.volkswagen.values import CAR

SECOND_NOTCH = 1.3  # the stock gap settings run 1.0, 1.3, 1.8, 2.4, 3.6 s
FOLLOW_TIMES = {1: 1.25, 2: 1.45, 3: 1.75}  # leadDistanceBars: T_FOLLOW, aggressive to relaxed
STOP_DISTANCE = 3.0  # the planner follows at T_FOLLOW * v + STOP_DISTANCE


@pytest.mark.parametrize("digital", [False, True])
@pytest.mark.parametrize("speed", [0.0, 10.0, 30.0])
def test_grows_with_distance_and_saturates_at_the_ends(digital, speed):
  codes = [lead_headway_index(d / 2, speed, digital) for d in range(1, 600)]
  closest, farthest = LEAD_INDEX_RANGE_DIGITAL if digital else LEAD_INDEX_RANGE_ANALOG
  assert codes == sorted(codes)
  assert (codes[0], codes[-1]) == (closest, farthest)


def test_reads_headway_not_distance():
  assert lead_headway_index(1.8 * 10.0, 10.0, False) == lead_headway_index(1.8 * 30.0, 30.0, False)
  assert lead_headway_index(30.0, 10.0, False) > lead_headway_index(30.0, 25.0, False)


def test_stopped_reads_distance_alone():
  assert lead_headway_index(5.0, 0.0, False) == lead_headway_index(5.0, LEAD_HEADWAY_MIN_SPEED, False)


def make_car(candidate):
  fingerprint = {bus: {} for bus in range(8)}
  cp = CarInterface.get_params(candidate, fingerprint, [], alpha_long=True, is_release=False, docs=False)
  cp_iq = CarInterface.get_params_iq(cp, candidate, fingerprint, [], alpha_long=True, is_release_iq=False, docs=False)
  car = CarInterface(cp, cp_iq)
  car.update([(0, [])])
  car.CS.out = structs.CarState()
  car.CS.out.cruiseState.available = True
  return car


def send_hud(car, *, lead_distance, speed, bars=2, lead_visible=True):
  car.CS.out.vEgo = speed
  cc = structs.CarControl()
  cc.enabled = cc.longActive = True
  cc.hudControl.leadVisible = lead_visible
  cc.hudControl.leadDistance = lead_distance
  cc.hudControl.leadDistanceBars = bars
  cc.hudControl.leadFollowTime = FOLLOW_TIMES[bars]
  car.CC.frame += -car.CC.frame % car.CC.CCP.ACC_HUD_STEP
  now_nanos = (car.CC.frame + 1) * 10_000_000
  _, messages = car.apply(cc.as_reader(), structs.IQCarControl(), now_nanos)
  return now_nanos, messages


def mqb_hud(car, **kwargs):
  now_nanos, messages = send_hud(car, **kwargs)
  parser = CANParser("vw_mqb", [("ACC_02", 0)], car.CC.CAN.pt)
  parser.update([(now_nanos, messages)])
  return parser.vl["ACC_02"]["ACC_Gesetzte_Zeitluecke"], parser.vl["ACC_02"]["ACC_Abstandsindex"]


def pq_hud(car, **kwargs):
  now_nanos, messages = send_hud(car, **kwargs)
  parser = CANParser("vw_pq", [("ACC_GRA_Anzeige", 0)], car.CC.CAN.pt)
  parser.update([(now_nanos, messages)])
  return parser.vl["ACC_GRA_Anzeige"]["ACA_Zeitluecke"], parser.vl["ACC_GRA_Anzeige"]["ACA_gemZeitl"]


PLATFORMS = [
  pytest.param(CAR.VOLKSWAGEN_GOLF_MK7, mqb_hud, False, id="mqb_analog"),
  pytest.param(CAR.VOLKSWAGEN_GOLF_MK7, mqb_hud, True, id="mqb_digital"),
  pytest.param(CAR.VOLKSWAGEN_PASSAT_NMS, pq_hud, False, id="pq"),
]


@pytest.mark.parametrize("candidate, hud, digital", PLATFORMS)
def test_reported_cases(candidate, hud, digital):
  car = make_car(candidate)
  car.CS.upscale_lead_car_signal = digital
  closest = (LEAD_INDEX_RANGE_DIGITAL if digital else LEAD_INDEX_RANGE_ANALOG)[0]
  speed = 45 * CV.MPH_TO_MS
  # About one second behind at 45 mph read the far end; it belongs at the closest notch, short of the second
  assert closest < hud(car, lead_distance=20.0, speed=speed)[1] < lead_headway_index(SECOND_NOTCH * speed, speed, digital)
  # Stopped 1.5 m behind a car read second-closest
  assert hud(car, lead_distance=1.5, speed=0.0)[1] == closest


@pytest.mark.parametrize("candidate, hud, digital", PLATFORMS)
def test_no_lead_sends_no_object(candidate, hud, digital):
  car = make_car(candidate)
  car.CS.upscale_lead_car_signal = digital
  assert hud(car, lead_distance=30.0, speed=20.0, lead_visible=False)[1] == 0
  assert hud(car, lead_distance=0.0, speed=20.0)[1] == 0


@pytest.mark.parametrize("candidate, hud, digital", PLATFORMS)
def test_personalities_get_distinct_notches_and_ordered_markers(candidate, hud, digital):
  car = make_car(candidate)
  car.CS.upscale_lead_car_signal = digital
  speed = 25.0
  notches, markers = zip(*(hud(car, bars=bars, lead_distance=FOLLOW_TIMES[bars] * speed + STOP_DISTANCE, speed=speed)
                           for bars in FOLLOW_TIMES), strict=True)
  assert notches == (2, 3, 4)
  assert markers[0] < markers[1] < markers[2]
