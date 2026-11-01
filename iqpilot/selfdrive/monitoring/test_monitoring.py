import numpy as np
import pytest

from iqpilot.cereal import log, car
from iqpilot.common.realtime import DT_DMON
from iqpilot.selfdrive.monitoring.helpers import DriverMonitoring, DRIVER_MONITOR_SETTINGS
from iqpilot.system.hardware import HARDWARE

EventName = log.OnroadEvent.EventName
dm_settings = DRIVER_MONITOR_SETTINGS(device_type=HARDWARE.get_device_type())

TEST_TIMESPAN = 120  # seconds
DISTRACTED_SECONDS_TO_ORANGE = dm_settings._DISTRACTED_TIME - dm_settings._DISTRACTED_PROMPT_TIME_TILL_TERMINAL + 1
DISTRACTED_SECONDS_TO_RED = dm_settings._DISTRACTED_TIME + 1
INVISIBLE_SECONDS_TO_ORANGE = dm_settings._AWARENESS_TIME - dm_settings._AWARENESS_PROMPT_TIME_TILL_TERMINAL + 1
INVISIBLE_SECONDS_TO_RED = dm_settings._AWARENESS_TIME + 1

def make_msg(face_detected, distracted=False, model_uncertain=False):
  ds = log.DriverStateV2.new_message()
  ds.leftDriverData.faceOrientation = [0., 0., 0.]
  ds.leftDriverData.facePosition = [0., 0.]
  ds.leftDriverData.faceProb = 1. * face_detected
  ds.leftDriverData.leftEyeProb = 1.
  ds.leftDriverData.rightEyeProb = 1.
  ds.leftDriverData.leftBlinkProb = 1. * distracted
  ds.leftDriverData.rightBlinkProb = 1. * distracted
  ds.leftDriverData.faceOrientationStd = [1.*model_uncertain, 1.*model_uncertain, 1.*model_uncertain]
  ds.leftDriverData.facePositionStd = [1.*model_uncertain, 1.*model_uncertain]
  # TODO: test both separately when e2e is used
  ds.leftDriverData.phoneProb = 0.
  return ds


def set_driver_data(data, face_prob=1.0, phone_prob=0.0):
  data.faceOrientation = [0., 0., 0.]
  data.facePosition = [0., 0.]
  data.faceProb = face_prob
  data.leftEyeProb = 1.
  data.rightEyeProb = 1.
  data.leftBlinkProb = 0.
  data.rightBlinkProb = 0.
  data.faceOrientationStd = [0., 0., 0.]
  data.facePositionStd = [0., 0.]
  data.phoneProb = phone_prob


# driver state from neural net, 10Hz
msg_NO_FACE_DETECTED = make_msg(False)
msg_ATTENTIVE = make_msg(True)
msg_DISTRACTED = make_msg(True, distracted=True)
msg_ATTENTIVE_UNCERTAIN = make_msg(True, model_uncertain=True)
msg_DISTRACTED_UNCERTAIN = make_msg(True, distracted=True, model_uncertain=True)
msg_DISTRACTED_BUT_SOMEHOW_UNCERTAIN = make_msg(True, distracted=True, model_uncertain=dm_settings._POSESTD_THRESHOLD*1.5)

# driver interaction with car
car_interaction_DETECTED = True
car_interaction_NOT_DETECTED = False

# some common state vectors
always_no_face = [msg_NO_FACE_DETECTED] * int(TEST_TIMESPAN / DT_DMON)
always_attentive = [msg_ATTENTIVE] * int(TEST_TIMESPAN / DT_DMON)
always_distracted = [msg_DISTRACTED] * int(TEST_TIMESPAN / DT_DMON)
always_true = [True] * int(TEST_TIMESPAN / DT_DMON)
always_false = [False] * int(TEST_TIMESPAN / DT_DMON)

class TestMonitoring:
  def _run_seq(self, msgs, interaction, engaged, standstill):
    DM = DriverMonitoring()
    events = []
    for idx in range(len(msgs)):
      DM._update_states(msgs[idx], [0, 0, 0], 0, engaged[idx], standstill[idx])
      # cal_rpy and car_speed don't matter here

      # evaluate events at 10Hz for tests
      DM._update_events(interaction[idx], engaged[idx], standstill[idx], 0, 0)
      events.append(DM.current_events)
    assert len(events) == len(msgs), f"got {len(events)} for {len(msgs)} driverState input msgs"
    return events, DM

  def _assert_no_events(self, events):
    assert all(not len(e) for e in events)

  # engaged, driver is attentive all the time
  def test_fully_aware_driver(self):
    events, _ = self._run_seq(always_attentive, always_false, always_true, always_false)
    self._assert_no_events(events)

  # engaged, driver is distracted and does nothing
  def test_fully_distracted_driver(self):
    events, d_status = self._run_seq(always_distracted, always_false, always_true, always_false)
    assert len(events[int((d_status.settings._DISTRACTED_TIME-d_status.settings._DISTRACTED_PRE_TIME_TILL_TERMINAL)/2/DT_DMON)]) == 0
    assert events[int((d_status.settings._DISTRACTED_TIME-d_status.settings._DISTRACTED_PRE_TIME_TILL_TERMINAL + \
                    ((d_status.settings._DISTRACTED_PRE_TIME_TILL_TERMINAL-d_status.settings._DISTRACTED_PROMPT_TIME_TILL_TERMINAL)/2))/DT_DMON)].names[0] == \
                    EventName.preDriverDistracted
    assert events[int((d_status.settings._DISTRACTED_TIME-d_status.settings._DISTRACTED_PROMPT_TIME_TILL_TERMINAL + \
                    ((d_status.settings._DISTRACTED_PROMPT_TIME_TILL_TERMINAL)/2))/DT_DMON)].names[0] == EventName.promptDriverDistracted
    assert events[int((d_status.settings._DISTRACTED_TIME + \
                    ((TEST_TIMESPAN-10-d_status.settings._DISTRACTED_TIME)/2))/DT_DMON)].names[0] == EventName.driverDistracted
    assert isinstance(d_status.awareness, float)

  # engaged, no face detected the whole time, no action
  def test_fully_invisible_driver(self):
    events, d_status = self._run_seq(always_no_face, always_false, always_true, always_false)
    assert len(events[int((d_status.settings._AWARENESS_TIME-d_status.settings._AWARENESS_PRE_TIME_TILL_TERMINAL)/2/DT_DMON)]) == 0
    assert events[int((d_status.settings._AWARENESS_TIME-d_status.settings._AWARENESS_PRE_TIME_TILL_TERMINAL + \
                      ((d_status.settings._AWARENESS_PRE_TIME_TILL_TERMINAL-d_status.settings._AWARENESS_PROMPT_TIME_TILL_TERMINAL)/2))/DT_DMON)].names[0] == \
                      EventName.preDriverUnresponsive
    assert events[int((d_status.settings._AWARENESS_TIME-d_status.settings._AWARENESS_PROMPT_TIME_TILL_TERMINAL + \
                      ((d_status.settings._AWARENESS_PROMPT_TIME_TILL_TERMINAL)/2))/DT_DMON)].names[0] == EventName.promptDriverUnresponsive
    assert events[int((d_status.settings._AWARENESS_TIME + \
                      ((TEST_TIMESPAN-10-d_status.settings._AWARENESS_TIME)/2))/DT_DMON)].names[0] == EventName.driverUnresponsive

  # engaged, down to orange, driver pays attention, back to normal; then down to orange, driver touches wheel
  #  - should have short orange recovery time and no green afterwards; wheel touch only recovers when paying attention
  def test_normal_driver(self):
    ds_vector = [msg_DISTRACTED] * int(DISTRACTED_SECONDS_TO_ORANGE/DT_DMON) + \
                [msg_ATTENTIVE] * int(DISTRACTED_SECONDS_TO_ORANGE/DT_DMON) + \
                [msg_DISTRACTED] * int((DISTRACTED_SECONDS_TO_ORANGE+2)/DT_DMON) + \
                [msg_ATTENTIVE] * (int(TEST_TIMESPAN/DT_DMON)-int((DISTRACTED_SECONDS_TO_ORANGE*3+2)/DT_DMON))
    interaction_vector = [car_interaction_NOT_DETECTED] * int(DISTRACTED_SECONDS_TO_ORANGE*3/DT_DMON) + \
                         [car_interaction_DETECTED] * (int(TEST_TIMESPAN/DT_DMON)-int(DISTRACTED_SECONDS_TO_ORANGE*3/DT_DMON))
    events, _ = self._run_seq(ds_vector, interaction_vector, always_true, always_false)
    assert len(events[int(DISTRACTED_SECONDS_TO_ORANGE*0.5/DT_DMON)]) == 0
    assert events[int((DISTRACTED_SECONDS_TO_ORANGE-0.1)/DT_DMON)].names[0] == EventName.promptDriverDistracted
    assert len(events[int(DISTRACTED_SECONDS_TO_ORANGE*1.5/DT_DMON)]) == 0
    assert events[int((DISTRACTED_SECONDS_TO_ORANGE*3-0.1)/DT_DMON)].names[0] == EventName.promptDriverDistracted
    assert events[int((DISTRACTED_SECONDS_TO_ORANGE*3+0.1)/DT_DMON)].names[0] == EventName.promptDriverDistracted
    assert len(events[int((DISTRACTED_SECONDS_TO_ORANGE*3+2.5)/DT_DMON)]) == 0

  # engaged, down to orange, driver dodges camera, then comes back still distracted, down to red, \
  #                          driver dodges, and then touches wheel to no avail, disengages and reengages
  #  - orange/red alert should remain after disappearance, and only disengaging clears red
  def test_biggest_comma_fan(self):
    _invisible_time = 2  # seconds
    ds_vector = always_distracted[:]
    interaction_vector = always_false[:]
    op_vector = always_true[:]
    ds_vector[int(DISTRACTED_SECONDS_TO_ORANGE/DT_DMON):int((DISTRACTED_SECONDS_TO_ORANGE+_invisible_time)/DT_DMON)] \
                                                        = [msg_NO_FACE_DETECTED] * int(_invisible_time/DT_DMON)
    ds_vector[int((DISTRACTED_SECONDS_TO_RED+_invisible_time)/DT_DMON):int((DISTRACTED_SECONDS_TO_RED+2*_invisible_time)/DT_DMON)] \
                                                        = [msg_NO_FACE_DETECTED] * int(_invisible_time/DT_DMON)
    interaction_vector[int((DISTRACTED_SECONDS_TO_RED+2*_invisible_time+0.5)/DT_DMON):int((DISTRACTED_SECONDS_TO_RED+2*_invisible_time+1.5)/DT_DMON)] \
                                                        = [True] * int(1/DT_DMON)
    op_vector[int((DISTRACTED_SECONDS_TO_RED+2*_invisible_time+2.5)/DT_DMON):int((DISTRACTED_SECONDS_TO_RED+2*_invisible_time+3)/DT_DMON)] \
                                                        = [False] * int(0.5/DT_DMON)
    events, _ = self._run_seq(ds_vector, interaction_vector, op_vector, always_false)
    assert events[int((DISTRACTED_SECONDS_TO_ORANGE+0.5*_invisible_time)/DT_DMON)].names[0] == EventName.promptDriverDistracted
    assert events[int((DISTRACTED_SECONDS_TO_RED+1.5*_invisible_time)/DT_DMON)].names[0] == EventName.driverDistracted
    assert events[int((DISTRACTED_SECONDS_TO_RED+2*_invisible_time+1.5)/DT_DMON)].names[0] == EventName.driverDistracted
    assert len(events[int((DISTRACTED_SECONDS_TO_RED+2*_invisible_time+3.5)/DT_DMON)]) == 0

  # engaged, invisible driver, down to orange, driver touches wheel; then down to orange again, driver appears
  #  - both actions should clear the alert, but momentary appearance should not
  def test_sometimes_transparent_commuter(self):
    _visible_time = np.random.choice([0.5, 10])
    ds_vector = always_no_face[:]*2
    interaction_vector = always_false[:]*2
    ds_vector[int((2*INVISIBLE_SECONDS_TO_ORANGE+1)/DT_DMON):int((2*INVISIBLE_SECONDS_TO_ORANGE+1+_visible_time)/DT_DMON)] = \
                                                                                             [msg_ATTENTIVE] * int(_visible_time/DT_DMON)
    interaction_vector[int((INVISIBLE_SECONDS_TO_ORANGE)/DT_DMON):int((INVISIBLE_SECONDS_TO_ORANGE+1)/DT_DMON)] = [True] * int(1/DT_DMON)
    events, _ = self._run_seq(ds_vector, interaction_vector, 2*always_true, 2*always_false)
    assert len(events[int(INVISIBLE_SECONDS_TO_ORANGE*0.5/DT_DMON)]) == 0
    assert events[int((INVISIBLE_SECONDS_TO_ORANGE-0.1)/DT_DMON)].names[0] == EventName.promptDriverUnresponsive
    assert len(events[int((INVISIBLE_SECONDS_TO_ORANGE+0.1)/DT_DMON)]) == 0
    if _visible_time == 0.5:
      assert events[int((INVISIBLE_SECONDS_TO_ORANGE*2+1-0.1)/DT_DMON)].names[0] == EventName.promptDriverUnresponsive
      assert events[int((INVISIBLE_SECONDS_TO_ORANGE*2+1+0.1+_visible_time)/DT_DMON)].names[0] == EventName.preDriverUnresponsive
    elif _visible_time == 10:
      assert events[int((INVISIBLE_SECONDS_TO_ORANGE*2+1-0.1)/DT_DMON)].names[0] == EventName.promptDriverUnresponsive
      assert len(events[int((INVISIBLE_SECONDS_TO_ORANGE*2+1+0.1+_visible_time)/DT_DMON)]) == 0

  # engaged, invisible driver, down to red, driver appears and then touches wheel, then disengages/reengages
  #  - only disengage will clear the alert
  def test_last_second_responder(self):
    _visible_time = 2  # seconds
    ds_vector = always_no_face[:]
    interaction_vector = always_false[:]
    op_vector = always_true[:]
    ds_vector[int(INVISIBLE_SECONDS_TO_RED/DT_DMON):int((INVISIBLE_SECONDS_TO_RED+_visible_time)/DT_DMON)] = [msg_ATTENTIVE] * int(_visible_time/DT_DMON)
    interaction_vector[int((INVISIBLE_SECONDS_TO_RED+_visible_time)/DT_DMON):int((INVISIBLE_SECONDS_TO_RED+_visible_time+1)/DT_DMON)] = [True] * int(1/DT_DMON)
    op_vector[int((INVISIBLE_SECONDS_TO_RED+_visible_time+1)/DT_DMON):int((INVISIBLE_SECONDS_TO_RED+_visible_time+0.5)/DT_DMON)] = [False] * int(0.5/DT_DMON)
    events, _ = self._run_seq(ds_vector, interaction_vector, op_vector, always_false)
    assert len(events[int(INVISIBLE_SECONDS_TO_ORANGE*0.5/DT_DMON)]) == 0
    assert events[int((INVISIBLE_SECONDS_TO_ORANGE-0.1)/DT_DMON)].names[0] == EventName.promptDriverUnresponsive
    assert events[int((INVISIBLE_SECONDS_TO_RED-0.1)/DT_DMON)].names[0] == EventName.driverUnresponsive
    assert events[int((INVISIBLE_SECONDS_TO_RED+0.5*_visible_time)/DT_DMON)].names[0] == EventName.driverUnresponsive
    assert events[int((INVISIBLE_SECONDS_TO_RED+_visible_time+0.5)/DT_DMON)].names[0] == EventName.driverUnresponsive
    assert len(events[int((INVISIBLE_SECONDS_TO_RED+_visible_time+1+0.1)/DT_DMON)]) == 0

  # disengaged, always distracted driver
  #  - dm should stay quiet when not engaged
  def test_pure_dashcam_user(self):
    events, _ = self._run_seq(always_distracted, always_false, always_false, always_false)
    assert sum(len(event) for event in events) == 0

  # engaged, car stops at traffic light, down to orange, no action, then car starts moving
  #  - should only reach green when stopped, but continues counting down on launch
  def test_long_traffic_light_victim(self):
    _redlight_time = 60  # seconds
    standstill_vector = always_true[:]
    standstill_vector[int(_redlight_time/DT_DMON):] = [False] * int((TEST_TIMESPAN-_redlight_time)/DT_DMON)
    events, d_status = self._run_seq(always_distracted, always_false, always_true, standstill_vector)
    assert events[int((d_status.settings._DISTRACTED_TIME-d_status.settings._DISTRACTED_PRE_TIME_TILL_TERMINAL+1)/DT_DMON)].names[0] == \
                                                                                                                    EventName.preDriverDistracted
    assert events[int((_redlight_time-0.1)/DT_DMON)].names[0] == EventName.preDriverDistracted
    assert events[int((_redlight_time+0.5)/DT_DMON)].names[0] == EventName.promptDriverDistracted

  # engaged, model is somehow uncertain and driver is distracted
  #  - should fall back to wheel touch after uncertain alert
  def test_somehow_indecisive_model(self):
    ds_vector = [msg_DISTRACTED_BUT_SOMEHOW_UNCERTAIN] * int(TEST_TIMESPAN/DT_DMON)
    interaction_vector = always_false[:]
    events, d_status = self._run_seq(ds_vector, interaction_vector, always_true, always_false)
    assert EventName.preDriverUnresponsive in \
                              events[int((INVISIBLE_SECONDS_TO_ORANGE-1+DT_DMON*d_status.settings._HI_STD_FALLBACK_TIME-0.1)/DT_DMON)].names
    assert EventName.promptDriverUnresponsive in \
                              events[int((INVISIBLE_SECONDS_TO_ORANGE-1+DT_DMON*d_status.settings._HI_STD_FALLBACK_TIME+0.1)/DT_DMON)].names
    assert EventName.driverUnresponsive in \
                              events[int((INVISIBLE_SECONDS_TO_RED-1+DT_DMON*d_status.settings._HI_STD_FALLBACK_TIME+0.1)/DT_DMON)].names

  def test_calibrated_pose_blink_and_phone_distraction(self):
    dm = DriverMonitoring()
    dm.pose.calibrated = True
    dm.pose.pitch = -1.0
    dm.pose.yaw = 1.0
    dm.blink.left = 1.0
    dm.blink.right = 1.0
    dm.phone.prob_calibrated = True
    dm.phone.prob = 1.0
    assert set(dm._get_distracted_types()) == {1, 2, 4}

  def test_state_update_calibrates_pose_and_preserves_rhd_while_engaged(self):
    dm = DriverMonitoring()
    dm.wheelpos.prob_offseter.filtered_stat.M = 1.0
    dm.wheelpos.prob_offseter.filtered_stat.n = dm.settings._WHEELPOS_FILTER_MIN_COUNT + 1
    ds = log.DriverStateV2.new_message()
    ds.wheelOnRightProb = 1.0
    set_driver_data(ds.rightDriverData)

    dm._update_states(ds, [0., 0., 0.], 20.0, False, False)
    assert dm.wheel_on_right
    assert dm.pose.pitch_offseter.filtered_stat.n > 0
    assert dm.pose.yaw_offseter.filtered_stat.n > 0
    assert dm.phone.prob_offseter.filtered_stat.n > 0

    dm.wheelpos.prob_offseter.filtered_stat.M = 0.0
    dm._update_states(ds, [0., 0., 0.], 20.0, True, False)
    assert dm.wheel_on_right

  def test_state_update_forces_rhd(self):
    dm = DriverMonitoring(force_rhd=True)
    dm.wheel_on_right_last = False
    dm.wheelpos.prob_offseter.filtered_stat.M = 0.0
    dm.wheelpos.prob_offseter.filtered_stat.n = dm.settings._WHEELPOS_FILTER_MIN_COUNT + 1
    ds = log.DriverStateV2.new_message()
    ds.wheelOnRightProb = 0.0
    set_driver_data(ds.rightDriverData)

    dm._update_states(ds, [0., 0., 0.], 20.0, True, False)

    assert dm.wheel_on_right
    assert dm.face_detected
    assert dm.wheelpos.prob_offseter.filtered_stat.n == dm.settings._WHEELPOS_FILTER_MIN_COUNT + 1
    assert dm.get_state_packet().driverMonitoringState.isRHD

  def test_state_update_adopts_rhd_while_engaged_when_only_the_default_was_lhd(self):
    dm = DriverMonitoring(rhd_saved=False)
    ds = log.DriverStateV2.new_message()
    ds.wheelOnRightProb = 1.0
    set_driver_data(ds.rightDriverData)
    set_driver_data(ds.leftDriverData, face_prob=0.0)

    for _ in range(dm.settings._WHEELPOS_FILTER_MIN_COUNT + 2):
      dm._update_states(ds, [0., 0., 0.], 20.0, True, False)

    assert dm.wheel_on_right
    assert dm.face_detected

  def test_state_update_latches_a_calibrated_flip_while_engaged(self):
    dm = DriverMonitoring(rhd_saved=False)
    ds = log.DriverStateV2.new_message()
    ds.wheelOnRightProb = 1.0
    set_driver_data(ds.rightDriverData)
    set_driver_data(ds.leftDriverData)

    for _ in range(dm.settings._WHEELPOS_FILTER_MIN_COUNT + 2):
      dm._update_states(ds, [0., 0., 0.], 20.0, True, False)
    assert dm.wheel_on_right

    dm.wheelpos.prob_offseter.filtered_stat.M = 0.0
    dm._update_states(ds, [0., 0., 0.], 20.0, True, False)
    assert dm.wheel_on_right

  def test_state_update_uses_the_saved_side_before_calibration(self):
    dm = DriverMonitoring(rhd_saved=True)
    ds = log.DriverStateV2.new_message()
    ds.wheelOnRightProb = 1.0
    set_driver_data(ds.rightDriverData)
    set_driver_data(ds.leftDriverData, face_prob=0.0)

    dm._update_states(ds, [0., 0., 0.], 20.0, True, False)

    assert not dm.wheelpos.prob_calibrated
    assert dm.wheel_on_right
    assert dm.face_detected

  def test_state_update_rejects_incomplete_driver_data(self):
    dm = DriverMonitoring()
    dm._update_states(log.DriverStateV2.new_message(), [0., 0., 0.], 0.0, False, False)
    assert not dm.face_detected

  def test_demo_step_and_state_packet(self):
    dm = DriverMonitoring()
    ds = log.DriverStateV2.new_message()
    set_driver_data(ds.leftDriverData)
    dm.run_step({"driverStateV2": ds}, demo=True)
    packet = dm.get_state_packet(valid=False)
    assert not packet.valid
    assert packet.driverMonitoringState.faceDetected
    assert packet.driverMonitoringState.isActiveMode


@pytest.mark.parametrize("enabled_state, lat_active_state, expected", [
  (False, False, False), # Both Disabled
  (True, False, True),   # OP Enabled, Lat Inactive
  (False, True, True),   # OP Disabled, Lat Active (e.g. AOL)
  (True, True, True)     # Both Active
])
def test_enabled_states(enabled_state, lat_active_state, expected):
  """
  Test DriverMonitoring.run_step with all 4 combinations of:
  - selfdriveState.enabled (True/False)
  - carControl.latActive (True/False)
  """
  cs = car.CarState.new_message()
  cs.vEgo = 30.0
  cs.gearShifter = car.CarState.GearShifter.drive
  cs.standstill = False
  cs.steeringPressed = False
  cs.gasPressed = False

  ss = log.SelfdriveState.new_message()
  ss.enabled = enabled_state

  cc = car.CarControl.new_message()
  cc.latActive = lat_active_state

  mv2 = log.ModelDataV2.new_message()
  mv2.meta.disengagePredictions.brakeDisengageProbs = [0.0]

  lc = log.ExtrinsicsCalibration.new_message()
  lc.rpyCalib = [0.0, 0.0, 0.0]

  ds = make_msg(False)

  sm = {
      'carState': cs,
      'selfdriveState': ss,
      'carControl': cc,
      'modelV2': mv2,
      'extrinsicsCalibration': lc,
      'driverStateV2': ds
  }

  driver_monitoring = DriverMonitoring()

  # run_test doesn't assign enabled to a variable, so we need to spy on _update_events to see its value
  captured_args = []
  original_update_events = driver_monitoring._update_events

  def spy_update_events(driver_engaged, op_engaged, standstill, wrong_gear, car_speed):
      captured_args.append(op_engaged)
      return original_update_events(driver_engaged, op_engaged, standstill, wrong_gear, car_speed)

  driver_monitoring._update_events = spy_update_events

  driver_monitoring.run_step(sm, demo=False)

  # Assertion
  assert len(captured_args) == 1, "Expected _update_events to be called exactly once"
  actual_enabled = captured_args[0]

  assert actual_enabled == expected, f"Expected op_engaged={expected}, but got {actual_enabled}"
