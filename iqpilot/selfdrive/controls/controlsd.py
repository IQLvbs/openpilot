#!/usr/bin/env python3
import math
import os
import time
from numbers import Number

from iqpilot.cereal import car, log
import iqpilot.cereal.messaging as messaging
from iqpilot.common.constants import CV
from iqpilot.common.iq_perf import PerfSample, PerfTraceEmitter, PerfTraceRing
from iqpilot.common.params import Params
from iqpilot.common.pt2 import PT2Filter
from iqpilot.common.realtime import config_realtime_process, lock_memory, DT_CTRL, Priority, Ratekeeper
from iqpilot.common.swaglog import cloudlog

from iqdbc.car.car_helpers import interfaces
from iqdbc.car.vehicle_model import VehicleModel
from iqpilot.common.steer_delay import lateral_action_delay
from iqpilot.selfdrive.controls.lib.drive_helpers import clip_curvature
from iqpilot.selfdrive.controls.lib.actuator_feedback import SteeringActuatorFeedback, car_output_supports_pairing, steering_feedback_from_output
from iqpilot.selfdrive.controls.lib.hkg_tuning import uses_hkg_tuning
from iqpilot.selfdrive.controls.lib.curvature_lookahead import select_lateral_curvature
from iqpilot.selfdrive.controls.lib.latcontrol import LatControl
from iqpilot.selfdrive.controls.lib.latcontrol_pid import LatControlPID
from iqpilot.selfdrive.controls.lib.latcontrol_angle import LatControlAngle, STEER_ANGLE_SATURATION_THRESHOLD
from iqpilot.selfdrive.controls.lib.latcontrol_torque import LatControlTorque
from iqpilot.selfdrive.controls.lib.latcontrol_torque_pq import LatControlTorquePQ
from iqpilot.selfdrive.controls.lib.latcontrol_torque_v0 import LatControlTorqueV0, is_vw_mqb_torque
from iqpilot.selfdrive.controls.lib.longcontrol import LongControl
from iqpilot.selfdrive.controls.lib.stop_offset import CONTROLLER_VERSION
from iqpilot.selfdrive.controls.lib.longitudinal_comfort import LongitudinalComfort
from iqpilot.selfdrive.controls.steering_fault_recovery import SteeringFaultRecovery
from iqpilot.system.proprietary_runtime._verified_import import import_verified_module
from iqpilot.selfdrive.locationd.helpers import PoseCalibrator, Pose

from iqpilot.selfdrive.controls.iq_controls_layer import IQControlsLayer

NavTurnSignalController = import_verified_module(
  "iqpilot_navd_private", "iqpilot_private.navd.nav_turn_signals"
).NavTurnSignalController

State = log.SelfdriveState.OpenpilotState
REPLAY = "REPLAY" in os.environ
EventName = log.OnroadEvent.EventName
LaneChangeState = log.LaneChangeState
LaneChangeDirection = log.LaneChangeDirection

ACTUATOR_FIELDS = tuple(car.CarControl.Actuators.schema.fields.keys())
CTRL_ESSENTIAL_SERVICES = ("carState", "modelV2", "longitudinalPlan", "vehicleParameters", "selfdriveState")
CTRL_LOOP_WARN_US = 15_000
CTRL_LOOP_BAD_US = 25_000
CTRL_LOOP_SEVERE_US = 50_000
CTRL_PHASE_WARN_US = 8_000
CTRL_TAIL_WARN_US = 8_000
CTRL_FLAG_MISSING_INPUTS = 1 << 0
CTRL_FLAG_RK_OVERRUN = 1 << 1
SMOOTH_STEER_W0 = 46.0
SMOOTH_STEER_ZETA = 1.0
SMOOTH_STEER_HOLD_FRAMES = 50
LAT_SMOOTH_SECONDS = 0.0


class Controls(IQControlsLayer):
  def __init__(self) -> None:
    self.params = Params()
    cloudlog.info("controlsd is waiting for CarParams")
    self.CP = messaging.log_from_bytes(self.params.get("CarParams", block=True), car.CarParams)
    cloudlog.info("controlsd got CarParams")

    # Initialize iqpilot controlsd extension and base model state
    IQControlsLayer.__init__(self, self.CP, self.params)

    self.CI = interfaces[self.CP.carFingerprint](self.CP, self.CP_IQ)

    self.sm = messaging.SubMaster(['lateralDelay', 'vehicleParameters', 'lateralTorqueParameters', 'modelV2', 'selfdriveState',
                                   'extrinsicsCalibration', 'deviceMotion', 'longitudinalPlan', 'lateralManeuverPlan',
                                   'carState', 'carOutput', 'driverMonitoringState', 'onroadEvents',
                                   'driverAssistance', 'lateralDelay'] + self.iq_sub_services,
                                  poll='selfdriveState')
    self.pm = messaging.PubMaster(['carControl', 'controlsState', 'iqPerfTrace'] + self.iq_pub_services)

    self.steer_limited_by_safety = False
    self.steering_fault_recovery = SteeringFaultRecovery()
    self.curvature = 0.0
    self.desired_curvature = 0.0
    self.lateral_action_selection = log.ControlsState.LateralActionSelection.inactive
    self.curvature_lookahead_enabled = self.params.get_bool("IQLateralCurvatureLookahead")
    self.roll_compensation = 0.0

    self._perf = PerfTraceEmitter("controlsd", pubmaster=self.pm)
    self._perf_ring = PerfTraceRing()

    self._param_update_time = 0.0
    self.enable_curvature_controller = False
    self.enable_smooth_steer = False
    self.enable_speed_limit_control = False
    self.enable_speed_limit_predicative = False
    self.enable_pred_react_to_speed_limits = False
    self.enable_pred_react_to_curves = False
    self.enable_long_comfort_mode = False
    self.force_rhd_for_bsm = False
    self.navigation_enabled = False
    self.nav_exit_lane_change = False
    self._update_params()

    self.nav_turn_signal_controller = NavTurnSignalController(self.CP)

    self.pose_calibrator = PoseCalibrator()
    self.calibrated_pose: Pose | None = None

    self.LoC = LongControl(self.CP, self.CP_IQ)
    self.longitudinal_comfort = LongitudinalComfort(self.CP_IQ.longitudinalActuation)
    self.VM = VehicleModel(self.CP)

    self.smooth_steer = PT2Filter(SMOOTH_STEER_W0, SMOOTH_STEER_ZETA, DT_CTRL)
    self.smooth_steer_inactive_frames = SMOOTH_STEER_HOLD_FRAMES
    self.is_curvature_car = self.CP.steerControlType == car.CarParams.SteerControlType.curvatureDEPRECATED
    if self.is_curvature_car and self.params.get("EnableSmoothSteer") is None:
      self.params.put_bool("EnableSmoothSteer", True)
    self.LaC: LatControl
    if self.CP.steerControlType in (car.CarParams.SteerControlType.angle, car.CarParams.SteerControlType.curvatureDEPRECATED):
      self.LaC = LatControlAngle(self.CP, self.CP_IQ, self.CI, DT_CTRL)
    elif self.CP.lateralTuning.which() == 'pid':
      self.LaC = LatControlPID(self.CP, self.CP_IQ, self.CI, DT_CTRL)
    elif self.CP.lateralTuning.which() == 'torque':
      if self._use_pq_torque():
        try:
          self.LaC = LatControlTorquePQ(self.CP, self.CP_IQ, self.CI, DT_CTRL)
        except Exception:
          cloudlog.exception("LatControlTorquePQ init failed; using generic torque")
          self.LaC = LatControlTorque(self.CP, self.CP_IQ, self.CI, DT_CTRL)
      elif self._use_mqb_torque_v0():
        try:
          self.LaC = LatControlTorqueV0(self.CP, self.CP_IQ, self.CI, DT_CTRL)
        except Exception:
          cloudlog.exception("LatControlTorqueV0 init failed; using generic torque")
          self.LaC = LatControlTorque(self.CP, self.CP_IQ, self.CI, DT_CTRL)
      else:
        self.LaC = LatControlTorque(self.CP, self.CP_IQ, self.CI, DT_CTRL)

  def _use_pq_torque(self) -> bool:
    try:
      if self.CP.brand != 'volkswagen':
        return False
      from iqdbc.car.volkswagen.values import VolkswagenFlags
      return bool(self.CP.flags & VolkswagenFlags.PQ)
    except Exception:
      cloudlog.exception("pq torque selection failed; using generic torque")
      return False

  def _use_mqb_torque_v0(self) -> bool:
    try:
      return is_vw_mqb_torque(self.CP)
    except Exception:
      cloudlog.exception("mqb torque v0 selection failed; using generic torque")
      return False

  def _update_params(self) -> None:
    self.enable_curvature_controller = self.params.get_bool("EnableCurvatureController")
    self.enable_smooth_steer = self.params.get_bool("EnableSmoothSteer")
    self.enable_speed_limit_control = self.params.get_bool("EnableSpeedLimitControl")
    self.enable_speed_limit_predicative = self.params.get_bool("EnableSpeedLimitPredicative")
    self.enable_pred_react_to_speed_limits = self.params.get_bool("EnableSLPredReactToSL")
    self.enable_pred_react_to_curves = self.params.get_bool("EnableSLPredReactToCurves")
    self.enable_long_comfort_mode = self.params.get_bool("EnableLongComfortMode")
    self.force_rhd_for_bsm = self.params.get_bool("ForceRHDForBSM")
    self.navigation_enabled = self.params.get_bool("NavigationEnabled")
    self.nav_exit_lane_change = self.params.get_bool("NavExitLaneChange")

  def update(self):
    self.sm.update(15)
    if self.sm.updated["extrinsicsCalibration"]:
      self.pose_calibrator.feed_live_calib(self.sm['extrinsicsCalibration'])
    if self.sm.updated["deviceMotion"]:
      device_pose = Pose.from_live_pose(self.sm['deviceMotion'])
      self.calibrated_pose = self.pose_calibrator.build_calibrated_pose(device_pose)
    if time.monotonic() - self._param_update_time > 3:
      self._update_params()
      self._param_update_time = time.monotonic()

  def state_control(self):
    CS = self.sm['carState']

    # Update VehicleModel
    lp = self.sm['vehicleParameters']
    x = max(lp.stiffnessFactor, 0.1)
    sr = max(lp.steerRatio, 0.1)
    self.VM.update_params(x, sr)

    steer_angle_without_offset = math.radians(CS.steeringAngleDeg - lp.angleOffsetDeg)
    self.curvature = -self.VM.calc_curvature(steer_angle_without_offset, CS.vEgo, lp.roll)
    self.roll_compensation = -self.VM.roll_compensation(lp.roll, CS.vEgo)

    nnff_assist = getattr(self.LaC, "nnff_assist", None)
    if nnff_assist is not None:
      nnff_assist.update_model_v2(self.sm['modelV2'] if self.sm.all_checks(['modelV2']) else None)

    if self.CP.lateralTuning.which() == 'torque' and hasattr(self.LaC, "update_live_torque_params"):
      torque_params = self.sm['lateralTorqueParameters']
      if self.sm.all_checks(['lateralTorqueParameters']) and torque_params.useParams:
        if hasattr(self.LaC, "set_live_torque_valid"):
          self.LaC.set_live_torque_valid(torque_params.valid)
        self.LaC.update_live_torque_params(torque_params.latAccelFactorFiltered, torque_params.latAccelOffsetFiltered,
                                           torque_params.frictionCoefficientFiltered)

    long_plan = self.sm['longitudinalPlan']
    model_v2 = self.sm['modelV2']

    CC = car.CarControl.new_message()
    CC.enabled = self.sm['selfdriveState'].enabled

    # Check which actuators can be enabled
    standstill = abs(CS.vEgo) <= max(self.CP.minSteerSpeed, 0.3) or CS.standstill

    # Get which state to use for active lateral control
    _lat_active = self.iq_lateral_allowed(self.sm)

    steering_fault_recovered = self.steering_fault_recovery.update(CS.steerFaultTemporary, CS.steerFaultPermanent)
    CC.latActive = _lat_active and steering_fault_recovered and \
                   (not standstill or self.CP.steerAtStandstill)
    # long control may stay active through a gas override on platforms that opt in
    override_longitudinal = any(e.overrideLongitudinal for e in self.sm['onroadEvents'])
    long_through_override = self.CP_IQ.longActiveWithGasOverride and self.CP.openpilotLongitudinalControl
    CC.longActive = CC.enabled and not getattr(CS, 'cruiseFaultLateralMode', False) and \
                    (not override_longitudinal or long_through_override) and \
                    (self.CP.openpilotLongitudinalControl or not self.CP_IQ.pcmCruiseSpeed)

    CC.leftBlinker, CC.rightBlinker = self.nav_turn_signal_controller.update(
      self.sm['iqNavState'] if self.sm.alive['iqNavState'] else None,
      self.navigation_enabled,
      self.nav_exit_lane_change,
      CC.latActive,
      CS.leftBlinker,
      CS.rightBlinker,
      CS.vEgo,
      CS.yawRate,
      CS.steeringAngleDeg,
      self.sm.alive['iqNavState'] and self.sm.valid['iqNavState'],
    )

    actuators = CC.actuators
    actuators.longControlState = self.LoC.long_control_state
    actuators.speed = float(max(long_plan.speeds, default=0.0))

    if not CC.latActive:
      self.LaC.reset()

    if not CC.longActive:
      self.LoC.reset()

    # accel PID loop
    pid_accel_limits = self.CI.get_pid_accel_limits(self.CP, self.CP_IQ, CS.vEgo, CS.vCruise * CV.KPH_TO_MS)
    v_target_now = long_plan.speeds[0] if len(long_plan.speeds) else CS.vEgo
    stop_target = long_plan.stopOffset
    stop_target_valid = stop_target.version == CONTROLLER_VERSION and stop_target.valid
    actuators.accel = float(self.LoC.update(CC.longActive, CS, long_plan.aTarget, long_plan.shouldStop, pid_accel_limits,
                                            long_plan.leadDistance, long_plan.hasLead, gas_override=override_longitudinal,
                                            v_target_now=v_target_now,
                                            stop_distance=stop_target.distance if stop_target_valid else None,
                                            stop_event_id=stop_target.eventId if stop_target_valid else 0,
                                            comfort=self.longitudinal_comfort, comfort_enabled=self.enable_long_comfort_mode,
                                            fcw=long_plan.fcw, plan_source=long_plan.longitudinalPlanSource))

    # Steering PID loop and lateral MPC
    # Reset desired curvature to current to avoid violating the limits on engage
    maneuver_curvature = self.sm['lateralManeuverPlan'].desiredCurvature if self.sm.valid['lateralManeuverPlan'] else None
    new_desired_curvature, self.lateral_action_selection = select_lateral_curvature(
      model_v2, CC.latActive, self.curvature, maneuver_curvature,
      self.curvature_lookahead_enabled, self.LaC.supports_legacy_curvature_lookahead,
    )

    self.smooth_steer_inactive_frames = 0 if CC.latActive else self.smooth_steer_inactive_frames + 1
    if self.is_curvature_car and self.enable_smooth_steer and self.smooth_steer_inactive_frames < SMOOTH_STEER_HOLD_FRAMES:
      new_desired_curvature = self.smooth_steer.update(new_desired_curvature)
    else:
      self.smooth_steer.reset(new_desired_curvature)

    lat_accel_override = bool(CS.gasPressed) or bool(self.sm['iqState'].aol.active)
    self.desired_curvature, curvature_limited = clip_curvature(CS.vEgo, self.desired_curvature, new_desired_curvature, lp.roll, lat_accel_override)
    lat_delay = lateral_action_delay(self.params, self.CP, self.sm["lateralDelay"].lateralDelay) + LAT_SMOOTH_SECONDS
    if nnff_assist is not None:
      nnff_assist.update_lateral_lag(lat_delay)
    if self.CP.carFingerprint == "HYUNDAI_TUCSON_4TH_GEN":
      lat_delay = min(lat_delay, 0.15)

    actuators.curvature = self.desired_curvature
    steer, steeringAngleDeg, lac_log = self.LaC.update(CC.latActive, CS, self.VM, lp,
                                                       self.steer_limited_by_safety, self.desired_curvature,
                                                       self.calibrated_pose, curvature_limited, lat_delay)
    if self.CP.steerControlType in (car.CarParams.SteerControlType.angle, car.CarParams.SteerControlType.curvatureDEPRECATED):
      actuators.torque = 0.0
      actuators.steeringAngleDeg = float(steeringAngleDeg)
    else:
      actuators.torque = float(steer)
      actuators.steeringAngleDeg = float(steeringAngleDeg)
    # Ensure no NaNs/Infs
    for p in ACTUATOR_FIELDS:
      attr = getattr(actuators, p)
      if not isinstance(attr, Number):
        continue

      if not math.isfinite(attr):
        cloudlog.error(f"actuators.{p} not finite {actuators.to_dict()}")
        setattr(actuators, p, 0.0)

    return CC, lac_log

  def publish(self, CC, lac_log):
    CS = self.sm['carState']

    CC.curvatureControllerActive = self.enable_curvature_controller
    CC.steerLimited = self.steer_limited_by_safety
    CC.forceRHDForBSM = self.force_rhd_for_bsm
    CC.longComfortMode = self.enable_long_comfort_mode

    # Orientation and angle rates can be useful for carcontroller
    # Only calibrated (car) frame is relevant for the carcontroller
    CC.currentCurvature = self.curvature
    CC.rollCompensation = self.roll_compensation
    if self.calibrated_pose is not None:
      CC.orientationNED = self.calibrated_pose.orientation.xyz.tolist()
      CC.angularVelocity = self.calibrated_pose.angular_velocity.xyz.tolist()

    CC.cruiseControl.override = CC.enabled and not CC.longActive and (self.CP.openpilotLongitudinalControl or not self.CP_IQ.pcmCruiseSpeed)
    dm_lockout = self.CP.pcmCruise and any(e.name == EventName.tooDistracted for e in self.sm['onroadEvents'])
    CC.cruiseControl.cancel = CS.cruiseState.enabled and (not CC.enabled or not self.CP.pcmCruise) and not dm_lockout
    CC.cruiseControl.resume = CC.enabled and CS.cruiseState.standstill and not self.sm['longitudinalPlan'].shouldStop
    CC.cruiseControl.speedLimit = self.enable_speed_limit_control
    CC.cruiseControl.speedLimitPredicative = self.enable_speed_limit_predicative
    CC.cruiseControl.speedLimitPredReactToSL = self.enable_pred_react_to_speed_limits
    CC.cruiseControl.speedLimitPredReactToCurves = self.enable_pred_react_to_curves

    hudControl = CC.hudControl
    hudControl.setSpeed = float(CS.vCruiseCluster * CV.KPH_TO_MS)
    hudControl.speedVisible = CC.enabled
    hudControl.lanesVisible = CC.enabled
    hudControl.leadVisible = self.sm['longitudinalPlan'].hasLead
    hudControl.leadDistance = self.sm['longitudinalPlan'].leadDistance
    hudControl.leadDistanceBars = self.sm['selfdriveState'].personality.raw + 1
    if self.sm['selfdriveState'].personality.raw == log.LongitudinalPersonality.relaxed:
      hudControl.leadFollowTime = 1.75
    elif self.sm['selfdriveState'].personality.raw == log.LongitudinalPersonality.aggressive:
      hudControl.leadFollowTime = 1.25
    else:
      hudControl.leadFollowTime = 1.45
    hudControl.visualAlert = self.sm['selfdriveState'].alertHudVisual
    hudControl.audibleAlert = self.sm['selfdriveState'].alertSound
    hudControl.driverUnresponsive = self.sm['selfdriveState'].alertType.split('/', 1)[0] == 'driverUnresponsive'

    hudControl.rightLaneVisible = True
    hudControl.leftLaneVisible = True
    if self.sm.valid['driverAssistance']:
      hudControl.leftLaneDepart = self.sm['driverAssistance'].leftLaneDeparture
      hudControl.rightLaneDepart = self.sm['driverAssistance'].rightLaneDeparture

    actuator_feedback = SteeringActuatorFeedback(True)
    if CC.latActive:
      angle_control = self.CP.steerControlType in (car.CarParams.SteerControlType.angle, car.CarParams.SteerControlType.curvatureDEPRECATED)
      if uses_hkg_tuning(self.CP) and car_output_supports_pairing():
        # replay rewrites envelope times into the recorded clock without virtualizing the host clock
        feedback_now = max(self.sm.logMonoTime.values()) if REPLAY else time.monotonic_ns()
        actuator_feedback = steering_feedback_from_output(
          self.sm['carOutput'], feedback_valid=self.sm.all_checks(['carOutput']),
          output_mono_time=self.sm.logMonoTime['carOutput'], now_mono_time=feedback_now,
          angle_control=angle_control, angle_threshold=STEER_ANGLE_SATURATION_THRESHOLD,
        )
        self.steer_limited_by_safety = actuator_feedback.limited
      else:
        CO = self.sm['carOutput']
        if angle_control:
          self.steer_limited_by_safety = abs(CC.actuators.steeringAngleDeg - CO.actuatorsOutput.steeringAngleDeg) > \
                                                STEER_ANGLE_SATURATION_THRESHOLD
        else:
          self.steer_limited_by_safety = abs(CC.actuators.torque - CO.actuatorsOutput.torque) > 1e-2
    update_feedback = getattr(getattr(self, 'LaC', None), 'update_actuator_feedback', None)
    if callable(update_feedback):
      update_feedback(actuator_feedback)

    # TODO: both controlsState and carControl valids should be set by
    #       sm.all_checks(), but this creates a circular dependency

    # controlsState
    dat = messaging.new_message('controlsState')
    dat.valid = CS.canValid
    cs = dat.controlsState

    cs.curvature = self.curvature
    cs.longitudinalPlanMonoTime = self.sm.logMonoTime['longitudinalPlan']
    cs.lateralPlanMonoTime = self.sm.logMonoTime['modelV2']
    cs.desiredCurvature = self.desired_curvature
    cs.lateralActionSelection = self.lateral_action_selection
    cs.longControlState = self.LoC.long_control_state
    cs.upAccelCmd = float(self.LoC.pid.p)
    cs.uiAccelCmd = float(self.LoC.pid.i)
    cs.ufAccelCmd = float(self.LoC.pid.f)
    cs.forceDecel = bool((self.sm['driverMonitoringState'].awarenessStatus < 0.) or
                         (self.sm['selfdriveState'].state == State.softDisabling))

    lat_tuning = self.CP.lateralTuning.which()
    if self.CP.steerControlType in (car.CarParams.SteerControlType.angle, car.CarParams.SteerControlType.curvatureDEPRECATED):
      cs.lateralControlState.angleState = lac_log
    elif lat_tuning == 'pid':
      cs.lateralControlState.pidState = lac_log
    elif lat_tuning == 'torque':
      cs.lateralControlState.torqueState = lac_log

    self.pm.send('controlsState', dat)

    # carControl
    cc_send = messaging.new_message('carControl')
    cc_send.valid = CS.canValid
    cc_send.carControl = CC
    self.pm.send('carControl', cc_send)

  def _tail_work(self) -> int:
    started_ns = time.monotonic_ns()
    self.refresh_iq_params(self.sm)
    self.publish_iq_state(self.sm, self.pm, self.longitudinal_comfort.control)
    return (time.monotonic_ns() - started_ns) // 1000

  def _missing_services(self) -> list[str]:
    missing = []
    for service in CTRL_ESSENTIAL_SERVICES:
      if not self.sm.alive[service] or not self.sm.freq_ok[service] or not self.sm.valid[service]:
        missing.append(service)
    return missing

  def _emit_perf_trace(self, frame_id: int, loop_dt_us: int, update_us: int, state_control_us: int,
                       publish_us: int, tail_work_us: int, rk_remaining_us: int) -> None:
    missing_services = self._missing_services()
    flags = 0
    if missing_services:
      flags |= CTRL_FLAG_MISSING_INPUTS
    if rk_remaining_us < 0:
      flags |= CTRL_FLAG_RK_OVERRUN

    sample = PerfSample(
      frame_id=frame_id,
      loop_dt_us=loop_dt_us,
      update_us=update_us,
      state_control_us=state_control_us,
      publish_us=publish_us,
      tail_work_us=tail_work_us,
      rk_remaining_us=rk_remaining_us,
      flags=flags,
    )
    self._perf_ring.push(sample)

    slow_phase = max(update_us, state_control_us, publish_us, tail_work_us)
    if loop_dt_us < CTRL_LOOP_WARN_US and slow_phase < CTRL_PHASE_WARN_US and tail_work_us < CTRL_TAIL_WARN_US and flags == 0:
      return

    if loop_dt_us >= CTRL_LOOP_SEVERE_US:
      severity = "critical"
    elif loop_dt_us >= CTRL_LOOP_BAD_US or rk_remaining_us < -10_000:
      severity = "error"
    else:
      severity = "warning"

    detail = (
        f"update_us={update_us} state_control_us={state_control_us} publish_us={publish_us} " +
        f"tail_work_us={tail_work_us} rk_remaining_us={rk_remaining_us}"
    )
    self._perf.emit(
      "controlsd_slow_loop",
      severity=severity,
      frame_id=frame_id,
      total_time_us=loop_dt_us,
      rk_remaining_us=rk_remaining_us,
      flags=flags,
      samples=self._perf_ring.snapshot(),
      missing_services=missing_services,
      detail=detail,
      min_interval_s=0.25,
    )

  def run(self):
    rk = Ratekeeper(100, print_delay_threshold=None)
    while True:
      started_ns = time.monotonic_ns()
      checkpoint_ns = started_ns
      self.update()
      update_us = (time.monotonic_ns() - checkpoint_ns) // 1000
      checkpoint_ns = time.monotonic_ns()

      CC, lac_log = self.state_control()
      state_control_us = (time.monotonic_ns() - checkpoint_ns) // 1000
      checkpoint_ns = time.monotonic_ns()

      self.publish(CC, lac_log)
      publish_us = (time.monotonic_ns() - checkpoint_ns) // 1000
      checkpoint_ns = time.monotonic_ns()

      tail_work_us = self._tail_work()
      loop_dt_us = (time.monotonic_ns() - started_ns) // 1000
      rk_remaining_us = int(rk.remaining * 1_000_000)
      self._emit_perf_trace(self.sm.frame, int(loop_dt_us), int(update_us), int(state_control_us),
                            int(publish_us), int(tail_work_us), rk_remaining_us)
      rk.monitor_time()


def main():
  config_realtime_process(4, Priority.CTRL_HIGH)
  lock_memory()
  controls = Controls()
  controls.run()


if __name__ == "__main__":
  main()
