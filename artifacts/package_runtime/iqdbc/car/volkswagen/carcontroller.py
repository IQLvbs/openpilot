"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""
import sys
import os
import numpy as np
import random
from iqdbc.can import CANPacker
from iqdbc.car import Bus, DT_CTRL, make_tester_present_msg, structs
from iqdbc.car.lateral import apply_driver_steer_torque_limits, apply_steer_angle_limits_simple
from iqdbc.car.lateral import apply_std_curvature_limits
from iqdbc.car.common.conversions import Conversions as CV
from iqdbc.car.common.numpy_fast import clip, interp
from iqdbc.car.interfaces import CarControllerBase
from iqdbc.car.volkswagen import mlbcan, mqbcan, pqcan, mebcan
from iqdbc.car.volkswagen.hud import AccHudState
from iqdbc.car.volkswagen.pq_radar_handler import PQRadarHandler
from iqdbc.car.volkswagen.values import (
  CanBus, CarControllerParams, MQB_A0_CARS, VolkswagenFlags, VolkswagenFlagsIQ, apply_pq_stopping_accel,
)
from iqdbc.car.volkswagen.mebutils import longitudinal_comfort_values, CurvatureSmoother, LatControlCurvature, LongStopHold
from iqdbc.car.vehicle_model import VehicleModel

iqpilot_path = os.path.join(os.path.dirname(__file__), '..', '..', '..')
sys.path.insert(0, iqpilot_path)
try:
  from iqpilot.common.params import Params
except ImportError:
  pass

VisualAlert = structs.CarControl.HUDControl.VisualAlert
AudibleAlert = structs.CarControl.HUDControl.AudibleAlert
LongCtrlState = structs.CarControl.Actuators.LongControlState


def dVisual(CCS, CS):
  if CCS == mqbcan:
    decelV = CS.tsk_verzoeg_anf
  elif CCS == pqcan:
    decelV = CS.br8_acc_anf
  else:
    decelV = False
  return decelV

# The MQB ACC_02.ACC_Abstandsindex and PQ ACC_GRA_Anzeige.ACA_gemZeitl ("measured time gap") lead marker is
# a time headway on a log scale, not a distance. Fitted to the stock radar's own output against lead distance
# and speed on 8 stock-ACC cars (Golf, Jetta, Atlas, Taos, Arteon and Kodiaq digital clusters, Octavia analog,
# Sharan PQ): position = 0.17 + 0.465 * ln(headway), from about 0.7 s (closest) to 6 s (farthest). Stopped,
# the radar still reports a finite headway, as if the car were rolling at about 3.5 m/s.
LEAD_HEADWAY_POSITION_OFFSET = 0.17
LEAD_HEADWAY_POSITION_PER_LOG_S = 0.465
LEAD_HEADWAY_MIN_SPEED = 3.5  # m/s
# Raw code at the closest and farthest positions: 4-bit analog MQB and PQ clusters use 1-15, digital MQB
# clusters (Kombi_03.KBI_Variante) the 10-bit 34-972 the stock radar sends. 0 means no lead on both.
LEAD_INDEX_RANGE_ANALOG = (1, 15)
LEAD_INDEX_RANGE_DIGITAL = (34, 972)


def lead_headway_index(lead_distance: float, v_ego: float, digital_cluster: bool) -> int:
  if lead_distance <= 0:
    return 0
  headway = lead_distance / max(v_ego, LEAD_HEADWAY_MIN_SPEED)
  position = clip(LEAD_HEADWAY_POSITION_OFFSET + LEAD_HEADWAY_POSITION_PER_LOG_S * np.log(headway), 0.0, 1.0)
  closest, farthest = LEAD_INDEX_RANGE_DIGITAL if digital_cluster else LEAD_INDEX_RANGE_ANALOG
  return int(round(closest + position * (farthest - closest)))


def accel_during_driver_override(accel: float, gas_pressed: bool, keep_long_active: bool) -> float:
  return 0.0 if gas_pressed and keep_long_active else accel


def ea_send_ready(stock_values, last_counter):
  return bool(stock_values) and stock_values["COUNTER"] != last_counter


EA_BLINKER_STEP = 2


def next_ea_counter(tx_counter, stock_counter):
  return ((stock_counter if tx_counter is None else tx_counter) + 1) % 16


def ea_blinker_command(left_request, right_request, left_active, right_active):
  blinker_active = left_active or right_active
  return left_request and not blinker_active, right_request and not blinker_active


class CarController(CarControllerBase):
  def __init__(self, dbc_names, CP, CP_IQ):
    try:
      from iqpilot.system.proprietary_runtime._verified_import import import_verified_module
      self._iq_lvbs_alc = import_verified_module("iqpilot_alc_private", "iqpilot_private.konn3kt.iqlvbs.alc")
    except Exception:
      self._iq_lvbs_alc = None
    super().__init__(dbc_names, CP, CP_IQ)
    self._params = Params()
    self.CCP = CarControllerParams(CP)
    self.CAN = CanBus(CP)
    self.packer_pt = CANPacker(dbc_names[Bus.pt])

    self._pt_tx_bus = self.CAN.pt
    if CP.flags & VolkswagenFlags.PQ:
      self.CCS = pqcan
      if CP.flags & VolkswagenFlagsIQ.IQ_PQ_LOWLINE:
        self._pt_tx_bus = self.CAN.aux
    elif CP.flags & VolkswagenFlags.MLB:
      self.CCS = mlbcan
      if CP.flags & VolkswagenFlagsIQ.IQ_MLB_NO_ECAN:
        self._pt_tx_bus = self.CAN.aux
    elif CP.flags & (VolkswagenFlags.MEB | VolkswagenFlags.MQB_EVO):
      self.CCS = mebcan
    else:
      self.CCS = mqbcan

    self.accel = 0
    self.apply_torque_last = 0
    self.apply_curvature_last = 0.
    self.apply_angle_last = 0
    self.ALC_entryCounter = 0
    self.ALC_driverExit = False
    self.ALC_reentry_blocked = False
    self.ALC_override_last = False
    self.ALC_override_counter = 0
    self.entering = False
    self.active = False
    self.CSLH3_SignLast = 0
    self.CSsteeringAngleDegLast = 0
    self.steering_power_last = 0
    self.curvature_smoother = (
      CurvatureSmoother(self.CCP.CURVATURE_SMOOTH_SPEED_BP, self.CCP.CURVATURE_SMOOTH_TAU_V, DT_CTRL * self.CCP.STEER_STEP)
      if self.CP.flags & (VolkswagenFlags.MEB | VolkswagenFlags.MQB_EVO) else None
    )
    self.long_stop_hold = LongStopHold(dt=(DT_CTRL * self.CCP.ACC_CONTROL_STEP)) if self.CP.flags & (VolkswagenFlags.MEB | VolkswagenFlags.MQB_EVO) else None
    self.gra_acc_counter_last = None
    self.gra_cancel_ticks = 0
    self.motor3_frame_last = None
    self.motor3_was_stopping = False
    self.motor3_resuming = False
    self.sng_handoff_active = False
    self.acc_counters_seeded: set[str] = set()
    self.klr_counter_last = None
    self.ea_counter_last = None
    self.ea_tx_counter = None
    self.eps_timer_soft_disable_alert = False
    self.hca_frame_timer_running = 0
    self.hca_frame_same_torque = 0
    self.accel_last = 0
    self.HCA_Status = 3
    self.leadDistanceBars = 0
    self.lead_distance_bars_last = None
    self.distance_bar_frame = 0
    self.acc_hud_state = AccHudState()
    self.mlb_hud_text = 0
    self.mlb_hud_text_frame = 0
    self.mlb_set_speed_last = 0
    self.mlb_lead_distance_bars_last = None
    self.speed_limit_last = 0
    self.speed_limit_changed_timer = 0
    self.blinkerActive = None
    self.hide_ea_error = False
    self.radar_disabled_warning_timer = 0
    # Check once at init whether the DBC includes MEB_AWV_01 (AEB HUD for radar-disabled camera harness cars)
    self._has_aeb_hud_msg = "MEB_AWV_01" in self.packer_pt.dbc.name_to_msg
    self.eps_timer_workaround = bool(CP.flags & (VolkswagenFlags.MLB | VolkswagenFlagsIQ.IQ_PQ_TIMEBOMB))
    self._pq_patch_checked = not bool(CP.flags & VolkswagenFlags.PQ) or self.eps_timer_workaround
    self.hca_frame_timer_resetting = 0
    self.hca_frame_low_torque = 0
    self.hca_reset_count = 0
    self.acc_hold_type_last = mebcan.ACC_HMS_NO_REQUEST
    self.acc_hold_ramp_counter = 0
    self.mqb_long_state = mqbcan.MqbLongStateMachine(CP.mass, self.CCP.ACCEL_MIN) if self.CCS == mqbcan else None
    self.radar_handler = PQRadarHandler(self.CAN) if self.CCS is pqcan else None
    self.blend_stock_radar = False
    self.unavailable = False
    self.unavailable_hold = 0
    self.VM = VehicleModel(CP)
    self.is_mqb_a0 = self._is_mqb_a0_car(CP.carFingerprint)
    self.LateralController = (
      LatControlCurvature(self.CCP.CURVATURE_PID, self.CCP.CURVATURE_LIMITS.CURVATURE_MAX, 1 / (DT_CTRL * self.CCP.STEER_STEP))
      if (CP.flags & (VolkswagenFlags.MEB | VolkswagenFlags.MQB_EVO))
      else None
    )

  @staticmethod
  def _is_mqb_a0_car(candidate) -> bool:
    return candidate in MQB_A0_CARS

  @property
  def hca_session_sec(self) -> float:
    return self.hca_frame_timer_running * DT_CTRL

  def _get_mqb_steering_torque_scale(self, v_ego: float, enabled: bool) -> float:
    if enabled and self.CCS == mqbcan:
      return float(np.interp(v_ego, [0.4, 3.5, 4.0], [0.8, 0.95, 1.0]))
    return 1.0

  def _mlb_acc_hud_text(self, hud_control, set_speed: float) -> int:
    # ACC_02 primary display text, briefly surfaced on a follow distance or set speed change
    if hud_control.leadDistanceBars != self.mlb_lead_distance_bars_last:
      self.mlb_hud_text_frame = self.frame
      self.mlb_hud_text = self.CCP.ACC_HUD_TEXT_DISTANCE.get(hud_control.leadDistanceBars, self.CCP.ACC_HUD_TEXTS["none"])
    elif set_speed != self.mlb_set_speed_last and hud_control.speedVisible:
      self.mlb_hud_text_frame = self.frame
      self.mlb_hud_text = self.CCP.ACC_HUD_TEXTS["setSpeed"]
    elif self.frame - self.mlb_hud_text_frame >= self.CCP.ACC_HUD_TEXT_STEP:
      self.mlb_hud_text = self.CCP.ACC_HUD_TEXTS["none"]
    self.mlb_lead_distance_bars_last = hud_control.leadDistanceBars
    self.mlb_set_speed_last = set_speed
    return self.mlb_hud_text

  def _tap_gra_cancel(self, cancel_req: bool, gra_send_ready: bool) -> bool:
    if not cancel_req:
      self.gra_cancel_ticks = 0
      return False

    period = self.CCP.GRA_CANCEL_TAP_ON + self.CCP.GRA_CANCEL_TAP_OFF
    if self.gra_cancel_ticks >= period * self.CCP.GRA_CANCEL_MAX_TAPS:
      return False

    pressed = self.gra_cancel_ticks % period < self.CCP.GRA_CANCEL_TAP_ON
    if gra_send_ready:
      self.gra_cancel_ticks += 1
    return pressed

  def _should_spam_mqb_a0_resume(self, CS, enabled: bool) -> bool:
    return bool(
      enabled and
      self.is_mqb_a0 and
      self.CCS == mqbcan and
      CS.out.standstill and
      self.frame % 50 < 15
    )

  def update(self, CC, CC_IQ, CS, now_nanos):
    actuators = CC.actuators
    hud_control = CC.hudControl
    can_sends = []
    output_torque = 0
    apply_torque = 0
    pqhca5or7Toggle = self._params.get_bool("pqhca5or7Toggle")
    eBrakeActive = self._params.get_bool("eBrakeActive")
    iq_mqb_steering_lockout = self._params.get_bool("iqMqbSteeringLockout")
    iq_mqb_acc_resume = self._params.get_bool("iqMqbAccResume")
    self.blend_stock_radar = self._params.get_bool("IQDynamicBlendStockRadar")
    if not self._pq_patch_checked:
      self._pq_patch_checked = True
      self.eps_timer_workaround = not self._params.get_bool("VwPqEpsPatched")
    blend_active = bool(self.blend_stock_radar and self.radar_handler is not None and self.CP.openpilotLongitudinalControl)
    AngleLateralControl = self._iq_lvbs_alc.angle_lateral_control_enabled(self, CS) if self._iq_lvbs_alc else False
    aol_active = CC_IQ.aol.available and CC_IQ.aol.enabled and CC_IQ.aol.active
    self.entering = CS.vw_iq_lvbs_alc_entering
    self.active = CS.vw_iq_lvbs_alc_active

    if hud_control.audibleAlert == AudibleAlert.refuse:
      self.unavailable_hold = self.CCP.IQ_PQ_UNAVAILABLE_HUD_FRAMES
    else:
      self.unavailable_hold = max(0, self.unavailable_hold - 1)
    self.unavailable = self.unavailable_hold > 0

    CS.force_rhd_for_bsm = getattr(CC, "forceRHDForBSM", False)
    CS.enable_predicative_speed_limit = getattr(CC.cruiseControl, "speedLimitPredicative", False)
    CS.enable_pred_react_to_speed_limits = getattr(CC.cruiseControl, "speedLimitPredReactToSL", False)
    CS.enable_pred_react_to_curves = getattr(CC.cruiseControl, "speedLimitPredReactToCurves", False)

    if self.frame % self.CCP.STEER_STEP == 0:
      if self.CCS is pqcan and self._iq_lvbs_alc is None:
        # Stock torque steering needs APD even without the optional ALC bundle.
        # Send it first so Panda sees the current authorization before HCA.
        can_sends.append(pqcan.create_apd_control(self.packer_pt, self.CAN.aux, self.CP, CC_IQ.angleOffsetDeg,
                                                CC.latActive and aol_active))
      if self.CP.flags & (VolkswagenFlags.MEB | VolkswagenFlags.MQB_EVO):
        if CC.latActive:
          hca_enabled = True
          if CC.curvatureControllerActive:
            apply_curvature = self.LateralController.update(CS.out, CC, actuators.curvature)
            apply_curvature = apply_curvature + (CS.out.steeringCurvature - (CC.currentCurvature - CC.rollCompensation))
          else:
            apply_curvature = actuators.curvature + (CS.out.steeringCurvature - CC.currentCurvature)
          apply_curvature = self.curvature_smoother.update(apply_curvature, CS.out.vEgo)
          apply_curvature = apply_std_curvature_limits(apply_curvature, self.apply_curvature_last, CS.out.vEgoRaw, CS.out.steeringCurvature,
                                                       CS.out.steeringPressed, self.CCP.STEER_STEP, CC.latActive, self.CCP.CURVATURE_LIMITS)

          min_power = max(self.steering_power_last - self.CCP.STEERING_POWER_STEP, self.CCP.STEERING_POWER_MIN)
          max_power = min(self.steering_power_last + self.CCP.STEERING_POWER_STEP, self.CCP.STEERING_POWER_MAX)
          target_power_driver = int(np.interp(abs(CS.out.steeringTorque), [self.CCP.STEER_DRIVER_ALLOWANCE, self.CCP.STEER_DRIVER_MAX],
                                                                          [self.CCP.STEERING_POWER_MAX, self.CCP.STEERING_POWER_MIN]))
          target_power = int(np.interp(CS.out.vEgo, [0., 0.5], [self.CCP.STEERING_POWER_MIN, target_power_driver]))
          steering_power = min(max(target_power, min_power), max_power)
        else:
          if self.LateralController is not None:
            self.LateralController.reset()
          if self.steering_power_last > 0:
            hca_enabled = True
            handoff_curvature = self.apply_curvature_last + (CS.out.steeringCurvature - self.apply_curvature_last) * self.CCP.CURVATURE_HANDOFF_RATE
            apply_curvature = apply_std_curvature_limits(handoff_curvature, self.apply_curvature_last, CS.out.vEgoRaw, CS.out.steeringCurvature,
                                                         CS.out.steeringPressed, self.CCP.STEER_STEP, True, self.CCP.CURVATURE_LIMITS)
            steering_power = max(self.steering_power_last - self.CCP.STEERING_POWER_STEP, 0)
          else:
            hca_enabled = False
            apply_curvature = 0.
            steering_power = 0
          self.curvature_smoother.reset(apply_curvature)

        can_sends.append(self.CCS.create_steering_control(self.packer_pt, self.CAN.pt, apply_curvature, hca_enabled, steering_power))
        self.apply_curvature_last = apply_curvature
        self.steering_power_last = steering_power
      else:
        # A paused AOL session is still enabled; only active AOL authorizes torque
        # without cruise. Keep the immediate cutoff when both authorities drop.
        if CC.latActive and (CC.enabled or aol_active) and not AngleLateralControl:
          torque_scale = self._get_mqb_steering_torque_scale(CS.out.vEgo, iq_mqb_steering_lockout)
          new_torque = int(round(actuators.torque * self.CCP.STEER_MAX * torque_scale))
          apply_torque = apply_driver_steer_torque_limits(new_torque, self.apply_torque_last, CS.out.steeringTorque, self.CCP)
          self.hca_frame_timer_running += self.CCP.STEER_STEP
          if self.apply_torque_last == apply_torque:
            self.hca_frame_same_torque += self.CCP.STEER_STEP
            if self.hca_frame_same_torque > self.CCP.STEER_TIME_STUCK_TORQUE / DT_CTRL:
              apply_torque -= (1, -1)[apply_torque < 0]
              self.hca_frame_same_torque = 0
          else:
            self.hca_frame_same_torque = 0
          hca_enabled = abs(apply_torque) > 0
          if self.eps_timer_workaround and self.hca_frame_timer_running >= self.CCP.STEER_TIME_BM / DT_CTRL:
            if abs(apply_torque) <= self.CCP.STEER_LOW_TORQUE:
              self.hca_frame_low_torque += self.CCP.STEER_STEP
              if self.hca_frame_low_torque >= self.CCP.STEER_TIME_LOW_TORQUE / DT_CTRL:
                hca_enabled = False
            else:
              self.hca_frame_low_torque = 0
              if self.hca_frame_timer_resetting > 0:
                apply_torque = 0
        else:
          self.hca_frame_low_torque = 0
          hca_enabled = False
          apply_torque = 0

        if hca_enabled:
          output_torque = apply_torque
          self.hca_frame_timer_resetting = 0
        else:
          output_torque = 0
          self.hca_frame_timer_resetting += self.CCP.STEER_STEP
          if self.hca_frame_timer_resetting >= self.CCP.STEER_TIME_RESET / DT_CTRL or not self.eps_timer_workaround:
            if self.hca_frame_timer_running > 0:
              self.hca_reset_count += 1
            self.hca_frame_timer_running = 0
            apply_torque = 0

        if hca_enabled and abs(apply_torque) > 0:
          if pqhca5or7Toggle and (self.CP.flags & (VolkswagenFlags.PQ | VolkswagenFlags.MLB)):
            self.HCA_Status = 7
          else:
            self.HCA_Status = 5
        else:
          self.HCA_Status = 3

        self.eps_timer_soft_disable_alert = self.eps_timer_workaround and (
          self.hca_frame_timer_running > self.CCP.STEER_TIME_ALERT / DT_CTRL or CS.eps_session.age_s > self.CCP.STEER_TIME_ALERT)
        self.apply_torque_last = apply_torque
        if not (AngleLateralControl and self.CCS in (mqbcan, pqcan, mlbcan)):
          can_sends.append(self.CCS.create_hca_steering_control(self.packer_pt, self._pt_tx_bus, output_torque, self.HCA_Status))

      if self.CP.flags & VolkswagenFlags.STOCK_HCA_PRESENT and self.CCS == mqbcan:
        ea_simulated_torque = float(np.clip(apply_torque * 2, -self.CCP.STEER_MAX, self.CCP.STEER_MAX))
        if abs(CS.out.steeringTorque) > abs(ea_simulated_torque):
          ea_simulated_torque = CS.out.steeringTorque
        can_sends.append(self.CCS.create_eps_update(self.packer_pt, self.CAN.cam, CS.eps_stock_values, ea_simulated_torque))

    if self._iq_lvbs_alc:
      self._iq_lvbs_alc.update_vw_alc(self, CC, CS, actuators, can_sends, apply_torque)
    if self._iq_lvbs_alc and self.frame % self.CCP.STEER_STEP == 0:
      self._iq_lvbs_alc.append_private_apd(self, CC_IQ, can_sends)

    if self.CP.flags & (VolkswagenFlags.MEB | VolkswagenFlags.MQB_EVO) and self.CP.flags & VolkswagenFlags.STOCK_KLR_PRESENT:
      if CS.klr_stock_values:
        klr_send_ready = CS.klr_stock_values["COUNTER"] != self.klr_counter_last
        if klr_send_ready:
          can_sends.append(mebcan.create_capacitive_wheel_touch(self.packer_pt, self.CAN.cam, CC.latActive, CS.klr_stock_values))
          can_sends.append(mebcan.create_capacitive_wheel_touch(self.packer_pt, self.CAN.pt, CC.latActive, CS.klr_stock_values))
        self.klr_counter_last = CS.klr_stock_values["COUNTER"]

    if self.CP.flags & (VolkswagenFlags.MEB | VolkswagenFlags.MQB_EVO):
      if CS.ea_hud_stock_values and self.frame % EA_BLINKER_STEP == 0:
        self.ea_tx_counter = next_ea_counter(self.ea_tx_counter, CS.ea_hud_stock_values["COUNTER"])
        left_blinker, right_blinker = ea_blinker_command(
          CC.leftBlinker, CC.rightBlinker, CS.left_blinker_active, CS.right_blinker_active,
        )
        can_sends.append(self.CCS.create_blinker_control(self.packer_pt, self.CAN.pt, CS.ea_hud_stock_values, CS.ea_control_stock_values,
                                                         left_blinker, right_blinker, self.hide_ea_error, self.ea_tx_counter))
        self.ea_counter_last = CS.ea_hud_stock_values["COUNTER"]

    if self.CP.openpilotLongitudinalControl and self.CCS in (mqbcan, mlbcan):
      seed_msgs = ("ACC_01", "ACC_02") if self.CCS is mlbcan else ("ACC_02", "ACC_06", "ACC_07", "ACC_10")
      for name in seed_msgs:
        if name in CS.acc_stock_counters and name not in self.acc_counters_seeded:
          addr = self.packer_pt.dbc.name_to_msg[name].address
          self.packer_pt.counters[addr] = (CS.acc_stock_counters[name] + 1) % 16
          self.acc_counters_seeded.add(name)

    if self.frame % self.CCP.ACC_CONTROL_STEP == 0 and self.CP.openpilotLongitudinalControl and not CS.out.radarDisableFailed:
      stopping = actuators.longControlState == LongCtrlState.stopping
      if self.CP.flags & (VolkswagenFlags.MEB | VolkswagenFlags.MQB_EVO):
        long_override = CC.cruiseControl.override or CS.out.gasPressed
        starting = mebcan.acc_starting(CC.longActive, actuators.longControlState, actuators.accel,
                                       CS.esp_hold_confirmation, CS.out.vEgo, long_override)
        accel = float(np.clip(actuators.accel, self.CCP.ACCEL_MIN, self.CCP.ACCEL_MAX) if CC.enabled else 0)

        stopping, starting, esp_hold, accel = self.long_stop_hold.update(
          CC.longActive, stopping, starting, accel, CS.esp_hold_confirmation, CS.out.vEgo, long_override)

        jerk_up, jerk_down, band_oben, band_unten = longitudinal_comfort_values(CC, CC_IQ, legacy_band=0.0)

        acc_control = self.CCS.acc_control_value(CS.out.cruiseState.available, CS.out.accFaulted, CC.enabled, long_override)
        acc_hold_type, self.acc_hold_ramp_counter = self.CCS.acc_hold_type(
          CS.out.cruiseState.available, CS.out.accFaulted, CC.enabled, starting, stopping,
          esp_hold, CS.out.vEgo, self.acc_hold_type_last, self.acc_hold_ramp_counter)
        self.acc_hold_type_last = acc_hold_type
        can_sends.extend(self.CCS.create_acc_accel_control(
          self.packer_pt, self.CAN.pt, self.CP, CS.acc_type, CC.enabled,
          jerk_up, jerk_down, band_oben, band_unten,
          accel, acc_control, acc_hold_type, stopping, starting, esp_hold,
          CS.out.vEgoRaw * CV.MS_TO_KPH, long_override, CS.travel_assist_available,
        ))
        self.accel_last = accel
      else:
        stopping = actuators.longControlState == LongCtrlState.stopping
        starting = actuators.longControlState == LongCtrlState.pid and (CS.esp_hold_confirmation or CS.out.vEgo < 0.25)
        long_active = CC.longActive
        accel = actuators.accel
        esp_override = None

        if self.mqb_long_state is not None:
          long_active, accel, stopping, starting, esp_override = self.mqb_long_state.update(CS, CC)

        accel = accel_during_driver_override(accel, CS.out.gasPressed, self.CP_IQ.longActiveWithGasOverride)

        acc_control = self.CCS.acc_control_value(CS.out.cruiseState.available, long_active, CC.cruiseControl.override, CS.out.accFaulted)
        accel = float(np.clip(accel, self.CCP.ACCEL_MIN, self.CCP.ACCEL_MAX) if (long_active or CC.cruiseControl.override) else 0)
        self.accel_last = accel

        use_radar_accel = blend_active and getattr(CC_IQ, "useRadarAccel", False) and CS.acc_radar_sta_adr == 1 and not self.radar_handler.failed
        if use_radar_accel:
          accel = float(np.clip(CS.acc_radar_sollbeschl, self.CCP.ACCEL_MIN, self.CCP.ACCEL_MAX))
          self.accel_last = accel

        if self.CCS == mqbcan:
          jerk_up, jerk_down, band_oben, band_unten = longitudinal_comfort_values(CC, CC_IQ, legacy_band=0.2)
          can_sends.extend(self.CCS.create_acc_accel_control(
            self.packer_pt, self.CAN.pt, CS.acc_type, accel, acc_control, stopping, starting, CS.esp_hold_confirmation,
            eBrakeActive, esp_override=esp_override, upper_jerk=jerk_up, lower_jerk=jerk_down,
            upper_control_limit=band_oben, lower_control_limit=band_unten,
          ))
        elif self.CCS == mlbcan:
          can_sends.extend(self.CCS.create_acc_accel_control(self.packer_pt, self.CAN.pt, accel, acc_control, stopping))
        else:
          if use_radar_accel:
            long_jerklimit, long_deviation = CS.acc_radar_aendgrad, CS.acc_radar_regelabw
          else:
            long_jerklimit = float(interp(accel, [-1.5, -0.3, 0.05], [2.0, 0.52, 0.30]))
            long_deviation = float(interp(accel, [-0.3, 0.1], [0.0, 0.20]))
          accel = apply_pq_stopping_accel(self.CP.carFingerprint, accel, stopping)

          sng_ecd_enabled = bool(self.CP.flags & VolkswagenFlagsIQ.IQ_PQ_SNG_ECD)
          low_speed = CS.out.vEgo <= self.CCP.SNG_HANDOFF_SPEED
          if sng_ecd_enabled and long_active and low_speed and accel <= 0.05 and not starting:
            self.sng_handoff_active = True
          else:
            self.sng_handoff_active = False
          sng_decel_req = float(np.clip(accel, self.CCP.ACCEL_MIN, self.CCP.SNG_HOLD_DECEL_MAX)) if self.sng_handoff_active else 0.0

          can_sends.extend(self.CCS.create_acc_accel_control(self.packer_pt, self.CAN.pt, CS.acc_type, accel, acc_control, stopping and not self.motor3_resuming, starting, CS.esp_hold_confirmation, long_deviation, long_jerklimit, eBrakeActive, sng_active=self.sng_handoff_active))
          if sng_ecd_enabled:
            can_sends.append(self.CCS.create_sng_handoff_control(self.packer_pt, self.CAN.aux, self.sng_handoff_active, sng_decel_req))

    if (self.CP.flags & VolkswagenFlags.DISABLE_RADAR) and self.CP.openpilotLongitudinalControl and not CS.out.radarDisableFailed:
      if self.CP.flags & (VolkswagenFlags.MEB | VolkswagenFlags.MQB_EVO):
        if self.radar_disabled_warning_timer < 600:
          self.radar_disabled_warning_timer += 1
        else:
          self.hide_ea_error = True

        if self.frame % self.CCP.AEB_CONTROL_STEP == 0:
          can_sends.append(make_tester_present_msg(0x700, self.CAN.pt, suppress_response=True))
          can_sends.append(self.CCS.create_aeb_control(self.packer_pt, self.CAN.pt, self.CP))

        if self.frame % self.CCP.AEB_HUD_STEP == 0 and self._has_aeb_hud_msg:
          can_sends.append(self.CCS.create_aeb_hud(self.packer_pt, self.CAN.pt, self.radar_disabled_warning_timer < 600))

        if self.frame % 4 == 0:
          can_sends.append(self.CCS.create_radar_objects(self.packer_pt, self.CAN.pt))

    if self.frame % self.CCP.LDW_STEP == 0:
      hud_alert = 0
      if hud_control.visualAlert in (VisualAlert.steerRequired, VisualAlert.ldw) or CS.out.steerFaultTemporary:
        hud_alert = self.CCP.LDW_MESSAGES["laneAssistTakeOver"]
      steering_pressed_hud = (self.frame // 2) % 2 == 0 if self.entering else CS.out.steeringPressed
      if self.CP.flags & (VolkswagenFlags.MEB | VolkswagenFlags.MQB_EVO):
        disable_alerts = getattr(CC, "disableCarSteerAlerts", False)
        sound_alert = self.CCP.LDW_SOUNDS["Chime"] if hud_alert != 0 and not disable_alerts else self.CCP.LDW_SOUNDS["None"]
        can_sends.append(self.CCS.create_lka_hud_control(self.packer_pt, self.CAN.pt, CS.ldw_stock_values, CC.latActive, CS.out.steeringPressed,
                                                         hud_alert, hud_control, sound_alert))
      else:
        can_sends.append(self.CCS.create_lka_hud_control(self.packer_pt, self._pt_tx_bus, CS.ldw_stock_values, CC.latActive, steering_pressed_hud,
                                                         hud_alert, hud_control, self.entering, self.CCS is pqcan and AngleLateralControl, self.active))

    if hud_control.leadDistanceBars != self.lead_distance_bars_last:
      self.distance_bar_frame = self.frame

    hud_engaged = CC.enabled and CS.out.cruiseState.available and not CS.out.accFaulted
    hud_override = hud_engaged and (CC.cruiseControl.override or CS.out.gasPressed)
    lead_visible, priority_boost = self.acc_hud_state.update(self.frame, hud_engaged, hud_control.leadVisible,
                                                          hud_control.leadDistanceBars)

    if self.frame % self.CCP.ACC_HUD_STEP == 0 and self.CP.openpilotLongitudinalControl:
      fcw_alert = hud_control.visualAlert == VisualAlert.fcw
      d_unresponsive = hud_control.driverUnresponsive
      if self.CP.flags & (VolkswagenFlags.MEB | VolkswagenFlags.MQB_EVO):
        show_distance_bars = self.frame - self.distance_bar_frame < 400
        gap = max(8, CS.out.vEgo * hud_control.leadFollowTime)
        distance = max(8, hud_control.leadDistance) if hud_control.leadDistance != 0 else 0
        acc_hud_status = self.CCS.acc_hud_status_value(CS.out.cruiseState.available, CS.out.accFaulted, CC.enabled,
                                                       CC.cruiseControl.override or CS.out.gasPressed)

        sl_predicative_active = CC.cruiseControl.speedLimitPredicative and CS.out.cruiseState.speedLimitPredicative != 0
        if CC.cruiseControl.speedLimit and CS.out.cruiseState.speedLimit != 0 and self.speed_limit_last != CS.out.cruiseState.speedLimit:
          self.speed_limit_changed_timer = self.frame
        self.speed_limit_last = CS.out.cruiseState.speedLimit
        sl_active = self.frame - self.speed_limit_changed_timer < 400
        speed_limit = CS.out.cruiseState.speedLimitPredicative if sl_predicative_active else (CS.out.cruiseState.speedLimit if sl_active else 0)

        acc_hud_event = self.CCS.acc_hud_event(acc_hud_status, CS.esp_hold_confirmation, sl_predicative_active,
                                               CS.speed_limit_predicative_type, sl_active)

        can_sends.append(self.CCS.create_acc_hud_control(self.packer_pt, self.CAN.pt, acc_hud_status, hud_control.setSpeed * CV.MS_TO_KPH,
                                                         hud_control.leadVisible, hud_control.leadDistanceBars + 1, show_distance_bars,
                                                         CS.esp_hold_confirmation, distance, gap, fcw_alert, acc_hud_event, speed_limit))
      else:
        # MLB scales the raw lead distance against the set follow gap in the packer, MQB and PQ send a headway index
        leadDistance = hud_control.leadDistance if self.CCS is mlbcan else \
          lead_headway_index(hud_control.leadDistance, CS.out.vEgo, CS.upscale_lead_car_signal)
        self.leadDistanceBars = min(3, hud_control.leadDistanceBars)
        acc_hud_status = self.CCS.acc_hud_status_value(CS.out.cruiseState.available, CS.out.accFaulted, CC.longActive and hud_engaged,
                                                       hud_override)
        set_speed = hud_control.setSpeed * CV.MS_TO_KPH
        decel = dVisual(self.CCS, CS)
        hud_kwargs = {"hud_text": self._mlb_acc_hud_text(hud_control, set_speed),
                      "desired_distance": max(8.0, CS.out.vEgo * hud_control.leadFollowTime)} if self.CCS is mlbcan else {}
        hud_kwargs["priority_boost"] = priority_boost
        can_sends.append(self.CCS.create_acc_hud_control(self.packer_pt, self.CAN.pt, acc_hud_status, set_speed, leadDistance,
                                                         self.leadDistanceBars, fcw_alert, lead_visible or fcw_alert, self.unavailable,
                                                         decel, d_unresponsive, **hud_kwargs))

    if self._iq_lvbs_alc and self.CP.flags & VolkswagenFlags.PQ:
      self._iq_lvbs_alc.update_turn_signals(self, CC, CS, can_sends)

    if self.CP.openpilotLongitudinalControl and (self.CP.flags & VolkswagenFlags.PQ):
      if blend_active:
        can_sends.extend(self.radar_handler.update(
          self.packer_pt, self.frame, CS,
          blend_active=True,
          engage_req=getattr(CC_IQ, "radarEngageReq", False),
          cancel_req=getattr(CC_IQ, "radarCancelReq", False),
          set_speed_kph=getattr(CC_IQ, "radarSetSpeedKph", 0.0),
          gap_bars=getattr(CC_IQ, "radarGapBars", 0),
          v_ego=CS.out.vEgo,
        ))
      elif self.frame % 2 == 0:
        can_sends.append(self.CCS.filter_motor2(self.packer_pt, self.CAN.ext, CS.motor2_stock))
        can_sends.append(self.CCS.filter_motor5(self.packer_pt, self.CAN.ext, CS.motor5_stock))

    gra_send_ready = self.CP.pcmCruise and CS.gra_stock_values["COUNTER"] != self.gra_acc_counter_last
    if self.CP.flags & (VolkswagenFlags.MEB | VolkswagenFlags.MQB_EVO):
      main_cruise_latching = not bool(CS.gra_stock_values["GRA_Typ_Hauptschalter"])
      stock_cancel_pressed = bool(CS.gra_stock_values["GRA_Abbrechen"] if main_cruise_latching else CS.gra_stock_values["GRA_Hauptschalter"])
    elif self.CP.flags & VolkswagenFlags.MLB:
      stock_cancel_pressed = bool(CS.gra_stock_values["LS_Abbrechen"])
    else:
      stock_cancel_pressed = bool(CS.gra_stock_values["GRA_Abbrechen"])

    cancel_cmd = stock_cancel_pressed or self._tap_gra_cancel(CC.cruiseControl.cancel, gra_send_ready)
    stalk_resume_allowed = not (self.CP.openpilotLongitudinalControl and self.CP.flags & (VolkswagenFlags.MEB | VolkswagenFlags.MQB_EVO))
    resume_cmd = (CC.cruiseControl.resume and stalk_resume_allowed) or self._should_spam_mqb_a0_resume(CS, iq_mqb_acc_resume)
    if gra_send_ready and (cancel_cmd or resume_cmd):
      stalk_on_powertrain = self.CP.flags & VolkswagenFlags.PQ or self.CP.flags & VolkswagenFlagsIQ.IQ_MLB_NO_ECAN
      bus_send = self.CAN.aux if stalk_on_powertrain else self.CAN.ext
      can_sends.append(self.CCS.create_acc_buttons_control(self.packer_pt, bus_send, CS.gra_stock_values,
                                                           cancel=cancel_cmd, resume=resume_cmd))

    if self.CP.openpilotLongitudinalControl and self.CCS == pqcan and not blend_active:
      if self.frame % 3:
        can_sends.append(self.CCS.create_gra_neu(self.packer_pt, self.CAN.ext, CS.gra_stock_values, CC.longActive))

    if self.CP.flags & VolkswagenFlagsIQ.IQ_PQ_ACC_FTS_EPB:
      is_stopping = actuators.longControlState == LongCtrlState.stopping
      if CS.out.vEgo > 0.5 or not CC.longActive:
        self.motor3_resuming = False
      elif (self.motor3_was_stopping and not is_stopping and self.CP.openpilotLongitudinalControl and CC.longActive):
        self.motor3_resuming = True
      if self.motor3_resuming and CS.motor3_stock:
        can_sends.append(self.CCS.create_motor3_resume(self.packer_pt, self.CAN.aux, CS.motor1_stock, CS.motor3_stock, resume=True))
      self.motor3_was_stopping = is_stopping

    new_actuators = actuators.as_builder()
    new_actuators.torque = self.apply_torque_last / self.CCP.STEER_MAX
    new_actuators.torqueOutputCan = self.apply_torque_last
    if self.CP.steerControlType == structs.CarParams.SteerControlType.angle:
      new_actuators.steeringAngleDeg = float(self.apply_angle_last)
    new_actuators.curvature = float(self.apply_curvature_last)
    new_actuators.accel = self.accel_last
    new_actuators.speed = float(getattr(actuators, "speed", 0.0))

    self.lead_distance_bars_last = hud_control.leadDistanceBars
    self.gra_acc_counter_last = CS.gra_stock_values["COUNTER"]
    self.motor3_frame_last = CS.motor3_frame
    self.frame += 1
    return new_actuators, can_sends
