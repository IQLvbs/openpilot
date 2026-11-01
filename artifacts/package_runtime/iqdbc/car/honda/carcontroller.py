import numpy as np
import math

from iqdbc.can import CANPacker
from iqdbc.car import ACCELERATION_DUE_TO_GRAVITY, Bus, DT_CTRL, rate_limit, make_tester_present_msg, structs
from iqdbc.car.common.pid import PIDController
from iqdbc.car.honda import dash_lane, dash_objects, hondacan
from iqdbc.car.honda.values import CAR, CruiseButtons, CruiseSettings, HONDA_BOSCH, HONDA_BOSCH_CANFD, HONDA_BOSCH_RADARLESS, \
                                     HONDA_BOSCH_TJA_CONTROL, HONDA_NIDEC_ALT_PCM_ACCEL, CarControllerParams
from iqdbc.car.interfaces import CarControllerBase

from iqdbc.lvbs.car.honda.aol import AolCarController
from iqdbc.lvbs.car.honda.gas_interceptor import GasInterceptorCarController

VisualAlert = structs.CarControl.HUDControl.VisualAlert
LongCtrlState = structs.CarControl.Actuators.LongControlState


def compute_gb_honda_bosch(accel, speed):
  # TODO returns 0s, is unused
  return 0.0, 0.0


def compute_gb_honda_nidec(accel, speed):
  creep_brake = 0.0
  creep_speed = 2.3
  creep_brake_value = 0.15
  if speed < creep_speed:
    creep_brake = (creep_speed - speed) / creep_speed * creep_brake_value
  gb = float(accel) / 4.8 - creep_brake
  return np.clip(gb, 0.0, 1.0), np.clip(-gb, 0.0, 1.0)


def compute_gas_brake(accel, speed, fingerprint):
  if fingerprint in HONDA_BOSCH:
    return compute_gb_honda_bosch(accel, speed)
  else:
    return compute_gb_honda_nidec(accel, speed)


# TODO not clear this does anything useful
def actuator_hysteresis(brake, braking, brake_steady, v_ego, car_fingerprint):
  # hyst params
  brake_hyst_on = 0.02    # to activate brakes exceed this value
  brake_hyst_off = 0.005  # to deactivate brakes below this value
  brake_hyst_gap = 0.01   # don't change brake command for small oscillations within this value

  # *** hysteresis logic to avoid brake blinking. go above 0.1 to trigger
  if (brake < brake_hyst_on and not braking) or brake < brake_hyst_off:
    brake = 0.
  braking = brake > 0.

  # for small brake oscillations within brake_hyst_gap, don't change the brake command
  if brake == 0.:
    brake_steady = 0.
  elif brake > brake_steady + brake_hyst_gap:
    brake_steady = brake - brake_hyst_gap
  elif brake < brake_steady - brake_hyst_gap:
    brake_steady = brake + brake_hyst_gap
  brake = brake_steady

  return brake, braking, brake_steady


def brake_pump_hysteresis(apply_brake, apply_brake_last, last_pump_ts, ts):
  pump_on = False

  # reset pump timer if:
  # - there is an increment in brake request
  # - we are applying steady state brakes and we haven't been running the pump
  #   for more than 20s (to prevent pressure bleeding)
  if apply_brake > apply_brake_last or (ts - last_pump_ts > 20. and apply_brake > 0):
    last_pump_ts = ts

  # once the pump is on, run it for at least 0.2s
  if ts - last_pump_ts < 0.2 and apply_brake > 0:
    pump_on = True

  return pump_on, last_pump_ts


def process_hud_alert(hud_alert):
  alert_fcw = False
  alert_steer_required = False

  # Make sure FCW is prioritized over steering required
  # TODO: implement separate available LDW alert
  if hud_alert == VisualAlert.fcw:
    alert_fcw = True
  elif hud_alert in (VisualAlert.steerRequired, VisualAlert.ldw):
    alert_steer_required = True

  return alert_fcw, alert_steer_required


class CarController(CarControllerBase, AolCarController, GasInterceptorCarController):
  def __init__(self, dbc_names, CP, CP_IQ):
    CarControllerBase.__init__(self, dbc_names, CP, CP_IQ)
    AolCarController.__init__(self)
    GasInterceptorCarController.__init__(self, CP, CP_IQ)
    self.packer = CANPacker(dbc_names[Bus.pt])
    self.params = CarControllerParams(CP)
    self.CAN = hondacan.CanBus(CP)
    self.tja_control = CP.carFingerprint in HONDA_BOSCH_TJA_CONTROL

    self.lane_renderer = dash_lane.LanePathRenderer()
    self.dash_object_author = dash_objects.DashObjectAuthor()
    self.rendered_lane = dash_lane.RenderedLane()
    self.lkas_hud_key = None
    self.lkas_state_change_frames = 0

    self.braking = False
    self.brake_steady = 0.
    self.brake_last = 0.
    self.apply_brake_last = 0
    self.last_pump_ts = 0.
    self.stopping_counter = 0

    self.accel = 0.0
    self.speed = 0.0
    self.gas = 0.0
    self.brake = 0.0
    self.last_torque = 0.0
    self.bosch_last_gas = 0

    self.lkas_button_send_remaining = 0
    self.last_lkas_button_frame = 0
    self.radar_disable_counter = 0
    self.radar_mux = 0
    # stock RADAR_HUD_CANFD raises its CMBS bit only for a short burst after ACC engages; 10Hz hud ticks
    self.radar_hud_pulse = 0
    self.last_acc_enabled = False

    self.gasfactor = 1.0
    self.gasfactor_before_maxgas = 1.0
    self.windfactor = 1.0
    self.windfactor_before_maxgas = 1.0
    self.windfactor_before_brake = 0.0
    self.pitch = 0.0

    self.brake_pid = PIDController(k_p=0.0, k_i=1.0, pos_limit=0.0, neg_limit=-2.0, rate=50)
    self.brake_pid.reset()

  def update(self, CC, CC_IQ, CS, now_nanos):
    AolCarController.update(self, self.CP, CC, CC_IQ)
    gas_pedal_force = 0.0
    min_gas = self.params.BOSCH_GAS_LOOKUP_BP[0]
    actuators = CC.actuators
    hud_control = CC.hudControl
    hud_v_cruise = hud_control.setSpeed / CS.v_cruise_factor if hud_control.speedVisible else 255
    pcm_cancel_cmd = CC.cruiseControl.cancel

    if len(CC.orientationNED) == 3:
      self.pitch = CC.orientationNED[1]
    hill_brake = math.sin(self.pitch) * ACCELERATION_DUE_TO_GRAVITY

    if CC.longActive:
      accel = actuators.accel
      gas, brake = compute_gas_brake(actuators.accel + hill_brake, CS.out.vEgo, self.CP.carFingerprint)
    else:
      accel = 0.0
      gas, brake = 0.0, 0.0

    # *** rate limit steer ***
    limited_torque = rate_limit(actuators.torque, self.last_torque, -self.params.STEER_DELTA_DOWN * DT_CTRL,
                                self.params.STEER_DELTA_UP * DT_CTRL)
    self.last_torque = limited_torque

    # *** apply brake hysteresis ***
    pre_limit_brake, self.braking, self.brake_steady = actuator_hysteresis(brake, self.braking, self.brake_steady,
                                                                           CS.out.vEgo, self.CP.carFingerprint)

    # *** rate limit after the enable check ***
    self.brake_last = rate_limit(pre_limit_brake, self.brake_last, -2., 3 * DT_CTRL)

    # vehicle hud display, wait for one update from 10Hz 0x304 msg
    alert_fcw, alert_steer_required = process_hud_alert(hud_control.visualAlert)

    # **** process the car messages ****

    # steer torque is converted back to CAN reference (positive when steering right)
    apply_torque = int(np.interp(-limited_torque * self.params.STEER_MAX,
                                 self.params.STEER_LOOKUP_BP, self.params.STEER_LOOKUP_V))

    # Send CAN commands
    can_sends = []

    if self.CP.carFingerprint in (HONDA_BOSCH - HONDA_BOSCH_RADARLESS) and self.CP.openpilotLongitudinalControl:
      if self.CP.carFingerprint in HONDA_BOSCH_CANFD and CS.stock_acc_alive:
        # CAN FD: the radar is silenced from here rather than from CarInterface.init(), and only once
        # the comma relay is confirmed open: init() ran under the ELM327 safety mode, so the
        # replacement ACC_CONTROL stream was blocked until the safety-mode switch landed, and whenever
        # that took longer than ~110ms after radar silence the brake module latched CRUISE_FAULT for
        # the whole drive. With the relay open the replacement stream starts within a few frames of
        # radar silence (see CS.stock_acc_alive), well inside the fault threshold
        if CS.canfd_relay_open:
          if self.radar_disable_counter % 50 == 0:
            # UDS extended diagnostic session, required before CommunicationControl
            can_sends.append((0x18DAB0F1, b'\x02\x10\x03\x00\x00\x00\x00\x00', self.CAN.pt))
          elif self.radar_disable_counter % 50 == 5:
            # UDS CommunicationControl disableRxAndTx (0x80 suppresses the response), retried every
            # 0.5s until the radar goes silent
            can_sends.append((0x18DAB0F1, b'\x03\x28\x83\x03\x00\x00\x00\x00', self.CAN.pt))
          self.radar_disable_counter += 1
      elif self.frame % 10 == 0:
        # tester present - w/ no response (keeps radar disabled)
        can_sends.append(make_tester_present_msg(0x18DAB0F1, self.CAN.pt, suppress_response=True))

    # simulate the disabled canfd radar to prevent faults. These look-alikes are consumed by both the
    # camera (behind the relay, on the camera bus) and the powertrain: openpilot's own TX is not
    # forwarded across the open relay, so each frame is packed exactly once (the packer's
    # counter/checksum only advance once per cycle) and the identical bytes are mirrored onto both
    # buses (re-packing would double-increment the counter and desync the buses). While the stock
    # radar is still transmitting it authors all of these itself
    if self.CP.carFingerprint in HONDA_BOSCH_CANFD and self.CP.openpilotLongitudinalControl and not CS.stock_acc_alive:
      if CC.enabled and not self.last_acc_enabled:
        self.radar_hud_pulse = 30  # ~3s at 10Hz, matching the stock 2-6s engage burst
      self.last_acc_enabled = CC.enabled
      radar_msgs = []
      if CS.hud_tick:
        radar_msgs.append(hondacan.create_radar_hud_canfd(self.packer, self.CAN.pt, CC.enabled, self.radar_hud_pulse > 0))
        if self.radar_hud_pulse > 0:
          self.radar_hud_pulse -= 1
      if CS.supp_tick:
        radar_msgs.append(hondacan.create_canfd_supplemental(self.packer, self.CAN.pt))
      if CS.radar_50hz_tick:
        # Cycle the radar MUX through the stock banks: 1-10, 17-26, 33-42, 49-58. This counter also
        # drives the LANE_PATH/HUD_OBJECTS mux below: it advances exactly one step per transmitted
        # frame, so the sweep stays contiguous even when a tick is missed (a frame-derived mux left
        # holes in the sweep the stock radar never produces).
        # These must be elif: a bare `if` at a bank start would fall through to the increment,
        # skipping the bank-start values (17, 33, 49)
        if self.radar_mux >= 58:
          self.radar_mux = 1
        elif self.radar_mux == 10:
          self.radar_mux = 17
        elif self.radar_mux == 26:
          self.radar_mux = 33
        elif self.radar_mux == 42:
          self.radar_mux = 49
        else:
          self.radar_mux += 1
      if CS.radar_5hz_tick:
        # RADAR_LEAD's LANE_PATH_LENGTH must track the valid-point count of the LANE_PATH sweep being
        # authored, and LEFT_LANE/RIGHT_LANE the per-side line-detected status, in lockstep with the
        # stock radar's behavior or the dash won't draw the lane lines
        radar_msgs.extend(hondacan.create_canfd_5hz_radar_messages(self.packer, self.CAN.pt, CS.radar_ref_counter,
                                                                   dash_lane.canfd_lane_length(self.rendered_lane),
                                                                   dash_lane.LANE_LINE_ON if self.rendered_lane.left_line else 0,
                                                                   dash_lane.LANE_LINE_ON if self.rendered_lane.right_line else 0))

      for addr, dat, _ in radar_msgs:
        can_sends.append((addr, dat, self.CAN.pt))
        can_sends.append((addr, dat, self.CAN.camera))

    # Send steering command.
    can_sends.append(hondacan.create_steering_control(self.packer, self.CAN, apply_torque, CC.latActive, self.tja_control))

    # wind brake from air resistance decel at high speed
    wind_brake = np.interp(CS.out.vEgo, [0.0, 2.3, 35.0], [0.001, 0.002, 0.15]) * self.windfactor  # not in m/s2 units
    wind_brake_ms2 = np.interp(CS.out.vEgo, [0.0, 13.4, 22.4, 31.3, 40.2], [0.000, 0.049, 0.136, 0.267, 0.441])  # in m/s2 units
    # all of this is only relevant for HONDA NIDEC
    max_accel = np.interp(CS.out.vEgo, self.params.NIDEC_MAX_ACCEL_BP, self.params.NIDEC_MAX_ACCEL_V)
    # TODO this 1.44 is just to maintain previous behavior
    pcm_speed_BP = [-wind_brake,
                    -wind_brake * (3 / 4),
                    0.0,
                    0.5]
    # The Honda ODYSSEY seems to have different PCM_ACCEL
    # msgs, is it other cars too?
    if self.CP_IQ.enableGasInterceptor or not CC.longActive:
      pcm_speed = 0.0
      pcm_accel = int(0.0)
    elif self.CP.carFingerprint in HONDA_NIDEC_ALT_PCM_ACCEL:
      pcm_speed_V = [0.0,
                     np.clip(CS.out.vEgo - 3.0, 0.0, 100.0),
                     np.clip(CS.out.vEgo + 0.0, 0.0, 100.0),
                     np.clip(CS.out.vEgo + 5.0, 0.0, 100.0)]
      pcm_speed = float(np.interp(gas - brake, pcm_speed_BP, pcm_speed_V))
      pcm_accel = int(1.0 * self.params.NIDEC_GAS_MAX)
    else:
      pcm_speed_V = [0.0,
                     np.clip(CS.out.vEgo - 2.0, 0.0, 100.0),
                     np.clip(CS.out.vEgo + 2.0, 0.0, 100.0),
                     np.clip(CS.out.vEgo + 5.0, 0.0, 100.0)]
      pcm_speed = float(np.interp(gas - brake, pcm_speed_BP, pcm_speed_V))
      pcm_accel = int(np.clip((accel / 1.44) / max_accel, 0.0, 1.0) * self.params.NIDEC_GAS_MAX)

    if not self.CP.openpilotLongitudinalControl:
      if self.frame % 2 == 0 and self.CP.carFingerprint not in HONDA_BOSCH_RADARLESS | HONDA_BOSCH_CANFD:
        can_sends.append(hondacan.create_bosch_supplemental_1(self.packer, self.CAN))
      # If using stock ACC, spam cancel command to kill gas when OP disengages.
      if pcm_cancel_cmd:
        can_sends.append(hondacan.spam_buttons_command(self.packer, self.CAN, CruiseButtons.CANCEL, 0, CS.scm_ambient_light,
                                                       self.CP.carFingerprint))
      elif CC.cruiseControl.resume:
        can_sends.append(hondacan.spam_buttons_command(self.packer, self.CAN, CruiseButtons.RES_ACCEL, 0, CS.scm_ambient_light,
                                                       self.CP.carFingerprint))

    else:
      # Send gas and brake commands.
      if self.frame % 2 == 0:
        ts = self.frame * DT_CTRL

        if self.CP.carFingerprint in HONDA_BOSCH:
          # low-speed extra brake: the fixed accel command under-delivers approaching a stop, so an
          # integral-only term closes the gap, releasing at 1 m/s^3 once out of the window
          if (accel < min_gas) and (CS.out.vEgo < 3.0) and not (-1e-3 < CS.out.vEgo < 1e-3):
            brake_addon = self.brake_pid.update(error=accel - CS.out.aEgo, speed=CS.out.vEgo)
            target_accel = min(accel, accel + brake_addon)
          else:
            if (self.brake_pid.i < 0.0) and (accel < min_gas):
              self.brake_pid.i = min(0.0, self.brake_pid.i + 0.02)
            else:
              self.brake_pid.reset()
            target_accel = min(accel, accel + self.brake_pid.i)

          self.accel = float(np.clip(target_accel, self.params.BOSCH_ACCEL_MIN, self.params.BOSCH_ACCEL_MAX))
          # not using self.accel since the brake pid resets with the gas pedal
          gas_pedal_force = accel + wind_brake_ms2 * self.windfactor + hill_brake

          # Live-learn gas pedal adjustments when openpilot is controlling gas.
          if (actuators.longControlState == LongCtrlState.pid) and (not CS.out.gasPressed):
            gas_error = accel - CS.out.aEgo
            if gas_error != 0.0 and gas_pedal_force > min_gas:
              if self.CP.carFingerprint in (CAR.HONDA_INSIGHT, CAR.HONDA_CIVIC_BOSCH):  # gas pedal reacts too slowly
                learn_speed = 150
              elif self.CP.carFingerprint == CAR.ACURA_RDX_3G:  # prevent overreacting to turbo lag
                learn_speed = 300
              else:
                learn_speed = 50
              self.gasfactor = np.clip(self.gasfactor + gas_error / learn_speed * (gas_pedal_force - min_gas), 0.01, 3.0)
            if gas_error != 0.0 and (not CS.out.brakePressed) and (CS.out.vEgo > 0.0):
              wind_learn_speed = 100 if self.CP.carFingerprint == CAR.ACURA_RDX_3G else 1000
              wind_adjust = 1 + wind_brake_ms2 / wind_learn_speed
              self.windfactor = np.clip(self.windfactor * (wind_adjust if (gas_error > 0) else 1.0 / wind_adjust), 0.1, 3.0)
            if gas_pedal_force <= min_gas:
              self.windfactor = max(self.windfactor, self.windfactor_before_brake)
            else:
              self.windfactor_before_brake = self.windfactor
            if gas_pedal_force >= self.params.BOSCH_ACCEL_MAX:
              self.gasfactor = min(self.gasfactor, self.gasfactor_before_maxgas)
              self.windfactor = min(self.windfactor, self.windfactor_before_maxgas)
            else:
              self.gasfactor_before_maxgas = self.gasfactor
              self.windfactor_before_maxgas = self.windfactor
          self.gas = float(np.interp((gas_pedal_force - min_gas) * self.gasfactor + min_gas,
                                     self.params.BOSCH_GAS_LOOKUP_BP, self.params.BOSCH_GAS_LOOKUP_V))

          # limit gas ramp to 60 units per frame, matches stock; higher sometimes makes the powertrain ignore the command
          max_gas = max(60, self.bosch_last_gas + 60)
          self.gas = min(self.gas, max_gas)
          self.bosch_last_gas = self.gas

          stopping = actuators.longControlState == LongCtrlState.stopping
          self.stopping_counter = self.stopping_counter + 1 if stopping else 0
          # CAN FD: never overlap the stock radar's own ACC_CONTROL stream; ours starts within a few
          # frames of the radar going silent (see the deferred radar disable above)
          if not (self.CP.carFingerprint in HONDA_BOSCH_CANFD and CS.stock_acc_alive):
            can_sends.extend(hondacan.create_acc_commands(self.packer, self.CAN, CC.enabled, CC.longActive, self.accel, self.gas,
                                                          self.stopping_counter, self.CP, gas_pedal_force))
        else:
          apply_brake = np.clip(self.brake_last - wind_brake, 0.0, 1.0)
          apply_brake = int(np.clip(apply_brake * self.params.NIDEC_BRAKE_MAX, 0, self.params.NIDEC_BRAKE_MAX - 1))
          pump_on, self.last_pump_ts = brake_pump_hysteresis(apply_brake, self.apply_brake_last, self.last_pump_ts, ts)

          pcm_override = True
          can_sends.append(hondacan.create_brake_command(self.packer, self.CAN, apply_brake, pump_on,
                                                         pcm_override, pcm_cancel_cmd, alert_fcw,
                                                         self.CP.carFingerprint, CS.stock_brake, self.CP_IQ))
          self.apply_brake_last = apply_brake
          self.brake = apply_brake / self.params.NIDEC_BRAKE_MAX

          gas_error = actuators.accel - CS.out.aEgo
          if (not CS.out.gasPressed) and (actuators.longControlState == LongCtrlState.pid) and self.CP_IQ.enableGasInterceptor:
            if gas_error != 0.0 and gas > 0.0:
              self.gasfactor = np.clip(self.gasfactor + gas_error / 50 * (gas * 4.8), 0.1, 3.0)
            if gas_error != 0.0 and (not CS.out.brakePressed) and (CS.out.vEgo > 0.0):
              wind_adjust = 1 + (wind_brake * 4.8) / 1000
              self.windfactor = np.clip(self.windfactor * (wind_adjust if (gas_error > 0) else 1.0 / wind_adjust), 0.1, 5.0)
            if gas <= 0.0:
              self.windfactor = max(self.windfactor, self.windfactor_before_brake)
            else:
              self.windfactor_before_brake = self.windfactor

          can_sends.extend(GasInterceptorCarController.update(self, CC, CS, gas * self.gasfactor, brake, wind_brake, self.packer, self.frame))

    # Send dashboard UI commands. On CAN FD, ACC_HUD is a radar look-alike that openpilot only owns
    # once it has disabled the radar; it rides the phase-locked 10Hz hud tick instead of frame % 10
    if (self.CP.carFingerprint in HONDA_BOSCH_CANFD and CS.hud_tick and
        self.CP.openpilotLongitudinalControl and not CS.stock_acc_alive):
      can_sends.append(hondacan.create_acc_hud(self.packer, self.CAN.pt, self.CP, CC.enabled, pcm_speed, actuators.accel,
                                               hud_control, hud_v_cruise, CS.is_metric, CS.acc_hud))

    if self.frame % 10 == 0:
      if self.CP.openpilotLongitudinalControl and self.CP.carFingerprint not in HONDA_BOSCH_CANFD:
        # On Nidec, this also controls longitudinal positive acceleration
        can_sends.append(hondacan.create_acc_hud(self.packer, self.CAN.pt, self.CP, CC.enabled, pcm_speed, pcm_accel,
                                                 hud_control, hud_v_cruise, CS.is_metric, CS.acc_hud))

      steering_available = CS.out.cruiseState.available and CS.out.vEgo > self.CP.minSteerSpeed
      reduced_steering = CS.out.steeringPressed

      lkas_state_change = None
      if self.CP.carFingerprint in HONDA_BOSCH_CANFD:
        # The key must contain exactly the signals that change the LKAS_HUD payload, nothing more:
        # a flickering input (like steer saturation) re-triggers the pulse continuously, which keeps
        # LKAS_STATE_CHANGE high and suppresses the dash lane lines entirely
        hud_key = (bool(CC.latActive), bool(self.dashed_lanes), bool(alert_steer_required), bool(CS.out.steerFaultPermanent))
        if hud_key != self.lkas_hud_key:
          self.lkas_hud_key = hud_key
          self.lkas_state_change_frames = 30  # 3s at the 10Hz LKAS_HUD rate, matching the stock pulse length
        lkas_state_change = self.lkas_state_change_frames > 0
        self.lkas_state_change_frames = max(0, self.lkas_state_change_frames - 1)

      can_sends.extend(hondacan.create_lkas_hud(self.packer, self.CAN.lkas, self.CP, hud_control, CC.latActive,
                                                steering_available, reduced_steering, alert_steer_required, CS.lkas_hud, self.dashed_lanes,
                                                steer_fault_permanent=CS.out.steerFaultPermanent, lkas_state_change=lkas_state_change))

      if self.CP.openpilotLongitudinalControl:
        # TODO: combining with create_acc_hud block above will change message order and will need replay logs regenerated
        if self.CP.carFingerprint in (HONDA_BOSCH - HONDA_BOSCH_RADARLESS - HONDA_BOSCH_CANFD):
          can_sends.append(hondacan.create_radar_hud(self.packer, self.CAN.pt))
        if self.CP.carFingerprint == CAR.HONDA_CIVIC_BOSCH:
          can_sends.append(hondacan.create_legacy_brake_command(self.packer, self.CAN.pt))
        if self.CP.carFingerprint not in HONDA_BOSCH:
          self.speed = pcm_speed
          if not self.CP_IQ.enableGasInterceptor:
            self.gas = pcm_accel / self.params.NIDEC_GAS_MAX

    # Render OP's lane and lead cars on the dash. On CAN FD these are radar look-alikes that only
    # exist (and are only allowed by panda safety) when the radar is disabled. Radarless keeps the
    # camera as the dash authority (known-good), so OP does not author these there
    if (CS.radar_50hz_tick and self.CP.carFingerprint in HONDA_BOSCH_CANFD and self.CP.openpilotLongitudinalControl
        and not CS.stock_acc_alive):
      leads = dash_objects.leads_from_model(self.model, CS.out.vEgo)
      lead = leads[0]
      lead_d = lead.dRel if lead.status else 0.0
      self.rendered_lane = self.lane_renderer.update(self.model, CS.out.vEgo, lead_d)
      mux = self.radar_mux
      # no LKAS_HUD_2 on CAN FD: the dash reads the lane length from the in-band terminator, so the
      # path is reshaped into the terminated-prefix form
      lane_offsets = dash_lane.canfd_lane_offsets(self.rendered_lane)
      lane_msg = dash_lane.create_lane_path(self.packer, self.CAN.lkas, lane_offsets, mux)
      can_sends.append(lane_msg)

      # CAN FD cars have no camera HUD_OBJECTS to poll (the disabled radar owned it): author OP's
      # lead in slot 0 with the other slots blank (tracks=None)
      tracks = CS.camera_object_tracker.snapshot() if CS.camera_object_tracker is not None else None
      hud_msg = self.dash_object_author.create(self.packer, self.CAN.lkas, lead, tracks, mux, now_nanos * 1e-9,
                                               extra_leads=leads[1:])
      can_sends.append(hud_msg)

      # the camera (behind the relay) also consumes these; mirror the identical packed bytes onto the
      # camera bus (packed once, so the counter/checksum stay in lockstep)
      for addr, dat, _ in (lane_msg, hud_msg):
        can_sends.append((addr, dat, self.CAN.camera))

    # CAN FD: when stock LKAS is active, the touch-steering-wheel nag eventually forces an ACC
    # disengagement (a brake tap from the VSA). Disable LKAS automatically and block the driver's LKAS
    # button by taking over SCM_BUTTONS on the camera bus while engaged (panda blocks the forwarded
    # stock SCM_BUTTONS while this stream flows). Radarless keeps the stock camera LKAS untouched
    if self.CP.carFingerprint in HONDA_BOSCH_CANFD and CC.enabled and self.frame % 4 == 0 and \
        not pcm_cancel_cmd and not CC.cruiseControl.resume:
      if self.lkas_button_send_remaining == 0 and CS.lkas_hud["LKAS_READY"] and self.frame >= self.last_lkas_button_frame + 500:
        self.lkas_button_send_remaining = 3

      if self.lkas_button_send_remaining > 0:
        self.last_lkas_button_frame = self.frame
        self.lkas_button_send_remaining -= 1
        cruise_setting = CruiseSettings.LKAS
      elif CS.cruise_setting == CruiseSettings.LKAS:
        cruise_setting = 0  # block the driver's LKAS button press
      else:
        cruise_setting = CS.cruise_setting

      can_sends.append(hondacan.spam_buttons_command(self.packer, self.CAN, CS.cruise_buttons, cruise_setting,
                                                     CS.scm_ambient_light, self.CP.carFingerprint, bus=self.CAN.camera))

    # Finalize actuator state for downstream consumers
    new_actuators = actuators.as_builder()
    new_actuators.speed = self.speed
    new_actuators.accel = self.accel
    new_actuators.gas = self.gas
    new_actuators.brake = self.brake
    new_actuators.torque = self.last_torque
    new_actuators.torqueOutputCan = apply_torque

    self.frame += 1
    return new_actuators, can_sends
