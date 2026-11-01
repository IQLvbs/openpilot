# Copyright (c) 2026 IQ.Lvbs. All rights reserved.
import importlib.util
import math
import sys
from collections import defaultdict
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np

from iqdbc.car.car_helpers import interfaces
from iqdbc.car.vehicle_model import VehicleModel
from iqpilot.cereal import car, log
from iqpilot.common.params import Params
from iqpilot.common.pt2 import PT2Filter
from iqpilot.common.realtime import DT_CTRL
from iqpilot.selfdrive.car import interfaces as iqpilot_interfaces
from iqpilot.selfdrive.car.helpers import convert_to_capnp
from iqpilot.selfdrive.controls.lib.latcontrol_angle import LatControlAngle
from iqpilot.selfdrive.controls.lib.latcontrol_pid import LatControlPID
from iqpilot.selfdrive.controls.lib.latcontrol_torque import LatControlTorque
from iqpilot.selfdrive.controls.lib.latcontrol_torque_pq import LatControlTorquePQ
from iqpilot.selfdrive.controls.lib.latcontrol_torque_v0 import LatControlTorqueV0
from iqpilot.selfdrive.controls.steering_fault_recovery import SteeringFaultRecovery
from iqpilot.selfdrive.iqmodeld.config import ModelConstants


CONTROL_CASES = {
  "pq": ("VOLKSWAGEN_PASSAT_NMS", LatControlTorquePQ),
  "torque": ("TOYOTA_RAV4", LatControlTorque),
  "tss2": ("TOYOTA_RAV4_TSS2", LatControlTorque),
  "gm": ("CHEVROLET_BOLT_EUV", LatControlTorque),
  "angle": ("NISSAN_LEAF", LatControlAngle),
  "curvature": ("VOLKSWAGEN_ID4_MK1", LatControlAngle),
  "mqb_v0": ("VOLKSWAGEN_PASSAT_MK8", LatControlTorqueV0),
  "pid": ("HONDA_CIVIC", LatControlPID),
}
HYUNDAI_CASES = {"hkg": ("HYUNDAI_KONA_EV", LatControlTorque)}
SCENARIO_FRAMES = 360


def load_controls_module(monkeypatch):
  module_name = "iqpilot.system.proprietary_runtime._verified_import"
  stub = ModuleType(module_name)
  stub.import_verified_module = lambda *args: SimpleNamespace(NavTurnSignalController=object)
  spec = importlib.util.spec_from_file_location("lateral_controlsd_under_test", Path(__file__).parents[1] / "controlsd.py")
  module = importlib.util.module_from_spec(spec)
  with monkeypatch.context() as context:
    context.setitem(sys.modules, module_name, stub)
    spec.loader.exec_module(module)
  return module


class ScenarioMessages(dict):
  def __init__(self):
    super().__init__()
    self.valid = defaultdict(bool)
    self.alive = defaultdict(bool)

  def all_checks(self, services):
    return all(self.valid[service] and self.alive[service] for service in services)


class ScenarioNetwork:
  friction_override = False

  def __init__(self):
    self.queries = 0

  def evaluate(self, values):
    self.queries += 1
    return float(np.tanh(values[1] * 0.2 + values[2] * 0.03 + values[3] * 0.1))


def build_controls(module, case, lookahead=False, slew=False, smooth=False, nnff=False):
  params = Params()
  params.put_bool("IQLateralCurvatureLookahead", lookahead)
  params.put_bool("IQLateralAccelSlew", slew)
  params.put_bool("NeuralNetworkFeedForward", False)
  params.put_bool("IQLiveSteerDelay", False)
  params.put("IQSoftwareSteerDelay", "0.10")
  car_name, controller_type = {**CONTROL_CASES, **HYUNDAI_CASES}[case]
  interface_type = interfaces[car_name]
  cp = interface_type.get_non_essential_params(car_name)
  cp_iq = interface_type.get_non_essential_params_iq(cp, car_name)
  ci = interface_type.__new__(interface_type)
  ci.CP = cp
  ci.CP_IQ = cp_iq
  iqpilot_interfaces.apply_iq_car_config(ci)

  controls = module.Controls.__new__(module.Controls)
  controls.CP = cp.as_reader()
  controls.CP_IQ = convert_to_capnp(cp_iq).as_reader()
  controls.CI = ci
  controls.params = params
  controls.LaC = controller_type(controls.CP, controls.CP_IQ, ci, DT_CTRL)
  if nnff:
    controls.LaC.nnff_assist.enabled = True
    controls.LaC.nnff_assist.has_nn_model = True
    controls.LaC.nnff_assist.model = ScenarioNetwork()
  controls.VM = VehicleModel(cp)
  controls.curvature_lookahead_enabled = lookahead
  controls.steer_limited_by_safety = False
  controls.steering_fault_recovery = SteeringFaultRecovery()
  controls.curvature = 0.0
  controls.desired_curvature = 0.0
  controls.roll_compensation = 0.0
  controls.smooth_steer = PT2Filter(module.SMOOTH_STEER_W0, module.SMOOTH_STEER_ZETA, DT_CTRL)
  controls.smooth_steer_inactive_frames = module.SMOOTH_STEER_HOLD_FRAMES
  controls.is_curvature_car = cp.steerControlType == car.CarParams.SteerControlType.curvatureDEPRECATED
  controls.enable_smooth_steer = smooth
  controls.calibrated_pose = None
  controls.navigation_enabled = False
  controls.nav_exit_lane_change = False
  controls.nav_turn_signal_controller = SimpleNamespace(update=lambda *args: (False, False))
  controls.LoC = SimpleNamespace(long_control_state="off", reset=lambda: None, update=lambda *args, **kwargs: 0.125)
  controls.enable_long_comfort_mode = False
  controls.longitudinal_comfort = module.LongitudinalComfort(controls.CP_IQ.longitudinalActuation)
  controls.iq_lateral_allowed = lambda sm: sm["selfdriveState"].active
  controls.sm = ScenarioMessages()
  return controls


def set_frame(controls, frame):
  speed = [0.0, 4.0, 12.0, 23.05, 35.0, 18.0][frame // 60]
  active = not (frame < 8 or 220 <= frame < 240)
  controls.sm["carState"] = car.CarState.new_message(
    vEgo=speed, steeringAngleDeg=1.5 * math.sin(frame / 19), steeringRateDeg=0.2 * math.cos(frame / 19),
    steeringPressed=160 <= frame < 175, gasPressed=260 <= frame < 280, standstill=speed == 0,
  )
  controls.sm["vehicleParameters"] = log.VehicleParameters.new_message(
    stiffnessFactor=1.0, steerRatio=controls.CP.steerRatio, angleOffsetDeg=0.15, roll=0.015 * math.sin(frame / 31),
  )
  controls.sm["lateralTorqueParameters"] = log.LateralTorqueParameters.new_message()
  controls.sm["selfdriveState"] = SimpleNamespace(enabled=active, active=active)
  controls.sm["longitudinalPlan"] = SimpleNamespace(speeds=[speed], aTarget=0.125, leadDistance=30.0, hasLead=True, shouldStop=False,
                                                   fcw=False, longitudinalPlanSource='cruise', stopOffset=SimpleNamespace(version=0, valid=False))
  controls.sm["onroadEvents"] = []
  controls.sm["iqState"] = SimpleNamespace(aol=SimpleNamespace(active=280 <= frame < 300))
  controls.sm["lateralDelay"] = SimpleNamespace(lateralDelay=0.4 if frame < 300 else 0.15)
  controls.sm["lateralManeuverPlan"] = SimpleNamespace(desiredCurvature=-0.0015)
  controls.sm.valid["lateralManeuverPlan"] = 180 <= frame < 200
  times = np.asarray(ModelConstants.T_IDXS)
  target = [0.0, 0.02, -0.02, -0.0012, 0.0008, -0.0006][frame // 60]
  controls.sm["modelV2"] = log.ModelDataV2.new_message(
    frameId=frame, action={"desiredCurvature": target},
    orientation={"z": (0.02 * times ** 2).tolist()}, orientationRate={"z": (0.04 * times).tolist()},
  )
  nnff = getattr(controls.LaC, "nnff_assist", None)
  if nnff is not None and nnff.has_nn_model:
    controls.sm["modelV2"].orientation.x = (0.003 * np.sin(times)).tolist()
    controls.sm["modelV2"].orientation.y = (0.002 * np.cos(times)).tolist()
    controls.sm["modelV2"].acceleration.y = (0.1 * np.sin(times)).tolist()
  controls.sm.valid["modelV2"] = True
  controls.sm.alive["modelV2"] = True
  controls.steer_limited_by_safety = 140 <= frame < 160


def run_scenario(controls):
  rows = []
  for frame in range(SCENARIO_FRAMES):
    set_frame(controls, frame)
    cc, controller_log = controls.state_control()
    values = {key: getattr(controller_log, key) for key in controller_log.schema.fields}
    rows.append([
      controls.desired_curvature, cc.actuators.curvature, cc.actuators.torque, cc.actuators.steeringAngleDeg,
      cc.actuators.accel, cc.latActive, cc.longActive, controls.LaC.sat_time,
      *[values[key] for key in sorted(values)],
    ])
  return np.asarray(rows, dtype=np.float64)
