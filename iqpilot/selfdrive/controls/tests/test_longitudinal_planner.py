import numpy as np
import pytest
from types import SimpleNamespace

from iqpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LEAD_T_IDXS_MODEL, T_IDXS
from iqpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LongitudinalMpc
from iqpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LongitudinalPlanSource
from iqpilot.selfdrive.controls.lib.longitudinal_planner import A_CRUISE_MIN, A_CRUISE_MIN_NAV, J_CRUISE_BP, J_CRUISE_VALS, get_accel_candidates, get_cruise_accel, get_e2e_accel


def model_velocity(v_ego, v_future):
  return np.interp(T_IDXS, [T_IDXS[0], T_IDXS[-1]], [v_ego, v_future])


class TestCruiseAccel:
  @pytest.mark.parametrize("v_cruise,v_ego,a_cruise_prev,direction", [
    (10.0, 30.0, 0.4, -1.0),
    (40.0, 30.0, -0.4, 1.0),
  ])
  def test_e2e_rate_limit(self, v_cruise, v_ego, a_cruise_prev, direction):
    dt = 0.05
    accel, _ = get_cruise_accel(True, v_cruise, v_ego, a_cruise_prev, a_cruise_prev, 0.0, SimpleNamespace(), dt, 0.0, True)

    assert accel == pytest.approx(a_cruise_prev + direction * np.interp(v_ego, J_CRUISE_BP, J_CRUISE_VALS) * dt)

  def test_rate_limit_is_gentler_at_highway_speed(self):
    dt = 0.05
    city, _ = get_cruise_accel(True, 30.0, 5.0, 0.0, 0.0, 0.0, SimpleNamespace(), dt, 0.0, True)
    highway, _ = get_cruise_accel(True, 45.0, 30.0, 0.0, 0.0, 0.0, SimpleNamespace(), dt, 0.0, True)

    assert 0.0 < highway < city

  def test_lowering_set_speed_while_braking_does_not_wait_for_cruise_to_ramp_down(self):
    dt = 0.05
    accel, _ = get_cruise_accel(True, 10.0, 30.0, 0.4, -1.0, 0.0, SimpleNamespace(), dt, 0.0, True)

    assert accel < 0.0

  @pytest.mark.parametrize("e2e", [False, True])
  def test_nav_target_brakes_harder_than_cruise(self, e2e):
    def settle(min_accel):
      accel = 0.0
      for _ in range(100):
        accel, _ = get_cruise_accel(e2e, 13.9, 19.4, accel, accel, 0.0, SimpleNamespace(steerRatio=15.3, wheelbase=2.64), 0.05, 0.0, True, min_accel)
      return accel

    assert settle(A_CRUISE_MIN) == pytest.approx(A_CRUISE_MIN)
    assert settle(A_CRUISE_MIN_NAV) == pytest.approx(A_CRUISE_MIN_NAV)


class TestE2eCruiseConvergence:
  def test_converges_when_model_wants_to_accelerate(self):
    assert get_e2e_accel(20.0, 30.0, model_velocity(20.0, 25.0), 0.1, False) == pytest.approx(0.5)

  def test_scales_down_near_cruise_speed(self):
    assert get_e2e_accel(28.5, 30.0, model_velocity(28.5, 30.0), 0.0, False) == pytest.approx(0.05)

  def test_preserves_active_model_deceleration(self):
    assert get_e2e_accel(20.0, 30.0, model_velocity(20.0, 25.0), -0.05, False) == pytest.approx(-0.05)

  def test_preserves_future_model_slowdown(self):
    assert get_e2e_accel(20.0, 30.0, model_velocity(20.0, 18.0), 0.1, False) == pytest.approx(0.1)

  @pytest.mark.parametrize("v_ego, v_cruise, should_stop", [
    (30.0, 30.0, False),
    (31.0, 30.0, False),
    (20.0, 30.0, True),
  ])
  def test_never_overrides_cruise_or_stop(self, v_ego, v_cruise, should_stop):
    assert get_e2e_accel(v_ego, v_cruise, model_velocity(v_ego, v_ego + 5.0), -0.2, should_stop) == pytest.approx(-0.2)


class TestAccelCandidates:
  MPC = (-0.2, LongitudinalPlanSource.lead0, True)
  CRUISE = (0.5, LongitudinalPlanSource.cruise, False)
  E2E = (0.1, LongitudinalPlanSource.e2e, False)

  def test_e2e_without_lead_frees_model_from_mpc(self):
    candidates = get_accel_candidates(True, False, self.MPC, self.CRUISE, self.E2E)
    assert candidates == [self.CRUISE, self.E2E]
    assert min(candidates, key=lambda c: c[0])[1] == LongitudinalPlanSource.e2e
    assert not any(should_stop for _, _, should_stop in candidates)

  def test_e2e_with_lead_keeps_mpc_safety_constraint(self):
    candidates = get_accel_candidates(True, True, self.MPC, self.CRUISE, self.E2E)
    assert candidates == [self.MPC, self.CRUISE, self.E2E]
    assert min(candidates, key=lambda c: c[0])[1] == LongitudinalPlanSource.lead0
    assert any(should_stop for _, _, should_stop in candidates)

  def test_acc_without_lead_keeps_mpc_policy(self):
    candidates = get_accel_candidates(False, False, self.MPC, self.CRUISE, self.E2E)
    assert candidates == [self.MPC, self.CRUISE]


class TestExperimentalLeadMpc:
  @staticmethod
  def mpc(v_ego=20.0):
    mpc = object.__new__(LongitudinalMpc)
    mpc.x0 = np.array([0.0, v_ego, 0.0])
    return mpc

  @staticmethod
  def model_lead(prob=0.9, x=None, v=None):
    return SimpleNamespace(
      prob=prob,
      x=np.asarray(x if x is not None else [30.0, 66.0, 98.0, 126.0, 150.0, 170.0]),
      v=np.asarray(v if v is not None else [20.0, 18.0, 16.0, 14.0, 12.0, 10.0]),
    )

  @staticmethod
  def radar_lead(status=True, model_prob=0.9, radar=True):
    return SimpleNamespace(
      status=status,
      dRel=28.0,
      vLead=19.0,
      aLeadK=-0.5,
      aLeadTau=1.5,
      vRel=-1.0,
      modelProb=model_prob,
      radar=radar,
    )

  def test_uses_valid_trajectory_with_radar_anchor(self):
    lead_xv = self.mpc().process_lead(self.model_lead(), self.radar_lead())

    assert lead_xv[0, 0] == pytest.approx(28.0)
    assert lead_xv[0, 1] == pytest.approx(19.0)
    assert lead_xv[-1, 0] == pytest.approx(168.0)
    assert lead_xv[-1, 1] == pytest.approx(9.0)
    assert np.all(np.diff(lead_xv[:, 0]) >= 0.0)

  def test_uses_valid_vision_only_trajectory(self):
    radar_lead = self.radar_lead(radar=False)
    lead_xv = self.mpc().process_lead(self.model_lead(), radar_lead)

    assert lead_xv[-1, 0] == pytest.approx(168.0)
    assert lead_xv[-1, 1] == pytest.approx(9.0)

  @pytest.mark.parametrize("model_lead, radar_lead", [
    (model_lead.__func__(prob=0.5), radar_lead.__func__()),
    (model_lead.__func__(x=[30.0, 66.0]), radar_lead.__func__()),
    (model_lead.__func__(v=[20.0, 18.0]), radar_lead.__func__()),
    (model_lead.__func__(x=[30.0, 66.0, 98.0, np.nan, 150.0, 170.0]), radar_lead.__func__()),
    (model_lead.__func__(v=[20.0, 18.0, 16.0, np.inf, 12.0, 10.0]), radar_lead.__func__()),
    (model_lead.__func__(), radar_lead.__func__(model_prob=0.0)),
    (None, radar_lead.__func__()),
  ])
  def test_falls_back_to_radar_extrapolation(self, model_lead, radar_lead):
    mpc = self.mpc()

    assert np.array_equal(mpc.process_lead(model_lead, radar_lead), mpc.process_lead_legacy(radar_lead))

  def test_prevents_backward_position_trajectory(self):
    model_lead = self.model_lead(x=[30.0, 40.0, 38.0, 60.0, 80.0, 100.0])
    lead_xv = self.mpc().process_lead(model_lead, self.radar_lead())

    assert np.all(np.diff(lead_xv[:, 0]) >= 0.0)

  def test_model_time_shape_matches_expected_horizon(self):
    assert LEAD_T_IDXS_MODEL.shape == (6,)

  @staticmethod
  def stopped_radar_lead(v_rel=-4.8, a_lead=-0.1):
    return SimpleNamespace(status=True, dRel=6.2, vLead=0.1, aLeadK=a_lead, aLeadTau=1.5, vRel=v_rel, modelProb=1.0, radar=True)

  def test_stopped_lead_is_not_predicted_to_drive_off(self):
    model_lead = self.model_lead(prob=1.0, x=[8.2, 8.9, 11.4, 16.0, 22.7, 31.6], v=[0.2, 0.7, 1.8, 2.9, 3.9, 4.8])
    lead_xv = self.mpc(v_ego=3.9).process_lead(model_lead, self.stopped_radar_lead())

    assert lead_xv[0, 0] == pytest.approx(6.2)
    assert lead_xv[-1, 0] <= 6.2 + 0.1 * 10.0 + 1e-6
    assert np.all(lead_xv[:, 1] <= 0.1 + 1e-6)

  def test_radar_confirmed_pullaway_keeps_model_acceleration(self):
    model_lead = self.model_lead(prob=1.0, x=[8.0, 12.0, 20.0, 32.0, 48.0, 68.0], v=[1.0, 3.0, 5.0, 7.0, 9.0, 11.0])
    radar_lead = SimpleNamespace(status=True, dRel=8.0, vLead=3.0, aLeadK=1.0, aLeadTau=1.5, vRel=2.0, modelProb=1.0, radar=True)
    lead_xv = self.mpc(v_ego=1.0).process_lead(model_lead, radar_lead)

    assert lead_xv[-1, 0] == pytest.approx(68.0)
    assert lead_xv[-1, 1] == pytest.approx(13.0)
