# Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
import pytest

from iqpilot.iqvd_private_src import iqvd
from iqpilot.selfdrive.ui.mici.onroad import augmented_road_view as mici
from iqpilot.selfdrive.ui.onroad import augmented_road_view as big


@pytest.mark.parametrize("view", [big, mici, iqvd])
def test_experimental_mode_wide_cam_gates_match_on_both_uis_and_iqvd(view):
  assert view.WIDE_CAM_MAX_SPEED == 5.0
  assert view.ROAD_CAM_MIN_SPEED == 10.0
