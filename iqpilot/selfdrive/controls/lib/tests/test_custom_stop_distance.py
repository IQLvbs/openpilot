from types import SimpleNamespace

import pytest

from iqpilot.selfdrive.controls.lib.custom_stop_distance import CustomStopDistance


@pytest.mark.parametrize('offset', (0, 1, 2, 4, 10))
@pytest.mark.parametrize('speed,distance', ((0.0, 0.5), (0.0, 10.0), (1.0, 10.0), (15.0, 40.0)))
def test_nonnegative_offsets_preserve_raw_perception(offset, speed, distance):
  controller = CustomStopDistance(distance=offset)
  lead = {'status': True, 'dRel': distance, 'vLead': speed, 'aLeadK': -1.0}
  assert controller.apply_lead(dict(lead)) == lead


@pytest.mark.parametrize('offset', (0, 1, 2, 5, 10))
def test_production_range_accepts_supported_requests(offset):
  params = SimpleNamespace(get=lambda *args, **kwargs: offset)
  controller = CustomStopDistance(params=params)
  assert controller.valid
  assert controller.distance == offset


@pytest.mark.parametrize('offset', (-2, -1, 11, float('nan'), float('inf'), 'invalid'))
def test_invalid_production_request_is_not_silently_clamped_or_rewritten(offset):
  params = SimpleNamespace(get=lambda *args, **kwargs: offset)
  controller = CustomStopDistance(params=params)
  assert not controller.valid
  assert controller.distance == 0.0


@pytest.mark.parametrize('offset', (-2, -1, 11))
def test_explicit_offset_uses_the_same_supported_range(offset):
  with pytest.raises(ValueError, match='unsupported stopping offset'):
    CustomStopDistance(distance=offset)


def test_configuration_is_not_reread_during_engagement():
  value = [2]
  params = SimpleNamespace(get=lambda *args, **kwargs: value[0])
  controller = CustomStopDistance(params=params)
  value[0] = 0
  for _ in range(100):
    controller.update(active=True)
    assert controller.distance == 2.0
