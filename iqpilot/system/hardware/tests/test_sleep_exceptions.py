from unittest.mock import Mock

import pytest

from iqpilot.cereal import car
from iqdbc.car.hyundai.values import CAR as HYUNDAI
from iqdbc.car.volkswagen.values import CAR as VOLKSWAGEN, VolkswagenFlags
from iqpilot.system.hardware.hardwared import _CarParamsCache


def car_params(brand, fingerprint="", flags=0):
  return car.CarParams.new_message(brand=brand, carFingerprint=fingerprint, flags=int(flags)).to_bytes()


def update(cache, values):
  cache.update(Mock(get=lambda key: values.get(key)))


@pytest.mark.parametrize("candidate", list(HYUNDAI))
def test_only_ev6_variants_are_exempt_in_hkg(candidate):
  cache = _CarParamsCache(refresh_s=0)
  update(cache, {"CarParams": car_params("hyundai", candidate, candidate.config.flags)})
  assert cache.no_sleep == (candidate in (HYUNDAI.KIA_EV6, HYUNDAI.KIA_EV6_PE))


@pytest.mark.parametrize("candidate", list(VOLKSWAGEN))
def test_volkswagen_exceptions_unchanged(candidate):
  cache = _CarParamsCache(refresh_s=0)
  update(cache, {"CarParams": car_params("volkswagen", candidate, candidate.config.flags)})
  assert cache.no_sleep == bool(candidate.config.flags & VolkswagenFlags.MEB)


@pytest.mark.parametrize("brand,expected", (
  ("tesla", True), ("toyota", False), ("honda", False), ("subaru", False),
  ("ford", False), ("rivian", False), ("gm", False), ("mock", False),
))
def test_other_brands_unchanged(brand, expected):
  cache = _CarParamsCache(refresh_s=0)
  update(cache, {"CarParams": car_params(brand, HYUNDAI.KIA_EV6)})
  assert cache.no_sleep == expected


@pytest.mark.parametrize("brand,fingerprint,flags", (
  ("hyundai", HYUNDAI.KIA_EV6, 0),
  ("hyundai", HYUNDAI.KIA_EV6_PE, 0),
  ("tesla", "", 0),
  ("volkswagen", "", VolkswagenFlags.MEB),
))
def test_exception_available_before_first_onroad_after_restart(brand, fingerprint, flags):
  cache = _CarParamsCache(refresh_s=0)
  update(cache, {"CarParamsPersistent": car_params(brand, fingerprint, flags)})
  assert cache.no_sleep


def test_current_car_overrides_previous_car_and_updates_cache():
  cache = _CarParamsCache(refresh_s=0)
  values = {"CarParamsPersistent": car_params("hyundai", HYUNDAI.KIA_EV6)}
  update(cache, values)
  assert cache.no_sleep
  values["CarParams"] = car_params("hyundai", HYUNDAI.HYUNDAI_IONIQ_5)
  update(cache, values)
  assert not cache.no_sleep
  values["CarParams"] = car_params("hyundai", HYUNDAI.KIA_EV6_PE)
  update(cache, values)
  assert cache.no_sleep
  update(cache, values)
  assert cache.no_sleep


def test_missing_and_malformed_params_do_not_enable_exception():
  cache = _CarParamsCache(refresh_s=0)
  update(cache, {})
  assert not cache.no_sleep
  update(cache, {"CarParams": car_params("tesla")})
  assert cache.no_sleep
  update(cache, {"CarParams": b"invalid", "CarParamsPersistent": car_params("tesla")})
  assert not cache.no_sleep
  update(cache, {"CarParams": car_params("hyundai", HYUNDAI.KIA_EV6)})
  assert cache.no_sleep


def test_empty_params_during_onroad_transition_retain_exception():
  cache = _CarParamsCache(refresh_s=0)
  update(cache, {"CarParams": car_params("hyundai", HYUNDAI.KIA_EV6)})
  update(cache, {})
  assert cache.no_sleep


def test_refresh_interval_preserved(mocker):
  clock = mocker.patch("iqpilot.system.hardware.hardwared.time.monotonic", return_value=10.)
  cache = _CarParamsCache()
  update(cache, {"CarParams": car_params("hyundai", HYUNDAI.KIA_EV6)})
  assert cache.no_sleep
  clock.return_value = 14.9
  update(cache, {"CarParams": car_params("hyundai", HYUNDAI.HYUNDAI_IONIQ_5)})
  assert cache.no_sleep
  clock.return_value = 15.
  update(cache, {"CarParams": car_params("hyundai", HYUNDAI.HYUNDAI_IONIQ_5)})
  assert not cache.no_sleep
