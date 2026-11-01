# Copyright (c) 2026 IQ.Lvbs. All rights reserved.
import gc
import weakref

from iqpilot.cereal import messaging
from iqpilot.konn3kt.shared_readers import SharedMessageReaders


SERVICES = ('deviceState', 'managerState', 'selfdriveState', 'modelV2', 'controlsState', 'pandaStates', 'iqNavState')


def service_set(value):
  return ('carState', *(service for index, service in enumerate(SERVICES) if value & (1 << index)))


def test_changing_telemetry_groups_releases_idle_native_readers():
  readers = SharedMessageReaders()
  refs = []
  for value in range(128):
    reader, _ = readers.get(service_set(value), transient=True)
    refs.append(weakref.ref(reader))
  del reader
  gc.collect()
  assert sum(ref() is not None for ref in refs) <= 8
  assert len(readers._recent) <= 8


def test_eviction_preserves_active_groups_and_canonicalizes_services():
  readers = SharedMessageReaders(recent_limit=2)
  active, lock = readers.get(('deviceState', 'carState'), transient=True)
  for value in range(2, 24):
    readers.get(service_set(value), transient=True)
  again, again_lock = readers.get(('carState', 'deviceState', 'carState'), transient=True)
  assert again is active
  assert again_lock is lock


def test_fixed_snapshot_survives_dynamic_subscription_churn():
  publisher = messaging.PubMaster(['carState'])
  readers = SharedMessageReaders()
  fixed, _ = readers.get(('carState',))
  message = messaging.new_message('carState')
  message.carState.vEgo = 12.5
  publisher.send('carState', message)
  fixed.update(100)
  assert fixed['carState'].vEgo == 12.5
  for value in range(1, 80):
    readers.get(service_set(value), transient=True)
  again, _ = readers.get(('carState',))
  assert again is fixed
  assert again['carState'].vEgo == 12.5
