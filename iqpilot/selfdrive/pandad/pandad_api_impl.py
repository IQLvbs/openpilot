import time
from iqpilot.cereal import log

NO_TRAVERSAL_LIMIT = 2**64 - 1

_cached_reader_fields = None
_cached_writer_fields = None


def _get_reader_fields(schema):
  global _cached_reader_fields
  if _cached_reader_fields is None:
    fields = schema.fields
    _cached_reader_fields = (fields['address'], fields['dat'], fields['src'])
  return _cached_reader_fields


def _get_writer_fields(schema):
  global _cached_writer_fields
  if _cached_writer_fields is None:
    fields = schema.fields
    _cached_writer_fields = (fields['address'], fields['dat'], fields['src'])
  return _cached_writer_fields


def can_list_to_can_capnp(can_msgs, msgtype='can', valid=True):
  global _cached_writer_fields

  dat = log.Event.new_message(valid=valid, logMonoTime=int(time.monotonic() * 1e9))
  can_data = dat.init(msgtype, len(can_msgs))
  if _cached_writer_fields is None and len(can_msgs) > 0:
    _cached_writer_fields = _get_writer_fields(can_data[0].schema)

  if _cached_writer_fields is not None:
    addr_f, dat_f, src_f = _cached_writer_fields
    for i, msg in enumerate(can_msgs):
      f = can_data[i]
      f._set_by_field(addr_f, msg[0])
      f._set_by_field(dat_f, msg[1])
      f._set_by_field(src_f, msg[2])

  return dat.to_bytes()


def can_capnp_to_list(strings, msgtype='can'):
  global _cached_reader_fields
  result = []

  for s in strings:
    with log.Event.from_bytes(s, traversal_limit_in_words=NO_TRAVERSAL_LIMIT) as event:
      frames = getattr(event, msgtype)
      if _cached_reader_fields is None and len(frames) > 0:
        _cached_reader_fields = _get_reader_fields(frames[0].schema)

      if _cached_reader_fields is not None:
        addr_f, dat_f, src_f = _cached_reader_fields
        frame_list = [(f._get_by_field(addr_f), f._get_by_field(dat_f), f._get_by_field(src_f)) for f in frames]
      else:
        frame_list = []

      result.append((event.logMonoTime, frame_list))
  return result
