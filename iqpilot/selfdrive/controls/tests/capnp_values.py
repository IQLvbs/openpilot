# Copyright (c) 2026 IQ.Lvbs. All rights reserved.
import capnp


def plain(value):
  if isinstance(value, (capnp.lib.capnp._DynamicStructReader, capnp.lib.capnp._DynamicStructBuilder)):
    schema = value.schema
    names = list(schema.non_union_fields)
    if schema.union_fields:
      names.append(value.which())
    return {name: plain(getattr(value, name)) for name in names}
  if isinstance(value, (capnp.lib.capnp._DynamicListReader, capnp.lib.capnp._DynamicListBuilder)):
    return [plain(item) for item in value]
  if isinstance(value, capnp.lib.capnp._DynamicEnum):
    return str(value)
  return value
