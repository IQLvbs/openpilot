"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos
"""
import base64
import gzip
import json

from iqpilot.common.params import Params, ParamKeyType


def encode_param(name: str, params=None, use_default: bool = False) -> bytes | None:
  params = params or Params()
  raw = params.get_default_value(name) if use_default else params.get(name)
  if raw is None:
    return None

  ktype = params.get_type(name)
  if ktype == ParamKeyType.BYTES:
    return bytes(raw)
  if ktype == ParamKeyType.JSON:
    return json.dumps(raw).encode("utf-8")
  return str(raw).encode("utf-8")


_FROM_TEXT = {
  ParamKeyType.STRING: lambda s: s,
  ParamKeyType.BOOL: lambda s: s.lower() in ("true", "1", "yes"),
  ParamKeyType.INT: int,
  ParamKeyType.FLOAT: float,
  ParamKeyType.TIME: str,
  ParamKeyType.JSON: json.loads,
}


def restore_param_from_base64(name: str, b64_data: str, compressed: bool = False) -> None:
  params = Params()
  ktype = params.get_type(name)

  blob = base64.b64decode(b64_data)
  if compressed:
    blob = gzip.decompress(blob)

  if ktype == ParamKeyType.BYTES:
    value = blob
  else:
    value = _FROM_TEXT.get(ktype, lambda s: s)(blob.decode("utf-8"))

  params.put(name, value)
