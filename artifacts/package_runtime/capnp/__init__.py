"""Dynamic Cap'n Proto serialization for openpilot."""

from .version import version as __version__
from .lib.capnp import *
from .lib.capnp import (
    _DynamicEnum,
    _DynamicListBuilder,
    _DynamicListReader,
    _DynamicStructBuilder,
    _DynamicStructReader,
    _ListSchema,
    _MallocMessageBuilder,
    _StructModule,
    _init_capnp_api,
)

_init_capnp_api()
