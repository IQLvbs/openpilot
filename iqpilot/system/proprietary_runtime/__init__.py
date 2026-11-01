import os

from iqpilot.common.basedir import BASEDIR


os.environ.setdefault("OPENPILOT_BASEDIR", BASEDIR)
os.environ.setdefault("IQPILOT_PROPRIETARY_ROOT", os.path.join(BASEDIR, "artifacts"))

_ROOTFS_PKG = "/usr/libexec/iqpilot/python/openpilot/system/proprietary_runtime"
if os.path.isdir(_ROOTFS_PKG):
  if _ROOTFS_PKG in __path__:
    __path__.remove(_ROOTFS_PKG)
  __path__.insert(0, _ROOTFS_PKG)
