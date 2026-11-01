import os
import random
import shutil
from pathlib import Path

import iqpilot.system.loggerd.deleter as deleter
from iqpilot.common.params import Params
from iqpilot.system.hardware.hw import Paths
from iqpilot.system.loggerd.xattr_cache import setxattr


def create_random_file(file_path: Path, size_mb: float, lock: bool = False) -> None:
  file_path.parent.mkdir(parents=True, exist_ok=True)

  if lock:
    lock_path = str(file_path) + ".lock"
    os.close(os.open(lock_path, os.O_CREAT | os.O_EXCL))

  chunks = 128
  chunk_bytes = int(size_mb * 1024 * 1024 / chunks)
  data = os.urandom(chunk_bytes)

  with open(file_path, "wb") as f:
    for _ in range(chunks):
      f.write(data)


class DeleterTestCase:
  f_type = "UNKNOWN"

  root: Path
  seg_num: int
  seg_format: str
  seg_format2: str
  seg_dir: str

  def setup_method(self):
    shutil.rmtree(Paths.log_root(), ignore_errors=True)
    Path(Paths.log_root()).mkdir(parents=True, exist_ok=True)
    self.seg_num = random.randint(1, 300)
    self.seg_format = "00000004--0ac3964c96--{}"
    self.seg_format2 = "00000005--4c4e99b08b--{}"
    self.seg_dir = self.seg_format.format(self.seg_num)

    self.params = Params()
    self.params.put("IsOffroad", True)
    self.params.put("DongleId", "0000000000000000")

  def make_file_with_data(self, f_dir: str, fn: str, size_mb: float = .1, lock: bool = False,
                          preserve_xattr: bytes | None = None) -> Path:
    file_path = Path(Paths.log_root()) / f_dir / fn
    create_random_file(file_path, size_mb, lock)

    if preserve_xattr is not None:
      setxattr(str(file_path.parent), deleter.PRESERVE_ATTR_NAME, preserve_xattr)

    return file_path
