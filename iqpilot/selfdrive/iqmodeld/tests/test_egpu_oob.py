"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""
import os
import pickle

import numpy as np
import pytest

os.environ["DEV"] = "CPU"

from iqpilot.selfdrive.iqmodeld.egpu_policy import dump_oob, is_oob, load_bundle


def _bundle():
  from tinygrad import Tensor
  w = Tensor(np.arange(4096, dtype=np.float32).reshape(64, 64), device="CPU").realize()
  return {"format": 2, "weights": w, "spec": {"a": ((1, 2), "float32")}, "blob": os.urandom(100_000)}


def test_oob_round_trip_matches_plain_pickle(tmp_path):
  b = _bundle()
  oob = tmp_path / "b.oob"
  with open(oob, "wb") as f:
    dump_oob(b, f)
  assert is_oob(str(oob))
  got = load_bundle(str(oob))
  np.testing.assert_array_equal(got["weights"].numpy(), b["weights"].numpy())
  assert got["blob"] == b["blob"] and got["spec"] == b["spec"] and got["format"] == 2
  plain = tmp_path / "b.pkl"
  with open(plain, "wb") as f:
    pickle.dump({"x": 1, "blob": b["blob"]}, f, protocol=pickle.HIGHEST_PROTOCOL)
  assert not is_oob(str(plain))
  assert load_bundle(str(plain))["blob"] == b["blob"]


def test_memory_guard_raises_when_starved(monkeypatch):
  from iqpilot.selfdrive.iqmodeld import iqegpumodeld as d
  monkeypatch.setattr(d, "_mem_available_mb", lambda: 90)
  monkeypatch.setattr(d, "MEMORY_WAIT_S", 0.0)
  with pytest.raises(RuntimeError, match="insufficient memory"):
    d._wait_for_memory(350)
  monkeypatch.setattr(d, "_mem_available_mb", lambda: 900)
  d._wait_for_memory(350)


def test_opcode_rewrite_equals_oob_load(tmp_path):
  from tinygrad import Tensor
  from iqpilot.selfdrive.iqmodeld.tools.oob_rewrite import rewrite_oob
  big = Tensor(np.random.default_rng(0).standard_normal((512, 512)).astype(np.float32), device="CPU").realize()
  small = Tensor(np.arange(16, dtype=np.float32), device="CPU").realize()
  b = {"format": 2, "w": big, "s": small, "meta": {"k": "v"}, "raw": os.urandom(200_000)}
  plain = tmp_path / "plain.pkl"
  with open(plain, "wb") as f:
    pickle.dump(b, f, protocol=5)
  oob = tmp_path / "oob.pkl"
  moved, _ = rewrite_oob(str(plain), str(oob))
  assert moved >= 2 and is_oob(str(oob))
  got = load_bundle(str(oob))
  np.testing.assert_array_equal(got["w"].numpy(), b["w"].numpy())
  np.testing.assert_array_equal(got["s"].numpy(), b["s"].numpy())
  assert got["raw"] == b["raw"] and got["meta"] == {"k": "v"}
