"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""
import os
import re
import signal
from pathlib import Path

from iqpilot.selfdrive.iqmodeld import egpu_helpers as eh

IQMODELD = Path(__file__).resolve().parents[1]


def _fake_proc(root: Path, pid: int, cmd: str, fds: dict[str, str]) -> None:
  d = root / str(pid)
  (d / "fd").mkdir(parents=True)
  (d / "cmdline").write_bytes(cmd.replace(" ", "\0").encode() + b"\0")
  for fd, target in fds.items():
    os.symlink(target, d / "fd" / fd)


def _lock(tmp: Path) -> str:
  lock = tmp / "am_usb:4-2.lock"
  lock.write_text("")
  return str(lock)


def test_the_process_holding_the_dock_lock_is_found(tmp_path):
  proc, tmp = tmp_path / "proc", tmp_path / "tmp"
  tmp.mkdir()
  lock = _lock(tmp)
  _fake_proc(proc, 101, "python -m iqpilot.selfdrive.iqmodeld.egpu_prefetch", {"3": "/dev/null", "7": lock})
  _fake_proc(proc, 102, "python -m iqpilot.selfdrive.controls.plannerd", {"3": "/dev/null"})
  assert eh.dock_lock_holders(str(proc), str(tmp)) == [(101, "python -m iqpilot.selfdrive.iqmodeld.egpu_prefetch")]


def test_no_lock_file_means_nobody_holds_the_dock(tmp_path):
  (tmp_path / "proc").mkdir()
  assert eh.dock_lock_holders(str(tmp_path / "proc"), str(tmp_path)) == []


def test_our_own_stale_holders_are_killed_and_the_lock_freed(tmp_path):
  # a prefetcher that outlived its offroad window kept the flock and the eGPU sat on "Loading" for the whole drive
  proc, tmp = tmp_path / "proc", tmp_path / "tmp"
  tmp.mkdir()
  lock = _lock(tmp)
  _fake_proc(proc, 101, "python -m iqpilot.selfdrive.iqmodeld.egpu_prefetch", {"7": lock})
  _fake_proc(proc, 102, "python -m iqpilot.selfdrive.iqmodeld.tools.compile_egpu_model --model x", {"4": lock})
  killed = []

  def kill(pid, sig):
    killed.append((pid, sig))
    import shutil
    shutil.rmtree(proc / str(pid))

  assert eh.free_dock_lock(timeout_s=1.0, proc_root=str(proc), tmp_dir=str(tmp), kill=kill, sleep=lambda s: None) == []
  assert sorted(killed) == [(101, signal.SIGKILL), (102, signal.SIGKILL)]


def test_a_foreign_holder_is_reported_not_killed(tmp_path):
  proc, tmp = tmp_path / "proc", tmp_path / "tmp"
  tmp.mkdir()
  lock = _lock(tmp)
  _fake_proc(proc, 555, "python3 some_user_script.py", {"5": lock})
  killed, t = [], [0.0]

  def clock():
    t[0] += 0.5
    return t[0]

  held = eh.free_dock_lock(timeout_s=1.0, proc_root=str(proc), tmp_dir=str(tmp), kill=lambda *a: killed.append(a),
                           sleep=lambda s: None, clock=clock)
  assert held == [(555, "python3 some_user_script.py")] and killed == []


def test_the_prefetcher_never_holds_the_dock_itself():
  src = (IQMODELD / "egpu_prefetch.py").read_text()
  warm = src[src.index("def warm_firmware"):src.index("def main")]
  assert 'Device["AMD"]' not in warm, "the offroad prefetcher must open the dock in a short-lived child, not in-process"
  assert "subprocess.run" in warm and "die_with_parent" in warm


def test_a_failed_dock_init_restarts_instead_of_retrying_against_its_own_lock():
  src = (IQMODELD / "iqegpumodeld.py").read_text()
  loop = src[src.index("opened_dock = False"):src.index('params.put_bool("UsbGpuLoading", False)\n  params.put_bool("UsbGpuCompiled"')]
  assert loop.index("free_dock_lock()") < loop.index("_load_infer_fn(")
  assert re.search(r"if opened_dock:\n(\s+#.*\n)*\s+cloudlog\.error\(.*\)\n\s+sys\.exit\(1\)", loop)


def test_compile_children_die_with_the_daemon():
  src = (IQMODELD / "iqegpumodeld.py").read_text()
  assert "preexec_fn=_compile_child_setup" in src and "die_with_parent()" in src
