from iqpilot.common.params import Params
from iqpilot.common.basedir import BASEDIR
from iqpilot.system.version import BuildMetadata, OpenpilotMetadata
from iqpilot.system.updated.updated import Updater, display_commit_date


def test_display_commit_date():
  assert display_commit_date("'1786378834 2026-08-10 11:20:34 -0500'") == "2026-08-10 11:20:34"
  assert display_commit_date("1786378834 2026-08-10 11:20:34 -0500") == "2026-08-10 11:20:34"
  assert display_commit_date("'1786378834 2026-08-10 11:20:34 -0500'\n") == "2026-08-10 11:20:34"
  assert display_commit_date("Jul 02") == "Jul 02"


def test_tici_branch_unchanged(mocker):
  params = Params()
  params.put("UpdaterTargetBranch", "master-dev")
  mocker.patch("iqpilot.system.updated.updated.HARDWARE.get_device_type", return_value="tici")

  try:
    assert Updater().target_branch == "master-dev"
  finally:
    params.remove("UpdaterTargetBranch")


def test_non_tici_branch_unchanged(mocker):
  params = Params()
  params.put("UpdaterTargetBranch", "master-dev")
  mocker.patch("iqpilot.system.updated.updated.HARDWARE.get_device_type", return_value="tizi")

  try:
    assert Updater().target_branch == "master-dev"
  finally:
    params.remove("UpdaterTargetBranch")


def test_non_git_baked_deployment_uses_build_metadata(mocker):
  mocker.patch("iqpilot.system.updated.updated.has_git_repo", return_value=False)
  mocker.patch("iqpilot.system.updated.updated.HARDWARE.get_device_type", return_value="tici")
  mocker.patch(
    "iqpilot.system.updated.updated.get_build_metadata",
    return_value=BuildMetadata(
      "release3",
      OpenpilotMetadata(
        version="1.2.3",
        release_notes="notes",
        git_commit="abcdef1234567890",
        git_origin="github.com/IQLvbs/openpilot",
        git_commit_date="Jul 02",
        build_style="release",
        is_dirty=False,
      ),
    ),
  )

  updater = Updater()

  assert updater.git_mode is False
  assert updater.get_branch(BASEDIR) == "release3"
  assert updater.get_commit_hash() == "abcdef1234567890"
  assert updater.target_branch == "release3"
  assert updater.update_available is False
  assert updater.update_ready is False
