import pathlib

from iqpilot.system.updated.casync import tar


def is_not_git(path: pathlib.Path) -> bool:
  return ".git" not in path.parts


def create_casync_tar_package(target_dir: pathlib.Path, output_path: pathlib.Path):
  tar.create_tar_archive(output_path, target_dir, is_not_git)
