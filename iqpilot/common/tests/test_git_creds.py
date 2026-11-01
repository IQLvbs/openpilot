import base64
import os
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

import pytest

from iqpilot.common import git_creds


RELEASE_ORIGIN = "https://git.konn3kt.com/IQ.Lvbs/IQ.Pilot.git"
PREVIEW_ORIGIN = "https://git.konn3kt.com/IQ.Lvbs/IQ.Pilot-Preview.git"
GITHUB_MIRROR_ORIGIN = "https://github.com/iqinc/openpilot.git"
GIT_HOSTS = [
  "git.konn3kt.com", "git-ssh.konn3kt.com", "ssh-git.konn3kt.xyz",
  "gitlvb.teallvbs.xyz", "ssh.gitlvb.teallvbs.xyz",
]
MIGRATING_ORIGINS = [
  template.format(host=host, repo=repo)
  for host in GIT_HOSTS
  for template in ("https://{host}/IQ.Lvbs/{repo}.git", "git@{host}:IQ.Lvbs/{repo}.git", "ssh://git@{host}/IQ.Lvbs/{repo}.git")
  for repo in ("IQ.Pilot", "IQ.Pilot-Preview")
] + [GITHUB_MIRROR_ORIGIN, "git@github.com:iqinc/openpilot.git", "ssh://git@github.com/iqinc/openpilot.git"]


@pytest.fixture
def device_repo(tmp_path, monkeypatch):
  monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "gitconfig"))
  monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
  for key in ("GIT_CONFIG_COUNT", "GIT_CONFIG_PARAMETERS", "GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR",
              "GIT_CONFIG_KEY_0", "GIT_CONFIG_VALUE_0", "GIT_CONFIG_KEY_1", "GIT_CONFIG_VALUE_1"):
    monkeypatch.delenv(key, raising=False)
  repo = tmp_path / "openpilot"
  repo.mkdir()
  subprocess.run(["git", "init", "-q", "-b", "release", str(repo)], check=True)
  monkeypatch.setattr(git_creds, "DEFAULT_REPO_DIR", str(repo))
  monkeypatch.setattr(git_creds, "PARAMS_DIR", str(tmp_path / "params"))
  monkeypatch.setattr(git_creds, "KEY_DIR", str(tmp_path / "konn3kt"))
  monkeypatch.setattr(git_creds, "KEY_PATH", str(tmp_path / "konn3kt" / "git_auth.key"))
  monkeypatch.setattr(git_creds, "HELPER_PATH", str(tmp_path / "konn3kt" / "git_credential_helper.py"))

  def unavailable_params():
    raise ImportError("Compiled params unavailable during bootstrap")

  monkeypatch.setattr(git_creds, "Params", unavailable_params)
  return repo


@pytest.fixture
def preview_access(monkeypatch):
  access = SimpleNamespace(returncode=0, timeout=False, requests=[])
  run_git = subprocess.run

  def run(args, **kwargs):
    if "ls-remote" in args:
      access.requests.append((args, kwargs))
      if access.timeout:
        raise subprocess.TimeoutExpired(args, kwargs["timeout"])
      return subprocess.CompletedProcess(args, access.returncode, "a" * 40 + "\trefs/heads/release\n", "rejected")
    return run_git(args, **kwargs)

  monkeypatch.setattr(subprocess, "run", run)
  return access


def set_origin(repo, origin):
  subprocess.run(["git", "-C", str(repo), "config", "remote.origin.url", origin], check=True)


def get_origin(repo):
  return subprocess.run(["git", "-C", str(repo), "config", "--get", "remote.origin.url"], capture_output=True, text=True, check=True).stdout.strip()


@pytest.mark.parametrize("origin", MIGRATING_ORIGINS)
def test_save_valid_token_migrates_to_current_preview(device_repo, preview_access, origin):
  set_origin(device_repo, origin)
  git_creds.set_credentials(" tester ", " preview-token ")

  assert get_origin(device_repo) == PREVIEW_ORIGIN
  assert git_creds.get_credentials() == ("tester", "preview-token")
  assert b"preview-token" not in (device_repo.parent / "params" / "GitAuthBlob").read_bytes()
  branch = subprocess.run(["git", "-C", str(device_repo), "symbolic-ref", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
  assert branch == "release"
  assert len(preview_access.requests) == 1
  args, kwargs = preview_access.requests[0]
  assert args[-1] == PREVIEW_ORIGIN
  assert kwargs["env"]["GIT_CONFIG_KEY_0"] == f"http.{PREVIEW_ORIGIN}/.extraHeader"
  assert "preview-token" not in " ".join(args)
  assert kwargs["env"]["GIT_CONFIG_VALUE_0"] == "Authorization: Basic " + base64.b64encode(b"tester:preview-token").decode()


@pytest.mark.parametrize("origin", MIGRATING_ORIGINS)
@pytest.mark.parametrize("existing", [False, True])
def test_rejected_token_preserves_origin_and_credentials(device_repo, preview_access, origin, existing):
  if existing:
    git_creds.set_credentials("previous", "previous-token")
  set_origin(device_repo, origin)
  preview_access.returncode = 128

  with pytest.raises(ValueError, match="Unable to read IQ.Pilot-Preview"):
    git_creds.set_credentials("tester", "invalid-token")

  assert get_origin(device_repo) == origin
  assert git_creds.get_credentials() == (("previous", "previous-token") if existing else None)


def test_validation_uses_entered_token_without_ambient_git_auth(device_repo, preview_access, monkeypatch):
  set_origin(device_repo, RELEASE_ORIGIN)
  monkeypatch.setenv("GIT_CONFIG_COUNT", "2")
  monkeypatch.setenv("GIT_CONFIG_KEY_0", "http.extraHeader")
  monkeypatch.setenv("GIT_CONFIG_VALUE_0", "Authorization: Basic fleet-token")
  monkeypatch.setenv("GIT_CONFIG_KEY_1", "credential.helper")
  monkeypatch.setenv("GIT_CONFIG_VALUE_1", "store")
  monkeypatch.setenv("GIT_CONFIG_PARAMETERS", "'credential.helper'='store'")
  monkeypatch.setenv("GIT_DIR", str(device_repo / ".git"))
  monkeypatch.setenv("GIT_ASKPASS", "/existing-askpass")
  monkeypatch.setenv("SSH_ASKPASS", "/existing-ssh-askpass")
  preview_access.returncode = 128

  with pytest.raises(ValueError):
    git_creds.set_credentials("tester", "invalid-token")

  _, kwargs = preview_access.requests[0]
  env = kwargs["env"]
  assert kwargs["cwd"] == "/"
  assert env["GIT_CONFIG_GLOBAL"] == "/dev/null"
  assert env["GIT_CONFIG_NOSYSTEM"] == "1"
  assert env["GIT_CONFIG_COUNT"] == "1"
  assert env["GIT_TERMINAL_PROMPT"] == "0"
  assert env["GIT_CONFIG_VALUE_0"] == "Authorization: Basic " + base64.b64encode(b"tester:invalid-token").decode()
  assert not {"GIT_CONFIG_PARAMETERS", "GIT_DIR", "GIT_ASKPASS", "SSH_ASKPASS"} & env.keys()
  assert get_origin(device_repo) == RELEASE_ORIGIN


def test_timeout_preserves_existing_credentials_and_origin(device_repo, preview_access):
  git_creds.set_credentials("previous", "previous-token")
  origin = RELEASE_ORIGIN
  set_origin(device_repo, origin)
  preview_access.timeout = True

  with pytest.raises(RuntimeError, match="timed out"):
    git_creds.set_credentials("tester", "preview-token")

  assert get_origin(device_repo) == origin
  assert git_creds.get_credentials() == ("previous", "previous-token")


@pytest.mark.parametrize("existing", [False, True])
def test_failed_origin_write_restores_previous_credentials(device_repo, preview_access, existing):
  if existing:
    git_creds.set_credentials("previous", "previous-token")
  origin = RELEASE_ORIGIN
  set_origin(device_repo, origin)
  (device_repo / ".git" / "config.lock").touch()

  with pytest.raises(RuntimeError, match="Existing credentials were restored"):
    git_creds.set_credentials("tester", "preview-token")

  assert get_origin(device_repo) == origin
  assert git_creds.get_credentials() == (("previous", "previous-token") if existing else None)


@pytest.mark.parametrize("origin", [
  "https://github.com/commaai/openpilot.git",
  "https://github.com/iqinc/openpilot-other.git",
  "https://github.com.evil.example/iqinc/openpilot.git",
  "https://github.com/iqinc/openpilot.git?other=repo",
  "https://git.konn3kt.com/teal/openpilot.git",
  "https://git.konn3kt.com/IQ.Lvbs/IQ.Pilot.git-other",
  "https://git.konn3kt.com.evil.example/IQ.Lvbs/IQ.Pilot.git",
  "git@ssh.gitlvb.teallvbs.xyz.evil.example:IQ.Lvbs/IQ.Pilot.git",
  "https://git.konn3kt.com:8443/IQ.Lvbs/IQ.Pilot.git",
  "https://git.konn3kt.com/IQ.Lvbs/IQ.Pilot.git?other=repo",
  "https://git.konn3kt.com/IQ.Lvbs/IQ.Pilot.git#other",
  "https://username@git.konn3kt.com/IQ.Lvbs/IQ.Pilot.git",
  "http://git.konn3kt.com/IQ.Lvbs/IQ.Pilot.git",
])
def test_other_origins_keep_existing_save_behavior(device_repo, preview_access, origin):
  set_origin(device_repo, origin)
  git_creds.set_credentials("tester", "token")
  assert get_origin(device_repo) == origin
  assert git_creds.get_credentials() == ("tester", "token")
  assert preview_access.requests == []


@pytest.mark.parametrize("username", ["", "tester"])
def test_empty_token_never_switches_origin(device_repo, preview_access, username):
  git_creds.set_credentials("previous", "previous-token")
  origin = RELEASE_ORIGIN
  set_origin(device_repo, origin)
  git_creds.set_credentials(username, " ")
  assert get_origin(device_repo) == origin
  assert git_creds.get_credentials() == ((username, "") if username else None)
  assert preview_access.requests == []


@pytest.mark.parametrize("origin", [
  "https://gitlvb.teallvbs.xyz/IQ.Lvbs/IQ.Pilot",
  "https://GIT.KONN3KT.COM/iq.lvbs/iq.pilot.git/",
  "ssh://git@git-ssh.konn3kt.com:22/IQ.Lvbs/IQ.Pilot.git",
  "https://github.com/iqinc/openpilot",
  "https://GITHUB.COM/IQINC/OPENPILOT.git/",
])
def test_equivalent_release_urls_migrate(device_repo, preview_access, origin):
  set_origin(device_repo, origin)
  git_creds.set_credentials("tester", "token")
  assert get_origin(device_repo) == PREVIEW_ORIGIN
  assert preview_access.requests[0][0][-1] == PREVIEW_ORIGIN


@pytest.mark.parametrize("origin,expected", [
  ("https://gitlvb.teallvbs.xyz/IQ.Lvbs/dependencies.git", "https://git.konn3kt.com/IQ.Lvbs/dependencies.git"),
  ("git@git-ssh.konn3kt.com:teal/openpilot.git", "https://git.konn3kt.com/teal/openpilot.git"),
  ("ssh://git@ssh-git.konn3kt.xyz/teal/openpilot.git", "https://git.konn3kt.com/teal/openpilot.git"),
  ("git@ssh.gitlvb.teallvbs.xyz:teal/openpilot.git", "https://git.konn3kt.com/teal/openpilot.git"),
  ("git@github.com:owner/repo.git", "https://github.com/owner/repo.git"),
  ("https://github.com/owner/repo.git", "https://github.com/owner/repo.git"),
])
def test_git_host_normalization_preserves_repository_path(origin, expected):
  assert git_creds.ssh_to_https(origin) == expected


@pytest.mark.parametrize("origin", [
  "https://gitlvb.teallvbs.xyz/IQ.Lvbs/IQ.Pilot-Preview.git",
  "git@ssh.gitlvb.teallvbs.xyz:IQ.Lvbs/IQ.Pilot-Preview.git",
  "git@git-ssh.konn3kt.com:IQ.Lvbs/IQ.Pilot-Preview.git",
  "ssh://git@ssh-git.konn3kt.xyz/IQ.Lvbs/IQ.Pilot-Preview.git",
])
def test_updater_with_stored_credentials_normalizes_origin(device_repo, preview_access, monkeypatch, origin):
  git_creds.set_credentials("tester", "token")
  set_origin(device_repo, origin)
  for key in ("GIT_CONFIG_COUNT", "GIT_CONFIG_KEY_0", "GIT_CONFIG_VALUE_0"):
    monkeypatch.setenv(key, "0" if key == "GIT_CONFIG_COUNT" else "unused")

  git_creds.configure(str(device_repo))

  assert get_origin(device_repo) == PREVIEW_ORIGIN
  assert git_creds.os.environ["GIT_CONFIG_KEY_0"] == f"http.{PREVIEW_ORIGIN}/.extraHeader"
  assert preview_access.requests == []


@pytest.fixture
def updater_access(device_repo, monkeypatch):
  public_repo = device_repo.parent / "public"
  subprocess.run(["git", "init", "-q", "-b", "release", str(public_repo)], check=True)
  (public_repo / "build").write_text("public build")
  subprocess.run(["git", "-C", str(public_repo), "add", "build"], check=True)
  subprocess.run(["git", "-C", str(public_repo), "-c", "user.name=Tester", "-c", "user.email=test@example.com",
                  "commit", "-qm", "published"], check=True)
  subprocess.run(["git", "-C", str(public_repo), "branch", "feature/shared"], check=True)
  subprocess.run(["git", "-C", str(public_repo), "branch", "alpha"], check=True)
  mirror_repo = device_repo.parent / "github"
  subprocess.run(["git", "clone", "-q", "--bare", str(public_repo), str(mirror_repo)], check=True)
  subprocess.run(["git", "-C", str(device_repo), "-c", "user.name=Tester", "-c", "user.email=test@example.com",
                  "commit", "--allow-empty", "-qm", "installed"], check=True)
  subprocess.run(["git", "-C", str(device_repo), "config", "remote.origin.fetch", "+refs/heads/*:refs/remotes/origin/*"], check=True)
  git_creds.set_credentials("tester", "expired-token")
  set_origin(device_repo, PREVIEW_ORIGIN)
  for key, value in (("GIT_CONFIG_COUNT", "0"), ("GIT_CONFIG_KEY_0", "unused"), ("GIT_CONFIG_VALUE_0", "unused")):
    monkeypatch.setenv(key, value)

  access = SimpleNamespace(public_repo=public_repo, mirror_repo=mirror_repo, requests=[], failures={PREVIEW_ORIGIN: "auth"},
                           preview_auth=git_creds._auth_header("tester", "expired-token"))
  run_git = subprocess.run

  def failure_response(args, kwargs, failure):
    if failure == "timeout":
      raise subprocess.TimeoutExpired(args, kwargs.get("timeout", 10))
    diagnostic = {
      "auth": "Authentication failed",
      "offline": "Connection failed",
      "http500": "The requested URL returned error: 500",
      "http503": "The requested URL returned error: 503",
    }[failure]
    if kwargs.get("check"):
      raise subprocess.CalledProcessError(128, args, output=diagnostic)
    return subprocess.CompletedProcess(args, 128, "", diagnostic)

  def run(args, **kwargs):
    if "ls-remote" not in args and "fetch" not in args:
      return run_git(args, **kwargs)
    repo = kwargs["cwd"]
    env = dict(kwargs.get("env", os.environ))
    origin = next((arg for arg in args if arg.startswith(("https://", "http://"))), None)
    if origin is None:
      origin = get_origin(repo)
    access.requests.append((origin, args, env))
    if origin == RELEASE_ORIGIN:
      assert git_creds._auth_header("tester", "expired-token") not in env.values()
      assert run_git(["git", "-C", str(repo), "config", "--get", f"http.{RELEASE_ORIGIN}/.extraHeader"],
                     capture_output=True, text=True, check=True).stdout.strip() == "Authorization: Basic fleet-auth"
    elif origin == GITHUB_MIRROR_ORIGIN:
      assert git_creds._auth_header("tester", "expired-token") not in env.values()
      assert env["GIT_CONFIG_KEY_1"] == f"http.{GITHUB_MIRROR_ORIGIN}/.extraHeader"
      assert env["GIT_CONFIG_VALUE_1"] == ""
    if origin in (RELEASE_ORIGIN, GITHUB_MIRROR_ORIGIN):
      assert env["GIT_CONFIG_KEY_0"] == "credential.helper"
      assert env["GIT_CONFIG_VALUE_0"] == ""
    elif origin == PREVIEW_ORIGIN and access.preview_auth not in env.values():
      return failure_response(args, kwargs, "auth")
    failure = access.failures.get(origin, None if origin in (RELEASE_ORIGIN, GITHUB_MIRROR_ORIGIN) else "auth")
    if failure is not None:
      return failure_response(args, kwargs, failure)
    # Use real Git for refs and fetches, replacing only the transport URL.
    backing_repo = mirror_repo if origin == GITHUB_MIRROR_ORIGIN else public_repo
    index = int(env["GIT_CONFIG_COUNT"])
    env.update({
      "GIT_CONFIG_COUNT": str(index + 1),
      f"GIT_CONFIG_KEY_{index}": f"url.{backing_repo.as_uri()}.insteadOf",
      f"GIT_CONFIG_VALUE_{index}": origin,
    })
    return run_git(args, **{**kwargs, "env": env})

  def configure_auth(repo):
    if get_origin(repo) == RELEASE_ORIGIN:
      subprocess.run(["git", "-C", str(repo), "config", f"http.{RELEASE_ORIGIN}/.extraHeader",
                      "Authorization: Basic fleet-auth"], check=True)
    git_creds.configure(str(repo))

  access.configure_auth = configure_auth
  monkeypatch.setattr(subprocess, "run", run)
  return access


def test_recovery_fetch_returns_its_branch_when_another_request_is_queued(device_repo, updater_access, monkeypatch):
  git_creds._params_put("UpdaterTargetBranch", b"feature/preview-only")
  replace_target = git_creds._replace_target

  def queue_next_request(previous, branch):
    replace_target(previous, branch)
    git_creds._params_put("UpdaterTargetBranch", b"zenith")

  monkeypatch.setattr(git_creds, "_replace_target", queue_next_request)
  assert git_creds.fetch_remote(str(device_repo), "feature/preview-only", updater_access.configure_auth) == "alpha"
  assert git_creds._params_get("UpdaterTargetBranch") == b"zenith"
  assert subprocess.check_output(["git", "-C", str(device_repo), "rev-parse", "FETCH_HEAD"]) == subprocess.check_output(
    ["git", "-C", str(updater_access.public_repo), "rev-parse", "alpha"])


@pytest.mark.parametrize("branch", ["release", "feature/shared"])
def test_shared_branch_falls_back_without_changing_credentials_or_checkout(device_repo, updater_access, branch):
  previous_blob = (device_repo.parent / "params" / "GitAuthBlob").read_bytes()
  previous_head = subprocess.check_output(["git", "-C", str(device_repo), "rev-parse", "HEAD"])

  output = git_creds.run_remote(["git", "ls-remote", "--heads"], str(device_repo), branch, updater_access.configure_auth)

  assert f"refs/heads/{branch}" in output
  assert get_origin(device_repo) == RELEASE_ORIGIN
  assert (device_repo.parent / "params" / "GitAuthBlob").read_bytes() == previous_blob
  assert subprocess.check_output(["git", "-C", str(device_repo), "rev-parse", "HEAD"]) == previous_head
  assert [origin for origin, _, _ in updater_access.requests] == [PREVIEW_ORIGIN, PREVIEW_ORIGIN, RELEASE_ORIGIN, RELEASE_ORIGIN]


def test_fetch_retries_public_and_populates_origin_tracking_refs(device_repo, updater_access):
  previous_head = subprocess.check_output(["git", "-C", str(device_repo), "rev-parse", "HEAD"])
  git_creds.run_remote(["git", "fetch", "origin", "release"], str(device_repo), "release", updater_access.configure_auth)

  published_head = subprocess.check_output(["git", "-C", str(updater_access.public_repo), "rev-parse", "HEAD"])
  for ref in ("FETCH_HEAD", "refs/remotes/origin/release"):
    assert subprocess.check_output(["git", "-C", str(device_repo), "rev-parse", ref]) == published_head
  assert subprocess.check_output(["git", "-C", str(device_repo), "rev-parse", "HEAD"]) == previous_head
  assert get_origin(device_repo) == RELEASE_ORIGIN


@pytest.mark.parametrize("branch", ["feature/preview-only", "feature/sh", "release-extra"])
def test_missing_public_branch_uses_alpha(device_repo, updater_access, branch):
  git_creds._params_put("UpdaterTargetBranch", branch.encode())
  previous_blob = (device_repo.parent / "params" / "GitAuthBlob").read_bytes()
  output = git_creds.run_remote(["git", "ls-remote", "--heads"], str(device_repo), branch, updater_access.configure_auth)

  assert "refs/heads/alpha" in output
  assert get_origin(device_repo) == RELEASE_ORIGIN
  assert git_creds._params_get("UpdaterTargetBranch") == b"alpha"
  assert (device_repo.parent / "params" / "GitAuthBlob").read_bytes() == previous_blob


@pytest.mark.parametrize("clear_via_settings", [False, True])
@pytest.mark.parametrize("selected,current,expected", [
  ("feature/preview-only", "release", b"alpha"),
  (None, "feature/preview-only", b"alpha"),
  ("feature/shared", "feature/preview-only", b"feature/shared"),
  (None, "feature/shared", None),
])
def test_clear_credentials_leaves_preview_immediately(device_repo, updater_access, monkeypatch, clear_via_settings,
                                                     selected, current, expected):
  subprocess.run(["git", "-C", str(device_repo), "branch", "-m", current], check=True)
  if selected is not None:
    git_creds._params_put("UpdaterTargetBranch", selected.encode())
  configure = git_creds.configure

  def configure_auth(repo):
    if get_origin(repo) == RELEASE_ORIGIN:
      subprocess.run(["git", "-C", str(repo), "config", f"http.{RELEASE_ORIGIN}/.extraHeader",
                      "Authorization: Basic fleet-auth"], check=True)
    configure(repo)

  monkeypatch.setattr(git_creds, "configure", configure_auth)
  if clear_via_settings:
    git_creds.set_credentials("", "")
  else:
    git_creds.clear_credentials()

  assert git_creds.get_credentials() is None
  assert get_origin(device_repo) == RELEASE_ORIGIN
  assert git_creds._params_get("UpdaterTargetBranch") == expected
  assert subprocess.check_output(["git", "-C", str(device_repo), "branch", "--show-current"], text=True).strip() == current
  assert all(origin != PREVIEW_ORIGIN for origin, _, _ in updater_access.requests)
  assert git_creds._auth_header("tester", "expired-token") not in os.environ.values()


@pytest.mark.parametrize("failure", ["offline", "http503", "timeout"])
def test_clear_credentials_leaves_preview_even_when_offline(device_repo, updater_access, failure):
  updater_access.failures.update({RELEASE_ORIGIN: failure, GITHUB_MIRROR_ORIGIN: failure})
  git_creds.clear_credentials()
  assert get_origin(device_repo) == RELEASE_ORIGIN
  assert git_creds.get_credentials() is None


def test_preview_only_fetch_retries_alpha(device_repo, updater_access):
  branch = "feature/preview-only"
  git_creds._params_put("UpdaterTargetBranch", branch.encode())
  git_creds.run_remote(["git", "fetch", "origin", branch], str(device_repo), branch, updater_access.configure_auth)
  assert get_origin(device_repo) == RELEASE_ORIGIN
  assert git_creds._params_get("UpdaterTargetBranch") == b"alpha"
  assert subprocess.check_output(["git", "-C", str(device_repo), "rev-parse", "FETCH_HEAD"]) == subprocess.check_output(
    ["git", "-C", str(updater_access.public_repo), "rev-parse", "alpha"])


@pytest.mark.parametrize("failure", ["offline", "timeout"])
def test_failed_public_probe_preserves_preview(device_repo, updater_access, failure):
  updater_access.failures.update({RELEASE_ORIGIN: failure, GITHUB_MIRROR_ORIGIN: failure})
  with pytest.raises(subprocess.CalledProcessError):
    git_creds.run_remote(["git", "fetch", "origin", "release"], str(device_repo), "release", updater_access.configure_auth)
  assert get_origin(device_repo) == PREVIEW_ORIGIN
  assert git_creds.get_credentials() == ("tester", "expired-token")
  assert len(updater_access.requests) == 4


def test_valid_preview_access_does_not_fall_back(device_repo, updater_access):
  updater_access.failures[PREVIEW_ORIGIN] = None
  output = git_creds.run_remote(["git", "ls-remote", "--heads"], str(device_repo), "release", updater_access.configure_auth)
  assert "refs/heads/release" in output
  assert get_origin(device_repo) == PREVIEW_ORIGIN
  assert len(updater_access.requests) == 1


def test_local_ref_lock_with_valid_preview_access_preserves_origin(device_repo, updater_access):
  updater_access.failures[PREVIEW_ORIGIN] = None
  lock = device_repo / ".git/refs/remotes/origin/release.lock"
  lock.parent.mkdir(parents=True, exist_ok=True)
  lock.touch()
  previous_blob = (device_repo.parent / "params" / "GitAuthBlob").read_bytes()

  with pytest.raises(subprocess.CalledProcessError) as error:
    git_creds.run_remote(["git", "fetch", "origin", "release"], str(device_repo), "release", updater_access.configure_auth)

  assert "cannot lock ref" in error.value.output
  assert get_origin(device_repo) == PREVIEW_ORIGIN
  assert (device_repo.parent / "params" / "GitAuthBlob").read_bytes() == previous_blob
  assert [origin for origin, _, _ in updater_access.requests] == [PREVIEW_ORIGIN, PREVIEW_ORIGIN]


@pytest.mark.parametrize("failure", ["offline", "http500", "http503", "timeout"])
def test_transient_preview_failure_preserves_origin(device_repo, updater_access, failure):
  git_creds._params_put("UpdaterTargetBranch", b"feature/preview-only")
  updater_access.failures[PREVIEW_ORIGIN] = failure
  error = subprocess.TimeoutExpired if failure == "timeout" else subprocess.CalledProcessError
  with pytest.raises(error):
    git_creds.run_remote(["git", "fetch", "origin", "release"], str(device_repo), "release", updater_access.configure_auth)
  assert get_origin(device_repo) == PREVIEW_ORIGIN
  assert git_creds.get_credentials() == ("tester", "expired-token")
  assert all(origin == PREVIEW_ORIGIN for origin, _, _ in updater_access.requests)
  assert git_creds._params_get("UpdaterTargetBranch") == b"feature/preview-only"


@pytest.mark.parametrize("origin", [RELEASE_ORIGIN, GITHUB_MIRROR_ORIGIN])
def test_public_updates_ignore_rejected_preview_token(device_repo, updater_access, origin):
  set_origin(device_repo, origin)
  git_creds.run_remote(["git", "ls-remote", "--heads"], str(device_repo), "release", updater_access.configure_auth)
  assert get_origin(device_repo) == origin
  assert len(updater_access.requests) == 1
  assert git_creds.get_credentials() == ("tester", "expired-token")


@pytest.mark.parametrize("origin,expected", [
  ("https://git.konn3kt.com/IQ.Lvbs/./IQ.Pilot.git", RELEASE_ORIGIN),
  ("https://git.konn3kt.com/IQ.Lvbs/other/../IQ.Pilot.git", RELEASE_ORIGIN),
  ("https://github.com/iqinc/%6fpenpilot.git", GITHUB_MIRROR_ORIGIN),
  ("https://github.com/iqinc/other/../openpilot.git", GITHUB_MIRROR_ORIGIN),
  ("https://tester@git.konn3kt.com/IQ.Lvbs/IQ.Pilot.git", RELEASE_ORIGIN),
  ("https://git@github.com/iqinc/openpilot.git", GITHUB_MIRROR_ORIGIN),
  ("https://git.konn3kt.com./IQ.Lvbs/IQ.Pilot.git", RELEASE_ORIGIN),
  ("https://github.com./iqinc/openpilot.git", GITHUB_MIRROR_ORIGIN),
  ("https://git.konn3kt.com/IQ.Lvbs/IQ.Pilot.git?source=public", RELEASE_ORIGIN),
  ("https://github.com:8443/iqinc/openpilot.git", GITHUB_MIRROR_ORIGIN),
  ("https://git。konn3kt.com/IQ.Lvbs/IQ.Pilot.git", RELEASE_ORIGIN),
  ("https://github。com/iqinc/openpilot.git", GITHUB_MIRROR_ORIGIN),
  ("https://github.com/iqinc/iqpilot.git", RELEASE_ORIGIN),
  ("git@github.com:iqinc/iqpilot.git", RELEASE_ORIGIN),
  ("ssh://git@github.com/iqinc/iqpilot.git", RELEASE_ORIGIN),
  ("https://github.com/iqinc/%69qpilot.git", RELEASE_ORIGIN),
  ("https://github.com/iqinc/other/../iqpilot.git", RELEASE_ORIGIN),
  ("https://github.com/iqinc/./iqpilot.git/", RELEASE_ORIGIN),
  ("https://tester@github.com/iqinc/iqpilot.git", RELEASE_ORIGIN),
  ("https://github.com:8443/iqinc/iqpilot.git?source=public#ref", RELEASE_ORIGIN),
  ("https://github.com./iqinc/iqpilot.git", RELEASE_ORIGIN),
  ("https://github。com/iqinc/iqpilot.git", RELEASE_ORIGIN),
])
def test_public_url_aliases_never_receive_tester_auth(device_repo, updater_access, origin, expected):
  set_origin(device_repo, origin)
  git_creds.configure(str(device_repo))
  assert get_origin(device_repo) == expected
  assert git_creds.get_credentials_for_repo(str(device_repo)) is None
  assert git_creds._auth_header("tester", "expired-token") not in os.environ.values()


def test_custom_repository_never_falls_back(device_repo, updater_access):
  origin = "https://git.konn3kt.com/teal/openpilot.git"
  set_origin(device_repo, origin)
  with pytest.raises(subprocess.CalledProcessError):
    git_creds.run_remote(["git", "ls-remote", "--heads"], str(device_repo), "release", updater_access.configure_auth)
  assert get_origin(device_repo) == origin
  assert len(updater_access.requests) == 1


def test_locked_origin_preserves_preview_after_successful_public_probe(device_repo, updater_access):
  (device_repo / ".git" / "config.lock").touch()
  with pytest.raises(subprocess.CalledProcessError):
    git_creds.run_remote(["git", "ls-remote", "--heads"], str(device_repo), "release", updater_access.configure_auth)
  assert get_origin(device_repo) == PREVIEW_ORIGIN
  assert git_creds.get_credentials() == ("tester", "expired-token")
  assert len(updater_access.requests) == 3


@pytest.mark.parametrize("failure", ["offline", "timeout"])
@pytest.mark.parametrize("cmd", [["git", "ls-remote", "--heads"], ["git", "fetch", "origin", "release"]])
def test_gitea_failure_recovers_through_anonymous_github_mirror(device_repo, updater_access, failure, cmd):
  updater_access.failures[RELEASE_ORIGIN] = failure
  previous_head = subprocess.check_output(["git", "-C", str(device_repo), "rev-parse", "HEAD"])

  git_creds.run_remote(cmd, str(device_repo), "release", updater_access.configure_auth)

  assert get_origin(device_repo) == GITHUB_MIRROR_ORIGIN
  assert git_creds.get_credentials() == ("tester", "expired-token")
  assert subprocess.check_output(["git", "-C", str(device_repo), "rev-parse", "HEAD"]) == previous_head
  assert [origin for origin, _, _ in updater_access.requests] == [
    PREVIEW_ORIGIN, PREVIEW_ORIGIN, RELEASE_ORIGIN, GITHUB_MIRROR_ORIGIN, GITHUB_MIRROR_ORIGIN,
  ]
  if "fetch" in cmd:
    mirror_head = subprocess.check_output(["git", "-C", str(updater_access.mirror_repo), "rev-parse", "HEAD"])
    for ref in ("FETCH_HEAD", "refs/remotes/origin/release"):
      assert subprocess.check_output(["git", "-C", str(device_repo), "rev-parse", ref]) == mirror_head


def test_branch_missing_on_gitea_can_fall_back_to_github(device_repo, updater_access):
  subprocess.run(["git", "-C", str(updater_access.public_repo), "branch", "-D", "feature/shared"], capture_output=True, check=True)

  output = git_creds.run_remote(["git", "ls-remote", "--heads"], str(device_repo), "feature/shared", updater_access.configure_auth)

  assert "refs/heads/feature/shared" in output
  assert get_origin(device_repo) == GITHUB_MIRROR_ORIGIN


@pytest.mark.parametrize("origin,expected", [
  (PREVIEW_ORIGIN, "Basic " + base64.b64encode(b"tester:preview-token").decode()),
  (RELEASE_ORIGIN, "Basic test-fleet"),
  (GITHUB_MIRROR_ORIGIN, None),
])
def test_remote_requests_send_only_the_correct_authorization_header(device_repo, origin, expected):
  git_creds.set_credentials("tester", "preview-token")
  set_origin(device_repo, origin)
  headers = []

  class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
      headers.append(self.headers.get_all("Authorization"))
      self.send_response(403)
      self.end_headers()

    def log_message(self, *args):
      pass

  with HTTPServer(("127.0.0.1", 0), Handler) as server:
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
      url = f"http://127.0.0.1:{server.server_port}/repo.git"
      subprocess.run(["git", "-C", str(device_repo), "config", f"http.{url}/.extraHeader", "Authorization: Basic test-fleet"], check=True)
      git_creds.configure(str(device_repo))
      env = os.environ.copy()
      # Exercise the configured auth over loopback without contacting Gitea.
      for key in env:
        if key.startswith("GIT_CONFIG_KEY_"):
          env[key] = env[key].replace(origin, url)
      env["GIT_TERMINAL_PROMPT"] = "0"
      subprocess.run(["git", "ls-remote", url], cwd=device_repo, env=env, capture_output=True, timeout=5)
      assert headers == [None if expected is None else [expected]]
    finally:
      server.shutdown()
      thread.join()


@pytest.mark.parametrize("status,rejected", [(401, True), (403, True), (404, True), (500, False), (503, False)])
def test_preview_probe_distinguishes_rejected_access_from_outage(device_repo, monkeypatch, status, rejected):
  git_creds.set_credentials("tester", "preview-token")
  headers = []

  class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
      headers.append(self.headers.get_all("Authorization"))
      self.send_response(status)
      if status == 401:
        self.send_header("WWW-Authenticate", 'Basic realm="test"')
      self.end_headers()

    def log_message(self, *args):
      pass

  with HTTPServer(("127.0.0.1", 0), Handler) as server:
    monkeypatch.setattr(git_creds, "_PREVIEW_ORIGIN", f"http://127.0.0.1:{server.server_port}/preview.git")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
      assert git_creds._preview_access_rejected() is rejected
      expected = "Basic " + base64.b64encode(b"tester:preview-token").decode()
      assert headers and all(header == [expected] for header in headers)
    finally:
      server.shutdown()
      thread.join()


@pytest.mark.parametrize("another_fetch", [False, True])
def test_fetch_keeps_fetched_branch_when_a_new_target_arrives(device_repo, monkeypatch, another_fetch):
  from iqpilot.system.updated import updated
  source = device_repo.parent / "source"
  subprocess.run(["git", "init", "-q", "-b", "release", str(source)], check=True)
  for branch, build in (("release", "public build"), ("zenith", "preview build")):
    subprocess.run(["git", "-C", str(source), "checkout", "-q", "-B", branch], check=True)
    (source / "build").write_text(build)
    subprocess.run(["git", "-C", str(source), "add", "build"], check=True)
    subprocess.run(["git", "-C", str(source), "-c", "user.name=Tester", "-c", "user.email=test@example.com",
                    "commit", "-qm", build], check=True)
  subprocess.run(["git", "-C", str(device_repo), "-c", "user.name=Tester", "-c", "user.email=test@example.com",
                  "commit", "--allow-empty", "-qm", "installed"], check=True)
  set_origin(device_repo, str(source))
  values = {"UpdaterTargetBranch": "release"}
  updater = updated.Updater.__new__(updated.Updater)
  updater.params = SimpleNamespace(get=lambda key, **_kwargs: values.get(key), put=values.__setitem__, put_bool=values.__setitem__)
  monkeypatch.setattr(updated, "BASEDIR", str(device_repo))
  monkeypatch.setattr(updated, "setup_git_options", lambda _repo: None)
  monkeypatch.setattr(updated, "configure_git_auth", lambda _repo: None)

  def checkpoint(repo):
    if another_fetch:
      subprocess.run(["git", "fetch", "origin", "zenith"], cwd=repo, check=True, capture_output=True)

  monkeypatch.setattr(updated, "checkpoint", checkpoint)
  monkeypatch.setattr(updated, "cleanup_stale_prebuilt_marker", lambda _repo, _branch: None)
  monkeypatch.setattr(updated, "AGNOS", False)
  remote_output = git_creds._remote_output

  def request_next_branch(args, repo_dir, timeout):
    output = remote_output(args, repo_dir, timeout)
    if args == ["git", "fetch", "origin", "release"]:
      values["UpdaterTargetBranch"] = "zenith"
    return output

  monkeypatch.setattr(git_creds, "_remote_output", request_next_branch)
  updater.fetch_update()
  assert updater.get_branch(str(device_repo)) == "release"
  assert (device_repo / "build").read_text() == "public build"
  assert values["UpdaterTargetBranch"] == "zenith"


def test_legacy_source_remotes_are_removed_without_touching_user_remotes(device_repo):
  set_origin(device_repo, PREVIEW_ORIGIN)
  remotes = {
    "iqpilot-public": RELEASE_ORIGIN,
    "iqpilot-preview": PREVIEW_ORIGIN,
    "iqpilot-public-2": GITHUB_MIRROR_ORIGIN,
    "iqpilot-preview-fork": PREVIEW_ORIGIN,
    "iqpilot-public-3": "https://example.com/someone/openpilot.git",
    "upstream": RELEASE_ORIGIN,
  }
  for name, url in remotes.items():
    subprocess.run(["git", "-C", str(device_repo), "remote", "add", name, url], check=True)
  subprocess.run(["git", "-C", str(device_repo), "config", "remote.iqpilot-preview-2.fetch", "+refs/heads/*:refs/x/*"], check=True)
  lock = device_repo / ".git" / "iqpilot-sources.lock"
  lock.write_text("")

  git_creds.remove_legacy_source_remotes(str(device_repo))

  left = subprocess.check_output(["git", "-C", str(device_repo), "remote"], text=True).split()
  assert sorted(left) == ["iqpilot-preview-2", "iqpilot-preview-fork", "iqpilot-public-3", "origin", "upstream"]
  assert get_origin(device_repo) == PREVIEW_ORIGIN
  assert not lock.exists()
  git_creds.remove_legacy_source_remotes(str(device_repo))


def test_fallback_keeps_a_branch_selected_while_it_probes(device_repo, updater_access):
  git_creds._params_put("UpdaterTargetBranch", b"feature/preview-only")

  def select_during_probe(repo):
    if "iqpilot-public-" in str(repo):
      git_creds._params_put("UpdaterTargetBranch", b"feature/shared")
    updater_access.configure_auth(repo)

  assert git_creds.fetch_remote(str(device_repo), "feature/preview-only", select_during_probe) == "alpha"
  assert git_creds._params_get("UpdaterTargetBranch") == b"feature/shared"


@pytest.mark.parametrize("origin,target,stored,listing,expected", [
  (RELEASE_ORIGIN, "feature/preview-only", "feature/preview-only", {"alpha", "release"}, "alpha"),
  (GITHUB_MIRROR_ORIGIN, "feature/preview-only", None, {"alpha"}, "alpha"),
  (RELEASE_ORIGIN, "feature/preview-only", "feature/preview-only", {"release"}, "feature/preview-only"),
  (RELEASE_ORIGIN, "release", "release", {"alpha", "release"}, "release"),
  (PREVIEW_ORIGIN, "feature/preview-only", "feature/preview-only", {"alpha"}, "feature/preview-only"),
  (RELEASE_ORIGIN, "feature/preview-only", "feature/shared", {"alpha", "feature/shared"}, "feature/shared"),
])
def test_missing_public_target_moves_to_alpha_only_when_alpha_exists(device_repo, origin, target, stored, listing, expected):
  set_origin(device_repo, origin)
  if stored is not None:
    git_creds._params_put("UpdaterTargetBranch", stored.encode())
  git_creds.retarget_missing_public_branch(str(device_repo), target, listing)
  assert (git_creds._params_get("UpdaterTargetBranch") or target.encode()) == expected.encode()


def test_bounded_remote_kills_transport_children(tmp_path):
  import time
  import psutil

  child_file = tmp_path / 'transport.pid'
  with pytest.raises(subprocess.TimeoutExpired):
    git_creds.run_remote(['sh', '-c', 'sleep 45 & echo $! > "$1"; wait', '_', str(child_file)], str(tmp_path), 'alpha',
                         lambda _: None, timeout=0.5)
  child = int(child_file.read_text())
  # A dead child may briefly remain a zombie until PID 1 reaps it in containers.
  for _ in range(20):
    try:
      if psutil.Process(child).status() == psutil.STATUS_ZOMBIE:
        break
    except psutil.NoSuchProcess:
      break
    time.sleep(0.05)
  else:
    pytest.fail('remote transport child survived the timeout')
