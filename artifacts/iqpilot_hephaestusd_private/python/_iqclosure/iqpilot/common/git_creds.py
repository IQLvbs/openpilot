import base64
import fcntl
import json
import os
import posixpath
import re
import signal
import subprocess
import tempfile
from collections.abc import Callable, Container, Iterator
from contextlib import contextmanager
from urllib.parse import unquote, urlsplit

from iqpilot.common.params import Params

PARAM = "GitAuthBlob"
PARAMS_DIR = os.environ.get("PARAMS_DIR", "/data/params/d")
KEY_DIR = "/data/konn3kt"
KEY_PATH = os.path.join(KEY_DIR, "git_auth.key")
HELPER_PATH = os.path.join(KEY_DIR, "git_credential_helper.py")
DEFAULT_REPO_DIR = "/data/openpilot"
CREDENTIAL_HOSTS = ("git.konn3kt.com", "gitlvb.teallvbs.xyz")
_GIT_HTTPS_HOST = "git.konn3kt.com"
_GIT_HOST_ALIASES = {
  _GIT_HTTPS_HOST,
  "git-ssh.konn3kt.com",
  "ssh-git.konn3kt.xyz",
  "gitlvb.teallvbs.xyz",
  "ssh.gitlvb.teallvbs.xyz",
}
_PREVIEW_ORIGIN = f"https://{_GIT_HTTPS_HOST}/IQ.Lvbs/IQ.Pilot-Preview.git"
_PUBLIC_ORIGIN = f"https://{_GIT_HTTPS_HOST}/IQ.Lvbs/IQ.Pilot.git"
_GITHUB_MIRROR_ORIGIN = "https://github.com/iqinc/openpilot.git"
_PUBLIC_ORIGINS = (_PUBLIC_ORIGIN, _GITHUB_MIRROR_ORIGIN)
# Use release once release installs again.
_PUBLIC_FALLBACK_BRANCH = "alpha"
_AMBIENT_GIT_ENV = ("GIT_CONFIG", "GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_ASKPASS", "SSH_ASKPASS")
_ACCESS_REJECTED = re.compile(
  "|".join((
    r"authentication failed",
    r"terminal prompts disabled",
    r"returned error: (?:401|403|404)\b",
    r"repository .* not found",
  )),
  re.IGNORECASE,
)

_HELPER_SCRIPT = '''#!/usr/bin/env python3
import json
import os
import sys

KEY_PATH = "{key_path}"


def main() -> None:
  if len(sys.argv) < 2 or sys.argv[1] != "get":
    return
  for line in sys.stdin:
    if not line.strip():
      break
  params_dir = os.environ.get("PARAMS_DIR", "/data/params/d")
  blob_path = os.path.join(params_dir, "GitAuthBlob")
  try:
    with open(KEY_PATH, "rb") as f:
      key = f.read().strip()
    with open(blob_path, "rb") as f:
      blob = f.read()
    if not blob:
      return
    from cryptography.fernet import Fernet
    data = json.loads(Fernet(key).decrypt(blob).decode())
    username = data.get("u", "")
    token = data.get("t", "")
    if username and token:
      sys.stdout.write("username=%s\\npassword=%s\\n" % (username, token))
  except Exception:
    return


if __name__ == "__main__":
  main()
'''


def _params_get(name: str) -> bytes | None:
  # During bootstrap, Params dependencies may not be installed yet.
  try:
    return Params().get(name)
  except Exception:
    try:
      with open(os.path.join(PARAMS_DIR, name), "rb") as f:
        return f.read()
    except OSError:
      return None


def _params_put(name: str, value: bytes) -> None:
  try:
    Params().put(name, value)
    return
  except Exception:
    pass
  os.makedirs(PARAMS_DIR, exist_ok=True)
  tmp = os.path.join(PARAMS_DIR, f".tmp_{name}")
  fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
  with os.fdopen(fd, "wb") as f:
    f.write(value)
    f.flush()
    os.fsync(f.fileno())
  os.replace(tmp, os.path.join(PARAMS_DIR, name))


def _params_remove(name: str) -> None:
  try:
    Params().remove(name)
    return
  except Exception:
    pass
  try:
    os.unlink(os.path.join(PARAMS_DIR, name))
  except OSError:
    pass


def _load_or_create_key() -> bytes:
  from cryptography.fernet import Fernet
  try:
    with open(KEY_PATH, "rb") as f:
      return f.read().strip()
  except FileNotFoundError:
    pass
  key = Fernet.generate_key()
  os.makedirs(KEY_DIR, exist_ok=True)
  tmp = KEY_PATH + ".tmp"
  fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
  with os.fdopen(fd, "wb") as f:
    f.write(key)
  os.replace(tmp, KEY_PATH)
  return key


def _origin_url(repo_dir: str) -> str:
  return subprocess.run(
    ["git", "-C", repo_dir, "config", "--get", "remote.origin.url"],
    capture_output=True, text=True, check=False,
  ).stdout.strip()


def _canonical_iqpilot_origin(origin: str) -> str | None:
  parsed = urlsplit(ssh_to_https(origin))
  if (parsed.scheme != "https" or parsed.hostname not in (_GIT_HTTPS_HOST, "github.com") or parsed.username is not None
      or parsed.port not in (None, 443) or parsed.query or parsed.fragment):
    return None
  path = posixpath.normpath("/" + unquote(parsed.path).lstrip("/")).rstrip("/").lower().removesuffix(".git")
  if parsed.hostname == "github.com":
    return _GITHUB_MIRROR_ORIGIN if path == "/iqinc/openpilot" else None
  return {"/iq.lvbs/iq.pilot": _PUBLIC_ORIGIN, "/iq.lvbs/iq.pilot-preview": _PREVIEW_ORIGIN}.get(path)


def _public_iqpilot_origin(origin: str) -> str | None:
  # Credential exclusion is broader than eligibility for automatic Preview
  # migration: userinfo, URL metadata and DNS aliases must never expose a token.
  parsed = urlsplit(ssh_to_https(origin))
  host = (parsed.hostname or "").encode("idna").decode("ascii").rstrip(".")
  if host in _GIT_HOST_ALIASES:
    host = _GIT_HTTPS_HOST
  if parsed.scheme != "https" or host not in (_GIT_HTTPS_HOST, "github.com"):
    return None
  canonical_url = parsed._replace(netloc=host, query="", fragment="").geturl()
  path = posixpath.normpath("/" + unquote(parsed.path).lstrip("/")).rstrip("/").lower().removesuffix(".git")
  if host == "github.com" and path == "/iqinc/iqpilot":
    return _PUBLIC_ORIGIN
  canonical = _canonical_iqpilot_origin(canonical_url)
  return canonical if canonical in _PUBLIC_ORIGINS else None


def _is_git_env(key: str) -> bool:
  return key in _AMBIENT_GIT_ENV or key.startswith("GIT_CONFIG_")


@contextmanager
def preserve_git_environment() -> Iterator[None]:
  """Restore Git settings after preparing authentication for a subprocess."""
  saved = {key: value for key, value in os.environ.items() if _is_git_env(key)}
  try:
    yield
  finally:
    for key in list(os.environ):
      if _is_git_env(key):
        os.environ.pop(key, None)
    os.environ.update(saved)


def _isolated_git_env() -> dict[str, str]:
  """Remove ambient Git configuration, repository context, and credential prompts."""
  env = os.environ.copy()
  for key in list(env):
    if _is_git_env(key):
      env.pop(key, None)
  env.update({
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_TERMINAL_PROMPT": "0",
    "LC_ALL": "C",
  })
  return env


def _probe_preview_access(username: str, token: str) -> subprocess.CompletedProcess[str]:
  # Fleet credentials and installed helpers must not validate a tester token.
  env = _isolated_git_env()
  env.update({
    "GIT_CONFIG_COUNT": "1",
    "GIT_CONFIG_KEY_0": f"http.{_PREVIEW_ORIGIN}/.extraHeader",
    "GIT_CONFIG_VALUE_0": _auth_header(username, token),
  })
  return subprocess.run(
    ["git", "ls-remote", "--exit-code", "--heads", _PREVIEW_ORIGIN],
    cwd="/", env=env, capture_output=True, text=True, check=False, timeout=10,
  )


def _validated_preview_origin(username: str, token: str) -> str | None:
  if _canonical_iqpilot_origin(_origin_url(DEFAULT_REPO_DIR)) is None:
    return None
  try:
    result = _probe_preview_access(username, token)
  except subprocess.TimeoutExpired:
    raise RuntimeError("Preview access check timed out. Check the device's connection and try again.") from None
  if result.returncode != 0:
    raise ValueError("Unable to read IQ.Pilot-Preview with this token. Check the token, repository access, and connection.")
  return _PREVIEW_ORIGIN


def set_credentials(username: str, token: str) -> None:
  """Save encrypted credentials and enable Preview after verifying access.

  Empty username AND token clears credentials.
  """
  username = (username or "").strip()
  token = (token or "").strip()
  if not username and not token:
    clear_credentials()
    return
  preview_origin = _validated_preview_origin(username, token) if token else None
  from cryptography.fernet import Fernet
  blob = Fernet(_load_or_create_key()).encrypt(
    json.dumps({"u": username, "t": token}).encode()
  )
  previous_blob = _params_get(PARAM) if preview_origin is not None else None
  _params_put(PARAM, blob)
  if preview_origin is not None:
    try:
      subprocess.run(
        ["git", "-C", DEFAULT_REPO_DIR, "config", "remote.origin.url", preview_origin],
        capture_output=True, text=True, check=True,
      )
    except (OSError, subprocess.SubprocessError):
      if previous_blob is None:
        _params_remove(PARAM)
      else:
        _params_put(PARAM, previous_blob)
      raise RuntimeError("Could not switch to IQ.Pilot-Preview. Existing credentials were restored.") from None
  try:
    install_credential_helper(DEFAULT_REPO_DIR)
  except Exception:
    pass


def get_credentials() -> tuple[str, str] | None:
  """Return (username, token), or None if unset / unreadable."""
  blob = _params_get(PARAM)
  if not blob:
    return None
  try:
    from cryptography.fernet import Fernet
    data = json.loads(Fernet(_load_or_create_key()).decrypt(blob).decode())
    return data.get("u", ""), data.get("t", "")
  except Exception:
    return None


def get_credentials_for_repo(repo_dir: str) -> tuple[str, str] | None:
  """Return tester credentials only for repositories that use them."""
  if _public_iqpilot_origin(_origin_url(repo_dir)) is not None:
    return None
  return get_credentials()


def clear_credentials() -> None:
  _params_remove(PARAM)
  if _canonical_iqpilot_origin(_origin_url(DEFAULT_REPO_DIR)) != _PREVIEW_ORIGIN:
    return

  # Leaving Preview is explicit; it must work even while the device is offline.
  subprocess.run(["git", "-C", DEFAULT_REPO_DIR, "config", "remote.origin.url", _PUBLIC_ORIGIN], capture_output=True, check=True)

  def configure_public_auth(repo_dir: str) -> None:
    try:
      from iqpilot.system.proprietary_runtime.bundle_modes import import_verified_module
      import_verified_module("iqpilot_updater_private", "iqpilot_private.updater.git_remote").configure(repo_dir)
    except Exception:
      pass
    configure(repo_dir)

  configure_public_auth(DEFAULT_REPO_DIR)
  selected = _params_get("UpdaterTargetBranch")
  if selected is None:
    branch = subprocess.run(["git", "-C", DEFAULT_REPO_DIR, "symbolic-ref", "--quiet", "--short", "HEAD"],
                            capture_output=True, text=True, check=False).stdout.strip()
  else:
    branch = selected.decode() if isinstance(selected, bytes) else selected
  _switch_to_public(DEFAULT_REPO_DIR, branch, configure_public_auth)
  configure_public_auth(DEFAULT_REPO_DIR)


def has_credentials() -> bool:
  return get_credentials() is not None


def _auth_header(username: str, token: str) -> str:
  return "Authorization: Basic " + base64.b64encode(f"{username}:{token}".encode()).decode()


def _set_git_config(entries: list[tuple[str, str]]) -> None:
  for key in list(os.environ):
    if key.startswith(("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_")):
      os.environ.pop(key, None)
  os.environ["GIT_CONFIG_COUNT"] = str(len(entries))
  for index, (key, value) in enumerate(entries):
    os.environ[f"GIT_CONFIG_KEY_{index}"] = key
    os.environ[f"GIT_CONFIG_VALUE_{index}"] = value


def _preview_access_rejected() -> bool:
  # Probe even without readable credentials: an outage must not cause fallback.
  try:
    result = _probe_preview_access(*(get_credentials() or ("", "")))
  except (OSError, subprocess.SubprocessError):
    return False
  if result.returncode == 0:
    return False
  diagnostic = result.stdout + "\n" + result.stderr
  return _ACCESS_REJECTED.search(diagnostic) is not None


def _switch_to_public(repo_dir: str, branch: str, configure_auth: Callable[[str], None]) -> str | None:
  # Probe public branches without changing the installed origin or its auth.
  target = None
  try:
    with preserve_git_environment(), tempfile.TemporaryDirectory(prefix="iqpilot-public-") as probe_dir:
      for key in list(os.environ):
        if _is_git_env(key):
          os.environ.pop(key, None)
      subprocess.run(["git", "init", "-q", probe_dir], capture_output=True, check=True)
      for candidate in _PUBLIC_ORIGINS:
        subprocess.run(["git", "-C", probe_dir, "config", "remote.origin.url", candidate], capture_output=True, check=True)
        configure_auth(probe_dir)
        env = _isolated_git_env()
        env.update({k: v for k, v in os.environ.items() if k == "GIT_CONFIG_COUNT" or k.startswith(("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_"))})
        try:
          result = subprocess.run(
            ["git", "-c", "credential.helper=", "ls-remote", "--exit-code", "--heads", "origin",
             f"refs/heads/{branch}", f"refs/heads/{_PUBLIC_FALLBACK_BRANCH}"],
            cwd=probe_dir, env=env, capture_output=True, text=True, check=False, timeout=10,
          )
        except (OSError, subprocess.SubprocessError):
          continue
        if result.returncode != 0:
          continue
        refs = {line.split()[1] for line in result.stdout.splitlines() if len(line.split()) == 2}
        if f"refs/heads/{branch}" in refs:
          target = candidate, branch
          break
        if target is None and f"refs/heads/{_PUBLIC_FALLBACK_BRANCH}" in refs:
          target = candidate, _PUBLIC_FALLBACK_BRANCH
  except (OSError, subprocess.SubprocessError):
    return None

  if target is None:
    return None
  public_origin, public_branch = target
  subprocess.run(["git", "-C", repo_dir, "config", "remote.origin.url", public_origin], capture_output=True, check=True)
  if public_branch != branch:
    _replace_target(branch, public_branch)
  return public_branch


def _target_paths() -> tuple[str, str]:
  """The updater target's file and the lock native Params holds while replacing values."""
  try:
    params = Params()
    return params.get_param_path("UpdaterTargetBranch"), os.path.join(os.path.dirname(os.path.normpath(params.get_param_path())), ".lock")
  except Exception:
    return os.path.join(PARAMS_DIR, "UpdaterTargetBranch"), os.path.join(os.path.dirname(os.path.normpath(PARAMS_DIR)), ".lock")


def _replace_target(previous: str, branch: str) -> None:
  """Move the updater target from previous to branch, unless the user picked another branch meanwhile.

  Compare and replace under the Params lock: a selection saved before this reads the target is kept, and
  one saved after it replaces this recovery.
  """
  target, lock_path = _target_paths()
  directory = os.path.dirname(lock_path)
  os.makedirs(os.path.dirname(target), exist_ok=True)
  fd, temporary = tempfile.mkstemp(prefix=".tmp_value_", dir=directory)
  try:
    with os.fdopen(fd, "wb") as stream:
      stream.write(branch.encode())
      stream.flush()
      os.fsync(stream.fileno())
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDONLY, 0o775)
    try:
      fcntl.flock(lock_fd, fcntl.LOCK_EX)
      try:
        with open(target, "rb") as stream:
          current = stream.read().decode()
      except FileNotFoundError:
        current = None
      if current not in (None, previous):
        return
      os.replace(temporary, target)
      parent = os.open(os.path.dirname(target), os.O_RDONLY)
      try:
        os.fsync(parent)
      finally:
        os.close(parent)
    finally:
      os.close(lock_fd)
  finally:
    if os.path.exists(temporary):
      os.unlink(temporary)


def _fallback_to_public(repo_dir: str, branch: str, configure_auth: Callable[[str], None]) -> str | None:
  if _canonical_iqpilot_origin(_origin_url(repo_dir)) != _PREVIEW_ORIGIN or not _preview_access_rejected():
    return None
  return _switch_to_public(repo_dir, branch, configure_auth)


def retarget_missing_public_branch(repo_dir: str, branch: str, available: Container[str]) -> None:
  """Move a public device off a branch public doesn't have, e.g. one kept after leaving Preview offline."""
  if branch in available or _PUBLIC_FALLBACK_BRANCH not in available:
    return
  if _public_iqpilot_origin(_origin_url(repo_dir)) is not None:
    _replace_target(branch, _PUBLIC_FALLBACK_BRANCH)


def _remote_output(cmd: list[str], repo_dir: str, timeout: float | None) -> str:
  if timeout is None:
    return subprocess.check_output(cmd, cwd=repo_dir, stderr=subprocess.STDOUT, encoding="utf8")
  # Git starts transports and credential helpers. Bound the entire process tree.
  with subprocess.Popen(cmd, cwd=repo_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        encoding="utf8", start_new_session=True) as proc:
    try:
      output, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
      try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
      except ProcessLookupError:
        pass
      proc.communicate()
      raise
  if proc.returncode:
    raise subprocess.CalledProcessError(proc.returncode, cmd, output)
  return output


def _run_remote(cmd: list[str], repo_dir: str, branch: str, configure_auth: Callable[[str], None], *, timeout: float | None = None) -> tuple[str, str]:
  """Retry rejected Preview access on the selected or fallback public branch."""
  configure_auth(repo_dir)
  try:
    return _remote_output(cmd, repo_dir, timeout), branch
  except subprocess.CalledProcessError:
    selected = _fallback_to_public(repo_dir, branch, configure_auth)
    if selected is None:
      raise
  # A fetch already in progress must follow the newly selected public target.
  if cmd == ["git", "fetch", "origin", branch]:
    cmd = ["git", "fetch", "origin", selected]
  configure_auth(repo_dir)
  return _remote_output(cmd, repo_dir, timeout), selected


def run_remote(cmd: list[str], repo_dir: str, branch: str, configure_auth: Callable[[str], None], *, timeout: float | None = None) -> str:
  return _run_remote(cmd, repo_dir, branch, configure_auth, timeout=timeout)[0]


def fetch_remote(repo_dir: str, branch: str, configure_auth: Callable[[str], None], *, timeout: float | None = None) -> str:
  """Return the branch actually fetched, independent of later target requests."""
  return _run_remote(["git", "fetch", "origin", branch], repo_dir, branch, configure_auth, timeout=timeout)[1]


# Older tester builds added these beside origin to merge the public and Preview branch lists.
_LEGACY_SOURCE_REMOTE = re.compile(r"iqpilot-(public|preview)(-[0-9]+)?")


def remove_legacy_source_remotes(repo_dir: str) -> None:
  """Leave origin as the only update source, keeping any remote a user added."""
  env = _isolated_git_env()
  names = subprocess.run(["git", "-C", repo_dir, "remote"], text=True, capture_output=True, check=False, env=env).stdout.split()
  for name in names:
    if not _LEGACY_SOURCE_REMOTE.fullmatch(name):
      continue
    url = subprocess.run(["git", "-C", repo_dir, "config", "--get", f"remote.{name}.url"],
                         text=True, capture_output=True, check=False, env=env).stdout.strip()
    if _canonical_iqpilot_origin(url) is not None or _public_iqpilot_origin(url) is not None:
      subprocess.run(["git", "-C", repo_dir, "remote", "remove", name], capture_output=True, check=False, env=env)
  git_dir = subprocess.run(["git", "-C", repo_dir, "rev-parse", "--absolute-git-dir"],
                           text=True, capture_output=True, check=False, env=env).stdout.strip()
  if git_dir:
    try:
      os.unlink(os.path.join(git_dir, "iqpilot-sources.lock"))
    except FileNotFoundError:
      pass


def ssh_to_https(url: str) -> str:
  """Convert SSH to HTTPS and normalize known Git hosts to git.konn3kt.com.

  Other HTTPS URLs are unchanged. Other SSH hosts lose a leading ssh. label.
  """
  url = url.strip()
  if url.startswith("https://"):
    parsed = urlsplit(url)
    if parsed.hostname in _GIT_HOST_ALIASES and parsed.username is None and parsed.port in (None, 443):
      return parsed._replace(netloc=_GIT_HTTPS_HOST).geturl()
    return url
  host = path = ""
  if url.startswith("ssh://"):
    rest = url[len("ssh://"):]
    rest = rest.split("@", 1)[-1]
    hostport, _, path = rest.partition("/")
    host = hostport.split(":", 1)[0]
  elif url.startswith("git@") or ("@" in url and ":" in url.split("@", 1)[-1] and "://" not in url):
    rest = url.split("@", 1)[-1]
    host, _, path = rest.partition(":")
  else:
    return url
  if host.lower() in _GIT_HOST_ALIASES:
    host = _GIT_HTTPS_HOST
  elif host.startswith("ssh."):
    host = host[len("ssh."):]
  return f"https://{host}/{path}"


def install_credential_helper(repo_dir: str = DEFAULT_REPO_DIR) -> None:
  if get_credentials() is None:
    return

  scopes = {f"https://{host}" for host in CREDENTIAL_HOSTS}
  origin = _origin_url(repo_dir)
  https = ssh_to_https(origin)
  if https.startswith("https://"):
    parts = urlsplit(https)
    if parts.hostname:
      scopes.add(f"{parts.scheme}://{parts.hostname}")

  try:
    os.makedirs(KEY_DIR, exist_ok=True)
    tmp = HELPER_PATH + ".tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o755)
    with os.fdopen(fd, "w") as f:
      f.write(_HELPER_SCRIPT.format(key_path=KEY_PATH))
    os.replace(tmp, HELPER_PATH)
  except Exception:
    return

  helper_cmd = f"!/usr/bin/env python3 {HELPER_PATH}"
  for scope in scopes:
    subprocess.run(
      ["git", "config", "--global", f"credential.{scope}.helper", helper_cmd],
      check=False, capture_output=True,
    )


def configure(repo_dir: str, *, remote: str = "") -> None:
  """Apply saved credentials before a Git remote operation.

  Public IQ.Pilot uses fleet auth, and its GitHub mirror uses anonymous access;
  both ignore tester credentials. Other repositories use saved credentials over
  HTTPS. Authorization headers are passed through the environment, not saved
  in .git/config. An explicit remote selects auth without changing origin.
  """
  origin = remote or _origin_url(repo_dir)
  public_origin = _public_iqpilot_origin(origin)
  if public_origin in _PUBLIC_ORIGINS:
    if not remote and origin != public_origin:
      subprocess.run(["git", "-C", repo_dir, "config", "remote.origin.url", public_origin], capture_output=True, check=True)
    # Prevent the credential helper from supplying a rejected tester token.
    config = [("credential.helper", "")]
    if public_origin == _GITHUB_MIRROR_ORIGIN:
      config.append((f"http.{public_origin}/.extraHeader", ""))
    _set_git_config(config)
    return

  creds = get_credentials()
  if creds is None:
    return
  username, token = creds

  try:
    install_credential_helper(repo_dir)
  except Exception:
    pass

  if not origin:
    return

  https = ssh_to_https(origin)
  if not remote and https != origin and https.startswith("https://"):
    subprocess.run(
      ["git", "-C", repo_dir, "config", "remote.origin.url", https],
      check=False, capture_output=True,
    )

  if not https.startswith("https://"):
    return

  key = https if https.endswith("/") else https + "/"
  # Git accumulates headers; clear fleet auth before supplying tester auth.
  header_key = f"http.{key}.extraHeader"
  _set_git_config([(header_key, ""), (header_key, _auth_header(username, token))])
