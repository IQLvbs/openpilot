#!/usr/bin/env bash
# Python env + proprietary runtime sync for the openpilot tree.
#
# Two callers, same steps:
#   - launch_chffrplus.sh sources this at boot and calls sync_python_env. When an
#     update was already prepared this is a warm no-op; when there is real work
#     (local dev edits, failed prep) it runs under the boot spinner as before.
#   - updated.py executes this standalone right after an update is applied to
#     disk, so the heavy uv/package work happens offroad with the stack idle
#     instead of at boot alongside everything else starting at once.
#
# Variables assigned here (VENV_SITE_PACKAGES, BASE_SITE_PACKAGES, ...) are
# intentionally global: the sourced caller composes PYTHONPATH from them.

_iq_maybe_sudo() {
  # On IQ.OS 4.9.9.2+ the boot-prepare oneshot already owns .venv to iq, so these defensive ops run unprivileged; pre-4.9.9.2 keeps sudo.
  if [ -f /run/iqpilot-boot-prepare.done ]; then
    "$@"
  else
    sudo "$@"
  fi
}

# a package_runtime left behind by a release tree outranks the installed packages, so only a prebuilt tree may use it
iq_package_runtime_root() {
  if [ -f "$1/prebuilt" ] && [ -d "$1/artifacts/package_runtime" ]; then
    printf '%s\n' "$1/artifacts/package_runtime"
  fi
}

IQOS_SERVICE_DROPINS_HELPER="/usr/libexec/iqos/iq-service-dropins"

# IQ.OS boot-prepare writes the hephaestusd/ble-transportd/flockd drop-ins (package PYTHONPATH, iqpilot_t domain) only for a tree present at boot, so a tree installed after boot runs those units bare: degraded and denied the iqkeeperd signer.
iq_service_dropins_missing() {
  [ -f "${1:-/run/iqpilot-boot-prepare.done}" ] && [ ! -f "${2:-/run/systemd/system/hephaestusd.service.d/iqpilot-packages.conf}" ]
}

sync_python_env() {
  local DIR
  DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." >/dev/null && pwd )"

  # Best-effort: install the verified runtime before anything starts the services that
  # need it. This only succeeds on a tree that has built _verified_import.so, so on a
  # prebuilt release install it is a no-op -- there the runtime arrives with the IQ.OS
  # flash, whose image bakes it in. What actually fixes the 20+ minute first-install hang
  # is --no-block on the service starts below: hephaestusd/ble-transportd/flockd each
  # ExecStartPre-wait up to 600s for /usr/libexec/iqpilot/iqpilot_bundle_runner.
  if [ -x "$DIR/iqpilot/system/proprietary_runtime/install_verified_runtime.sh" ]; then
    "$DIR/iqpilot/system/proprietary_runtime/install_verified_runtime.sh" || true
  fi

  # Install/update proprietary runtime bundles.
  if [ -f "$DIR/artifacts/runtime/ensure_private_installed.sh" ]; then
    bash "$DIR/artifacts/runtime/ensure_private_installed.sh" || true
  elif [ -x "$DIR/scripts/iqpilot/ensure_navd_private_installed.sh" ]; then
    # Backward-compat fallback for older trees.
    "$DIR/scripts/iqpilot/ensure_navd_private_installed.sh" || true
  fi

  PYTHONPATH="$DIR" /usr/local/venv/bin/python3 -c "from iqpilot.common.git_creds import install_credential_helper; install_credential_helper('$DIR')" 2>/dev/null || true
  PACKAGE_SOURCES_ID="$(git -C "$DIR" rev-parse --verify --quiet "HEAD:artifacts/package_sources" 2>/dev/null || true)"
  PACKAGE_LOCK_SHA="$(printf '%s\n%s\n' "$(sha256sum "$DIR/uv.lock" | awk '{print $1}')" "$PACKAGE_SOURCES_ID" | sha256sum | awk '{print $1}')"
  INSTALLED_PACKAGE_LOCK_SHA="$(cat "$DIR/.iqpilot-package-lock-sha256" 2>/dev/null || true)"
  mapfile -t RUNTIME_WHEELS < <(/usr/local/venv/bin/python3 "$DIR/iqpilot/system/runtime_wheel_requirements.py" "$DIR")
  mapfile -t RUNTIME_WHEEL_SOURCES < <(/usr/local/venv/bin/python3 "$DIR/iqpilot/system/runtime_wheel_requirements.py" "$DIR" --sources)
  RUNTIME_WHEEL_LOCK_SHA="$(printf '%s\n' "${RUNTIME_WHEELS[@]}" | sha256sum | awk '{print $1}')"
  INSTALLED_RUNTIME_WHEEL_LOCK_SHA="$(cat "$DIR/.iqpilot-runtime-wheel-lock-sha256" 2>/dev/null || true)"
  BASE_SITE_PACKAGES="$(/usr/local/venv/bin/python3 -c 'import site; print(site.getsitepackages()[0])')"
  PROJECT_RAYLIB="$("$DIR/.venv/bin/python3" -c 'import importlib.metadata; print(importlib.metadata.distribution("raylib").locate_file(""))' 2>/dev/null || true)"
  if [[ "$PROJECT_RAYLIB" = "$DIR/.venv"/* ]]; then
    _iq_maybe_sudo rm -rf "$DIR/.venv"
    rm -f "$DIR/.iqpilot-package-lock-sha256" "$DIR/.iqpilot-runtime-wheel-lock-sha256"
    INSTALLED_PACKAGE_LOCK_SHA=""
    INSTALLED_RUNTIME_WHEEL_LOCK_SHA=""
  fi
  VENV_SITE_PACKAGES="$("$DIR/.venv/bin/python3" -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null || true)"
  PACKAGES_READY=0
  if [ -n "$(iq_package_runtime_root "$DIR")" ] && PYTHONPATH="$DIR/artifacts/package_runtime" /usr/local/venv/bin/python3 -c "import iqdbc, msgq, panda, tinygrad, capnp" 2>/dev/null \
      && PYTHONPATH="$DIR/artifacts/package_runtime" /usr/local/venv/bin/python3 "$DIR/iqpilot/system/runtime_packages_verify.py" 2>/dev/null; then
    PACKAGES_READY=1
  elif [ "$PACKAGE_LOCK_SHA" = "$INSTALLED_PACKAGE_LOCK_SHA" ] && "$DIR/.venv/bin/python3" -c "import iqdbc, msgq, panda, tinygrad, capnp" 2>/dev/null \
      && "$DIR/.venv/bin/python3" "$DIR/iqpilot/system/runtime_packages_verify.py"; then
    # a top-level import passes on a partially extracted install (lazy backends),
    # so readiness also requires every wheel RECORD file to exist on disk
    PACKAGES_READY=1
  fi
  RUNTIME_WHEELS_READY=0
  if [ -n "$(iq_package_runtime_root "$DIR")" ] && PYTHONPATH="$DIR/artifacts/package_runtime" /usr/local/venv/bin/python3 -c "import libdatachannel" 2>/dev/null \
      && PYTHONPATH="$DIR:$DIR/artifacts/package_runtime" /usr/local/venv/bin/python3 "$DIR/iqpilot/system/runtime_wheels_verify.py" 2>/dev/null; then
    RUNTIME_WHEELS_READY=1
  elif [ "$RUNTIME_WHEEL_LOCK_SHA" = "$INSTALLED_RUNTIME_WHEEL_LOCK_SHA" ] && "$DIR/.venv/bin/python3" -c "import libdatachannel" 2>/dev/null \
      && PYTHONPATH="$DIR" "$DIR/.venv/bin/python3" "$DIR/iqpilot/system/runtime_wheels_verify.py"; then
    RUNTIME_WHEELS_READY=1
  fi
  if [ ! -f "$DIR/prebuilt" ] && [ ! -x "$DIR/.venv/bin/python3" ]; then
    PACKAGES_READY=0
    RUNTIME_WHEELS_READY=0
  fi
  if [ "$PACKAGES_READY" != "1" ] || [ "$RUNTIME_WHEELS_READY" != "1" ]; then
    # A source install fetches git+https packages from git.konn3kt.com; at early boot DNS, connectivity, or the device's stale RTC (months behind until NTP syncs, which fails TLS cert validity) make the fetch die and abort launch with no venv. Wait until the host actually answers over TLS before uv reaches the network.
    for _ in $(seq 1 90); do curl -sS --max-time 8 -o /dev/null https://git.konn3kt.com/ 2>/dev/null && break; sleep 2; done
    # Private components (git.konn3kt.com/teal/*) need auth; inject the device credential as an https extraHeader via GIT_CONFIG_* (HOME-independent, never persisted) so the uv git fetch authenticates. GitAuthBlob is decoded standalone because the Params->cereal chain is unavailable mid-bootstrap.
    IQ_GIT_AUTH_ENV="$(/usr/local/venv/bin/python3 - 2>/dev/null <<'PY'
import base64, json
try:
  from cryptography.fernet import Fernet
  key = open("/data/konn3kt/git_auth.key", "rb").read().strip()
  blob = open("/data/params/d/GitAuthBlob", "rb").read()
  d = json.loads(Fernet(key).decrypt(blob).decode())
  u, t = d.get("u", ""), d.get("t", "")
  if u and t:
    hdr = "Authorization: Basic " + base64.b64encode(("%s:%s" % (u, t)).encode()).decode()
    hosts = ("git.konn3kt.com", "gitlvb.teallvbs.xyz")
    print("export GIT_CONFIG_COUNT=%d" % len(hosts))
    for i, h in enumerate(hosts):
      print("export GIT_CONFIG_KEY_%d=%r" % (i, "http.https://%s/.extraHeader" % h))
      print("export GIT_CONFIG_VALUE_%d=%r" % (i, hdr))
except Exception:
  pass
PY
)"
    [ -n "$IQ_GIT_AUTH_ENV" ] && eval "$IQ_GIT_AUTH_ENV"
    UV_CACHE_DIR="$DIR/.uv-cache"
    # AGNOS removes owner root, so the /usr/comma/shims/uv wrapper (it sudo-remounts / and /tmp) cannot run; force uv off the PATH shim and keep its cache and TMPDIR on writable /data so the venv and wheels build unprivileged.
    mkdir -p "$UV_CACHE_DIR" "$DIR/.uv-tmp"
    if [ ! -x "$DIR/.venv/bin/python3" ]; then
      UV_CACHE_DIR="$UV_CACHE_DIR" TMPDIR="$DIR/.uv-tmp" PATH="/usr/local/venv/bin:/usr/bin:$PATH" uv venv --python /usr/local/venv/bin/python3 "$DIR/.venv" || return 1
      VENV_SITE_PACKAGES="$("$DIR/.venv/bin/python3" -c 'import site; print(site.getsitepackages()[0])')"
    fi
    _iq_maybe_sudo chown -R "$(id -u):$(id -g)" "$DIR/.venv" "$UV_CACHE_DIR" "$DIR/artifacts/package_sources" 2>/dev/null || true
    if [ "${#RUNTIME_WHEELS[@]}" = "0" ] || [ "${#RUNTIME_WHEEL_SOURCES[@]}" = "0" ]; then
      return 1
    fi
    if [ "$PACKAGES_READY" != "1" ]; then
      IQDBC_PACKAGE_SOURCE=""
      PYCAPNP_PACKAGE_SOURCE=""
      PACKAGE_SOURCES=()
      while IFS=$'\t' read -r package_name package_source; do
        if [ "$package_name" = "iqdbc" ]; then
          IQDBC_PACKAGE_SOURCE="$package_source"
          PACKAGE_SOURCES+=("$package_source")
        elif [ "$package_name" = "pycapnp" ]; then
          PYCAPNP_PACKAGE_SOURCE="$package_source"
        else
          PACKAGE_SOURCES+=("$package_source")
        fi
      done < <(/usr/local/venv/bin/python3 "$DIR/iqpilot/system/runtime_package_sources.py" "$DIR")
      if [ "${#PACKAGE_SOURCES[@]}" = "0" ] || [ -z "$IQDBC_PACKAGE_SOURCE" ] || [ -z "$PYCAPNP_PACKAGE_SOURCE" ]; then
        return 1
      fi
      PACKAGE_BUILD_PYTHONPATH="$BASE_SITE_PACKAGES:$VENV_SITE_PACKAGES"
      # The base AGNOS venv ships Eigen as a Python package instead of under
      # /usr/include, so source-package builds need its install directory on the compiler include path.
      # Resolve it through Python rather than pinning the Python minor version.
      EIGEN_INCLUDE_ROOT="$(/usr/local/venv/bin/python3 -c \
        'from pathlib import Path; import eigen; root = Path(eigen.__file__).resolve().parent / "install"; assert (root / "eigen3/Eigen/Dense").is_file(); print(root)' \
        2>/dev/null || true)"
      PACKAGE_BUILD_CPATH="${CPATH:-}"
      if [ -n "$EIGEN_INCLUDE_ROOT" ]; then
        PACKAGE_BUILD_CPATH="$EIGEN_INCLUDE_ROOT${PACKAGE_BUILD_CPATH:+:$PACKAGE_BUILD_CPATH}"
      fi
      UV_CACHE_DIR="$UV_CACHE_DIR" TMPDIR="$DIR/.uv-tmp" PYTHONPATH="$PACKAGE_BUILD_PYTHONPATH" PATH="/usr/local/venv/bin:/usr/bin:$PATH" \
        uv pip install --python "$DIR/.venv/bin/python" --no-build-isolation --no-deps --reinstall "$PYCAPNP_PACKAGE_SOURCE" || return 1
      UV_CACHE_DIR="$UV_CACHE_DIR" TMPDIR="$DIR/.uv-tmp" PYTHONPATH="$PACKAGE_BUILD_PYTHONPATH" CPATH="$PACKAGE_BUILD_CPATH" PATH="/usr/local/venv/bin:/usr/bin:$PATH" \
        uv pip install --python "$DIR/.venv/bin/python" --no-build-isolation --no-deps --reinstall "$IQDBC_PACKAGE_SOURCE" || return 1
      UV_CACHE_DIR="$UV_CACHE_DIR" TMPDIR="$DIR/.uv-tmp" PYTHONPATH="$PACKAGE_BUILD_PYTHONPATH" CPATH="$PACKAGE_BUILD_CPATH" PATH="/usr/local/venv/bin:/usr/bin:$PATH" \
        uv pip install --python "$DIR/.venv/bin/python" --no-build-isolation --no-deps --reinstall "${PACKAGE_SOURCES[@]}" || return 1
      printf '%s\n' "$PACKAGE_LOCK_SHA" > "$DIR/.iqpilot-package-lock-sha256"
    fi
    if [ "$RUNTIME_WHEELS_READY" != "1" ]; then
      UV_CACHE_DIR="$UV_CACHE_DIR" TMPDIR="$DIR/.uv-tmp" PATH="/usr/local/venv/bin:/usr/bin:$PATH" \
        uv pip install --python "$DIR/.venv/bin/python" --no-deps --reinstall "${RUNTIME_WHEEL_SOURCES[@]}" || return 1
      "$DIR/.venv/bin/python3" -c "import libdatachannel" || return 1
      PYTHONPATH="$DIR" "$DIR/.venv/bin/python3" "$DIR/iqpilot/system/runtime_wheels_verify.py" || return 1
      printf '%s\n' "$RUNTIME_WHEEL_LOCK_SHA" > "$DIR/.iqpilot-runtime-wheel-lock-sha256"
    fi
  fi
  if [ ! -f "$DIR/prebuilt" ] && [ ! -x "$DIR/.venv/bin/python3" ]; then
    return 1
  fi
  if [ -n "$VENV_SITE_PACKAGES" ]; then
    printf 'import site; site.addsitedir("%s")\n' "$BASE_SITE_PACKAGES" | _iq_maybe_sudo tee "$VENV_SITE_PACKAGES/iqpilot-system-venv.pth" >/dev/null
  fi
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  sync_python_env
  exit $?
fi

# NOTE: keep this file in lockstep with the launch-time expectations above --
# updated.py invokes it standalone and launch_chffrplus.sh sources it.
