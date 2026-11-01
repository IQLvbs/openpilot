#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUNTIME_ROOT="${IQPILOT_VERIFIED_RUNTIME_ROOT:-/usr/libexec/iqpilot}"
PYTHON_ROOT="${RUNTIME_ROOT}/python"
TARGET_EXT_DIR="${PYTHON_ROOT}/openpilot/system/proprietary_runtime"
MANIFEST_SRC="${ROOT}/iqpilot/system/proprietary_runtime/rootfs_integrity.json"
MANIFEST_SIG_SRC="${ROOT}/iqpilot/system/proprietary_runtime/rootfs_integrity.json.sig"

RUNNER_SRC="${ROOT}/iqpilot/system/proprietary_runtime/iqpilot_bundle_runner"
EXT_SRC="${ROOT}/iqpilot/system/proprietary_runtime/_verified_import.so"
STUB_ROOT="${ROOT}/iqpilot/system/proprietary_runtime/rootfs_python"

if [ ! -x "${RUNNER_SRC}" ]; then
  echo "missing runner: ${RUNNER_SRC}"
  exit 1
fi

if [ ! -f "${EXT_SRC}" ]; then
  exit 0
fi

if [ ! -f "${MANIFEST_SRC}" ] || [ ! -f "${MANIFEST_SIG_SRC}" ]; then
  if [ -n "${IQPILOT_SIGNING_KEY:-}" ]; then
    "${ROOT}/.venv/bin/python" \
      "${ROOT}/scripts/iqpilot/build_proprietary_runtime_manifest.py" \
      --repo-root "${ROOT}" \
      --output "${MANIFEST_SRC}"
  fi
fi

if [ ! -f "${MANIFEST_SRC}" ] || [ ! -f "${MANIFEST_SIG_SRC}" ]; then
  echo "missing rootfs runtime integrity manifest or signature"
  exit 1
fi

if [ -x "${RUNTIME_ROOT}/iqpilot_bundle_runner" ] && \
   [ -f "${TARGET_EXT_DIR}/_verified_import.so" ] && \
   [ -f "${RUNTIME_ROOT}/runtime_integrity.json" ]; then
  echo "verified runtime already installed at ${RUNTIME_ROOT}"
  exit 0
fi

if ! sudo -n mount -o remount,rw / 2>/dev/null; then
  echo "cannot remount / (no sudo access); runtime must be pre-installed in the OS image"
  exit 0
fi
trap 'sudo -n mount -o remount,ro / 2>/dev/null' EXIT

sudo -n mkdir -p "${RUNTIME_ROOT}" "${TARGET_EXT_DIR}"
sudo -n install -m 0755 "${RUNNER_SRC}" "${RUNTIME_ROOT}/iqpilot_bundle_runner"
sudo -n install -m 0755 "${EXT_SRC}" "${TARGET_EXT_DIR}/_verified_import.so"
sudo -n install -m 0644 "${MANIFEST_SRC}" "${RUNTIME_ROOT}/runtime_integrity.json"
sudo -n install -m 0644 "${MANIFEST_SIG_SRC}" "${RUNTIME_ROOT}/runtime_integrity.json.sig"
sudo -n install -m 0644 "${STUB_ROOT}/openpilot/__init__.py" "${PYTHON_ROOT}/openpilot/__init__.py"
sudo -n mkdir -p "${PYTHON_ROOT}/openpilot/system"
sudo -n install -m 0644 "${STUB_ROOT}/openpilot/system/__init__.py" "${PYTHON_ROOT}/openpilot/system/__init__.py"
sudo -n install -m 0644 "${STUB_ROOT}/openpilot/system/proprietary_runtime/__init__.py" "${TARGET_EXT_DIR}/__init__.py"

echo "installed verified runtime to ${RUNTIME_ROOT}"
