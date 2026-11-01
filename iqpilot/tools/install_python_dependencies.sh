#!/usr/bin/env bash
set -euo pipefail

# Increase the pip timeout to handle TimeoutError
export PIP_DEFAULT_TIMEOUT=200

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
ROOT="$DIR"/../../
cd "$ROOT"

if ! command -v "uv" > /dev/null 2>&1; then
  echo "installing uv..."
  curl -LsSf --retry 5 --retry-delay 5 --retry-all-errors https://astral.sh/uv/install.sh | sh
  UV_BIN="$HOME/.local/bin"
  PATH="$UV_BIN:$PATH"
fi

echo "updating uv..."
# ok to fail, can also fail due to installing with brew
uv self update || true

echo "installing python packages..."
UV_SYNC_ARGS=(--frozen)
if [[ "${IQPILOT_RUNTIME_DEPENDENCIES_ONLY:-0}" != "1" ]]; then
  UV_SYNC_ARGS+=(--all-extras)
elif [[ ! -f prebuilt ]]; then
  UV_SYNC_ARGS+=(--extra build --extra mpc-build)
fi
UV_SYNC_OK=0
for attempt in 1 2 3; do
  if uv sync "${UV_SYNC_ARGS[@]}"; then
    UV_SYNC_OK=1
    break
  fi
  [[ "${attempt}" -lt 3 ]] && sleep "$((attempt * 5))"
done
if [[ "${UV_SYNC_OK}" -ne 1 ]]; then
  exit 1
fi
source .venv/bin/activate

if [[ "$(uname)" == 'Darwin' ]]; then
  touch "$ROOT"/.env
  echo "export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES" >> "$ROOT"/.env
fi
