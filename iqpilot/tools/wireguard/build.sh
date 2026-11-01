#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
WIREGUARD_GO_REF="${WIREGUARD_GO_REF:-0.0.20230223}"
WIREGUARD_TOOLS_REF="${WIREGUARD_TOOLS_REF:-v1.0.20210914}"
GO_VERSION="${GO_VERSION:-go1.23.4}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

if [ "$(uname -s)" != "Linux" ] || [ "$(uname -m)" != "aarch64" ]; then
  echo "build.sh runs on the device (aarch64 Linux)" >&2
  exit 1
fi

if ! command -v go >/dev/null 2>&1; then
  curl -fsSL "https://go.dev/dl/${GO_VERSION}.linux-arm64.tar.gz" | tar -C "$TMP" -xz
  export PATH="$TMP/go/bin:$PATH"
fi

git clone -q --depth 1 --branch "$WIREGUARD_GO_REF" https://git.zx2c4.com/wireguard-go "$TMP/wireguard-go"
(cd "$TMP/wireguard-go" && CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o "$DIR/wireguard-go" .)

git clone -q --depth 1 --branch "$WIREGUARD_TOOLS_REF" https://git.zx2c4.com/wireguard-tools "$TMP/wireguard-tools"
make -s -C "$TMP/wireguard-tools/src" wg
install -m 0755 "$TMP/wireguard-tools/src/wg" "$DIR/wg"

echo "built $DIR/wireguard-go ($WIREGUARD_GO_REF) and $DIR/wg ($WIREGUARD_TOOLS_REF)"
