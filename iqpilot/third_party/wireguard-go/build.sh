#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
REF="${WIREGUARD_GO_REF:-0.0.20250522}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

git clone -q --depth 1 --branch "$REF" https://git.zx2c4.com/wireguard-go "$TMP/src"
(cd "$TMP/src" && CGO_ENABLED=0 GOOS=linux GOARCH=arm64 go build -trimpath -ldflags="-s -w -buildid=" -o "$DIR/larch64/wireguard-go" .)
chmod 755 "$DIR/larch64/wireguard-go"
shasum -a 256 "$DIR/larch64/wireguard-go" 2>/dev/null || sha256sum "$DIR/larch64/wireguard-go"
