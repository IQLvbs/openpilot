#!/usr/bin/env bash
set -euo pipefail

DEST="${IQ_WIREGUARD_ROOT:-/data/wg}"
IFACE="$(sed -n 's/^WG_IFACE=//p' "$DEST/wg.env" 2>/dev/null || true)"
IFACE="${IFACE:-wg0}"

sudo rm -f "$DEST/wireguard-go" "$DEST/wg"
if [ -e "/sys/class/net/$IFACE" ]; then
  sudo ip link del "$IFACE" || true
fi
sudo pkill -xf "$DEST/wireguard-go $IFACE" || true
sudo rm -f /dev/shm/iq_wireguard.json

if [ "${1:-}" = "--purge" ]; then
  sudo rm -rf "$DEST"
  echo "removed $DEST, including the device keys"
else
  echo "tunnel stopped and binaries removed; keys and wg.env kept in $DEST (use --purge to delete them)"
fi
