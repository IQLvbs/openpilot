#!/usr/bin/env bash
set -euo pipefail

SRC="$(cd "$(dirname "$0")" && pwd)"
DEST="${IQ_WIREGUARD_ROOT:-/data/wg}"

if [ "$(id -u)" = 0 ]; then
  echo "run install.sh as the device user, not root" >&2
  exit 1
fi
for bin in wireguard-go wg; do
  [ -x "$SRC/$bin" ] || { echo "missing $SRC/$bin, run ./build.sh first" >&2; exit 1; }
done
sudo -n true 2>/dev/null || { echo "wireguardd needs passwordless sudo for ip and wireguard-go" >&2; exit 1; }
[ -c /dev/net/tun ] || { echo "/dev/net/tun is missing, userspace WireGuard cannot run here" >&2; exit 1; }

if pgrep -f '[w]g-boot\.sh' >/dev/null 2>&1; then
  sudo pkill -f '[w]g-boot\.sh' || true
fi
if [ -f /data/continue.sh ] && grep -q 'wg-boot\.sh' /data/continue.sh; then
  sed -i '/wg-boot\.sh/d' /data/continue.sh
fi

sudo mkdir -p "$DEST"
sudo chown -R "$(id -u):$(id -g)" "$DEST"
rm -f "$DEST"/wg-up.sh "$DEST"/wg-watchdog.sh "$DEST"/wg-boot.sh "$DEST"/*.log "$DEST"/status.json "$DEST"/wg0.setconf
install -m 0755 "$SRC/wireguard-go" "$SRC/wg" "$DEST/"
[ -f "$DEST/wg.env" ] || install -m 0600 "$SRC/wg.env.example" "$DEST/wg.env"
if [ ! -f "$DEST/privatekey" ]; then
  (umask 077 && "$DEST/wg" genkey > "$DEST/privatekey")
fi
"$DEST/wg" pubkey < "$DEST/privatekey" > "$DEST/publickey"
chmod 0644 "$DEST/publickey"

echo "device public key: $(cat "$DEST/publickey")"
echo "next: fill in $DEST/wg.env, then turn on WireGuard in Settings > Network"
