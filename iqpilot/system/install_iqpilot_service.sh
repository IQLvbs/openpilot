#!/usr/bin/bash
# Copyright (c) 2026 IQ.Lvbs. All rights reserved.
set -e

SERVICE_NAME="$1"
SERVICE_FILE="/data/openpilot/iqpilot/system/${SERVICE_NAME}"
SERVICE_OVERRIDE="/etc/systemd/system/${SERVICE_NAME}"
SERVICE_BAKED="/lib/systemd/system/${SERVICE_NAME}"
SERVICE_DROPIN="/run/systemd/system/${SERVICE_NAME}.d"

echo "Installing ${SERVICE_NAME} systemd service..."

if [ -f "$SERVICE_BAKED" ] && grep -q "/usr/libexec/iqpilot/iqpilot_bundle_runner" "$SERVICE_BAKED"; then
    echo "Using IQ.OS baked ${SERVICE_NAME}; removing stale override if present..."
    sudo mount -o remount,rw /
    sudo rm -f "$SERVICE_OVERRIDE"
    sudo systemctl daemon-reload
    sudo mount -o remount,ro /
else
    if [ ! -f "$SERVICE_FILE" ]; then
        echo "ERROR: Service file not found at $SERVICE_FILE"
        exit 1
    fi

    echo "IQ.OS baked unit unavailable; installing fallback override into /etc/systemd/system..."
    sudo cp "$SERVICE_FILE" "$SERVICE_OVERRIDE"
    sudo systemctl daemon-reload
fi

sudo mkdir -p "$SERVICE_DROPIN"
printf '%s\n' '[Service]' 'Environment="PYTHONPATH=/usr/libexec/iqpilot/python:/data/openpilot/.venv/lib/python3.12/site-packages:/data/openpilot"' | sudo tee "$SERVICE_DROPIN/iqpilot-packages.conf" >/dev/null
sudo systemctl daemon-reload

echo "Enabling $SERVICE_NAME to start at boot..."
sudo systemctl enable "$SERVICE_NAME"

echo "Starting $SERVICE_NAME..."
sudo systemctl restart "$SERVICE_NAME"

echo ""
echo "Service status:"
sudo systemctl status "$SERVICE_NAME" --no-pager
