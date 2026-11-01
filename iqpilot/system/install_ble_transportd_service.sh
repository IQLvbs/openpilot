#!/usr/bin/bash
set -e

SERVICE_NAME="ble-transportd.service"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
bash "$SCRIPT_DIR/install_iqpilot_service.sh" "$SERVICE_NAME"

echo ""
echo "Useful commands:"
echo "  sudo systemctl status ble-transportd    - Check service status"
echo "  sudo systemctl restart ble-transportd   - Restart service"
echo "  sudo systemctl stop ble-transportd      - Stop service"
echo "  sudo journalctl -u ble-transportd -f    - View live logs"
