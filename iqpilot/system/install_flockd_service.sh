#!/usr/bin/bash
set -e

SERVICE_NAME="flockd.service"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
bash "$SCRIPT_DIR/install_iqpilot_service.sh" "$SERVICE_NAME"

echo ""
echo "Useful commands:"
echo "  sudo systemctl status flockd    - Check service status"
echo "  sudo systemctl restart flockd   - Restart service"
echo "  sudo journalctl -u flockd -f    - View live logs"
