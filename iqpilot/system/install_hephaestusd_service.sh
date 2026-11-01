#!/usr/bin/bash
set -e

SERVICE_NAME="hephaestusd.service"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
bash "$SCRIPT_DIR/install_iqpilot_service.sh" "$SERVICE_NAME"

echo ""
echo "Useful commands:"
echo "  sudo systemctl status hephaestusd    - Check service status"
echo "  sudo systemctl restart hephaestusd   - Restart service"
echo "  sudo systemctl stop hephaestusd      - Stop service"
echo "  sudo journalctl -u hephaestusd -f    - View live logs"
