#!/bin/bash
# deploy_to_board.sh - Deploy DMS application to FRDM-IMX93 board
# Usage: ./deploy_to_board.sh [BOARD_IP] [DMS_DIR]
#
# Defaults:
#   BOARD_IP=192.168.1.100
#   DMS_DIR=/opt/dms

set -e

BOARD_IP=${1:-192.168.1.100}
DMS_DIR=${2:-/opt/dms}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== DMS Deployment to FRDM-IMX93 ==="
echo "Board: root@${BOARD_IP}"
echo "Target: ${DMS_DIR}"
echo ""

# Test SSH connectivity
echo "[1/6] Testing SSH connection..."
if ! ssh -o ConnectTimeout=5 "root@${BOARD_IP}" "echo OK" 2>/dev/null; then
    echo "ERROR: Cannot connect to root@${BOARD_IP}"
    echo "  - Ensure board is on network (eth0 or wlan0)"
    echo "  - Check IP: ping ${BOARD_IP}"
    echo "  - Try: ssh root@${BOARD_IP}"
    exit 1
fi
echo "  SSH OK"

# Create target directory structure
echo "[2/6] Creating directory structure..."
ssh "root@${BOARD_IP}" "
    mkdir -p ${DMS_DIR}/{src/{behavioral,dashboard,calibration,utils},config,models,scripts,sounds,systemd}
"

# Copy Python source code
echo "[3/6] Copying source code..."
scp -r "${PROJECT_DIR}/src/"*.py "root@${BOARD_IP}:${DMS_DIR}/src/"
scp -r "${PROJECT_DIR}/src/behavioral/"*.py "root@${BOARD_IP}:${DMS_DIR}/src/behavioral/"
scp -r "${PROJECT_DIR}/src/dashboard/"*.py "root@${BOARD_IP}:${DMS_DIR}/src/dashboard/"
scp -r "${PROJECT_DIR}/src/calibration/"*.py "root@${BOARD_IP}:${DMS_DIR}/src/calibration/"
scp -r "${PROJECT_DIR}/src/utils/"*.py "root@${BOARD_IP}:${DMS_DIR}/src/utils/"

# Copy main app
echo "[4/6] Copying main app and config..."
scp "${PROJECT_DIR}/dms_app.py" "root@${BOARD_IP}:${DMS_DIR}/"
scp "${PROJECT_DIR}/config/config.yaml" "root@${BOARD_IP}:${DMS_DIR}/config/"
scp "${PROJECT_DIR}/config/zones.xml" "root@${BOARD_IP}:${DMS_DIR}/config/"

# Copy systemd service
echo "[5/6] Installing systemd service..."
scp "${PROJECT_DIR}/systemd/dms.service" "root@${BOARD_IP}:/etc/systemd/system/"
ssh "root@${BOARD_IP}" "
    systemctl daemon-reload
    systemctl enable dms.service
"

# Copy test scripts
echo "[6/6] Copying test scripts..."
scp "${PROJECT_DIR}/scripts/"test_*.py "root@${BOARD_IP}:${DMS_DIR}/scripts/"
scp "${PROJECT_DIR}/scripts/test_board_integration.py" "root@${BOARD_IP}:${DMS_DIR}/scripts/" 2>/dev/null || true

echo ""
echo "=== Deployment Complete ==="
echo ""
echo "Next steps on board:"
echo "  1. Copy models to ${DMS_DIR}/models/"
echo "  2. Run integration test: python3 ${DMS_DIR}/scripts/test_board_integration.py"
echo "  3. Start DMS: python3 ${DMS_DIR}/dms_app.py --config ${DMS_DIR}/config/config.yaml"
echo "  4. Or enable service: systemctl start dms"
