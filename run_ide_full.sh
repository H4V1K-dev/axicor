#!/bin/bash
# Quick startup script for IDE + both mock servers

set -e

WORKSPACE="/home/alex/Workflow/Genesis"
cd "$WORKSPACE"

echo "[setup] Killing old processes..."
killall python3 2>/dev/null || true
killall genesis-ide 2>/dev/null || true
sleep 1

echo "[setup] Starting GeometryServer (TCP 8001)..."
python3 genesis-ide/tests/geometry_protocol.py 2>&1 &
GEOM_PID=$!

echo "[setup] Starting TelemetryServer (WebSocket 8002)..."
python3 genesis-ide/tests/telemetry_mock.py 2>&1 &
TELEM_PID=$!

sleep 2

echo ""
echo "┌─────────────────────────────────────────────┐"
echo "│  Servers Ready:                             │"
echo "│                                             │"
echo "│  ✓ GeometryServer  (TCP 8001, PID $GEOM_PID)   │"
echo "│  ✓ TelemetryServer (WebSocket 8002, PID $TELEM_PID) │"
echo "│                                             │"
echo "│  Starting IDE...                            │"
echo "└─────────────────────────────────────────────┘"
echo ""

# Run IDE in foreground
cargo run -p genesis-ide 2>&1

# Cleanup
echo ""
echo "[cleanup] Stopping servers..."
kill $GEOM_PID $TELEM_PID 2>/dev/null || true
echo "[cleanup] Done."
