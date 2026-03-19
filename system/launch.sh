#!/bin/bash
set -euo pipefail

# -------- CONFIG --------
PI_RIGHT="pi@172.20.10.4"
PI_LEFT="pi@172.20.10.7"
PI_BOTTOM="pi@172.20.10.5"
PI4_D435="pi@172.20.10.2"

REMOTE_DIR="~/Documents/GreenhouseGuardians"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Starting all camera services in parallel..."

# Kill existing sessions and start fresh — all SSH calls run concurrently
ssh -o ConnectTimeout=5 "$PI_RIGHT"  \
    "tmux kill-session -t cam 2>/dev/null; \
     tmux new-session -d -s cam \"cd $REMOTE_DIR && python3 pi_zero_right.py 2>&1 | tee /tmp/cam.log\"" &

ssh -o ConnectTimeout=5 "$PI_LEFT"   \
    "tmux kill-session -t cam 2>/dev/null; \
     tmux new-session -d -s cam \"cd $REMOTE_DIR && python3 pi_zero_left.py 2>&1 | tee /tmp/cam.log\"" &

ssh -o ConnectTimeout=5 "$PI_BOTTOM" \
    "tmux kill-session -t cam 2>/dev/null; \
     tmux new-session -d -s cam \"cd $REMOTE_DIR && python3 pi_zero_bottom.py 2>&1 | tee /tmp/cam.log\"" &

ssh -o ConnectTimeout=5 "$PI4_D435"  \
    "tmux kill-session -t d435 2>/dev/null; \
     tmux new-session -d -s d435 \"cd $REMOTE_DIR && ./d435_server 2>&1 | tee /tmp/d435.log\"" &

# Wait for all SSH connections to complete
wait
echo "All remote services launched."

# Start dashboard locally
tmux kill-session -t dashboard 2>/dev/null || true
tmux new-session -d -s dashboard "cd '$SCRIPT_DIR' && python3 dashboard.py"
echo "Dashboard started at http://127.0.0.1:5050"
echo ""
echo "Attach to sessions:"
echo "  Dashboard : tmux attach -t dashboard"
echo "  Pi Zero   : ssh <pi-host> && tmux attach -t cam"
echo "  D435      : ssh $PI4_D435 && tmux attach -t d435"
