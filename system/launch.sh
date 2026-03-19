#!/bin/bash

echo "🚀 Starting Multi-Camera System..."

# -------- CONFIG --------
PI_RIGHT="pi@172.20.10.4"
PI_LEFT="pi@172.20.10.7"
PI_BOTTOM="pi@172.20.10.5"
PI4_D435="pi@172.20.10.2"

REMOTE_DIR="~/Documents/GreenhouseGuardians"

# -------- START PI ZERO CAMS --------
echo "Starting Pi Zero RIGHT..."
ssh $PI_RIGHT "tmux kill-session -t cam 2>/dev/null; tmux new -d -s cam 'cd $REMOTE_DIR && python3 pi_zero_right.py'"

echo "Starting Pi Zero LEFT..."
ssh $PI_LEFT "tmux kill-session -t cam 2>/dev/null; tmux new -d -s cam 'cd $REMOTE_DIR && python3 pi_zero_left.py'"

echo "Starting Pi Zero BOTTOM..."
ssh $PI_BOTTOM "tmux kill-session -t cam 2>/dev/null; tmux new -d -s cam 'cd $REMOTE_DIR && python3 pi_zero_bottom.py'"

# -------- START D435 SERVER --------
echo "Starting D435 (Pi 4)..."
ssh $PI4_D435 "tmux kill-session -t d435 2>/dev/null; tmux new -d -s d435 'cd $REMOTE_DIR && python3 d435_server.py'"

# -------- START DASHBOARD LOCALLY --------
echo "Starting Dashboard locally..."
tmux kill-session -t dashboard 2>/dev/null
tmux new -d -s dashboard "cd $REMOTE_DIR && python3 dashboard.py"

echo "✅ All systems launched!"
echo ""
echo "Attach to sessions with:"
echo "  tmux attach -t dashboard"
echo "  ssh pi@IP -> tmux attach -t cam"