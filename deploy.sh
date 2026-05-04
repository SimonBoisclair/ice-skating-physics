#!/bin/bash
# Manual deploy script — run on the RunPod server
# Usage: bash deploy.sh

set -e
cd /workspace

echo "=== Pulling latest code ==="
git pull origin main

echo "=== Building React frontend ==="
cd sandbox-ui
npm install --silent
npm run build
cd ..

echo "=== Restarting server ==="
pkill -f warp_server.py || true
sleep 2
nohup python3 warp_server.py > server.log 2>&1 &

echo "=== Waiting for server to start ==="
sleep 20
if pgrep -f warp_server.py > /dev/null; then
  echo "Server started successfully"
  tail -5 server.log
else
  echo "ERROR: Server failed to start"
  tail -20 server.log
  exit 1
fi
