#!/usr/bin/env bash
# Restart the MoCap labeller Node server (frees the port first).
set -e
cd "$(dirname "$0")"
PORT="${PORT:-8765}"
if command -v lsof >/dev/null 2>&1; then
  PIDS=$(lsof -t -i:"$PORT" 2>/dev/null || true)
  if [ -n "$PIDS" ]; then
    echo "Stopping process(es) on port $PORT: $PIDS"
    kill $PIDS 2>/dev/null || true
    sleep 0.4
  fi
fi
echo "Starting Node server on port $PORT…"
exec node server.mjs
