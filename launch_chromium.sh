#!/usr/bin/env bash
set -eu
# Try to enable pipefail where supported
(set -o pipefail) 2>/dev/null || true
# Launch a persistent Chromium instance with remote debugging (CDP) enabled on 0.0.0.0:9222
# Use the Playwright-installed Chromium binary
CHROMIUM_BIN="$(python -c 'import playwright.__main__, json, sys; from pathlib import Path; from playwright._impl._driver import compute_driver_executable;print(compute_driver_executable())' 2>/dev/null || true)"
# Fallback path typical inside playwright install
if [[ -z "$CHROMIUM_BIN" || ! -x "$CHROMIUM_BIN" ]]; then
  CHROMIUM_BIN="/root/.cache/ms-playwright/chromium*/chrome-linux/chrome"
fi
# Resolve glob if needed
if [[ "$CHROMIUM_BIN" == *"*"* ]]; then
  CHROMIUM_CANDIDATE=$(compgen -G "$CHROMIUM_BIN" | head -n1 || true)
  if [[ -n "$CHROMIUM_CANDIDATE" ]]; then
    CHROMIUM_BIN="$CHROMIUM_CANDIDATE"
  fi
fi

echo "[chromium] Using binary: $CHROMIUM_BIN" >&2

exec "$CHROMIUM_BIN" \
  --remote-debugging-address=0.0.0.0 \
  --remote-debugging-port=9222 \
  --no-first-run \
  --no-default-browser-check \
  --disable-dev-shm-usage \
  --disable-gpu \
  --no-sandbox \
  --headless=new \
  --hide-scrollbars \
  --disable-software-rasterizer \
  about:blank
