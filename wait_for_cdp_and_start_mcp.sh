#!/usr/bin/env bash
set -euo pipefail
CDP_ENDPOINT="${CDP_ENDPOINT:-http://127.0.0.1:9222}"
PORT="${MCP_PORT:-8931}"
TRIES=60
SLEEP=1
echo "[mcp-wrapper] Waiting for CDP endpoint $CDP_ENDPOINT ..."
for i in $(seq 1 $TRIES); do
  if curl -s "$CDP_ENDPOINT/json/version" | grep -q "Browser"; then
    echo "[mcp-wrapper] CDP endpoint is up after $i tries. Starting MCP server."
    exec npx @playwright/mcp@latest --port "$PORT" --cdp-endpoint "$CDP_ENDPOINT"
  fi
  sleep $SLEEP
done
echo "[mcp-wrapper] CDP endpoint not ready after $TRIES attempts; starting MCP anyway (may fail)."
exec npx @playwright/mcp@latest --port "$PORT" --cdp-endpoint "$CDP_ENDPOINT"
