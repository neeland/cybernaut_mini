#!/usr/bin/env bash
# Registers the hosted Nosible MCP server with Claude Code (local scope).
# Reads NOSIBLE_API_KEY from the .env file at the repo root (run `make env`
# first to sync it from ../nosible-py). Normally unnecessary — .mcp.json
# already defines this server project-wide; use this only for local-scope
# registration outside the project config.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/../.env"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Error: .env file not found at $ENV_FILE — run 'make env' first" >&2
  exit 1
fi

set -a
source "$ENV_FILE"
set +a

if [[ -z "${NOSIBLE_API_KEY:-}" ]]; then
  echo "Error: NOSIBLE_API_KEY is not set in $ENV_FILE" >&2
  exit 1
fi

# Re-runnable: replace any existing registration
claude mcp remove nosible >/dev/null 2>&1 || true

claude mcp add --transport http nosible https://nosible-mcp.onrender.com/mcp/ \
  --header "X-Nosible-Api-Key: ${NOSIBLE_API_KEY}"
