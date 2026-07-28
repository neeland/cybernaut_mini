#!/usr/bin/env bash
# Set up the Numerai MCP server auth token (interactive — opens a browser).
#
# The `numerai` MCP server itself is already declared in ../.mcp.json; this
# script only obtains the API token it needs (NUMERAI_MCP_AUTH="Token <key>").
# The official installer prompts you to create/choose a key at
# numer.ai/mcp/authorize and writes the token to your shell profile.
#
# After running, open a new shell (or `source` your profile) so Claude Code
# picks up NUMERAI_MCP_AUTH, then check with: claude mcp list
set -euo pipefail
cd "$(dirname "$0")/../.."

echo "==> Running the Numerai Claude MCP installer (interactive)"
echo "    This opens a browser to generate/select a Numerai API key."
curl -sL https://numer.ai/install-claude-mcp.sh | bash

echo
echo "Done. Restart your shell (or 'source ~/.bashrc') and run: claude mcp list"
