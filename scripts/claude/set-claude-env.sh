#!/usr/bin/env bash
# set-claude-env.sh — point Claude Code at a local LM Studio model instead of the
# Anthropic cloud, and revert back. Works when sourced from bash OR zsh.
#
#   source ./scripts/claude/set-claude-env.sh         # enable local (LM Studio / proxy)
#   source ./scripts/claude/set-claude-env.sh reset   # revert to Anthropic cloud
#   source ./scripts/claude/reset-claude-env.sh       # revert to Anthropic cloud (standalone)
#   claude_env_test                                   # check the endpoint is reachable
#
# Env is read by Claude Code at STARTUP, so after sourcing you must restart it:
#   • VS Code extension → "Developer: Reload Window", then a new chat
#   • CLI               → start a fresh `claude` session
#
# ──────────────────────────────────────────────────────────────────────────────
# READ THIS — protocol mismatch (the #1 reason "local Claude Code" silently fails)
#   Claude Code talks Anthropic's Messages API: it POSTs to  BASE_URL/v1/messages.
#   LM Studio's built-in server is OpenAI-compatible (/v1/chat/completions) and
#   does NOT serve /v1/messages — so pointing BASE_URL straight at LM Studio
#   typically returns 404. Two ways to bridge it; set LMSTUDIO_BASE_URL to
#   whichever one actually serves /v1/messages:
#     A) Translation proxy in front of LM Studio (recommended):
#          • claude-code-router : npm i -g @musistudio/claude-code-router
#          • LiteLLM proxy      : pip install 'litellm[proxy]'  (Anthropic passthrough)
#        Point BASE_URL at the proxy (e.g. http://host.docker.internal:4000).
#     B) An LM Studio build / plugin that exposes an Anthropic-compatible endpoint:
#        point BASE_URL straight at it.
# ──────────────────────────────────────────────────────────────────────────────

# ── guard: must be sourced (exports have to persist in your shell) ────────────
_sce_sourced=1
if [ -n "${ZSH_VERSION:-}" ]; then
  # When sourced, zsh's ZSH_EVAL_CONTEXT ends in ":file" (e.g. "cmdarg:file",
  # "toplevel:file"); when executed it's just "toplevel".
  case "${ZSH_EVAL_CONTEXT:-}" in *:file|*:file:*) ;; *) _sce_sourced=0 ;; esac
elif [ -n "${BASH_VERSION:-}" ]; then
  [ "${BASH_SOURCE[0]}" = "${0}" ] && _sce_sourced=0
fi
if [ "$_sce_sourced" = "0" ]; then
  echo "set-claude-env.sh must be SOURCED, not executed:" >&2
  echo "    source ./scripts/claude/set-claude-env.sh" >&2
  exit 1
fi
unset _sce_sourced

# ── revert path ───────────────────────────────────────────────────────────────
if [ "${1:-}" = "reset" ] || [ "${1:-}" = "off" ]; then
  unset ANTHROPIC_BASE_URL ANTHROPIC_AUTH_TOKEN ANTHROPIC_MODEL \
        ANTHROPIC_SMALL_FAST_MODEL ANTHROPIC_DEFAULT_OPUS_MODEL \
        ANTHROPIC_DEFAULT_SONNET_MODEL ANTHROPIC_DEFAULT_HAIKU_MODEL
  echo "[claude-env] reverted to Anthropic cloud (local overrides unset)."
  echo "[claude-env] a fresh login shell re-sources your .env (ANTHROPIC_API_KEY)."
  echo "[claude-env] restart Claude Code (reload window / new session) to apply."
  return 0 2>/dev/null || true
fi

# ── where is LM Studio (or your proxy) listening? ─────────────────────────────
# From inside a devcontainer the host's localhost is unreachable as "localhost";
# reach the host via host.docker.internal. Auto-detected, override with any of
# LMSTUDIO_BASE_URL / LMSTUDIO_HOST / LMSTUDIO_PORT before sourcing.
LMSTUDIO_PORT="${LMSTUDIO_PORT:-1234}"
if [ -z "${LMSTUDIO_HOST:-}" ]; then
  if [ -f /.dockerenv ] || [ -n "${REMOTE_CONTAINERS:-}" ] || [ -n "${CODESPACES:-}" ]; then
    LMSTUDIO_HOST="host.docker.internal"   # container → host
  else
    LMSTUDIO_HOST="localhost"              # running on the host directly
  fi
fi
LMSTUDIO_BASE_URL="${LMSTUDIO_BASE_URL:-http://${LMSTUDIO_HOST}:${LMSTUDIO_PORT}}"

# Model id exactly as it appears in `GET /v1/models` (LM Studio "API identifier").
# Override before sourcing:  LMSTUDIO_MODEL=my-model source ./set-claude-env.sh
LMSTUDIO_MODEL="${LMSTUDIO_MODEL:-google/gemma-4-e4b}"

# ── export the Claude Code overrides ──────────────────────────────────────────
export ANTHROPIC_BASE_URL="$LMSTUDIO_BASE_URL"
export ANTHROPIC_AUTH_TOKEN="${LMSTUDIO_TOKEN:-lmstudio}"   # any non-empty token
# Map every tier to the one local model so whichever tier Claude Code requests
# (opus/sonnet/haiku/background) resolves to your local model.
export ANTHROPIC_MODEL="$LMSTUDIO_MODEL"
export ANTHROPIC_SMALL_FAST_MODEL="$LMSTUDIO_MODEL"
export ANTHROPIC_DEFAULT_OPUS_MODEL="$LMSTUDIO_MODEL"
export ANTHROPIC_DEFAULT_SONNET_MODEL="$LMSTUDIO_MODEL"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="$LMSTUDIO_MODEL"
# Using a Bearer token locally — drop any cloud x-api-key so they don't conflict.
unset ANTHROPIC_API_KEY

echo "[claude-env] Claude Code → ${ANTHROPIC_BASE_URL}  (model: ${LMSTUDIO_MODEL})"
echo "[claude-env] run 'claude_env_test' to check reachability."
echo "[claude-env] restart Claude Code (reload window / new session) to apply."

# ── connectivity helper (left defined in your shell) ──────────────────────────
claude_env_test() {
  echo "[claude-env] GET ${ANTHROPIC_BASE_URL}/v1/models"
  if curl -fsS --max-time 5 "${ANTHROPIC_BASE_URL}/v1/models" \
       -H "Authorization: Bearer ${ANTHROPIC_AUTH_TOKEN:-lmstudio}"; then
    echo ""
    echo "[claude-env] endpoint reachable ✅  (this is the OpenAI /v1/models route)"
    echo "[claude-env] Claude Code will POST to ${ANTHROPIC_BASE_URL}/v1/messages —"
    echo "             if that route 404s, put a proxy in front (see script header)."
  else
    echo "[claude-env] ❌ cannot reach ${ANTHROPIC_BASE_URL}"
    echo "             • is LM Studio's server started (Developer ▸ Start Server)?"
    echo "             • in a container, the host needs host.docker.internal —"
    echo "               add  \"runArgs\": [\"--add-host=host.docker.internal:host-gateway\"]"
    echo "               to .devcontainer/devcontainer.json on non-Docker-Desktop hosts."
  fi
}
