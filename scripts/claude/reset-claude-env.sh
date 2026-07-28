#!/usr/bin/env bash
# reset-claude-env.sh — restore Claude Code's default (Anthropic cloud) model APIs
# by unsetting the local LM Studio / proxy overrides exported by set-claude-env.sh.
# Works when sourced from bash OR zsh.
#
#   source ./scripts/claude/reset-claude-env.sh    # revert to Anthropic cloud
#
# Env is read by Claude Code at STARTUP, so after sourcing you must restart it:
#   • VS Code extension → "Developer: Reload Window", then a new chat
#   • CLI               → start a fresh `claude` session
#
# This is the standalone twin of `set-claude-env.sh reset`. It only unsets the
# overrides — a fresh login shell re-sources your .env (ANTHROPIC_API_KEY) so the
# cloud default returns on its own.

# ── guard: must be sourced (unsets have to persist in your shell) ──────────────
_rce_sourced=1
if [ -n "${ZSH_VERSION:-}" ]; then
  case "${ZSH_EVAL_CONTEXT:-}" in *:file|*:file:*) ;; *) _rce_sourced=0 ;; esac
elif [ -n "${BASH_VERSION:-}" ]; then
  [ "${BASH_SOURCE[0]}" = "${0}" ] && _rce_sourced=0
fi
if [ "$_rce_sourced" = "0" ]; then
  echo "reset-claude-env.sh must be SOURCED, not executed:" >&2
  echo "    source ./scripts/claude/reset-claude-env.sh" >&2
  exit 1
fi
unset _rce_sourced

# ── revert: drop every override set-claude-env.sh exports ──────────────────────
unset ANTHROPIC_BASE_URL ANTHROPIC_AUTH_TOKEN ANTHROPIC_MODEL \
      ANTHROPIC_SMALL_FAST_MODEL ANTHROPIC_DEFAULT_OPUS_MODEL \
      ANTHROPIC_DEFAULT_SONNET_MODEL ANTHROPIC_DEFAULT_HAIKU_MODEL

echo "[claude-env] reverted to Anthropic cloud (local overrides unset)."
echo "[claude-env] a fresh login shell re-sources your .env (ANTHROPIC_API_KEY)."
echo "[claude-env] restart Claude Code (reload window / new session) to apply."
return 0 2>/dev/null || true
