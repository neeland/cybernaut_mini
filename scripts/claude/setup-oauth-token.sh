#!/usr/bin/env bash
# setup-oauth-token.sh — mint a Claude subscription OAuth token and persist it
# into .env, so the devcontainer picks it up at boot (runArgs --env-file).
#
#   ./scripts/claude/setup-oauth-token.sh                 # interactive: runs `claude setup-token`
#   ./scripts/claude/setup-oauth-token.sh --token sk-ant-oat01-...   # skip the flow, just write it
#
# What it does:
#   1. ensures .env exists / is aligned (delegates to .devcontainer/sync-env.sh)
#   2. runs `claude setup-token` under script(1) — the browser OAuth flow stays
#      fully interactive, but the terminal transcript is captured so the printed
#      sk-ant-oat01-... token can be scraped automatically afterwards
#   3. writes CLAUDE_CODE_OAUTH_TOKEN=<token> into .env (backing it up first)
#   4. blanks ANTHROPIC_API_KEY in .env — .env.example mandates exactly ONE auth
#      method, and the OAuth token (subscription billing) now takes over
#
# Run on your HOST machine (needs a browser). Inside a container the flow still
# works but you must open the printed URL manually on the host.
set -euo pipefail
cd "$(dirname "$0")/../.."

ENVFILE=".env"
log()  { echo "[oauth-token] $*"; }
die()  { echo "[oauth-token] ERROR: $*" >&2; exit 1; }

TOKEN="${2:-}"
case "${1:-}" in
  --token) [ -n "$TOKEN" ] || die "--token requires a value" ;;
  "") ;;
  *) die "unknown argument: $1 (usage: setup-oauth-token.sh [--token sk-ant-oat01-...])" ;;
esac

# 1. .env must exist and contain the CLAUDE_CODE_OAUTH_TOKEN key.
[ -f .devcontainer/sync-env.sh ] && bash .devcontainer/sync-env.sh >/dev/null
[ -f "$ENVFILE" ] || die "$ENVFILE not found and sync-env.sh could not create it"

# 2. Interactive mint (skipped when --token was given).
if [ -z "$TOKEN" ]; then
  command -v claude >/dev/null 2>&1 || die "claude CLI not on PATH"
  if [ -f /.dockerenv ] || [ -n "${REMOTE_CONTAINERS:-}" ] || [ -n "${CODESPACES:-}" ]; then
    log "NOTE: you appear to be inside a container — no browser here. Open the"
    log "      URL the flow prints on your HOST browser, then continue as normal."
  fi
  CAPTURE="$(mktemp)"
  trap 'rm -f "$CAPTURE"' EXIT
  log "launching 'claude setup-token' — complete the browser OAuth flow..."
  # script(1) keeps stdin/stdout attached to the TTY (the flow stays interactive)
  # while recording a transcript we can scrape. Flag syntax differs by OS.
  if [ "$(uname)" = "Darwin" ]; then
    script -q "$CAPTURE" claude setup-token || true
  else
    script -q -c "claude setup-token" "$CAPTURE" || true
  fi
  # Tokens are base64url-ish; take the LAST match (the final printed token).
  TOKEN="$(tr -d '\r' < "$CAPTURE" | grep -oE 'sk-ant-oat01-[A-Za-z0-9_-]+' | tail -1 || true)"
  if [ -n "$TOKEN" ]; then
    log "captured token from the setup-token output"
  else
    log "could not find a token in the transcript (flow cancelled or output format changed)"
    printf '[oauth-token] paste the sk-ant-oat01-... token (empty to abort): '
    IFS= read -r TOKEN
    [ -n "$TOKEN" ] || die "no token provided"
  fi
fi

case "$TOKEN" in
  sk-ant-oat01-*) ;;
  *) die "token does not look like a Claude OAuth token (expected sk-ant-oat01-...)" ;;
esac

# 3+4. Persist: back up, set the token, blank the API key (exactly ONE auth method).
BACKUP="$ENVFILE.bak-$(date +%Y%m%d%H%M%S)"
cp "$ENVFILE" "$BACKUP"
tmp="$(mktemp)"
sed -e "s|^CLAUDE_CODE_OAUTH_TOKEN=.*|CLAUDE_CODE_OAUTH_TOKEN=${TOKEN}|" \
    -e "s|^ANTHROPIC_API_KEY=.*|ANTHROPIC_API_KEY=|" \
    "$ENVFILE" > "$tmp"
mv "$tmp" "$ENVFILE"

grep -q "^CLAUDE_CODE_OAUTH_TOKEN=${TOKEN}$" "$ENVFILE" || die "failed to write token into $ENVFILE"
log "CLAUDE_CODE_OAUTH_TOKEN written to $ENVFILE (backup: $BACKUP)"
log "ANTHROPIC_API_KEY blanked — usage now bills to your Claude subscription"
log "rebuild/reopen the devcontainer (or restart Claude Code) to pick it up"
