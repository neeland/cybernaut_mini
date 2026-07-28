#!/usr/bin/env bash
# Install/update Claude Code + oh-my-claudecode: CLIs, OMC plugin, HUD
# statusline, hooks, and ~/.claude/settings.json niceties.
#
# Called by .devcontainer/post-create.sh on container create, and safe to
# re-run manually any time something looks broken:
#
#   scripts/setup_claude_omc.sh
#
# Both CLIs are also baked into the devcontainer IMAGE (see
# .devcontainer/Dockerfile), so `claude` exists even if this script never ran —
# this script's job is updating to latest and wiring the plugin/HUD/settings.
set -uo pipefail
log() { echo "[setup-claude-omc] $*"; }
export PATH="$HOME/.local/bin:$PATH"

# Canonical OMC marketplace URL — matches the host machine's settings.json.
OMC_MARKETPLACE_URL="https://github.com/Yeachan-Heo/oh-my-claudecode.git"

npm_g() {  # global npm install with sudo fallback (image prefix is root-owned)
  npm install -g "$@" 2>/dev/null || sudo npm install -g "$@"
}

# 1. Claude Code + omc CLI: update to latest (npm, matching the image install);
#    if npm is somehow unavailable, fall back to the native installer, which
#    puts a per-user binary in ~/.local/bin.
if command -v npm >/dev/null 2>&1; then
  log "installing/updating Claude Code + omc CLI (npm -g)"
  npm_g @anthropic-ai/claude-code@latest oh-my-claude-sisyphus@latest \
    || log "WARN: npm -g install failed"
fi
if ! command -v claude >/dev/null 2>&1; then
  log "claude not on PATH; trying native installer"
  curl -fsSL https://claude.ai/install.sh | bash || log "WARN: native install failed"
  export PATH="$HOME/.local/bin:$PATH"
fi
if ! command -v claude >/dev/null 2>&1; then
  log "ERROR: claude still not on PATH — aborting (omc setup needs it)"
  exit 1
fi
log "claude: $(claude --version 2>/dev/null | head -1)"

# 2. oh-my-claudecode plugin — provides the /oh-my-claudecode:* skills.
#    marketplace add fails harmlessly if it is already registered.
log "wiring oh-my-claudecode plugin"
claude plugin marketplace add "$OMC_MARKETPLACE_URL" 2>/dev/null || true
claude plugin marketplace update omc || true
claude plugin install oh-my-claudecode@omc || log "WARN: omc plugin install failed"

# 3. omc setup — installs the HUD statusline wrapper, hooks, agents and skills
#    into ~/.claude.
if command -v omc >/dev/null 2>&1; then
  log "running omc setup (HUD + hooks + agents + skills)"
  omc setup --quiet || log "WARN: omc setup failed"
else
  log "WARN: omc CLI unavailable; HUD/hooks may be incomplete"
fi

# 4. Merge the host's Claude niceties into ~/.claude/settings.json (jq merge,
#    existing keys survive). bypassPermissions is container-specific (sandboxed
#    vscode user); DISABLE_AUTOUPDATER because the npm -g install is root-owned
#    — updates happen by re-running this script, not by claude self-updating.
mkdir -p ~/.claude
SETTINGS="$HOME/.claude/settings.json"
[ -s "$SETTINGS" ] || echo '{}' > "$SETTINGS"
if command -v jq >/dev/null 2>&1; then
  tmp="$(mktemp)"
  jq \
    '.permissions.defaultMode = "bypassPermissions"
     | .env["CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS"] = "1"
     | .env["DISABLE_AUTOUPDATER"] = "1"
     | .statusLine = {"type": "command", "command": "node ${CLAUDE_CONFIG_DIR:-$HOME/.claude}/hud/omc-hud.mjs"}
     | .enabledPlugins["oh-my-claudecode@omc"] = true
     | .extraKnownMarketplaces.omc.source = {"source": "git", "url": "https://github.com/Yeachan-Heo/oh-my-claudecode.git"}
     | .editorMode = "vim"
     | .verbose = true
     | .teammateMode = "tmux"
     | .tui = "fullscreen"' \
    "$SETTINGS" > "$tmp" && mv "$tmp" "$SETTINGS"
  log "merged HUD + OMC niceties into ~/.claude/settings.json"
else
  log "WARN: jq missing; could not write ~/.claude/settings.json"
fi

log "done — run scripts/devcontainer_doctor.sh to verify"
