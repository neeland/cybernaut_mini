#!/usr/bin/env bash
# Health check for the devcontainer: verifies everything the container is
# supposed to ship (claude, omc, plugin, HUD, uv, .venv) and prints the fix
# for anything missing. Exits non-zero if any check fails.
#
#   scripts/devcontainer_doctor.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"

FAIL=0
ok()   { printf '  \033[32mok\033[0m    %s\n' "$1"; }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n      fix: %s\n' "$1" "$2"; FAIL=1; }

echo "devcontainer doctor"

command -v node >/dev/null 2>&1 \
  && ok "node $(node --version)" \
  || bad "node missing" "rebuild the devcontainer (Dockerfile installs Node LTS)"

command -v claude >/dev/null 2>&1 \
  && ok "claude $(claude --version 2>/dev/null | head -1)" \
  || bad "claude missing" "scripts/setup_claude_omc.sh"

command -v omc >/dev/null 2>&1 \
  && ok "omc $(command -v omc)" \
  || bad "omc CLI missing" "scripts/setup_claude_omc.sh"

if command -v claude >/dev/null 2>&1; then
  claude plugin list 2>/dev/null | grep -qi 'oh-my-claudecode' \
    && ok "oh-my-claudecode plugin installed" \
    || bad "oh-my-claudecode plugin missing" "scripts/setup_claude_omc.sh"
fi

[ -f "$HOME/.claude/hud/omc-hud.mjs" ] \
  && ok "OMC HUD statusline (~/.claude/hud/omc-hud.mjs)" \
  || bad "OMC HUD missing" "omc setup   (or scripts/setup_claude_omc.sh)"

[ -s "$HOME/.claude/settings.json" ] && grep -q statusLine "$HOME/.claude/settings.json" 2>/dev/null \
  && ok "~/.claude/settings.json has statusLine" \
  || bad "~/.claude/settings.json not wired" "scripts/setup_claude_omc.sh"

command -v uv >/dev/null 2>&1 \
  && ok "uv $(uv --version 2>/dev/null)" \
  || bad "uv missing" "curl -LsSf https://astral.sh/uv/install.sh | sh"

[ -x .venv/bin/python ] \
  && ok ".venv ($(.venv/bin/python --version 2>&1))" \
  || bad ".venv missing" "uv sync --extra viz"

if [ "$FAIL" -eq 0 ]; then
  echo "all checks passed"
else
  echo "some checks FAILED — run the fixes above, then re-run this script"
fi
exit "$FAIL"
