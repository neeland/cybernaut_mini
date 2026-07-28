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

# Entire: the repo's git + Claude Code hooks call `entire` unconditionally, so a
# missing binary means every commit prints a warning and captures nothing.
if command -v entire >/dev/null 2>&1; then
  ok "entire $(entire version 2>/dev/null | head -1)"
  # Hook matchers change between releases; stale ones stop firing silently and
  # only `entire doctor` notices.
  if entire doctor --force </dev/null 2>&1 | grep -qi "OUT OF DATE"; then
    bad "Entire agent hooks are out of date (they no longer fire)" \
        "scripts/entire_setup.sh"
  else
    ok "Entire hooks current"
  fi
else
  bad "entire missing (git hooks print a warning and capture nothing)" \
      "scripts/entire_setup.sh"
fi

[ -x .venv/bin/python ] \
  && ok ".venv ($(.venv/bin/python --version 2>&1))" \
  || bad ".venv missing" "uv sync --extra viz"

# ------------------------------------------------------------------ #
# Project checks — the pieces a from-scratch build must also produce  #
# ------------------------------------------------------------------ #

if [ -x .venv/bin/python ]; then
  # Kedro discovers project commands at cybernaut_mini.cli:cli. If that Click
  # group goes missing, every `kedro` subcommand dies with "Cannot load commands"
  # while the Typer CLI keeps working — so check the kedro side explicitly.
  .venv/bin/python -m kedro registry list >/dev/null 2>&1 \
    && ok "kedro CLI loads ($(.venv/bin/python -m kedro registry list 2>/dev/null | grep -c '^-') pipelines registered)" \
    || bad "kedro CLI cannot load project commands" "check cybernaut_mini.cli exposes a Click group named 'cli'"

  # Notebooks are executed by the test suite, which needs a resolvable kernel.
  .venv/bin/python -c "
from jupyter_client.kernelspec import KernelSpecManager
raise SystemExit(0 if 'python3' in KernelSpecManager().find_kernel_specs() else 1)
" >/dev/null 2>&1 \
    && ok "jupyter python3 kernelspec (notebooks executable)" \
    || bad "no python3 kernelspec" "uv sync --extra viz   (installs ipykernel from the dev group)"

  # The catalog is the project's only data entry point; a broken one breaks
  # every pipeline and every notebook at once.
  .venv/bin/python -c "
from cybernaut_mini.notebook import kedro_catalog
required = {'documents', 'judgments', 'shard_index', 'raw_corpus', 'raw_corpus_source'}
missing = required - set(kedro_catalog().keys())
raise SystemExit(1 if missing else 0)
" >/dev/null 2>&1 \
    && ok "Kedro catalog resolves all core datasets" \
    || bad "catalog is missing core datasets" "check conf/base/catalog.yml"
fi

if [ "$FAIL" -eq 0 ]; then
  echo "all checks passed"
else
  echo "some checks FAILED — run the fixes above, then re-run this script"
fi
exit "$FAIL"
