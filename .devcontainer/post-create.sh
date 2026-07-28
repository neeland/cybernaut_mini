#!/usr/bin/env bash
# Runs once on container create/rebuild. Sequential + idempotent so the whole
# environment (uv, deps, Claude Code, oh-my-claudecode, HUD, hooks, shell) is
# set up automatically with no manual steps. Safe to re-run.
#
# claude + omc are baked into the IMAGE (Dockerfile) so they exist even if this
# script fails; scripts/setup_claude_omc.sh (step 3) updates them and wires the
# plugin/HUD/settings, and can be re-run standalone. Verify any time with
# scripts/devcontainer_doctor.sh.
set -uo pipefail
log() { echo "[post-create] $*"; }
export PATH="$HOME/.local/bin:$PATH"

# 1. uv (for the vscode user; the Dockerfile installs it for root)
if ! command -v uv >/dev/null 2>&1; then
  log "installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# 2. Project dependencies — uv sync creates the project-local .venv (python
#    pinned by .python-version) with the dev dependency-group. The viz extra
#    adds kedro-viz for the pipeline graph on port 4141 (auto-forwarded).
if [ -f pyproject.toml ]; then
  log "installing project deps (uv sync --extra viz)"
  uv sync --extra viz || log "WARN: uv sync failed"
fi

# 2b. VisiData — interactive terminal explorer for parquet/csv/jsonl artifacts.
#     Installed isolated via `uv tool` so it never touches the project venv.
if command -v uv >/dev/null 2>&1; then
  log "installing visidata (uv tool)"
  uv tool install 'visidata[parquet]' 2>/dev/null \
    || uv tool install visidata \
    || log "WARN: visidata install failed (non-fatal)"
fi

# 3. Claude Code + oh-my-claudecode — update CLIs, install the OMC plugin, run
#    `omc setup` (HUD/hooks/agents/skills), merge ~/.claude/settings.json.
if [ -x scripts/setup_claude_omc.sh ] || [ -f scripts/setup_claude_omc.sh ]; then
  bash scripts/setup_claude_omc.sh \
    || log "WARN: claude/omc setup incomplete — re-run scripts/setup_claude_omc.sh"
else
  log "WARN: scripts/setup_claude_omc.sh missing; claude/omc left at image versions"
fi

# 3b. Entire CLI — session capture linked to commits. The repo ships the hooks
#     (.git/hooks, .claude/settings.json, .entire/settings.json) but they no-op
#     without the binary, which is what made every commit print
#     "[entire] Entire CLI is enabled but not installed or not on PATH".
#     The script also refreshes hooks, whose tool matchers change between
#     releases and silently stop firing when stale.
if [ -f scripts/entire_setup.sh ]; then
  bash scripts/entire_setup.sh \
    || log "WARN: entire setup incomplete — re-run scripts/entire_setup.sh"
else
  log "NOTE: scripts/entire_setup.sh missing; Entire hooks will no-op"
fi

# 4. pre-commit git hooks (only if the repo defines them)
if [ -f .pre-commit-config.yaml ] && command -v uv >/dev/null 2>&1; then
  log "installing pre-commit hooks"
  uv run pre-commit install || log "WARN: pre-commit install failed"
fi

# 5. Handy command cheatsheet (shown by ~/.zshrc on interactive login)
mkdir -p ~/.config/dev-bootstrap
cat > ~/.config/dev-bootstrap/commands.txt <<'EOF'
source .venv/bin/activate            # activate the project venv
make env                             # sync MCP/API keys into .env + .claude/settings.local.json
make check                           # lint + typecheck + test (includes notebook execution)
make build-sample                    # build the sample index (configs/tiny.yaml)
make search-sample                   # hybrid search over the sample index
make eval-sample                     # eval against data/sample/judgments.jsonl
pytest -q                            # test suite

# Kedro pipelines (all data flows through conf/base/catalog.yml)
make pipelines                       # list registered pipelines
make ingest                          # corpus_ingest: 01_raw -> 02_intermediate -> 03_primary
make build-prod                      # full production shard (needs `make install-prod` + pinned SHAs)
make eval-pipeline                   # evaluation -> data/08_reporting/eval_report.json
kedro viz                            # interactive pipeline graph (port 4141, auto-forwarded)

# Notebooks — each one reads through the catalog and is executed by the tests
make notebooks                       # run every notebook headlessly
make lab                             # JupyterLab with catalog/context/session injected (:8888)

make install-prod                    # add the hf + st extras for real corpus builds
make entire                          # install/refresh the Entire CLI + its hooks
make doctor                          # verify the whole container is wired up
vd artifacts/sample                  # browse index artifacts (VisiData)
scripts/devcontainer_doctor.sh       # verify claude/omc/uv/.venv/kedro/catalog are healthy
scripts/setup_claude_omc.sh          # (re)install/update Claude Code + OMC
EOF

# 6. Shell: link the shared dotfiles (zshrc / starship / tmux) from ~/env.
#    ~/env is bind-mounted read-only from the host (see devcontainer.json), so a
#    single source of truth drives the host and every container.
ENV_DIR="$HOME/env"
if [ -x "$ENV_DIR/install.sh" ]; then
  log "linking dotfiles from $ENV_DIR"
  "$ENV_DIR/install.sh" --link || log "WARN: dotfile link failed"
else
  log "NOTE: ~/env not mounted; using default shell config"
fi

# 7. zsh-history-substring-search (not packaged on Debian)
HSS="$HOME/.zsh/plugins/zsh-history-substring-search"
[ -d "$HSS" ] || git clone --depth=1 \
  https://github.com/zsh-users/zsh-history-substring-search "$HSS" || true

log "done — run scripts/devcontainer_doctor.sh to verify"
