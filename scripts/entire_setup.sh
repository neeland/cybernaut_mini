#!/usr/bin/env bash
# Install and fully configure the Entire CLI for this repository.
#
# Entire records agent sessions and links them to commits. This repo already ships
# its integration points — .entire/settings.json, five hooks in .git/hooks, and the
# Claude Code hooks in .claude/settings.json — but none of them do anything unless
# the `entire` binary is on PATH, which is why every commit printed:
#
#   [entire] Entire CLI is enabled but not installed or not on PATH.
#
# Called by .devcontainer/post-create.sh on container create, and safe to re-run:
#
#   scripts/entire_setup.sh              # install if missing, then (re)wire hooks
#   scripts/entire_setup.sh --upgrade    # reinstall the CLI even if present
#   scripts/entire_setup.sh --channel nightly
#   scripts/entire_setup.sh --no-doctor  # skip the stuck-session sweep
#
# Docs: https://docs.entire.io/installation
set -uo pipefail
log() { echo "[entire-setup] $*"; }

# The official installer always targets ~/.local/bin (no sudo). The devcontainer
# already puts that on PATH via remoteEnv, but a plain `bash script.sh` may not
# have it, so export it before any `command -v entire` check.
export PATH="$HOME/.local/bin:$PATH"

CHANNEL="stable"
UPGRADE=0
RUN_DOCTOR=1
AGENT="claude-code"

while [ $# -gt 0 ]; do
  case "$1" in
    --upgrade)   UPGRADE=1 ;;
    --no-doctor) RUN_DOCTOR=0 ;;
    --channel)   shift; CHANNEL="${1:-stable}" ;;
    --agent)     shift; AGENT="${1:-claude-code}" ;;
    -h|--help)   sed -n '2,18p' "$0"; exit 0 ;;
    *) log "unknown argument: $1 (try --help)"; exit 2 ;;
  esac
  shift
done

# ------------------------------------------------------------------ #
# 0. Always operate on this repo, and only on a real repo             #
# ------------------------------------------------------------------ #
# Matching the scripts/ convention, this cd's to its own repo root so it can be
# run from anywhere — the caller's cwd is never the target.
#
# That matters here beyond convenience: `entire enable` offers to *initialise* a
# git repo and create a GitHub remote when run outside one, and with -y it does
# so without asking. This script never passes -y, always lands in its own repo,
# and still hard-fails below if that somehow is not a git repo (e.g. the script
# was copied elsewhere).
cd "$(dirname "$0")/.." || exit 1
if ! git rev-parse --git-dir >/dev/null 2>&1; then
  log "ERROR: not inside a git repository — refusing to run"
  log "       (entire enable can create and push a new repo; that must be deliberate)"
  exit 1
fi
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT" || exit 1
log "repository: $REPO_ROOT"

# ------------------------------------------------------------------ #
# 1. The CLI                                                          #
# ------------------------------------------------------------------ #
if [ "$UPGRADE" -eq 1 ] || ! command -v entire >/dev/null 2>&1; then
  log "installing Entire CLI (channel: $CHANNEL)"
  # The installer verifies a published checksum before moving the binary
  # into place, and needs no sudo.
  curl -fsSL https://entire.io/install.sh | bash -s -- --channel "$CHANNEL" \
    || log "WARN: install script failed"
  export PATH="$HOME/.local/bin:$PATH"
else
  log "Entire CLI already present ($(command -v entire))"
fi

if ! command -v entire >/dev/null 2>&1; then
  log "ERROR: entire is still not on PATH after install"
  log "       expected at \$HOME/.local/bin/entire — check that dir is on PATH"
  exit 1
fi
log "version: $(entire version 2>/dev/null | head -1)"

# ------------------------------------------------------------------ #
# 2. Repository + agent hooks                                         #
# ------------------------------------------------------------------ #
# --agent puts enable into non-interactive mode (no TTY needed in post-create).
# --force reinstalls hooks: releases change the Claude Code tool matchers, and
# stale matchers silently stop firing — `entire doctor` reports exactly that.
# --no-init-repo is belt-and-braces with the git check above.
log "enabling Entire for agent '$AGENT' (reinstalling hooks)"
entire enable --force --agent "$AGENT" --no-init-repo \
  || log "WARN: entire enable failed — run it manually to see the prompt"

# ------------------------------------------------------------------ #
# 3. Stuck sessions                                                   #
# ------------------------------------------------------------------ #
# Plain `entire doctor` opens an interactive TUI and dies without a TTY
# ("could not open TTY"), so automation must pass --force, which condenses every
# fixable session instead of prompting per session.
if [ "$RUN_DOCTOR" -eq 1 ]; then
  log "sweeping stuck sessions (entire doctor --force)"
  entire doctor --force </dev/null 2>&1 | sed 's/^/[entire-setup]   /' \
    || log "NOTE: doctor reported issues it could not fix automatically"
fi

# ------------------------------------------------------------------ #
# 4. Report                                                           #
# ------------------------------------------------------------------ #
log "status:"
entire status 2>&1 | sed 's/^/[entire-setup]   /'

# An account is optional: hooks, checkpoints and local history all work logged
# out. Login only adds the hosted features (activity, search, dispatches).
# `auth status` exits 0 whether or not a session exists, so match on its output
# rather than its exit code.
if entire auth status 2>&1 | grep -qi "not logged in"; then
  log "not logged in — optional. Run 'entire login' for hosted features"
  log "(activity/search/dispatch); local session capture works without it."
fi

log "done — verify any time with: entire status  /  scripts/devcontainer_doctor.sh"
