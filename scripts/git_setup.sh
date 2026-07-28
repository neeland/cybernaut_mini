#!/usr/bin/env bash
# Installs the GitHub CLI (gh) and the Entire CLI, then walks through the
# interactive login for each. Safe to re-run: install and login steps are
# skipped when already done.

set -euo pipefail

info()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
ok()    { printf '\033[1;32m ok\033[0m %s\n' "$*"; }

if ! command -v brew >/dev/null 2>&1; then
    echo "Homebrew is required but not installed. Install it from https://brew.sh and re-run." >&2
    exit 1
fi

# --- GitHub CLI -------------------------------------------------------------

if command -v gh >/dev/null 2>&1; then
    ok "gh already installed ($(gh --version | head -1))"
else
    info "Installing GitHub CLI (gh)..."
    brew install gh
fi

if gh auth status >/dev/null 2>&1; then
    ok "gh already logged in"
else
    info "Logging in to GitHub — follow the prompts to finish authentication."
    gh auth login
fi

# --- Entire CLI -------------------------------------------------------------

if command -v entire >/dev/null 2>&1; then
    ok "entire already installed ($(entire --version 2>/dev/null | head -1))"
else
    info "Installing Entire CLI..."
    brew install entireio/tap/entire
fi

if entire auth status >/dev/null 2>&1; then
    ok "entire already logged in"
else
    info "Logging in to Entire — follow the prompts to finish authentication."
    entire login
fi

# Enable Entire session tracking in this repository.
repo_root="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
if [ -f "$repo_root/.entire/settings.json" ]; then
    ok "Entire already enabled in $repo_root"
else
    info "Enabling Entire in $repo_root..."
    (cd "$repo_root" && entire enable)
fi

ok "git setup complete"
