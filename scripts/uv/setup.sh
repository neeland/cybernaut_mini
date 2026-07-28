#!/usr/bin/env bash
# Bootstrap the project from a clean checkout.
# Creates the uv-managed .venv, installs runtime + dev deps from uv.lock,
# and wires up pre-commit. Safe to re-run any time.
set -euo pipefail
cd "$(dirname "$0")/../.."
unset VIRTUAL_ENV  # ignore any stale activated venv; uv targets ./.venv

echo "==> Syncing environment from uv.lock (runtime + dev group)"
uv sync                       # reads pyproject.toml + uv.lock, builds/updates .venv

echo "==> Selecting arch-specific parameters (conf/base/parameters.yml symlink)"
./scripts/dev/select-arch.sh auto

echo "==> Installing pre-commit git hooks"
uv run pre-commit install || echo "   (skipped: no .pre-commit-config.yaml yet)"

echo "==> Kedro project info"
uv run kedro info

echo
echo "Done. The env lives in ./.venv  —  run commands with:  uv run <cmd>"
