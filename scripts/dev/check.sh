#!/usr/bin/env bash
# Lint, format-check, and test — the same gate you'd run before committing.
set -euo pipefail
cd "$(dirname "$0")/../.."
unset VIRTUAL_ENV  # ignore any stale activated venv; uv targets ./.venv

echo "==> ruff lint"
uv run ruff check .

echo "==> ruff format (check only)"
uv run ruff format --check .

echo "==> pytest"
uv run pytest

echo "All checks passed."
