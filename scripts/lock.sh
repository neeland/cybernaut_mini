#!/usr/bin/env bash
# Re-resolve dependencies and refresh the lockfile, then export a plain
# requirements.txt for tools/CI that don't understand uv.lock.
# Run this whenever you change dependencies in pyproject.toml.
set -euo pipefail
cd "$(dirname "$0")/../.."
unset VIRTUAL_ENV  # ignore any stale activated venv; uv targets ./.venv

echo "==> Re-resolving and updating uv.lock"
uv lock

echo "==> Syncing .venv to match the new lock"
uv sync

echo "==> Exporting requirements.txt (runtime only, no hashes)"
uv export --no-dev --no-hashes --no-emit-project -o requirements.txt

echo "Done. uv.lock + requirements.txt are up to date."
