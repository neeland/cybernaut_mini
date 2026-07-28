#!/usr/bin/env bash
# Launch Kedro-Viz (interactive pipeline visualisation).
# Binds to 0.0.0.0 so it's reachable from the devcontainer host.
set -euo pipefail
cd "$(dirname "$0")/../.."
unset VIRTUAL_ENV  # ignore any stale activated venv; uv targets ./.venv
exec uv run kedro viz run --host 0.0.0.0 --port "${PORT:-4141}" --no-browser "$@"
