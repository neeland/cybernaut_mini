#!/usr/bin/env bash
# Launch JupyterLab with the Kedro extension/context loaded.
# `kedro jupyter lab` injects `catalog`, `context`, `session`, `pipelines`.
set -euo pipefail
cd "$(dirname "$0")/../.."
unset VIRTUAL_ENV  # ignore any stale activated venv; uv targets ./.venv
exec uv run kedro jupyter lab --ip 0.0.0.0 --no-browser --port "${PORT:-8888}" "$@"
