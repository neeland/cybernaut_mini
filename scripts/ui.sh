#!/usr/bin/env bash
# Launch the MLflow tracking UI against the SAME sqlite backend configured in
# conf/base/mlflow.yml. Must match, or the UI shows no runs.
set -euo pipefail
cd "$(dirname "$0")/../.."
unset VIRTUAL_ENV  # ignore any stale activated venv; uv targets ./.venv
exec uv run mlflow ui \
  --backend-store-uri "${MLFLOW_BACKEND:-sqlite:///mlflow.db}" \
  --host 0.0.0.0 --port "${PORT:-5000}" "$@"
