#!/usr/bin/env bash
# One-shot competition run: download real data, tune+train, predict live, submit.
# Usage:
#   ./scripts/compete.sh                # full run (uploads if NUMERAI_* keys set)
#   ./scripts/compete.sh --dry-run      # everything except the upload
#   ./scripts/compete.sh --submit-only  # reuse the trained model, just predict+submit
set -euo pipefail
cd "$(dirname "$0")/../.."

ARGS=()
for arg in "$@"; do
  case "$arg" in
    --dry-run)     ARGS+=(--params "submit.dry_run=true") ;;
    --submit-only) ARGS+=(--pipeline submission) ;;
    *)             ARGS+=("$arg") ;;
  esac
done

uv run kedro run "${ARGS[@]}"

echo
echo "── Submission receipt ─────────────────────────────"
cat data/08_reporting/submission_receipt.json 2>/dev/null || echo "(no receipt written)"
echo
