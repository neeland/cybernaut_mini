#!/usr/bin/env bash
# README/diagram sync guard (pre-commit hook — see .pre-commit-config.yaml and
# src/numeria/pipelines/README.md). If a commit changes a pipeline's code but NOT that pipeline's
# README.md, the mermaid diagram has almost certainly gone stale — so we block
# the commit with a clear message.
#
# Escape hatch (use sparingly, with a reason):  README_SYNC_SKIP=1 git commit ...
set -euo pipefail

if [ "${README_SYNC_SKIP:-0}" = "1" ]; then
  echo "readme-sync: skipped (README_SYNC_SKIP=1)"; exit 0
fi

staged="$(git diff --cached --name-only)"
[ -z "$staged" ] && exit 0

fail=0
check_dir() {
  local dir="$1"
  local code_changed readme_changed
  code_changed="$(echo "$staged" | grep -E "^${dir}/.*\.(py)$" || true)"
  readme_changed="$(echo "$staged" | grep -E "^${dir}/README\.md$" || true)"
  if [ -n "$code_changed" ] && [ -z "$readme_changed" ]; then
    echo "✗ ${dir}: code changed but ${dir}/README.md was not updated."
    echo "    Update its mermaid diagram to match (see pipelines/README.md). Files:"
    echo "$code_changed" | sed 's/^/      - /'
    fail=1
  fi
}

# Each pipeline directory must keep its README in lock-step with its nodes.
for p in data_processing modeling backtest submission; do
  check_dir "src/numeria/pipelines/${p}"
done

if [ "$fail" = "1" ]; then
  echo
  echo "Commit blocked: update the README(s) above (or README_SYNC_SKIP=1 to override)."
  exit 1
fi
echo "readme-sync: ok"
