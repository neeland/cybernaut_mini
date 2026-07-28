#!/usr/bin/env bash
# container-deps guard (pre-commit hook — see .pre-commit-config.yaml).
#
# The data-download pipeline uses aria2c for multi-connection downloads when it
# is available (src/numeria/pipelines/data_processing/nodes.py). If the
# devcontainer image stops installing aria2, the container silently loses that
# fast path, so we block the commit until the Dockerfile is fixed.
#
# Escape hatch (use sparingly, with a reason):  CONTAINER_DEPS_SKIP=1 git commit ...
set -euo pipefail

if [ "${CONTAINER_DEPS_SKIP:-0}" = "1" ]; then
  echo "container-deps: skipped (CONTAINER_DEPS_SKIP=1)"; exit 0
fi

DOCKERFILE=".devcontainer/Dockerfile"
if [ ! -f "$DOCKERFILE" ]; then
  echo "container-deps: ${DOCKERFILE} not found"; exit 1
fi

fail=0
require() {
  local pkg="$1" why="$2"
  # -w matches the whole word, so 'aria2' won't match 'aria2c'/'libaria2'.
  if ! grep -qw "$pkg" "$DOCKERFILE"; then
    echo "✗ ${DOCKERFILE} does not install '${pkg}' (needed for: ${why})."
    fail=1
  fi
}

# Runtime tools the pipeline relies on must be baked into the devcontainer image.
require aria2 "multi-connection dataset downloads in data_processing/nodes.py"

if [ "$fail" = "1" ]; then
  echo
  echo "Commit blocked: add the package(s) above to ${DOCKERFILE}'s apt-get install"
  echo "(or CONTAINER_DEPS_SKIP=1 to override)."
  exit 1
fi
echo "container-deps: ok"
