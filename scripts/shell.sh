#!/usr/bin/env bash
# Open a shell inside this project's running devcontainer (run from the HOST).
#
# Auto-detects the container by the devcontainer label that pins it to this repo
# folder, so it keeps working when Docker assigns a new random name. Overrides:
#   CONTAINER=<name|id>   target a specific container
#   USER_IN=<user>        user to exec as (default: vscode)
#   WORKDIR=<path>        working directory (default: /workspaces/<repo>)
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
NAME="$(basename "$REPO")"
USER_IN="${USER_IN:-vscode}"
WORKDIR="${WORKDIR:-/workspaces/$NAME}"

command -v docker >/dev/null 2>&1 || { echo "docker not found on PATH" >&2; exit 1; }

# Prefer the exact devcontainer label (set by the Dev Containers tooling to the
# host workspace folder); fall back to a container mounting /workspaces/<repo>.
CONTAINER="${CONTAINER:-$(docker ps --filter "label=devcontainer.local_folder=$REPO" --format '{{.ID}}' | head -1)}"
if [ -z "$CONTAINER" ]; then
  CONTAINER="$(docker ps --filter "volume=$WORKDIR" --format '{{.ID}}' | head -1)"
fi

if [ -z "$CONTAINER" ]; then
  echo "No running devcontainer found for $REPO." >&2
  echo "Start it (VS Code: 'Dev Containers: Reopen in Container'), or pass CONTAINER=<name>." >&2
  echo "Running containers:" >&2
  docker ps --format '  {{.Names}}  ({{.Image}})' >&2
  exit 1
fi

# Inject the repo .env at exec time so secrets written AFTER the container was
# created (e.g. CLAUDE_CODE_OAUTH_TOKEN from setup-oauth-token.sh) reach this
# session without a rebuild. docker run --env-file only applies at create time.
if [ -f "$REPO/.env" ]; then
  set -- --env-file "$REPO/.env"
else
  set --
fi

# Prefer zsh inside the container, fall back to bash. docker exec does not set
# $SHELL, and tools like installers use it to pick shell completions, so export
# it explicitly before exec'ing the shell.
exec docker exec -it -u "$USER_IN" -w "$WORKDIR" "$@" "$CONTAINER" \
  bash -lc 'if command -v zsh >/dev/null 2>&1; then SHELL="$(command -v zsh)"; else SHELL="$(command -v bash)"; fi; export SHELL; exec "$SHELL"'
