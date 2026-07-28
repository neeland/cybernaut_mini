#!/usr/bin/env bash
# Fetch Numerai's official example-scripts repo for REFERENCE only.
#
# This project does not track those scripts (they're ~83M and carry their own
# .git history, so `example-scripts/` is gitignored here). Run this whenever you
# want the upstream notebooks/models locally to learn from or copy patterns out
# of; nothing it produces is committed.
#
# Requires SSH access to GitHub (your ~/.ssh is mounted into the devcontainer).
set -euo pipefail
cd "$(dirname "$0")/../.."

REPO="git@github.com:numerai/example-scripts.git"
DEST="example-scripts"

if [ -d "$DEST/.git" ]; then
  echo "==> $DEST already present — pulling latest"
  git -C "$DEST" pull --ff-only
else
  echo "==> Cloning $REPO into ./$DEST (reference only, gitignored)"
  git clone "$REPO" "$DEST"
fi

echo
echo "Done. Reference material is in ./$DEST (not tracked by this repo)."
echo "Browse:  ls $DEST/{numerai,signals,crypto}"
