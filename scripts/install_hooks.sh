#!/usr/bin/env bash
# Install the tracked git hooks into this clone's .git/hooks.
#
# .git/hooks is not version-controlled, so a fresh clone has no hooks until they
# are installed. Run this once after cloning; `make install` does it for you.
#
# Symlinks rather than copies, so editing scripts/git-hooks/* takes effect
# immediately with no re-install and no chance of the hook drifting from the
# tracked file.
set -euo pipefail

cd "$(dirname "$0")/.."
repo_root="$(git rev-parse --show-toplevel)"

# --git-common-dir, not --git-dir: in a linked worktree the hooks directory is
# shared with the main checkout, and installing into the per-worktree private
# dir would leave the hook firing in only one of them.
hooks_dir="$(git rev-parse --git-common-dir)/hooks"
mkdir -p "$hooks_dir"

installed=0
for hook in scripts/git-hooks/*; do
  [ -f "$hook" ] || continue
  name="$(basename "$hook")"

  # Relative target so the symlink survives a moved or differently-mounted repo.
  # .git/hooks/<name> -> ../../scripts/git-hooks/<name>
  target="$(python3 -c 'import os,sys; print(os.path.relpath(sys.argv[1], sys.argv[2]))' \
    "$repo_root/$hook" "$hooks_dir")"

  chmod +x "$hook"
  ln -sfn "$target" "$hooks_dir/$name"
  echo "[hooks] installed $name -> $target"
  installed=$((installed + 1))
done

if [ "$installed" -eq 0 ]; then
  echo "[hooks] no hooks found in scripts/git-hooks/" >&2
  exit 1
fi
