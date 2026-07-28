#!/usr/bin/env bash
# Architecture selector — switches the active conf/base/parameters.yml symlink
# between the M1 (ARM64) and x86_64 parameter sets. The two source files differ
# only in LightGBM thread settings (model.lgbm_fixed); everything else is shared.
#
#   ./scripts/dev/select-arch.sh m1     # -> parameters_m1.yml  (8 threads)
#   ./scripts/dev/select-arch.sh x86    # -> parameters_x86.yml (4 threads)
#   ./scripts/dev/select-arch.sh auto   # detect from `uname -m`
#   ./scripts/dev/select-arch.sh status # show current
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE="$ROOT/conf/base"

host_arch() {
  case "$(uname -m)" in
    arm64|aarch64) echo m1 ;;
    x86_64|amd64)  echo x86 ;;
    *)             echo unknown ;;
  esac
}

link() {
  local target="parameters_$1.yml"
  if [ ! -f "$BASE/$target" ]; then
    echo "error: $BASE/$target not found" >&2; exit 1
  fi
  ( cd "$BASE" && ln -sf "$target" parameters.yml )
  echo "✓ active parameters: $target"
}

case "${1:-status}" in
  m1|arm64|arm) link m1 ;;
  x86|x86_64|intel) link x86 ;;
  auto) a=$(host_arch); [ "$a" = unknown ] && { echo "cannot detect arch"; exit 1; }; link "$a" ;;
  status)
    if [ -L "$BASE/parameters.yml" ]; then
      echo "active parameters: $(readlink "$BASE/parameters.yml")"
    else
      echo "parameters.yml is not a symlink (or missing)"
    fi
    echo "host arch: $(host_arch)"
    ;;
  *) echo "usage: $0 [m1|x86|auto|status]"; exit 1 ;;
esac
