#!/usr/bin/env bash
#
# Run a command with this machine's thread tuning already in the environment.
#
#   scripts/with-accel.sh uv run cybernaut-mini build ...
#   scripts/with-accel.sh make test
#
# BLAS libraries (Accelerate, OpenBLAS) and OpenMP read their thread counts once,
# when the shared library loads. Setting them from inside a running Python process
# is therefore too late — os.environ changes nothing that has already initialised.
# This wrapper sets them in the parent shell and then execs, which is the only
# point at which they still have an effect.
#
# Values come from cybernaut_mini.accel.tuning_env(), so the policy lives in one
# place: 4 threads and VECLIB_MAXIMUM_THREADS on Apple silicon (performance cores
# only — a GEMM spread onto efficiency cores waits on the slowest thread), all
# cores and OPENBLAS_NUM_THREADS elsewhere. Anything already exported wins.

set -euo pipefail

[[ $# -gt 0 ]] || {
  printf 'usage: %s <command> [args...]\n' "$0" >&2
  exit 2
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Resolve the tuning with a bare interpreter. `uv run` would import the package
# under the *untuned* environment, and the point is to set the variables before
# anything that reads them is loaded. accel.py imports only the stdlib at module
# scope, so this works without the project's dependencies installed.
while IFS='=' read -r key value; do
  [[ -n "$key" ]] || continue
  # Existing values win: an operator who exported one meant it, and silently
  # overriding it makes a benchmark move for reasons invisible in the command.
  [[ -n "${!key:-}" ]] && continue
  export "$key=$value"
done < <(
  PYTHONPATH="$ROOT/src" python3 -c '
from cybernaut_mini.accel import tuning_env
for k, v in tuning_env().items():
    print(f"{k}={v}")
' 2>/dev/null || true
)

if [[ -n "${CYBERNAUT_ACCEL_VERBOSE:-}" ]]; then
  printf '\033[90m[accel]'
  for key in OMP_NUM_THREADS VECLIB_MAXIMUM_THREADS OPENBLAS_NUM_THREADS \
    PYTORCH_ENABLE_MPS_FALLBACK TOKENIZERS_PARALLELISM; do
    [[ -n "${!key:-}" ]] && printf ' %s=%s' "$key" "${!key}"
  done
  printf '\033[0m\n' >&2
fi

exec "$@"
