#!/usr/bin/env bash
#
# Native macOS (Apple silicon) setup — the default development environment.
#
#   scripts/setup_mac.sh            install and verify
#   scripts/setup_mac.sh --check    verify only, change nothing
#
# The devcontainer still works and is kept for a possible move to AWS, but it
# cannot reach Metal, the Neural Engine, or Accelerate: Docker Desktop's VM is
# native arm64 Linux, which is not the same thing as Apple silicon. Running here
# is what makes those paths available at all.
#
# What this gets you that the container cannot:
#   * numpy/scipy linked against Accelerate rather than OpenBLAS
#   * torch with the MPS backend (embedding.device=auto resolves to 'mps')
#   * thread counts tuned to performance cores instead of all 8 logical cores
#
# What it does NOT fix: the build is dominated by joblib process-pool overhead
# and Kedro MemoryDataset deepcopy, neither of which is hardware-bound. Expect a
# better development experience here, not a dramatically faster build.

set -euo pipefail

CHECK_ONLY=0
[[ "${1:-}" == "--check" ]] && CHECK_ONLY=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

info() { printf '\033[36m==>\033[0m %s\n' "$1"; }
warn() { printf '\033[33mwarning:\033[0m %s\n' "$1" >&2; }
die() {
  printf '\033[31merror:\033[0m %s\n' "$1" >&2
  exit 1
}

# ------------------------------------------------------------------ #
# Platform                                                            #
# ------------------------------------------------------------------ #

[[ "$(uname -s)" == "Darwin" ]] || die "not macOS (uname -s = $(uname -s)).
This script is the native-Mac path. Inside the devcontainer use 'make install'."

if [[ "$(uname -m)" != "arm64" ]]; then
  warn "uname -m = $(uname -m), expected arm64.
On an Intel Mac there is no MPS and no AMX; everything below still installs,
but embedding.device will resolve to 'cpu'."
fi

info "macOS $(sw_vers -productVersion 2>/dev/null || echo '?') on $(uname -m)"

# ------------------------------------------------------------------ #
# Toolchain                                                           #
# ------------------------------------------------------------------ #

command -v uv >/dev/null 2>&1 || die "uv not found.
  install: curl -LsSf https://astral.sh/uv/install.sh | sh
  or     : brew install uv"

info "uv $(uv --version | awk '{print $2}')"

if ((CHECK_ONLY == 0)); then
  # 'st' brings sentence-transformers; 'mps' adds the macOS arm64 torch wheel
  # carrying the Metal backend. 'hf' is corpus acquisition from the Hub. The
  # marker on 'mps' means this same command is harmless on Linux.
  info "syncing dependencies (st + mps + hf)"
  uv sync --extra st --extra mps --extra hf
fi

# ------------------------------------------------------------------ #
# Verify what we actually got                                         #
# ------------------------------------------------------------------ #

info "hardware acceleration report"
uv run python -c "from cybernaut_mini.accel import describe; print(describe().render())"

info "checking for the two things that silently go wrong"

uv run python - <<'PY'
import sys
from cybernaut_mini.accel import describe

report = describe()
problems = []

# 1. numpy on OpenBLAS means the Accelerate/AMX path is not in play. The wheel
#    should link Accelerate on macOS arm64 for numpy>=2.0 / scipy>=1.14.
if report.apple_silicon and "openblas" in report.blas.lower():
    problems.append(
        "numpy linked OpenBLAS, not Accelerate. Reinstall numpy>=2.0 from a\n"
        "    wheel rather than a source build: uv pip install --reinstall numpy"
    )

# 2. torch present but no MPS means the wrong wheel — usually a CPU-only or an
#    x86 build running under Rosetta.
if report.apple_silicon and report.torch_version and report.device != "mps":
    problems.append(
        f"torch {report.torch_version} resolved device {report.device!r}, not 'mps'.\n"
        "    Check you are not running an x86 Python under Rosetta:\n"
        "      python -c 'import platform; print(platform.machine())'  # want arm64"
    )

if problems:
    print()
    for p in problems:
        print(f"  [!] {p}")
    sys.exit(1)
print("  ok")
PY

# ------------------------------------------------------------------ #
# Environment                                                         #
# ------------------------------------------------------------------ #

info "thread tuning for this machine"
uv run python - <<'PY'
from cybernaut_mini.accel import tuning_env
for key, value in tuning_env().items():
    print(f"  export {key}={value}")
PY

cat <<'EOF'

These must be set BEFORE python starts — BLAS libraries read them once at load
time, so exporting them from inside the process has no effect. Either add them
to your shell profile, or use the Makefile targets, which do it for you.

Next:
  make check          lint + typecheck + tests
  make accel          re-print the report above
  make build-sample   a 63-document index, ~seconds

EOF
