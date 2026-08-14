"""Hardware acceleration: what this machine offers, and what to ask it for.

The repo's compute is spread across four layers, and only one of them is torch:

1. **BLAS** — MiniBatchKMeans, centroid cosine, and model2vec's pooling are all
   numpy calls that land in whatever BLAS numpy was linked against. On macOS
   arm64 that is Apple's Accelerate; on Linux it is OpenBLAS. This is the layer
   that matters for the *default* install, because the default embedder
   (model2vec) never touches torch at all.
2. **torch** — only the ``sentence_transformers`` provider. MPS on Apple
   silicon, CUDA on a GPU box, CPU everywhere else.
3. **Native extensions** — usearch, rbloom, zstandard, ahocorasick-rs,
   PyStemmer and tokenizers ship compiled ARM code and need no configuration.
   They are listed in :data:`NATIVE_EXTENSIONS` so a survey can tell them apart
   from the pure-Python dependencies, which no accelerator can help.
4. **Pure Python** — rank-bm25 and networkx. Named in :data:`PURE_PYTHON_HOT`
   because they are the parts a faster chip does *not* fix, and knowing that is
   the point of this module.

Nothing here changes results on its own. :func:`resolve_device` answers "which
torch device", :func:`tuning_env` answers "which thread counts", and
:func:`describe` answers "what did I actually get", which is the one worth
printing before a long build.

Determinism
-----------
Byte-identical rebuilds are a **per-device** guarantee, not a cross-device one.
The same corpus embedded on MPS and on CPU will not produce bit-identical
vectors: different kernels accumulate in different orders, and no flag makes a
GPU reduction match a CPU one bit for bit. :func:`device_fingerprint` renders the
chosen device as a short string so a caller can record it alongside an index;
comparing two indexes byte-for-byte is only meaningful when those fingerprints
agree. Wiring it into ``IndexMeta`` is deliberately left to the caller — this
module has no opinion about the artifact schema.

The ``hash`` provider is exempt: it is pure integer/numpy code with no BLAS
reduction wide enough to reorder, which is why it stays the offline
reproducibility anchor.

Blog ref: no paragraph of https://nosible.com/blog/the-road-to-cybernaut-1
    corresponds to this module, and that is the honest reason it exists. The post
    describes a federation of 250,000 shards served from NOSIBLE's own
    infrastructure and never discusses the reader's hardware. This replica runs
    on one laptop, so the question the post never has to answer — "what can this
    machine actually do, and which of it is reachable from here?" — has to be
    answered somewhere. Local copy: ``docs/blog-archive/the-road-to-cybernaut-1.md``.

Assumptions:
    - ``sys.platform == "darwin"`` plus an arm64 machine is the only signal that
      Metal and the Neural Engine are reachable. Checking ``platform.machine()``
      alone is wrong and actively harmful: this repo's own devcontainer reports
      ``aarch64`` on Linux, so an arch-only check would select MPS and tune for
      performance cores inside a VM that has neither.
    - Thread count on Apple silicon is capped at 4 rather than at
      ``os.cpu_count()``. The base M-series parts pair 4 performance cores with 4
      efficiency cores, ``cpu_count`` counts both, and a BLAS that spreads one
      GEMM across the two kinds waits on the slowest thread. Four is the
      performance-core count on every M-series part from the M1 up; a machine
      with more will be left some headroom rather than mistuned.
    - Environment variables are returned, not set, by :func:`tuning_env`. BLAS and
      OpenMP read them when the shared library loads, so a caller that has already
      imported numpy cannot benefit — the decision about *when* to apply them is
      not this module's to make. :func:`apply_tuning` exists for the callers that
      can still act on it, and never overrides a value an operator exported.
    - Device detection may import torch, so every import of it is deferred inside
      a function. The default install has no torch at all, and asking which device
      to use must not be the thing that requires a 500 MB dependency.

Alternatives considered:
    - Detecting the accelerator once at import and caching it in a module global:
      one probe instead of many, but it makes the answer depend on import order
      and leaves no way to ask "what would 'cpu' resolve to?" in a test. Every
      function here is pure with respect to its arguments instead, which is what
      lets the Apple-silicon branches be tested from Linux.
    - Raising when an explicitly requested device is unavailable. Rejected because
      one config file is meant to travel between a Mac and a Linux box: degrading
      to CPU with the reason recorded in :func:`describe` keeps a build working
      where raising would fail it for a machine difference the config cannot know.
    - Reporting the specific chip (``M1 Pro``, ``M2 Max``) in the fingerprint.
      More informative, but wrong for its purpose: two CPUs of the same
      architecture and BLAS produce identical bytes, so encoding the model number
      would make comparable indexes look incomparable.
    - Wrapping MLX or Core ML/ANE here as well. Both are real Apple-only paths and
      the ANE is the strongest one for a BERT-class encoder, but each needs a model
      conversion step rather than a device string, so neither fits behind this
      interface. Left out until the embedding step is worth optimising — measured
      at 2.6s of a ~135s build, it is not yet.
"""

from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass, field
from typing import Literal

DeviceName = Literal["auto", "mps", "cuda", "cpu"]

#: Dependencies shipping compiled code. Already native on arm64; no tuning knobs.
NATIVE_EXTENSIONS = (
    "numpy",
    "scipy",
    "sklearn",
    "usearch",
    "rbloom",
    "zstandard",
    "ahocorasick_rs",
    "Stemmer",
    "tokenizers",
)

#: Interpreted hot paths. A faster chip does not help these; only different code
#: does. ``rank_bm25`` builds one ``BM25Okapi`` per shard and ``networkx`` builds
#: one term graph per shard, so both scale with shard count, not just corpus size.
PURE_PYTHON_HOT = ("rank_bm25", "networkx", "jieba", "pysbd")


def is_apple_silicon() -> bool:
    """True on an arm64 macOS host — *not* in a Linux VM on that same Mac.

    Docker Desktop's VM reports ``Linux``/``aarch64``: native ARM, but with no
    Metal, no Neural Engine and no Accelerate. The platform check is therefore
    on the OS as well as the machine, because arm64 alone does not imply any of
    the Apple-specific paths are reachable.
    """
    return sys.platform == "darwin" and platform.machine() in {"arm64", "aarch64"}


def resolve_device(requested: DeviceName = "auto") -> str:
    """Resolve a configured device name to a concrete torch device string.

    ``auto`` prefers MPS on Apple silicon, then CUDA, then CPU. An explicit
    request is honoured when the backend is actually usable and otherwise falls
    back to CPU with the reason attached to :func:`describe`, because a build
    that quietly runs on the wrong device is worse than one that says so.

    Importing torch is deferred: the default install has no torch, and asking
    for the device must not be what pulls in a 500 MB dependency.
    """
    if requested == "cpu":
        return "cpu"

    try:
        import torch
    except ImportError:
        return "cpu"

    if requested in {"auto", "mps"} and torch.backends.mps.is_available():
        return "mps"
    if requested == "mps":
        return "cpu"  # asked for, not available
    if requested in {"auto", "cuda"} and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def device_fingerprint(device: str) -> str:
    """The string an index records to say what produced its vectors.

    Deliberately coarse — ``mps``, ``cuda``, ``cpu-darwin-arm64`` — rather than a
    full hardware id. Two CPUs of the same architecture and BLAS produce the same
    bytes, so encoding the specific chip would make indexes look incomparable
    when they are not.
    """
    if device in {"mps", "cuda"}:
        return device
    return f"cpu-{sys.platform}-{platform.machine()}"


def tuning_env(*, cores: int | None = None) -> dict[str, str]:
    """Thread-count and fallback settings to export before a build.

    Returned rather than applied, so a caller decides whether to set them; BLAS
    libraries read most of these once at load time, so they must be in the
    environment before numpy is imported to have any effect.

    On Apple silicon the performance and efficiency cores are not
    interchangeable for this workload. ``os.cpu_count()`` counts both, and
    letting a BLAS spread a GEMM across efficiency cores makes the whole call
    wait on the slowest thread. M-series parts from the M1 onward have 4
    performance cores in the base configuration, so the thread count is capped
    there rather than at the total.
    """
    env: dict[str, str] = {}
    total = cores if cores is not None else (os.cpu_count() or 1)

    if is_apple_silicon():
        threads = max(1, min(4, total))
        # Accelerate's own knob. OPENBLAS_NUM_THREADS is meaningless here
        # because the macOS arm64 numpy wheels do not link OpenBLAS.
        env["VECLIB_MAXIMUM_THREADS"] = str(threads)
        # Not every op has an MPS kernel. Without this a missing one raises
        # instead of running on the CPU, which fails a build for a single layer.
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    else:
        threads = max(1, total)
        env["OPENBLAS_NUM_THREADS"] = str(threads)

    env["OMP_NUM_THREADS"] = str(threads)
    env["TOKENIZERS_PARALLELISM"] = "false"  # silences the fork warning
    return env


def apply_tuning(env: dict[str, str] | None = None) -> dict[str, str]:
    """Set the tuning variables that are not already set, and return the result.

    Existing values always win: an operator who exported ``OMP_NUM_THREADS``
    meant it, and silently overriding it would make a benchmark unreproducible
    for reasons invisible in the command line.
    """
    chosen = env if env is not None else tuning_env()
    applied: dict[str, str] = {}
    for key, value in chosen.items():
        if os.environ.get(key):
            applied[key] = os.environ[key]
            continue
        os.environ[key] = value
        applied[key] = value
    return applied


@dataclass(frozen=True)
class AccelReport:
    """What this machine actually offers, resolved rather than assumed."""

    platform: str
    machine: str
    apple_silicon: bool
    cpu_count: int
    blas: str
    simd: str
    torch_version: str | None
    torch_backend: str
    device: str
    fingerprint: str
    notes: tuple[str, ...] = field(default=())

    def render(self) -> str:
        lines = [
            f"platform       {self.platform} / {self.machine}"
            + ("  (Apple silicon)" if self.apple_silicon else ""),
            f"cpu count      {self.cpu_count}",
            f"numpy BLAS     {self.blas}",
            f"numpy SIMD     {self.simd or 'unreported'}",
            f"torch          {self.torch_version or 'not installed'}",
            f"torch backend  {self.torch_backend}",
            f"device         {self.device}",
            f"fingerprint    {self.fingerprint}",
        ]
        lines.extend(f"note           {note}" for note in self.notes)
        return "\n".join(lines)


def _numpy_build() -> tuple[str, str]:
    """``(blas description, SIMD extensions)`` for the installed numpy.

    Read from ``numpy.__config__.CONFIG``, the dict numpy 2.x populates at build
    time. There is no public accessor — ``show_config()`` only prints — so the
    dict is read directly and every lookup is defensive: this is diagnostics, and
    a numpy that reorganises its config must not break a build.

    The BLAS name is the single most useful line in the report. On macOS arm64 it
    should read ``accelerate``; if it says ``openblas`` there, the wheel did not
    pick up Apple's BLAS and the AMX path is not in play.
    """
    try:
        import numpy as np
    except ImportError:
        return "numpy not installed", ""
    try:
        cfg = getattr(np.__config__, "CONFIG", {})
        blas = cfg.get("Build Dependencies", {}).get("blas", {}) or {}
        name = str(blas.get("name") or "unknown")
        version = str(blas.get("version") or "")
        simd = cfg.get("SIMD Extensions", {}) or {}
        found = simd.get("found") or []
        return f"{name} {version}".strip(), ",".join(str(x) for x in found)
    except Exception:
        return "unknown", ""


def describe(requested: DeviceName = "auto") -> AccelReport:
    """Resolve everything and explain the gaps, for printing before a long run."""
    notes: list[str] = []
    apple = is_apple_silicon()

    torch_version: str | None = None
    backend = "none (torch not installed)"
    try:
        import torch

        torch_version = torch.__version__
        if torch.backends.mps.is_available():
            backend = "mps"
        elif torch.cuda.is_available():
            backend = "cuda"
        else:
            backend = "cpu"
            if platform.machine() in {"arm64", "aarch64"} and sys.platform != "darwin":
                notes.append(
                    "arm64 Linux: no Metal, no Neural Engine. A '+cu' torch build "
                    "here carries CUDA kernels that cannot run."
                )
    except ImportError:
        notes.append(
            "torch absent — fine for the 'hash' and 'model2vec' providers, which "
            "never use it. Only 'sentence_transformers' needs it."
        )

    device = resolve_device(requested)
    if requested not in {"auto", device}:
        notes.append(f"requested device {requested!r} unavailable; using {device!r}")
    if apple and device == "cpu" and torch_version:
        notes.append("MPS unavailable despite Apple silicon — check the torch build")

    blas, simd = _numpy_build()
    if apple and "openblas" in blas.lower():
        notes.append(
            "numpy linked OpenBLAS on Apple silicon; the Accelerate wheel is the "
            "faster path for kmeans and cosine (numpy>=2.0, scipy>=1.14)"
        )

    return AccelReport(
        platform=sys.platform,
        machine=platform.machine(),
        apple_silicon=apple,
        cpu_count=os.cpu_count() or 1,
        blas=blas,
        simd=simd,
        torch_version=torch_version,
        torch_backend=backend,
        device=device,
        fingerprint=device_fingerprint(device),
        notes=tuple(notes),
    )
