"""Tests for hardware detection and device resolution.

The point of these tests is that the Apple-silicon path has to be verifiable from
a machine that is not a Mac. Every platform-dependent branch is therefore reached
by monkeypatching ``sys.platform``/``platform.machine`` and a stub torch module,
so CI on Linux exercises the same code an M-series Mac will run. What cannot be
faked — whether MPS actually produces correct vectors — is not asserted here.
"""

from __future__ import annotations

import dataclasses
import sys
import types

import pytest

from cybernaut_mini import accel


class _Backends:
    def __init__(self, mps_available: bool) -> None:
        self.mps = types.SimpleNamespace(
            is_available=lambda: mps_available,
            is_built=lambda: mps_available,
        )


def _fake_torch(
    *, mps: bool = False, cuda: bool = False, version: str = "2.13.0"
) -> types.ModuleType:
    mod = types.ModuleType("torch")
    mod.__version__ = version  # type: ignore[attr-defined]
    mod.backends = _Backends(mps)  # type: ignore[attr-defined]
    mod.cuda = types.SimpleNamespace(is_available=lambda: cuda)  # type: ignore[attr-defined]
    return mod


@pytest.fixture
def no_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``import torch`` raise, as on the default install."""
    monkeypatch.setitem(sys.modules, "torch", None)


def _install_torch(monkeypatch: pytest.MonkeyPatch, **kwargs: object) -> None:
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(**kwargs))  # type: ignore[arg-type]


def _apple(monkeypatch: pytest.MonkeyPatch, *, on: bool = True) -> None:
    monkeypatch.setattr(sys, "platform", "darwin" if on else "linux")
    monkeypatch.setattr(accel.platform, "machine", lambda: "arm64" if on else "x86_64")


# ------------------------------------------------------------------ #
# is_apple_silicon                                                    #
# ------------------------------------------------------------------ #


def test_apple_silicon_true_on_darwin_arm64(monkeypatch: pytest.MonkeyPatch) -> None:
    _apple(monkeypatch)
    assert accel.is_apple_silicon() is True


def test_apple_silicon_false_in_linux_vm_on_the_same_mac(monkeypatch: pytest.MonkeyPatch) -> None:
    """The container case: native aarch64, but no Metal, no ANE, no Accelerate.

    This is the branch that matters — arm64 alone must not imply Apple silicon,
    or a devcontainer on an M1 would try to select MPS and tune for P-cores.
    """
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(accel.platform, "machine", lambda: "aarch64")
    assert accel.is_apple_silicon() is False


def test_apple_silicon_false_on_intel_mac(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(accel.platform, "machine", lambda: "x86_64")
    assert accel.is_apple_silicon() is False


# ------------------------------------------------------------------ #
# resolve_device                                                      #
# ------------------------------------------------------------------ #


def test_cpu_request_never_imports_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """'cpu' short-circuits, so asking for it works with no torch installed."""
    monkeypatch.setitem(sys.modules, "torch", None)
    assert accel.resolve_device("cpu") == "cpu"


def test_auto_without_torch_is_cpu(no_torch: None) -> None:
    assert accel.resolve_device("auto") == "cpu"


def test_auto_prefers_mps_on_apple_silicon(monkeypatch: pytest.MonkeyPatch) -> None:
    _apple(monkeypatch)
    _install_torch(monkeypatch, mps=True)
    assert accel.resolve_device("auto") == "mps"


def test_auto_prefers_mps_over_cuda_when_both_present(monkeypatch: pytest.MonkeyPatch) -> None:
    _apple(monkeypatch)
    _install_torch(monkeypatch, mps=True, cuda=True)
    assert accel.resolve_device("auto") == "mps"


def test_auto_falls_to_cuda_without_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    _apple(monkeypatch, on=False)
    _install_torch(monkeypatch, mps=False, cuda=True)
    assert accel.resolve_device("auto") == "cuda"


def test_auto_falls_to_cpu_without_any_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_torch(monkeypatch, mps=False, cuda=False)
    assert accel.resolve_device("auto") == "cpu"


def test_explicit_mps_degrades_to_cpu_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A config written on a Mac must still build on a Linux box, not raise."""
    _install_torch(monkeypatch, mps=False, cuda=False)
    assert accel.resolve_device("mps") == "cpu"


def test_explicit_cuda_degrades_to_cpu_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_torch(monkeypatch, mps=False, cuda=False)
    assert accel.resolve_device("cuda") == "cpu"


# ------------------------------------------------------------------ #
# device_fingerprint                                                  #
# ------------------------------------------------------------------ #


def test_accelerator_fingerprints_are_bare_device_names() -> None:
    assert accel.device_fingerprint("mps") == "mps"
    assert accel.device_fingerprint("cuda") == "cuda"


def test_cpu_fingerprint_carries_platform_and_arch(monkeypatch: pytest.MonkeyPatch) -> None:
    _apple(monkeypatch)
    assert accel.device_fingerprint("cpu") == "cpu-darwin-arm64"


def test_mac_and_container_cpu_fingerprints_differ(monkeypatch: pytest.MonkeyPatch) -> None:
    """Byte-identity is per-device; the fingerprint has to be able to say so."""
    _apple(monkeypatch)
    mac = accel.device_fingerprint("cpu")
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(accel.platform, "machine", lambda: "aarch64")
    assert accel.device_fingerprint("cpu") != mac


# ------------------------------------------------------------------ #
# tuning_env                                                          #
# ------------------------------------------------------------------ #


def test_apple_caps_threads_at_performance_cores(monkeypatch: pytest.MonkeyPatch) -> None:
    """8 logical cores on an M1 is 4 P + 4 E; a GEMM waits on the slowest thread."""
    _apple(monkeypatch)
    env = accel.tuning_env(cores=8)
    assert env["OMP_NUM_THREADS"] == "4"
    assert env["VECLIB_MAXIMUM_THREADS"] == "4"


def test_apple_does_not_set_openblas_knob(monkeypatch: pytest.MonkeyPatch) -> None:
    """macOS arm64 numpy links Accelerate, so OPENBLAS_NUM_THREADS does nothing."""
    _apple(monkeypatch)
    assert "OPENBLAS_NUM_THREADS" not in accel.tuning_env(cores=8)


def test_apple_enables_mps_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without this a single unimplemented op fails a whole build."""
    _apple(monkeypatch)
    assert accel.tuning_env(cores=8)["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"


def test_non_apple_uses_all_cores_and_openblas(monkeypatch: pytest.MonkeyPatch) -> None:
    _apple(monkeypatch, on=False)
    env = accel.tuning_env(cores=8)
    assert env["OMP_NUM_THREADS"] == "8"
    assert env["OPENBLAS_NUM_THREADS"] == "8"
    assert "VECLIB_MAXIMUM_THREADS" not in env


def test_thread_count_never_drops_below_one(monkeypatch: pytest.MonkeyPatch) -> None:
    _apple(monkeypatch, on=False)
    assert accel.tuning_env(cores=0)["OMP_NUM_THREADS"] == "1"


def test_apple_thread_cap_does_not_exceed_available_cores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 2-core VM must not be told to use 4 threads."""
    _apple(monkeypatch)
    assert accel.tuning_env(cores=2)["OMP_NUM_THREADS"] == "2"


# ------------------------------------------------------------------ #
# apply_tuning                                                        #
# ------------------------------------------------------------------ #


def test_apply_tuning_respects_an_operator_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """An exported value was meant; overriding it hides why a benchmark moved."""
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    applied = accel.apply_tuning({"OMP_NUM_THREADS": "8"})
    assert applied["OMP_NUM_THREADS"] == "3"


def test_apply_tuning_sets_what_is_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TOKENIZERS_PARALLELISM", raising=False)
    applied = accel.apply_tuning({"TOKENIZERS_PARALLELISM": "false"})
    assert applied["TOKENIZERS_PARALLELISM"] == "false"


def test_apply_tuning_treats_empty_string_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "")
    assert accel.apply_tuning({"OMP_NUM_THREADS": "4"})["OMP_NUM_THREADS"] == "4"


# ------------------------------------------------------------------ #
# describe                                                            #
# ------------------------------------------------------------------ #


def test_describe_reports_no_torch_without_raising(no_torch: None) -> None:
    report = accel.describe()
    assert report.device == "cpu"
    assert report.torch_version is None
    assert any("torch absent" in note for note in report.notes)


def test_describe_flags_dead_cuda_build_on_arm64_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    """The exact situation in this repo's devcontainer: torch+cu on aarch64."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(accel.platform, "machine", lambda: "aarch64")
    _install_torch(monkeypatch, mps=False, cuda=False, version="2.13.0+cu130")
    report = accel.describe()
    assert report.device == "cpu"
    assert any("cannot run" in note for note in report.notes)


def test_describe_notes_unavailable_explicit_request(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_torch(monkeypatch, mps=False, cuda=False)
    report = accel.describe("mps")
    assert any("unavailable" in note for note in report.notes)


def test_describe_warns_when_apple_numpy_linked_openblas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On a Mac, OpenBLAS means the Accelerate/AMX path is not in play."""
    _apple(monkeypatch)
    monkeypatch.setattr(accel, "_numpy_build", lambda: ("openblas 0.3.33", "NEON"))
    _install_torch(monkeypatch, mps=True)
    assert any("Accelerate" in note for note in accel.describe().notes)


def test_describe_renders_every_field(monkeypatch: pytest.MonkeyPatch) -> None:
    _apple(monkeypatch)
    _install_torch(monkeypatch, mps=True)
    text = accel.describe().render()
    for label in ("platform", "numpy BLAS", "numpy SIMD", "torch", "device", "fingerprint"):
        assert label in text


def test_report_is_frozen() -> None:
    """A report describes one resolution; mutating it would misattribute a build."""
    report = accel.describe("cpu")
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.device = "mps"  # type: ignore[misc]
