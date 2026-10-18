"""PES-N models: baselines, encoder, and research task heads.

``models`` belongs to the runtime-reachable set, so imports here stay
backend-light: the rule/constant baselines are pure numpy, and torch is
imported lazily by the modules that need it (encoder/heads/MLP baseline)
with typed unsupported errors when the optional backend is missing.
"""

from __future__ import annotations


class BackendMissingError(RuntimeError):
    """The optional ML backend is not installed (typed, never a crash)."""

    def __init__(self, backend: str = "torch", purpose: str = "model forward") -> None:
        super().__init__(
            f"optional backend {backend!r} is required for {purpose} but not "
            "installed; install requirements-pesn-runtime.txt"
        )


def require_torch():
    try:
        import torch  # noqa: F401 -- availability probe

        return torch
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise BackendMissingError("torch", "model forward") from exc


__all__ = ["BackendMissingError", "require_torch"]
