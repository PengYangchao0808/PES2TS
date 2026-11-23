"""Loss-component plugin registry.

A *loss component* is one named, independently-weighted training objective.  The
registry is the only place names in ``loss.components`` (experiment YAML) are
resolved; a component that has no implementation is registered with
``implemented=False`` and raises :class:`LossNotImplementedError` when used, so a
config can never silently train on a fabricated objective.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Protocol, runtime_checkable

import torch


class LossNotImplementedError(NotImplementedError):
    """Typed failure for a registered-but-unimplemented loss component."""

    def __init__(self, code: str, message: str):
        super().__init__(f"[{code}] {message}")
        self.code = str(code)
        self.message = str(message)


@runtime_checkable
class LossTerm(Protocol):
    """One loss term: ``(model_out, batch) -> scalar tensor``."""

    name: str
    #: ``False`` = registered for documentation only; calling it raises.
    implemented: bool

    def __call__(self, out: dict, batch: dict) -> torch.Tensor:  # pragma: no cover
        ...


@dataclass(frozen=True)
class LossSpec:
    name: str
    cls: type
    description: str
    implemented: bool


LOSS_REGISTRY: dict[str, LossSpec] = {}


def register_loss(name: str, *, description: str = "", implemented: bool = True):
    """Register a loss-term class under ``name`` (duplicates fail)."""
    key = str(name).strip()
    if not key:
        raise ValueError("loss name must be non-empty")

    def _decorator(cls: type) -> type:
        if key in LOSS_REGISTRY:
            raise ValueError(
                f"loss {key!r} already registered as {LOSS_REGISTRY[key].cls.__name__}")
        LOSS_REGISTRY[key] = LossSpec(name=key, cls=cls,
                                      description=str(description),
                                      implemented=bool(implemented))
        cls.name = key
        cls.implemented = bool(implemented)
        return cls

    return _decorator


def build_loss(name: str, **kwargs: Any) -> LossTerm:
    """Instantiate a registered loss component (raises ``KeyError`` if unknown)."""
    key = str(name).strip()
    if key not in LOSS_REGISTRY:
        raise KeyError(f"unknown loss component {name!r}; "
                       f"registered={sorted(LOSS_REGISTRY)}")
    return LOSS_REGISTRY[key].cls(**kwargs)


def list_losses(*, implemented_only: bool = False) -> list[str]:
    return sorted(name for name, spec in LOSS_REGISTRY.items()
                  if spec.implemented or not implemented_only)


def describe_losses() -> list[dict]:
    return [{
        "name": spec.name,
        "class": spec.cls.__name__,
        "implemented": spec.implemented,
        "description": spec.description,
    } for spec in sorted(LOSS_REGISTRY.values(), key=lambda s: s.name)]


def build_components(components: Iterable[tuple[str, float]] | None = None,
                     ) -> list[tuple[LossTerm, float]]:
    """Build ``[(term, weight)]`` from ``[(name, weight)]`` pairs."""
    out: list[tuple[LossTerm, float]] = []
    for name, weight in (components or ()):
        weight = float(weight)
        if weight < 0.0:
            raise ValueError(f"loss weight for {name!r} must be >= 0, got {weight}")
        out.append((build_loss(name), weight))
    return out


__all__ = [
    "LOSS_REGISTRY",
    "LossNotImplementedError",
    "LossSpec",
    "LossTerm",
    "build_components",
    "build_loss",
    "describe_losses",
    "list_losses",
    "register_loss",
]
