"""Named loss components wrapping the existing masked multi-task loss math.

Implemented terms (all reuse :mod:`pes2ts_v5.models.heads` — no new math):

===================  ============  ============================================
component            legacy key    meaning
===================  ============  ============================================
``move_huber``       ``move``      Smooth-L1 on reaction-coordinate displacement
``process_ce``       ``process``   cross-entropy on path-stage class
``progress_huber``   ``progress``  Smooth-L1 on path progress
``trust_bce``        ``trust``     BCE-with-logits on the offline trust proxy
===================  ============  ============================================

Registered but **not implemented**: ``physics_residual`` — this repository has no
physics-informed residual objective; using it raises
:class:`~pes2ts_v5.losses.registry.LossNotImplementedError` with
``code="LOSS_NOT_IMPLEMENTED"`` instead of training on a fabricated loss.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import torch
import torch.nn as nn

from pes2ts_v5.losses.registry import (
    LossNotImplementedError,
    LossTerm,
    build_components,
    build_loss,
    register_loss,
)
from pes2ts_v5.models.heads import (
    move_huber_loss,
    process_ce_loss,
    progress_huber_loss,
    trust_bce_loss,
)

#: component name -> legacy ``compute_losses`` dict key (backward compatibility)
LEGACY_KEYS: dict[str, str] = {
    "move_huber": "move",
    "process_ce": "process",
    "progress_huber": "progress",
    "trust_bce": "trust",
}


@register_loss("move_huber", description="masked Smooth-L1 on Move displacement")
class MoveHuber:
    """Smooth-L1 on the controlled reaction-coordinate displacement."""

    implemented = True

    def __call__(self, out: dict, batch: dict) -> torch.Tensor:
        return move_huber_loss(out, batch)


@register_loss("process_ce", description="masked cross-entropy on path-stage class")
class ProcessCE:
    """Cross-entropy on the PROCESS path-stage class (TS-neighbourhood etc.)."""

    implemented = True

    def __call__(self, out: dict, batch: dict) -> torch.Tensor:
        return process_ce_loss(out, batch)


@register_loss("progress_huber", description="masked Smooth-L1 on path progress")
class ProgressHuber:
    """Smooth-L1 on normalised path progress."""

    implemented = True

    def __call__(self, out: dict, batch: dict) -> torch.Tensor:
        return progress_huber_loss(out, batch)


@register_loss("trust_bce", description="masked BCE-with-logits on trust proxy")
class TrustBCE:
    """BCE-with-logits on the offline trust / risk-proxy label (not xTB success)."""

    implemented = True

    def __call__(self, out: dict, batch: dict) -> torch.Tensor:
        return trust_bce_loss(out, batch)


@register_loss(
    "physics_residual",
    description="physics-informed residual objective — NOT implemented in this repo",
    implemented=False)
class PhysicsResidual:
    """Documented-but-unimplemented component.

    There is no force/energy/physics-residual loss implementation in this
    repository, so this component exists only so an experiment YAML can *declare*
    the intent; calling it raises a typed :class:`LossNotImplementedError`.
    """

    implemented = False

    def __call__(self, out: dict, batch: dict) -> torch.Tensor:
        raise LossNotImplementedError(
            "LOSS_NOT_IMPLEMENTED",
            "physics_residual has no implementation in this repository "
            "(no force/energy/physics loss module exists); refusing to fabricate "
            "a training objective — remove it from loss.components or implement "
            "it explicitly")


class WeightedSumLoss(nn.Module):
    """Sum of weighted registered components, driven by ``LossConfig.components``.

    ``forward`` returns a dict with ``total`` plus one key per component; the
    legacy keys (``move``/``process``/``progress``/``trust``) are also populated
    so existing evaluation/code paths keep working.
    """

    def __init__(self, components: Iterable[tuple[str, float]] | None = None,
                 *, name: str = "weighted_sum"):
        super().__init__()
        pairs = build_components(components)
        if not pairs:
            raise ValueError("WeightedSumLoss needs at least one component")
        self.loss_name = str(name)
        self._terms: list[tuple[LossTerm, float]] = pairs
        self.term_names: list[str] = [term.name for term, _ in pairs]
        self.weights: dict[str, float] = {term.name: float(weight)
                                          for term, weight in pairs}

    def forward(self, out: dict, batch: dict) -> dict:
        parts: dict[str, Any] = {}
        total: torch.Tensor | None = None
        for term, weight in self._terms:
            value = term(out, batch)
            if not torch.is_tensor(value):
                raise TypeError(f"loss component {term.name!r} did not return a tensor")
            parts[term.name] = value
            legacy = LEGACY_KEYS.get(term.name)
            if legacy is not None:
                parts[legacy] = value
            total = value * float(weight) if total is None else total + value * float(weight)
        if total is None:  # pragma: no cover - guarded in __init__
            raise ValueError("WeightedSumLoss has no components")
        parts["total"] = total
        return parts

    def describe(self) -> list[dict]:
        return [{"name": term.name, "weight": float(weight),
                 "implemented": bool(getattr(term, "implemented", True))}
                for term, weight in self._terms]


def _component_pairs(loss_cfg: Any) -> list[tuple[str, float]]:
    """Accept a ``LossConfig``, a mapping, or an iterable of pairs."""
    if loss_cfg is None:
        return []
    components = getattr(loss_cfg, "components", None)
    if components is not None:
        return [(c.name, float(c.weight)) for c in components]
    if isinstance(loss_cfg, Mapping):
        components = loss_cfg.get("components", [])
        return [(c["name"], float(c["weight"])) for c in components]
    return [(name, float(weight)) for name, weight in loss_cfg]


def build_weighted_loss_from_config(loss_cfg: Any) -> WeightedSumLoss:
    """Build a :class:`WeightedSumLoss` from ``LossConfig`` / YAML-shaped mapping."""
    return WeightedSumLoss(_component_pairs(loss_cfg))


def legacy_loss_weights(loss_cfg: Any) -> dict[str, float]:
    """Map ``loss.components`` onto the legacy ``{move, process, progress, trust}``."""
    weights = {"move": 0.0, "process": 0.0, "progress": 0.0, "trust": 0.0}
    for name, weight in _component_pairs(loss_cfg):
        legacy = LEGACY_KEYS.get(name)
        if legacy is not None:
            weights[legacy] = float(weight)
    return weights


__all__ = [
    "LEGACY_KEYS",
    "MoveHuber",
    "PhysicsResidual",
    "ProcessCE",
    "ProgressHuber",
    "TrustBCE",
    "WeightedSumLoss",
    "build_weighted_loss_from_config",
    "legacy_loss_weights",
]


# explicit re-export for callers that resolve by name
_ = build_loss
