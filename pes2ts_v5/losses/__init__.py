"""Named, registered loss components for the V5.6 training engine.

``loss.components`` in an experiment YAML resolves here; the implemented terms
reuse the existing masked multi-task math in :mod:`pes2ts_v5.models.heads`.
"""

from pes2ts_v5.losses.components import (
    LEGACY_KEYS,
    MoveHuber,
    PhysicsResidual,
    ProcessCE,
    ProgressHuber,
    TrustBCE,
    WeightedSumLoss,
    build_weighted_loss_from_config,
    legacy_loss_weights,
)
from pes2ts_v5.losses.registry import (
    LOSS_REGISTRY,
    LossNotImplementedError,
    LossSpec,
    LossTerm,
    build_components,
    build_loss,
    describe_losses,
    list_losses,
    register_loss,
)

__all__ = [
    "LEGACY_KEYS",
    "LOSS_REGISTRY",
    "LossNotImplementedError",
    "LossSpec",
    "LossTerm",
    "MoveHuber",
    "PhysicsResidual",
    "ProcessCE",
    "ProgressHuber",
    "TrustBCE",
    "WeightedSumLoss",
    "build_components",
    "build_loss",
    "build_weighted_loss_from_config",
    "describe_losses",
    "legacy_loss_weights",
    "list_losses",
    "register_loss",
]
