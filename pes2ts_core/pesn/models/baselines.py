"""Same-input baselines: constant, graph rule, and low-dim MLP (B3).

Every baseline consumes exactly the deployable input (current geometry +
task graph — no reference TS, no trajectory, no labels), so improvements of
the learned model over these numbers are honest same-input comparisons.  The
constant and rule baselines are pure numpy (backend-free); the MLP baseline
imports torch lazily and raises a typed error when the backend is absent.
"""

from __future__ import annotations

from typing import Final, Sequence

import numpy as np

from pes2ts_core.pesn.features.geometry import covalent_radius
from pes2ts_core.pesn.features.graph import (
    ROLE_BROKEN,
    ROLE_FORMED,
    ROLE_ORDER_CHANGED,
    GraphFeatures,
)
from pes2ts_core.pesn.models import require_torch

#: Distance window of the rule baseline: bonded anchor is the covalent-radius
#: sum, the far anchor adds this gap (angstrom); frozen for comparability.
RULE_FAR_GAP: Final[float] = 1.5


class BaselineError(RuntimeError):
    """Typed baseline failure."""


def _rule_progress_one(features: GraphFeatures) -> float:
    """Geometric-interpolation rule progress from driver distances alone."""
    z = features.atomic_numbers
    driver_edges = features.driver_edge_index
    progresses: list[float] = []
    for edge_slot in driver_edges:
        if features.edge_roles[edge_slot, ROLE_ORDER_CHANGED] > 0.5:
            continue
        broken = features.edge_roles[edge_slot, ROLE_BROKEN] > 0.5
        formed = features.edge_roles[edge_slot, ROLE_FORMED] > 0.5
        if not (broken or formed):
            continue
        src = int(features.edge_index[0, edge_slot])
        dst = int(features.edge_index[1, edge_slot])
        distance = float(features.edge_distance[edge_slot])
        bond_ref = covalent_radius(int(z[src])) + covalent_radius(int(z[dst]))
        far_ref = bond_ref + RULE_FAR_GAP
        clipped = float(np.clip((distance - bond_ref) / (far_ref - bond_ref), 0.0, 1.0))
        if broken:
            progresses.append(clipped)
        else:
            progresses.append(1.0 - clipped)
    if not progresses:
        return 0.0
    return 2.0 * float(np.mean(progresses)) - 1.0


class ConstantProgressBaseline:
    """Trivial localization baseline: a fixed signed-progress prediction."""

    def __init__(self, s_constant: float = 0.0) -> None:
        self.s_constant = float(s_constant)

    def predict_batch(self, features: Sequence[GraphFeatures]) -> dict[str, np.ndarray]:
        n = len(features)
        s_hat = np.full(n, self.s_constant, dtype=np.float64)
        return {
            "signed_progress": s_hat,
            "path_proximity": np.abs(s_hat),
            "path_score": s_hat.copy(),
        }


class RuleProgressBaseline:
    """Endpoint-task rule baseline on scaled driver coordinates."""

    def predict_batch(self, features: Sequence[GraphFeatures]) -> dict[str, np.ndarray]:
        s_hat = np.asarray(
            [_rule_progress_one(feature) for feature in features], dtype=np.float64
        )
        return {
            "signed_progress": s_hat,
            "path_proximity": np.abs(s_hat),
            "path_score": s_hat.copy(),
        }


class MlpProgressBaseline:
    """Same-input low-dim MLP over normalized pooled descriptors (torch)."""

    def __init__(self, *, n_features: int, hidden: int = 64) -> None:
        torch = require_torch()
        self.torch = torch
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_features, hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, 1),
            torch.nn.Tanh(),
        )

    def parameters(self):
        return self.net.parameters()

    def train_mode(self, flag: bool) -> None:
        self.net.train(flag)

    def predict_batch(self, features: Sequence[GraphFeatures]) -> dict[str, np.ndarray]:
        from pes2ts_core.pesn.features.graph import pooled_descriptors

        matrix = np.asarray([pooled_descriptors(f) for f in features])
        with self.torch.no_grad():
            s_hat = self.net(self.torch.as_tensor(matrix, dtype=self.torch.float32)).squeeze(-1)
        s_numpy = s_hat.numpy().astype(np.float64)
        return {
            "signed_progress": s_numpy,
            "path_proximity": np.abs(s_numpy),
            "path_score": s_numpy.copy(),
        }


__all__ = [
    "RULE_FAR_GAP",
    "BaselineError",
    "ConstantProgressBaseline",
    "MlpProgressBaseline",
    "RuleProgressBaseline",
]
