"""Direct-displacement graph features (WP2, plan v3 §6).

Builds the model-side graph for one direct state: spatial edges (5 A
cutoff) with a Gaussian distance basis, task edges for every declared
edit (formed/broken/order_changed roles, never cut off), per-atom
normalization scales, and the invariant scalar context the policy/critics
consume. Numpy only; torch conversion happens in the loader.

Identity discipline (plan §3.2/§10): features never see source labels,
reference progress, future TS distance, or file paths — only the current
geometry, the task graph, the observation (delta E, gradient), and the
logical/event state.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Mapping

import numpy as np

FEATURE_SPEC_VERSION: Final[str] = "pesn_direct_feature_spec_v1.0"
SPATIAL_CUTOFF_ANGSTROM: Final[float] = 5.0
N_BASIS: Final[int] = 32

#: node roles for task edges: (formed, broken, order_changed)
TASK_EDGE_ROLES: Final[tuple[str, ...]] = ("formed", "broken", "order_changed")
N_TASK_ROLES: Final[int] = len(TASK_EDGE_ROLES)
_TASK_ROLE_INDEX: Final[dict[str, int]] = {r: i for i, r in enumerate(TASK_EDGE_ROLES)}

#: graph-level scalar context (names fixed; order is the packing order)
GRAPH_SCALAR_NAMES: Final[tuple[str, ...]] = (
    "delta_e_hartree",            # E(X) - E(reset)
    "grad_norm_hartree_angstrom", # ||g||_2
    "steps_remaining_frac",       # steps_remaining / budget
    "goal_hits_frac",             # consecutive goal hits / K
    "no_progress_frac",           # no-progress count / window
    "best_potential",             # Phi best so far in [-1, 0]
)
N_GRAPH_SCALARS: Final[int] = len(GRAPH_SCALAR_NAMES)


class DirectFeatureError(RuntimeError):
    """Typed feature-build refusal."""


def gaussian_distance_basis(distances: np.ndarray, *, cutoff: float = SPATIAL_CUTOFF_ANGSTROM,
                            n_basis: int = N_BASIS) -> np.ndarray:
    """Gaussian expansion with the plan §6 fixed centers and shared scale."""
    centers = np.linspace(0.0, cutoff, n_basis)
    scale = cutoff / n_basis
    d = np.asarray(distances, dtype=float)[..., None]
    return np.exp(-0.5 * ((d - centers) / scale) ** 2)


@dataclass(frozen=True, slots=True)
class DirectGraphFeatures:
    """One state's model graph (numpy; torch conversion in the loader)."""

    atomic_numbers: np.ndarray        # (N,)
    coordinates: np.ndarray           # (N,3) angstrom
    gradient: np.ndarray              # (N,3) hartree/angstrom
    edge_index: np.ndarray            # (2, E) both directions, task edges included
    edge_distance: np.ndarray         # (E,)
    edge_basis: np.ndarray            # (E, n_basis)
    edge_is_spatial: np.ndarray       # (E,) bool
    edge_task_roles: np.ndarray       # (E, N_TASK_ROLES) multi-hot
    node_active: np.ndarray           # (N,) bool (all True in v1)
    graph_scalars: np.ndarray         # (N_GRAPH_SCALARS,)
    n_atoms: int
    feature_spec_version: str = FEATURE_SPEC_VERSION


def build_direct_features(
    *,
    atomic_numbers: np.ndarray,
    coordinates: np.ndarray,
    gradient: np.ndarray,
    task_graph: Mapping,
    delta_e_hartree: float,
    steps_remaining: int,
    budget_steps: int,
    goal_consecutive_hits: float,
    goal_hits_required: int,
    no_progress_count: float,
    no_progress_window: int,
    best_potential: float,
    spatial_cutoff: float = SPATIAL_CUTOFF_ANGSTROM,
) -> DirectGraphFeatures:
    z = np.asarray(atomic_numbers, dtype=int)
    x = np.asarray(coordinates, dtype=float)
    g = np.asarray(gradient, dtype=float).reshape(x.shape)
    if x.shape[0] != z.size or x.ndim != 2 or x.shape[1] != 3:
        raise DirectFeatureError(f"coordinates {x.shape} incompatible with Z {z.shape}")

    n = z.size
    diff = x[:, None, :] - x[None, :, :]
    dist = np.sqrt((diff * diff).sum(-1))

    src_list, dst_list, is_spatial = [], [], []
    seen = np.zeros((n, n), dtype=bool)
    iu = np.triu_indices(n, k=1)
    within = dist[iu] < spatial_cutoff
    for i, j, ok in zip(iu[0], iu[1], within):
        if not ok:
            continue
        for a, b in ((i, j), (j, i)):
            src_list.append(a)
            dst_list.append(b)
            is_spatial.append(True)
            seen[a, b] = True

    edits = task_graph.get("edits", [])
    if not isinstance(edits, list):
        raise DirectFeatureError("task_graph.edits malformed")
    task_roles_by_pair: dict[tuple[int, int], np.ndarray] = {}
    for edit in edits:
        kind = str(edit.get("kind", "")).lower()
        role_idx = _TASK_ROLE_INDEX.get(kind)
        if role_idx is None:
            raise DirectFeatureError(f"unknown edit kind {kind!r}")
        i, j = int(edit["i"]), int(edit["j"])
        if not (0 <= i < n and 0 <= j < n) or i == j:
            raise DirectFeatureError(f"edit indices out of range: {i},{j}")
        for a, b in ((i, j), (j, i)):
            vec = task_roles_by_pair.get((a, b))
            if vec is None:
                vec = np.zeros(N_TASK_ROLES, dtype=np.float32)
                task_roles_by_pair[(a, b)] = vec
            vec[role_idx] = 1.0
            if not seen[a, b]:
                src_list.append(a)
                dst_list.append(b)
                is_spatial.append(False)
                seen[a, b] = True

    edge_index = np.asarray([src_list, dst_list], dtype=np.int64)
    edge_distance = dist[edge_index[0], edge_index[1]]
    edge_basis = gaussian_distance_basis(edge_distance, cutoff=spatial_cutoff)
    edge_task = np.zeros((len(src_list), N_TASK_ROLES), dtype=np.float32)
    for e in range(len(src_list)):
        vec = task_roles_by_pair.get((src_list[e], dst_list[e]))
        if vec is not None:
            edge_task[e] = vec

    graph_scalars = np.asarray([
        float(delta_e_hartree),
        float(np.linalg.norm(g)),
        float(steps_remaining) / float(max(budget_steps, 1)),
        float(goal_consecutive_hits) / float(max(goal_hits_required, 1)),
        float(no_progress_count) / float(max(no_progress_window, 1)),
        float(best_potential),
    ], dtype=np.float64)
    if not np.isfinite(graph_scalars).all():
        raise DirectFeatureError("non-finite graph scalar context")

    return DirectGraphFeatures(
        atomic_numbers=z,
        coordinates=x,
        gradient=g,
        edge_index=edge_index,
        edge_distance=edge_distance,
        edge_basis=edge_basis.astype(np.float32),
        edge_is_spatial=np.asarray(is_spatial, dtype=bool),
        edge_task_roles=edge_task,
        node_active=np.ones(n, dtype=bool),
        graph_scalars=graph_scalars,
        n_atoms=n,
    )


__all__ = [
    "FEATURE_SPEC_VERSION",
    "GRAPH_SCALARS",
    "GRAPH_SCALAR_NAMES",
    "N_BASIS",
    "N_GRAPH_SCALARS",
    "N_TASK_ROLES",
    "SPATIAL_CUTOFF_ANGSTROM",
    "DirectFeatureError",
    "DirectGraphFeatures",
    "build_direct_features",
    "gaussian_distance_basis",
]
