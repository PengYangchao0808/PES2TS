"""Task-graph feature construction: authority graph to numeric features (B3).

Edge roles (v1, frozen in the feature spec).  An edge carries a **multi-hot
role vector** because task identities overlap structurally: a ``broken``
edit pair is by definition also a reactant bond, an ``order_changed`` pair is
bonded on both sides.  Roles:

* ``ROLE_SPATIAL``  — geometry neighbourhood within the 5 A cutoff, excluding
  pairs already carrying a task identity; these encode local environment;
* ``ROLE_R_BOND`` / ``ROLE_P_BOND`` — endpoint-graph bonds (always present,
  never cut off: a bond remote in the *current* geometry stays visible);
* ``ROLE_FORMED`` / ``ROLE_BROKEN`` / ``ROLE_ORDER_CHANGED`` — the declared
  exclusive edits (always present regardless of distance), each with the
  endpoint bond orders preserved on the edge.

Driver-coordinate edges are identified by index into the edge list, in
``CoordinateSpec`` order, so the direction/activity heads read exactly the
declared coordinates.  The output is plain numpy; torch conversion happens
in the model layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from pes2ts_core.pesn.contracts.inputs import PESNInput
from pes2ts_core.pesn.features.geometry import (
    gaussian_distance_basis,
    pairwise_distances,
)

FEATURE_SPEC_VERSION: Final[str] = "pesn_feature_spec_v1.0"

ROLE_SPATIAL: Final[int] = 0
ROLE_R_BOND: Final[int] = 1
ROLE_P_BOND: Final[int] = 2
ROLE_FORMED: Final[int] = 3
ROLE_BROKEN: Final[int] = 4
ROLE_ORDER_CHANGED: Final[int] = 5
N_EDGE_ROLES: Final[int] = 6
ROLE_NAMES: Final[tuple[str, ...]] = (
    "spatial",
    "r_bond",
    "p_bond",
    "formed",
    "broken",
    "order_changed",
)

DEFAULT_SPATIAL_CUTOFF: Final[float] = 5.0
DEFAULT_N_BASIS: Final[int] = 16

#: Fields of PESNInput the feature layer is allowed to read.
ALLOWED_INPUT_FIELDS: Final[tuple[str, ...]] = (
    "atomic_numbers",
    "coordinates",
    "task_graph",
    "coordinate_spec",
    "units",
)


class FeatureBuildError(RuntimeError):
    """Typed failure of feature construction (never a silent fallback)."""


@dataclass(frozen=True, slots=True)
class GraphFeatures:
    """Numeric view of one qualified input; plain numpy arrays."""

    atomic_numbers: np.ndarray  # (N,) int
    edge_index: np.ndarray  # (2, E) int, both directions included
    edge_roles: np.ndarray  # (E, 6) multi-hot float64
    edge_r_order: np.ndarray  # (E,) reactant bond order (0 when absent)
    edge_p_order: np.ndarray  # (E,) product bond order (0 when absent)
    edge_distance: np.ndarray  # (E,) float64 angstrom
    edge_distance_basis: np.ndarray  # (E, n_basis) float64
    driver_edge_index: np.ndarray  # (C,) int index into edge list
    driver_q_scaled: np.ndarray  # (C,) float64 current q_tilde values
    n_atoms: int
    input_hash: str
    feature_spec_version: str = FEATURE_SPEC_VERSION

    @property
    def n_edges(self) -> int:
        return int(self.edge_index.shape[1])

    def has_role(self, role: int) -> np.ndarray:
        return self.edge_roles[:, role] > 0.5


def build_features(
    prediction_input: PESNInput,
    *,
    spatial_cutoff: float = DEFAULT_SPATIAL_CUTOFF,
    n_basis: int = DEFAULT_N_BASIS,
) -> GraphFeatures:
    """Convert one qualified input into numeric graph features."""
    task_graph = prediction_input.task_graph
    n_atoms = prediction_input.n_atoms
    if task_graph.n_atoms != n_atoms:
        raise FeatureBuildError(
            f"task graph has {task_graph.n_atoms} atoms, input has {n_atoms}"
        )
    coords = np.asarray(prediction_input.coordinates, dtype=np.float64)
    distances = pairwise_distances(coords)

    roles: dict[tuple[int, int], np.ndarray] = {}
    orders: dict[tuple[int, int], list[float]] = {}

    def _edge(pair: tuple[int, int]) -> tuple[np.ndarray, list[float]]:
        if pair not in roles:
            roles[pair] = np.zeros(N_EDGE_ROLES, dtype=np.float64)
            orders[pair] = [0.0, 0.0]
        return roles[pair], orders[pair]

    for term in task_graph.reactant_bonds:
        pair = (min(term.i, term.j), max(term.i, term.j))
        role, order = _edge(pair)
        role[ROLE_R_BOND] = 1.0
        order[0] = float(term.order)
    for term in task_graph.product_bonds:
        pair = (min(term.i, term.j), max(term.i, term.j))
        role, order = _edge(pair)
        role[ROLE_P_BOND] = 1.0
        order[1] = float(term.order)
    edit_role = {
        "formed": ROLE_FORMED,
        "broken": ROLE_BROKEN,
        "order_changed": ROLE_ORDER_CHANGED,
    }
    for edit in task_graph.edits:
        edge_role = edit_role.get(edit.kind)
        if edge_role is None:
            raise FeatureBuildError(f"unknown edit kind {edit.kind!r}")
        pair = (min(edit.i, edit.j), max(edit.i, edit.j))
        role, order = _edge(pair)
        role[edge_role] = 1.0
        order[0] = max(order[0], float(edit.reactant_order))
        order[1] = max(order[1], float(edit.product_order))

    driver_lookup: dict[tuple[int, int], int] = {}
    for position, coord in enumerate(prediction_input.coordinate_spec.coords):
        if coord.kind != "distance" or len(coord.atoms) != 2:
            raise FeatureBuildError(
                f"v1 features support distance drivers only, got {coord.kind}"
            )
        a, b = sorted(int(x) for x in coord.atoms)
        driver_lookup[(a, b)] = position

    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            if (i, j) in roles:
                continue
            if distances[i, j] < spatial_cutoff:
                role, _ = _edge((i, j))
                role[ROLE_SPATIAL] = 1.0

    pairs = sorted(roles)
    n_pairs = len(pairs)
    edge_index = np.empty((2, 2 * n_pairs), dtype=np.int64)
    edge_roles = np.empty((2 * n_pairs, N_EDGE_ROLES), dtype=np.float64)
    edge_r_order = np.empty(2 * n_pairs, dtype=np.float64)
    edge_p_order = np.empty(2 * n_pairs, dtype=np.float64)
    driver_edge_index = np.full(
        len(prediction_input.coordinate_spec.coords), -1, dtype=np.int64
    )
    for slot, pair in enumerate(pairs):
        role, order = roles[pair], orders[pair]
        for direction, (src, dst) in enumerate(((pair[0], pair[1]), (pair[1], pair[0]))):
            flat = 2 * slot + direction
            edge_index[0, flat] = src
            edge_index[1, flat] = dst
            edge_roles[flat] = role
            edge_r_order[flat] = order[0]
            edge_p_order[flat] = order[1]
        if pair in driver_lookup:
            driver_edge_index[driver_lookup[pair]] = 2 * slot

    if (driver_edge_index < 0).any():
        missing = int((driver_edge_index < 0).sum())
        raise FeatureBuildError(f"{missing} driver coordinates missing from edge set")

    edge_distance = distances[edge_index[0, :], edge_index[1, :]]
    basis = gaussian_distance_basis(edge_distance, cutoff=spatial_cutoff, n_basis=n_basis)
    driver_q = edge_distance[driver_edge_index]
    scales = np.array([c.scale_a for c in prediction_input.coordinate_spec.coords])
    refs = np.array([c.ref_q for c in prediction_input.coordinate_spec.coords])
    driver_q_scaled = (driver_q - refs) / scales

    return GraphFeatures(
        atomic_numbers=np.asarray(task_graph.atomic_numbers, dtype=np.int64),
        edge_index=edge_index,
        edge_roles=edge_roles,
        edge_r_order=edge_r_order,
        edge_p_order=edge_p_order,
        edge_distance=edge_distance,
        edge_distance_basis=basis,
        driver_edge_index=driver_edge_index,
        driver_q_scaled=driver_q_scaled,
        n_atoms=n_atoms,
        input_hash=prediction_input.input_hash,
    )


def swap_task_graph_features(features: GraphFeatures) -> GraphFeatures:
    """R/P-exchanged view for the symmetry loss: graph roles swap.

    Spatial edges and geometry are untouched; ``r_bond <-> p_bond``,
    ``formed <-> broken``, and the two bond-order columns swap — exactly the
    input transformation under which signed progress must flip sign.
    """
    swapped_roles = features.edge_roles[:, [ROLE_SPATIAL, ROLE_P_BOND, ROLE_R_BOND, ROLE_BROKEN, ROLE_FORMED, ROLE_ORDER_CHANGED]].copy()
    return GraphFeatures(
        atomic_numbers=features.atomic_numbers,
        edge_index=features.edge_index,
        edge_roles=swapped_roles,
        edge_r_order=features.edge_p_order,
        edge_p_order=features.edge_r_order,
        edge_distance=features.edge_distance,
        edge_distance_basis=features.edge_distance_basis,
        driver_edge_index=features.driver_edge_index,
        driver_q_scaled=features.driver_q_scaled,
        n_atoms=features.n_atoms,
        input_hash=features.input_hash,
    )


def pooled_descriptors(features: GraphFeatures) -> np.ndarray:
    """Low-dim descriptors for the same-input MLP baseline (no labels)."""
    z = features.atomic_numbers
    n_heavy = float(np.sum(z > 2))
    n_hydrogen = float(np.sum(z == 1))
    driver = features.edge_distance[features.driver_edge_index]
    r_mask = features.has_role(ROLE_R_BOND)
    p_mask = features.has_role(ROLE_P_BOND)
    return np.asarray(
        [
            float(features.n_atoms),
            n_heavy,
            n_hydrogen,
            float(len(features.driver_edge_index)),
            float(np.mean(driver)) if driver.size else 0.0,
            float(np.std(driver)) if driver.size else 0.0,
            float(np.min(driver)) if driver.size else 0.0,
            float(np.max(driver)) if driver.size else 0.0,
            float(np.mean(features.edge_distance[r_mask])) if r_mask.any() else 0.0,
            float(np.mean(features.edge_distance[p_mask])) if p_mask.any() else 0.0,
            float(np.mean(features.driver_q_scaled)) if driver.size else 0.0,
        ],
        dtype=np.float64,
    )


__all__ = [
    "ALLOWED_INPUT_FIELDS",
    "DEFAULT_N_BASIS",
    "DEFAULT_SPATIAL_CUTOFF",
    "FEATURE_SPEC_VERSION",
    "N_EDGE_ROLES",
    "ROLE_BROKEN",
    "ROLE_FORMED",
    "ROLE_NAMES",
    "ROLE_ORDER_CHANGED",
    "ROLE_P_BOND",
    "ROLE_R_BOND",
    "ROLE_SPATIAL",
    "FeatureBuildError",
    "GraphFeatures",
    "build_features",
    "pooled_descriptors",
    "swap_task_graph_features",
]
