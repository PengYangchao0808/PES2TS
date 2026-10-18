"""B3 acceptance: invariances, remote task edges, and no-future-visibility."""

from __future__ import annotations

import numpy as np
import pytest

from pes2ts_core.pesn.contracts.inputs import (
    CoordinateDef,
    CoordinateSpec,
    TaskEdit,
    TaskGraph,
    BondTerm,
    build_pesn_input,
)
from pes2ts_core.pesn.features.graph import (
    ROLE_BROKEN,
    ROLE_FORMED,
    ROLE_R_BOND,
    build_features,
    pooled_descriptors,
    swap_task_graph_features,
)
from pes2ts_core.pesn.features.geometry import (
    evaluate_coordinate_spec,
    gaussian_distance_basis,
    pairwise_distances,
)


def _input(coords: np.ndarray):
    task_graph = TaskGraph(
        schema_version="pesn_task_graph_v1",
        atomic_numbers=(8, 1, 1, 8),
        atom_map=(1, 2, 3, 4),
        reactant_bonds=(BondTerm(0, 1, 1.0),),
        product_bonds=(BondTerm(2, 3, 1.0),),
        edits=(TaskEdit("broken", 0, 1, 1.0, 0.0), TaskEdit("formed", 2, 3, 0.0, 1.0)),
        source="test",
    )
    coord_spec = CoordinateSpec(
        schema_version="pesn_coordinate_spec_v1",
        coords=(
            CoordinateDef("distance", (0, 1), "angstrom", "driver", 1.0, 0.0, "AH"),
            CoordinateDef("distance", (2, 3), "angstrom", "driver", 1.0, 0.0, "HB"),
        ),
        spec_version="t",
    )
    return build_pesn_input(
        input_id="f", task_graph=task_graph, coordinate_spec=coord_spec, coordinates=coords
    )


def _rigid(coords: np.ndarray, angle: float, shift: np.ndarray) -> np.ndarray:
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    return coords @ rotation.T + shift


def test_translation_and_rotation_invariance() -> None:
    rng = np.random.default_rng(3)
    coords = rng.normal(scale=1.5, size=(4, 3))
    base = build_features(_input(coords))
    moved = build_features(_input(_rigid(coords, 1.1, np.asarray([4.0, -3.0, 9.0]))))
    np.testing.assert_allclose(base.driver_q_scaled, moved.driver_q_scaled, atol=1e-9)
    np.testing.assert_allclose(
        pooled_descriptors(base), pooled_descriptors(moved), atol=1e-9
    )
    order = np.argsort(base.edge_index[0] * 1000 + base.edge_index[1])
    order_moved = np.argsort(moved.edge_index[0] * 1000 + moved.edge_index[1])
    np.testing.assert_allclose(
        base.edge_distance[order], moved.edge_distance[order_moved], atol=1e-9
    )
    np.testing.assert_allclose(base.edge_roles[order], moved.edge_roles[order_moved])


def test_atom_permutation_consistency() -> None:
    coords = np.asarray(
        [[0.0, 0, 0], [1.0, 0, 0], [3.0, 0, 0], [4.2, 0, 0]], dtype=float
    )
    base = build_features(_input(coords))
    permutation = [3, 2, 1, 0]
    permuted = build_features(_input(coords[permutation]))
    # driver values survive as a set (coordinate order is spec-defined)
    assert sorted(base.driver_q_scaled.tolist()) == pytest.approx(
        sorted(permuted.driver_q_scaled.tolist())
    )
    assert base.n_edges == permuted.n_edges
    assert sorted(base.edge_roles.sum(axis=1).tolist()) == sorted(permuted.edge_roles.sum(axis=1).tolist())


def test_remote_task_edges_survive_cutoff() -> None:
    # The broken pair (0,1) itself sits 30 A apart in the current geometry.
    coords = np.asarray(
        [[0.0, 0, 0], [30.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]], dtype=float
    )
    features = build_features(_input(coords))
    roles = features.edge_roles
    assert (roles[:, ROLE_R_BOND] > 0).any(), "reactant bond edge must exist"
    assert (roles[:, ROLE_FORMED] > 0).any(), "formed task edge must exist"
    assert (roles[:, ROLE_BROKEN] > 0).any()
    long_task = (features.edge_distance > 10.0) & (roles[:, ROLE_BROKEN] > 0)
    assert long_task.any(), "task edges at 30 A separation are retained"


def test_swap_features_exchange_graph_roles_only() -> None:
    coords = np.asarray([[0.0, 0, 0], [1.0, 0, 0], [2.5, 0, 0], [3.5, 0, 0]], dtype=float)
    base = build_features(_input(coords))
    swapped = swap_task_graph_features(base)
    np.testing.assert_allclose(base.edge_distance, swapped.edge_distance)
    np.testing.assert_allclose(base.driver_q_scaled, swapped.driver_q_scaled)
    assert swapped.has_role(ROLE_R_BOND).sum() == base.has_role(ROLE_R_BOND).sum()
    assert swapped.has_role(ROLE_FORMED).sum() == base.has_role(ROLE_BROKEN).sum()
    np.testing.assert_allclose(base.edge_p_order, swapped.edge_r_order)


def test_feature_build_never_sees_label_material() -> None:
    import inspect

    source = inspect.getsource(build_features)
    for forbidden in ("signed_progress", "path_proximity", "reference_ts", "neighbor"):
        assert forbidden not in source


def test_distance_basis_shape_and_monotonicity() -> None:
    distances = np.asarray([0.5, 1.0, 4.0])
    basis = gaussian_distance_basis(distances, cutoff=5.0, n_basis=16)
    assert basis.shape == (3, 16)
    assert np.all(basis >= 0) and np.all(basis <= 1.000001)


def test_coordinate_evaluation_matches_direct_distances() -> None:
    coords = np.asarray([[0.0, 0, 0], [1.3, 0, 0], [5.0, 0, 0], [6.4, 0, 0]])
    prediction_input = _input(coords)
    q = evaluate_coordinate_spec(coords, prediction_input.coordinate_spec)
    distances = pairwise_distances(coords)
    assert q[0] == pytest.approx(distances[0, 1])
    assert q[1] == pytest.approx(distances[2, 3])
    features = build_features(prediction_input)
    np.testing.assert_allclose(features.driver_q_scaled, q, atol=1e-9)
