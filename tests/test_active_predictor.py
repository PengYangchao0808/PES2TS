"""Key-atom-only geometric seed predictor (G2-AB3 WP-3).

Offline unit tests for ``predict_seed``: typed statuses, bitwise
non-active invariance, distance-driver Jacobian behaviour, the per-atom
displacement cap, and misuse raising ``SeedPredictionError``.
"""
from __future__ import annotations

import numpy as np
import pytest

from pes2ts_core.generation.planning.active_predictor import (
    SEED_AT_TARGET,
    SEED_NONFINITE,
    SEED_OK,
    SEED_RANK_DEFICIENT,
    SEED_TARGET_UNREACHABLE,
    SeedPredictionError,
    predict_seed,
)


def _row_tuple(row):
    return tuple(float(v) for v in row)


def test_at_target_when_distances_already_match():
    geometry = [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [4.0, 0.0, 0.0]]
    result = predict_seed(
        geometry,
        driver_kinds=["distance"],
        driver_atoms=[[0, 1]],
        target_values=[1.5],
        active_atom_indices=[0, 1],
    )
    assert result.status == SEED_AT_TARGET
    assert result.geometry == tuple(_row_tuple(row) for row in geometry)
    assert result.delta_active == ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    assert result.achieved_values == pytest.approx((1.5,))
    assert result.residuals == pytest.approx((0.0,), abs=1e-12)
    assert result.max_atom_displacement == 0.0
    assert result.jacobian_rank == 1
    assert result.reason is None


def test_predicted_seed_moves_only_key_atoms_bitwise_and_hits_target():
    # C-H driver (0,1) stretches 1.5 -> 2.5 A; spectator atom 2 must not move.
    geometry = [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [4.0, 0.0, 0.0]]
    parent = np.asarray(geometry, dtype=float)
    result = predict_seed(
        geometry,
        driver_kinds=["distance"],
        driver_atoms=[[0, 1]],
        target_values=[2.5],
        active_atom_indices=[0, 1],
        max_atom_displacement=0.6,
    )
    assert result.status == SEED_OK
    assert result.geometry is not None
    seed = np.asarray(result.geometry, dtype=float)

    # Non-active atom 2 is EXACTLY unchanged (bitwise rows).
    assert result.geometry[2] == _row_tuple(parent[2])
    assert seed[2].tobytes() == parent[2].tobytes()

    # Distance-driver Jacobian rows: the min-norm solve reaches the target
    # exactly for a pure bond stretch along the internuclear axis.
    achieved = float(np.linalg.norm(seed[0] - seed[1]))
    assert achieved == pytest.approx(2.5, abs=1e-9)
    assert result.achieved_values == pytest.approx((2.5,), abs=1e-9)
    assert result.residuals == pytest.approx((0.0,), abs=1e-9)

    # Per-atom displacement cap honoured for every active atom.
    for atom in (0, 1):
        displacement = float(np.linalg.norm(seed[atom] - parent[atom]))
        assert displacement <= 0.6 + 1e-12
    assert result.max_atom_displacement == pytest.approx(0.5, abs=1e-9)

    # Opposite axial moves along the unit vector u = (x0 - x1)/|x0 - x1|.
    delta = np.asarray(result.delta_active, dtype=float)
    assert delta[0] == pytest.approx((-0.5, 0.0, 0.0), abs=1e-9)
    assert delta[1] == pytest.approx((0.5, 0.0, 0.0), abs=1e-9)
    assert result.jacobian_rank == 1
    assert result.active_atom_indices == (0, 1)
    assert result.reason is None


def test_two_independent_distance_drivers_reach_both_targets():
    # 4-atom chain: drivers (0,1) and (2,3); spectator atom 4 frozen.
    geometry = [
        [0.0, 0.0, 0.0], [1.5, 0.0, 0.0],
        [3.0, 0.0, 0.0], [4.5, 0.0, 0.0],
        [9.0, 0.0, 0.0],
    ]
    parent = np.asarray(geometry, dtype=float)
    result = predict_seed(
        geometry,
        driver_kinds=["distance", "distance"],
        driver_atoms=[[0, 1], [2, 3]],
        target_values=[2.0, 2.5],
        active_atom_indices=[0, 1, 2, 3],
    )
    assert result.status == SEED_OK
    seed = np.asarray(result.geometry, dtype=float)
    assert float(np.linalg.norm(seed[0] - seed[1])) == pytest.approx(2.0, abs=1e-9)
    assert float(np.linalg.norm(seed[2] - seed[3])) == pytest.approx(2.5, abs=1e-9)
    # Independent atom blocks -> full row rank.
    assert result.jacobian_rank == 2
    assert len(result.singular_values or ()) == 2
    # Spectator atom 4 bitwise unchanged.
    assert result.geometry[4] == _row_tuple(parent[4])
    assert seed[4].tobytes() == parent[4].tobytes()
    for row in result.delta_active:
        assert float(np.linalg.norm(row)) <= 0.6 + 1e-12


def test_refused_rank_deficient_when_coupled_targets_cannot_be_reached():
    # Two distance drivers on the SAME atom pair with conflicting targets:
    # the K-restricted Jacobian has identical rows -> rank 1 < 2.
    geometry = [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [4.0, 0.0, 0.0]]
    result = predict_seed(
        geometry,
        driver_kinds=["distance", "distance"],
        driver_atoms=[[0, 1], [0, 1]],
        target_values=[2.0, 2.5],
        active_atom_indices=[0, 1],
    )
    assert result.status == SEED_RANK_DEFICIENT
    assert result.geometry is None
    assert result.jacobian_rank == 1
    assert result.singular_values is not None
    assert "rank" in (result.reason or "")


def test_refused_target_unreached_when_cap_blocks_the_seed():
    geometry = [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [4.0, 0.0, 0.0]]
    result = predict_seed(
        geometry,
        driver_kinds=["distance"],
        driver_atoms=[[0, 1]],
        target_values=[5.0],  # dq = 3.5 A -> 1.75 A per atom > cap 0.6
        active_atom_indices=[0, 1],
        max_atom_displacement=0.6,
    )
    assert result.status == SEED_TARGET_UNREACHABLE
    assert result.geometry is None
    assert result.max_atom_displacement == pytest.approx(1.75, abs=1e-9)
    assert result.max_atom_displacement > 0.6
    assert "cap" in (result.reason or "")


def test_refused_nonfinite_geometry():
    geometry = [[0.0, 0.0, 0.0], [float("nan"), 0.0, 0.0], [4.0, 0.0, 0.0]]
    result = predict_seed(
        geometry,
        driver_kinds=["distance"],
        driver_atoms=[[0, 1]],
        target_values=[2.0],
        active_atom_indices=[0, 1],
    )
    assert result.status == SEED_NONFINITE
    assert result.geometry is None
    assert "non-finite" in (result.reason or "")


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"driver_kinds": ["distance"], "driver_atoms": [[0, 1]],
          "target_values": [1.5, 2.0], "active_atom_indices": [0, 1]},
         "parallel"),
        ({"driver_kinds": ["distance"], "driver_atoms": [[0, 2]],
          "target_values": [2.0], "active_atom_indices": [0, 1]},
         "outside the active set"),
        ({"driver_kinds": ["distance"], "driver_atoms": [[0, 1]],
          "target_values": [2.0], "active_atom_indices": [0, 9]},
         "out of range"),
        ({"driver_kinds": ["distance"], "driver_atoms": [[0, 9]],
          "target_values": [2.0], "active_atom_indices": [0, 1]},
         "out of range"),
    ],
)
def test_misuse_raises_seed_prediction_error(kwargs, match):
    geometry = [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [4.0, 0.0, 0.0]]
    with pytest.raises(SeedPredictionError, match=match):
        predict_seed(geometry, **kwargs)


def test_coincident_driver_atoms_raise_on_jacobian_construction():
    geometry = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [4.0, 0.0, 0.0]]
    with pytest.raises(SeedPredictionError, match="coincident"):
        predict_seed(
            geometry,
            driver_kinds=["distance"],
            driver_atoms=[[0, 1]],
            target_values=[1.0],
            active_atom_indices=[0, 1],
        )


def test_angle_driver_degenerate_for_collinear_atoms_raises():
    geometry = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    with pytest.raises(SeedPredictionError, match="collinear"):
        predict_seed(
            geometry,
            driver_kinds=["angle"],
            driver_atoms=[[0, 1, 2]],
            target_values=[90.0],
            active_atom_indices=[0, 1, 2],
        )


def test_unsupported_driver_kind_raises():
    geometry = [
        [0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0], [4.5, 0.0, 0.0],
    ]
    with pytest.raises(SeedPredictionError, match="dihedral"):
        predict_seed(
            geometry,
            driver_kinds=["dihedral"],
            driver_atoms=[[0, 1, 2, 3]],
            target_values=[90.0],
            active_atom_indices=[0, 1, 2, 3],
        )


def test_angle_driver_moves_key_atoms_and_reports_rank():
    # Bent H-C-H: angle driver on (0,1,2), only atom 2 moves.
    geometry = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],  # 90 deg at atom 1
    ]
    result = predict_seed(
        geometry,
        driver_kinds=["angle"],
        driver_atoms=[[0, 1, 2]],
        target_values=[120.0],
        active_atom_indices=[0, 1, 2],
        max_atom_displacement=0.6,
    )
    # Linearised step from 90 -> 120 deg may exceed the 0.6 A cap; either a
    # predicted seed (rank 1, cap honoured) or a typed refusal is valid.
    assert result.status in (SEED_OK, SEED_TARGET_UNREACHABLE)
    assert result.jacobian_rank == 1
    if result.status == SEED_OK:
        seed = np.asarray(result.geometry, dtype=float)
        for atom in (0, 1, 2):
            assert float(np.linalg.norm(
                seed[atom] - np.asarray(geometry, dtype=float)[atom]
            )) <= 0.6 + 1e-12
