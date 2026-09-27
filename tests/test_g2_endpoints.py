"""Tests for the G2 endpoint tables, side validation, and rigid alignment.

Fixtures are synthetic G1 documents plus inventory component records built in
memory (never the real ``inventory.parquet``): the mapped bond-breaking
reaction ``[F:1][C:2]([H:3])([Cl:4])[O:5][H:6]>>[F:1][C:2]([H:3])([Cl:4])[O:5].[H:6]``
with the stored invariant (the atom at local XYZ row ``i`` carries map
``i + 1``).  The reaction center shell is ``{2, 5, 6}``, so the shared maps
outside the shell are exactly ``{1, 3, 4}`` -- three non-collinear Kabsch
anchors.
"""

# allow: SIZE_OK -- fixture builders plus 19 focused tests; the plan names
# exactly one test file for this task.

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pytest

from pes2ts_core.g0.rejections import RejectionCode
from pes2ts_core.g2.endpoints import (
    EndpointError,
    ComponentGeometry,
    anchor_maps,
    build_side_atoms,
    endpoints_record,
    kabsch_transform,
    place_single,
    side_bond_pairs,
    validate_sides,
)

HAPPY_SMILES = "[F:1][C:2]([H:3])([Cl:4])[O:5][H:6]>>[F:1][C:2]([H:3])([Cl:4])[O:5].[H:6]"

R0_ELEMENTS = ("F", "C", "H", "Cl", "O", "H")
R0_Z = (9, 6, 1, 17, 8, 1)
R0_X = (
    (0.0, 0.0, 0.0),
    (1.35, 0.0, 0.0),
    (1.9, 0.6, 0.3),
    (1.7, -1.2, 0.6),
    (2.6, 0.7, -0.2),
    (2.2, 1.5, 0.2),
)

WITH_SHELL = (2, 5, 6)
#: Shared maps outside ``with_shell``: three non-collinear anchors.
ANCHORS = (1, 3, 4)


def _rows(n_atoms: int, elements: tuple[str, ...]) -> list[dict[str, Any]]:
    return [
        {"local_index": i, "map": i + 1, "element": elements[i], "global_index": i}
        for i in range(n_atoms)
    ]


def _doc(
    *,
    p0_z: tuple[int, ...] = R0_Z[:5],
    p0_elements: tuple[str, ...] = R0_ELEMENTS[:5],
    drop_p0_map: int | None = None,
) -> dict[str, Any]:
    """Return a synthetic G1 document for the happy fixture."""
    p0_rows = _rows(5, p0_elements)
    if drop_p0_map is not None:
        p0_rows = [row for row in p0_rows if row["map"] != drop_p0_map]
    return {
        "reaction_id": "RXN_TEST_0001",
        "reaction_smiles": HAPPY_SMILES,
        "reactants": [{"tag": "R0", "rows": _rows(6, R0_ELEMENTS)}],
        "products": [
            {"tag": "P0", "rows": p0_rows},
            {"tag": "P1", "rows": [{"local_index": 0, "map": 6, "element": "H", "global_index": 0}]},
        ],
        "reaction_center": {"core": [5, 6], "with_shell": list(WITH_SHELL)},
        "validation": {"status": "valid"},
    }


def _component(tag: str, atomic_numbers: tuple[int, ...], coords: tuple[tuple[float, float, float], ...]) -> dict[str, Any]:
    return {
        "tag": tag,
        "smiles": "",
        "atomic_numbers": list(atomic_numbers),
        "coordinates": [list(row) for row in coords],
    }


def _row(*, p0_z: tuple[int, ...] = R0_Z[:5], **overrides: Any) -> dict[str, Any]:
    row = {
        "reaction_id": "RXN_TEST_0001",
        "reaction_smiles": HAPPY_SMILES,
        "charge_total_reactants": 0,
        "charge_total_products": 0,
        "multiplicity_max": 1,
        "components": [
            _component("R0", R0_Z, R0_X),
            _component("P0", p0_z, R0_X[:5]),
            _component("P1", (1,), (R0_X[5],)),
        ],
    }
    row.update(overrides)
    return row


def _sides(doc: dict[str, Any], row: dict[str, Any]) -> tuple[list[Any], list[Any]]:
    components = row["components"]
    return build_side_atoms(doc["reactants"], components), build_side_atoms(doc["products"], components)


def _rotation(axis: np.ndarray, degrees: float) -> np.ndarray:
    """Return the Rodrigues rotation matrix about *axis*."""
    unit = axis / np.linalg.norm(axis)
    x, y, z = unit
    rad = math.radians(degrees)
    c, s = math.cos(rad), math.sin(rad)
    k = 1.0 - c
    return np.array(
        [
            [c + x * x * k, x * y * k - z * s, x * z * k + y * s],
            [y * x * k + z * s, c + y * y * k, y * z * k - x * s],
            [z * x * k - y * s, z * y * k + x * s, c + z * z * k],
        ]
    )


def _angle_deg(matrix: np.ndarray) -> float:
    cos = (np.trace(matrix) - 1.0) / 2.0
    return math.degrees(math.acos(float(np.clip(cos, -1.0, 1.0))))


# --- build_side_atoms -------------------------------------------------------


def test_build_side_atoms_returns_map_sorted_tables_with_component_coordinates() -> None:
    atoms_r, atoms_p = _sides(_doc(), _row())
    assert [atom.map for atom in atoms_r] == [1, 2, 3, 4, 5, 6]
    assert [atom.map for atom in atoms_p] == [1, 2, 3, 4, 5, 6]
    assert atoms_r[0].element == "F"
    assert atoms_r[0].coordinate == (0.0, 0.0, 0.0)
    assert atoms_r[3].element == "Cl"
    assert atoms_p[5].component == "P1"
    assert atoms_p[5].coordinate == (2.2, 1.5, 0.2)


def test_build_side_atoms_is_order_independent_over_input_rows() -> None:
    doc = _doc()
    doc["reactants"][0]["rows"] = list(reversed(doc["reactants"][0]["rows"]))
    atoms_sorted, _ = _sides(_doc(), _row())
    atoms_shuffled, _ = _sides(doc, _row())
    assert atoms_shuffled == atoms_sorted


def test_build_side_atoms_rejects_element_mismatch() -> None:
    doc = _doc()
    doc["reactants"][0]["rows"][3]["element"] = "F"
    with pytest.raises(EndpointError) as exc:
        _sides(doc, _row())
    assert exc.value.code is RejectionCode.G2_ENDPOINT_MISMATCH


def test_build_side_atoms_rejects_non_finite_coordinates() -> None:
    row = _row()
    row["components"][0]["coordinates"][2] = [1.0, float("nan"), 0.0]
    with pytest.raises(EndpointError) as exc:
        _sides(_doc(), row)
    assert exc.value.code is RejectionCode.G2_ENDPOINT_MISMATCH


# --- validate_sides ---------------------------------------------------------


def test_validate_sides_accepts_g1_valid_invariant() -> None:
    atoms_r, atoms_p = _sides(_doc(), _row())
    result = validate_sides(atoms_r, atoms_p, _row())
    assert result == {"charge": 0, "multiplicity": 1, "multiplicity_basis": "g1_valid_invariant"}


def test_validate_sides_rejects_map_set_inequality() -> None:
    doc = _doc(drop_p0_map=4)
    row = _row()
    atoms_r, atoms_p = _sides(doc, row)
    with pytest.raises(EndpointError) as exc:
        validate_sides(atoms_r, atoms_p, row)
    assert exc.value.code is RejectionCode.G2_ENDPOINT_MISMATCH


def test_validate_sides_rejects_per_map_element_mismatch() -> None:
    doc = _doc(p0_z=(9, 6, 1, 8, 8), p0_elements=("F", "C", "H", "O", "O"))
    row = _row(p0_z=(9, 6, 1, 8, 8))
    atoms_r, atoms_p = _sides(doc, row)
    with pytest.raises(EndpointError) as exc:
        validate_sides(atoms_r, atoms_p, row)
    assert exc.value.code is RejectionCode.G2_ENDPOINT_MISMATCH


def test_validate_sides_rejects_charge_inequality() -> None:
    atoms_r, atoms_p = _sides(_doc(), _row())
    with pytest.raises(EndpointError) as exc:
        validate_sides(atoms_r, atoms_p, _row(charge_total_products=1))
    assert exc.value.code is RejectionCode.G2_ENDPOINT_MISMATCH


def test_validate_sides_rejects_non_singlet_multiplicity() -> None:
    atoms_r, atoms_p = _sides(_doc(), _row())
    with pytest.raises(EndpointError) as exc:
        validate_sides(atoms_r, atoms_p, _row(multiplicity_max=3))
    assert exc.value.code is RejectionCode.G2_ENDPOINT_MISMATCH


# --- side_bond_pairs --------------------------------------------------------


def test_side_bond_pairs_builds_map_space_bond_sets() -> None:
    reactant_pairs, product_pairs = side_bond_pairs(HAPPY_SMILES)
    assert reactant_pairs == frozenset({(1, 2), (2, 3), (2, 4), (2, 5), (5, 6)})
    assert product_pairs == frozenset({(1, 2), (2, 3), (2, 4), (2, 5)})
    # "Non-bonded" = bonded in neither R nor P: the broken pair (5, 6) is
    # bonded on the R side only; (1, 6) is bonded on neither side.
    assert reactant_pairs - product_pairs == frozenset({(5, 6)})
    assert (1, 6) not in reactant_pairs
    assert (1, 6) not in product_pairs


# --- anchor_maps ------------------------------------------------------------


def test_anchor_maps_excludes_the_reaction_center_shell() -> None:
    assert anchor_maps(_doc()) == ANCHORS


# --- kabsch_transform -------------------------------------------------------


def test_kabsch_replay_recovers_synthetic_rigid_transform() -> None:
    points = np.array(R0_X, dtype=float)
    rotation_true = _rotation(np.array([1.0, 2.0, 3.0]), 40.0)
    translation_true = np.array([0.7, -1.2, 0.4])
    moved = points @ rotation_true.T + translation_true
    # Forward fit: recovers exactly the applied (R, t).
    forward = kabsch_transform(points, moved)
    assert forward.rmsd < 1e-9
    assert abs(_angle_deg(forward.rotation) - 40.0) < 1e-6
    assert np.allclose(forward.translation, translation_true, atol=1e-9)
    # Restore: the inverse fit maps the moved set back onto the original.
    back = kabsch_transform(moved, points)
    assert back.rmsd < 1e-9
    assert abs(_angle_deg(back.rotation) - 40.0) < 1e-6
    assert np.allclose(back.rotation, rotation_true.T, atol=1e-9)
    assert np.allclose(back.translation, -(rotation_true.T @ translation_true), atol=1e-9)


def test_kabsch_correction_keeps_rotation_proper_under_reflection() -> None:
    points = np.array(R0_X, dtype=float)
    mirrored = points @ np.diag([1.0, 1.0, -1.0])
    result = kabsch_transform(points, mirrored)
    assert np.linalg.det(result.rotation) == pytest.approx(1.0, abs=1e-12)


# --- place_single -----------------------------------------------------------


def _axis_fixture() -> tuple[ComponentGeometry, ComponentGeometry]:
    """Return (moving, onto) sharing five maps, three of them collinear (1,2,3)."""
    onto_coords = {
        1: (0.0, 0.0, 0.0),
        2: (1.0, 0.0, 0.0),
        3: (2.0, 0.0, 0.0),
        4: (0.0, 1.0, 0.0),
        5: (0.0, 0.0, 1.0),
    }
    rotation = _rotation(np.array([1.0, 2.0, 3.0]), 40.0)
    translation = np.array([0.7, -1.2, 0.4])
    moving_coords = {
        map_: tuple(rotation @ np.array(coord) + translation) for map_, coord in onto_coords.items()
    }
    return ComponentGeometry("P0", moving_coords), ComponentGeometry("R0", onto_coords)


def test_place_single_prefers_non_collinear_anchor_maps() -> None:
    moving, onto = _axis_fixture()
    placed, placement = place_single(moving, onto, anchors=(1, 4, 5))
    assert placement.basis == "anchor_maps"
    assert placement.anchor_rank_ok is True
    assert placement.n_shared == 5
    assert placement.rmsd_shared < 1e-9
    for map_, coord in onto.coords.items():
        assert np.allclose(placed[map_], coord, atol=1e-9)


def test_place_single_falls_back_to_all_shared_maps_on_collinear_anchors() -> None:
    moving, onto = _axis_fixture()
    placed, placement = place_single(moving, onto, anchors=(1, 2, 3))
    assert placement.basis == "all_shared_maps"
    assert placement.anchor_rank_ok is False
    assert placement.rmsd_shared < 1e-9
    for map_, coord in onto.coords.items():
        assert np.allclose(placed[map_], coord, atol=1e-9)


def test_place_single_translation_only_when_shared_maps_few() -> None:
    onto = ComponentGeometry("R0", {1: (0.0, 0.0, 0.0), 2: (1.0, 0.0, 0.0), 3: (5.0, 5.0, 5.0)})
    moving = ComponentGeometry("P0", {1: (10.0, 0.0, 0.0), 2: (11.0, 0.0, 0.0)})
    placed, placement = place_single(moving, onto, anchors=())
    assert placement.basis == "translation_only"
    assert placement.anchor_rank_ok is False
    assert placement.n_shared == 2
    shared_centroid_placed = np.mean([placed[1], placed[2]], axis=0)
    shared_centroid_onto = np.mean([onto.coords[1], onto.coords[2]], axis=0)
    assert np.allclose(shared_centroid_placed, shared_centroid_onto, atol=1e-12)


def test_place_single_translation_only_when_everything_is_collinear() -> None:
    onto = ComponentGeometry("R0", {1: (0.0, 0.0, 0.0), 2: (1.0, 0.0, 0.0), 3: (2.0, 0.0, 0.0)})
    moving = ComponentGeometry("P0", {1: (5.0, 5.0, 5.0), 2: (6.0, 5.0, 5.0), 3: (7.0, 5.0, 5.0)})
    placed, placement = place_single(moving, onto, anchors=(1, 2, 3))
    assert placement.basis == "translation_only"
    assert placement.anchor_rank_ok is False
    assert placement.rmsd_shared == pytest.approx(0.0, abs=1e-12)
    assert np.allclose(placed[1], onto.coords[1], atol=1e-12)


# --- endpoints_record -------------------------------------------------------


def _record() -> dict[str, Any]:
    doc = _doc()
    row = _row()
    atoms_r, atoms_p = _sides(doc, row)
    validation = validate_sides(atoms_r, atoms_p, row)
    moving = ComponentGeometry(
        "P0", {atom.map: atom.coordinate for atom in atoms_p if atom.component == "P0"}
    )
    onto = ComponentGeometry("R0", {atom.map: atom.coordinate for atom in atoms_r})
    _, placement = place_single(moving, onto, anchors=ANCHORS)
    return endpoints_record(validation, atoms_r, atoms_p, [placement])


def test_endpoints_record_is_deterministic_across_two_runs() -> None:
    first = json.dumps(_record(), sort_keys=True)
    second = json.dumps(_record(), sort_keys=True)
    assert first == second
    payload = json.loads(first)
    assert payload["validation"]["multiplicity_basis"] == "g1_valid_invariant"
    assert [atom["map"] for atom in payload["reactant_atoms"]] == [1, 2, 3, 4, 5, 6]
    assert payload["placements"][0]["basis"] == "anchor_maps"


def test_endpoints_record_payload_is_plain_json_scalars() -> None:
    payload = _record()
    assert set(payload) == {"validation", "reactant_atoms", "product_atoms", "placements"}
    atom = payload["reactant_atoms"][0]
    assert set(atom) == {"map", "element", "component", "local_index", "global_index", "coordinate"}
    assert set(payload["placements"][0]) == {
        "component", "onto", "n_shared", "basis", "rmsd_shared", "anchor_rank_ok",
    }
