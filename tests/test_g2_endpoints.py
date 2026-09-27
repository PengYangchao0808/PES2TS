"""Tests for the G2 endpoint tables, side validation, and rigid alignment.

Fixtures are synthetic G1 documents plus inventory component records built in
memory (never the real ``inventory.parquet``): the mapped bond-breaking
reaction ``[F:1][C:2]([H:3])([Cl:4])[O:5][H:6]>>[F:1][C:2]([H:3])([Cl:4])[O:5].[H:6]``
with the stored invariant (the atom at local XYZ row ``i`` carries map
``i + 1``).  The reaction center shell is ``{2, 5, 6}``, so the shared maps
outside the shell are exactly ``{1, 3, 4}`` -- three non-collinear Kabsch
anchors.
"""

# allow: SIZE_OK -- fixture builders plus the focused task-2/task-3 tests; the
# plan names exactly one test file for these tasks.

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pytest

from pes2ts_core.g0.rejections import RejectionCode
from pes2ts_core.g0.rp_checks import ELEMENT_SYMBOLS
from pes2ts_core.g1.index_map import COVALENT_RADII
from pes2ts_core.g2.endpoints import (
    EndpointError,
    ComponentGeometry,
    anchor_maps,
    assemble_endpoints,
    build_side_atoms,
    endpoints_record,
    kabsch_transform,
    place_single,
    separate_changed_pairs,
    side_bond_pairs,
    validate_sides,
    write_endpoint_files,
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


def test_place_single_degenerate_translation_when_shared_maps_few() -> None:
    onto = ComponentGeometry("R0", {1: (0.0, 0.0, 0.0), 2: (1.0, 0.0, 0.0), 3: (5.0, 5.0, 5.0)})
    moving = ComponentGeometry("P0", {1: (10.0, 0.0, 0.0), 2: (11.0, 0.0, 0.0)})
    placed, placement = place_single(moving, onto, anchors=())
    assert placement.basis == "degenerate_translation"
    assert placement.anchor_rank_ok is False
    assert placement.n_shared == 2
    shared_centroid_placed = np.mean([placed[1], placed[2]], axis=0)
    shared_centroid_onto = np.mean([onto.coords[1], onto.coords[2]], axis=0)
    assert np.allclose(shared_centroid_placed, shared_centroid_onto, atol=1e-12)


def test_place_single_degenerate_translation_when_everything_is_collinear() -> None:
    onto = ComponentGeometry("R0", {1: (0.0, 0.0, 0.0), 2: (1.0, 0.0, 0.0), 3: (2.0, 0.0, 0.0)})
    moving = ComponentGeometry("P0", {1: (5.0, 5.0, 5.0), 2: (6.0, 5.0, 5.0), 3: (7.0, 5.0, 5.0)})
    placed, placement = place_single(moving, onto, anchors=(1, 2, 3))
    assert placement.basis == "degenerate_translation"
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


# --- assemble_endpoints / write_endpoint_files (task 3) ----------------------


def _dist(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return math.dist(a, b)


ASSOC_SMILES = "[F:1][C:2]([H:3])([Cl:4])[O:5].[H:6]>>[F:1][C:2]([H:3])([Cl:4])[O:5][H:6]"


def _assoc_doc(*, shuffle: bool = False) -> dict[str, Any]:
    r_blocks = [
        {"tag": "R0", "rows": _rows(5, R0_ELEMENTS[:5])},
        {"tag": "R1", "rows": [{"local_index": 0, "map": 6, "element": "H", "global_index": 5}]},
    ]
    p_blocks = [{"tag": "Q0", "rows": _rows(6, R0_ELEMENTS)}]
    if shuffle:
        r_blocks = list(reversed(r_blocks))
        p_blocks = list(reversed(p_blocks))
    return {
        "reaction_id": "RXN_TEST_0002",
        "reaction_smiles": ASSOC_SMILES,
        "reactants": r_blocks,
        "products": p_blocks,
        "reaction_center": {"core": [5, 6], "with_shell": list(WITH_SHELL)},
        "validation": {"status": "valid"},
    }


def _assoc_row() -> dict[str, Any]:
    return {
        "reaction_id": "RXN_TEST_0002",
        "reaction_smiles": ASSOC_SMILES,
        "charge_total_reactants": 0,
        "charge_total_products": 0,
        "multiplicity_max": 1,
        "components": [
            _component("R0", R0_Z[:5], R0_X[:5]),
            _component("R1", (1,), (R0_X[5],)),
            _component("Q0", R0_Z, R0_X),
        ],
    }


def _dissoc_doc() -> dict[str, Any]:
    return {
        "reaction_id": "RXN_TEST_0001",
        "reaction_smiles": HAPPY_SMILES,
        "reactants": [{"tag": "R0", "rows": _rows(6, R0_ELEMENTS)}],
        "products": [
            {"tag": "P0", "rows": _rows(5, R0_ELEMENTS[:5])},
            {"tag": "P1", "rows": [{"local_index": 0, "map": 6, "element": "H", "global_index": 5}]},
        ],
        "reaction_center": {"core": [5, 6], "with_shell": list(WITH_SHELL)},
        "validation": {"status": "valid"},
    }


def _pair_distance(coords: dict[int, tuple[float, float, float]], a: int, b: int) -> float:
    return _dist(coords[a], coords[b])


def test_assemble_two_to_one_association_separates_formed_pairs() -> None:
    assembly = assemble_endpoints(_assoc_doc(), _assoc_row())
    r_coords, p_coords = assembly.reactant_coords, assembly.product_coords
    assert _pair_distance(r_coords, 5, 6) >= 2.0
    o_h_limit = COVALENT_RADII[8] + COVALENT_RADII[1] + 0.45
    assert _pair_distance(p_coords, 5, 6) <= o_h_limit
    record = assembly.record
    assert all("group_id" in group and "anchor_component" in group for group in record["groups"])
    evaluated = [entry for entry in record["separations"] if entry["separation_evaluated"]]
    assert any(
        entry["side"] == "R" and entry["kind"] == "formed" and entry["triggered"]
        for entry in evaluated
    )


def test_assemble_one_to_two_dissociation_keeps_topology_valid() -> None:
    doc = _dissoc_doc()
    row = _row()
    assembly = assemble_endpoints(doc, row)
    assert _pair_distance(assembly.product_coords, 5, 6) >= 2.0
    r_pairs, p_pairs = side_bond_pairs(HAPPY_SMILES)
    for a, b in sorted(r_pairs):
        z_a, z_b = ELEMENT_SYMBOLS.index(R0_ELEMENTS[a - 1]) + 1, ELEMENT_SYMBOLS.index(R0_ELEMENTS[b - 1]) + 1
        limit = COVALENT_RADII[z_a] + COVALENT_RADII[z_b] + 0.45
        assert _pair_distance(assembly.reactant_coords, a, b) <= limit
    assert any(
        entry["side"] == "P" and entry["kind"] == "broken" and entry["triggered"]
        for entry in assembly.record["separations"]
    )
    assert r_pairs - p_pairs == frozenset({(5, 6)})


def test_assemble_is_byte_identical_under_block_shuffling(tmp_path: Any) -> None:
    first = assemble_endpoints(_assoc_doc(), _assoc_row())
    second = assemble_endpoints(_assoc_doc(shuffle=True), _assoc_row())
    assert json.dumps(first.record, sort_keys=True) == json.dumps(second.record, sort_keys=True)
    files_a = write_endpoint_files(tmp_path / "a", first)
    files_b = write_endpoint_files(tmp_path / "b", second)
    for path_a, path_b in zip(files_a, files_b):
        assert path_a.read_bytes() == path_b.read_bytes()


def test_assemble_candidate_winner_is_predictable_and_prefers_larger_min_distance() -> None:
    assembly = assemble_endpoints(_assoc_doc(), _assoc_row())
    candidates = assembly.record["candidates"]
    assert [entry["candidate_index"] for entry in candidates] == [0, 1]
    winner = candidates[assembly.winner_index]
    for loser in candidates:
        if loser["candidate_index"] == assembly.winner_index:
            continue
        winner_min = round(winner["metrics"]["min_nonbonded_distance"], 6)
        loser_min = round(loser["metrics"]["min_nonbonded_distance"], 6)
        assert winner_min >= loser_min
        assert winner["score"] <= loser["score"]


DISCONNECTED_SMILES = "[C:1][C:2].[O:3][O:4]>>[C:1].[C:2].[O:3].[O:4]"


def _disconnected_doc(*, spectator_offset: tuple[float, float, float]) -> dict[str, Any]:
    offset = spectator_offset
    return {
        "reaction_id": "RXN_TEST_0003",
        "reaction_smiles": DISCONNECTED_SMILES,
        "reactants": [
            {"tag": "R0", "rows": [
                {"local_index": 0, "map": 1, "element": "C", "global_index": 0},
                {"local_index": 1, "map": 2, "element": "C", "global_index": 1},
            ]},
            {"tag": "R1", "rows": [
                {"local_index": 0, "map": 3, "element": "O", "global_index": 2},
                {"local_index": 1, "map": 4, "element": "O", "global_index": 3},
            ]},
        ],
        "products": [
            {"tag": "PA", "rows": [{"local_index": 0, "map": 1, "element": "C", "global_index": 0}]},
            {"tag": "PB", "rows": [{"local_index": 0, "map": 2, "element": "C", "global_index": 1}]},
            {"tag": "PC", "rows": [{"local_index": 0, "map": 3, "element": "O", "global_index": 2}]},
            {"tag": "PD", "rows": [{"local_index": 0, "map": 4, "element": "O", "global_index": 3}]},
        ],
        "reaction_center": {"core": [1, 2, 3, 4], "with_shell": [1, 2, 3, 4]},
        "validation": {"status": "valid"},
    }


def _disconnected_row(*, spectator_offset: tuple[float, float, float]) -> dict[str, Any]:
    off = spectator_offset
    c1, c2 = (0.0, 0.0, 0.0), (1.5, 0.0, 0.0)
    o3 = (10.0 + off[0], off[1], off[2])
    o4 = (11.5 + off[0], off[1], off[2])
    return {
        "reaction_id": "RXN_TEST_0003",
        "reaction_smiles": DISCONNECTED_SMILES,
        "charge_total_reactants": 0,
        "charge_total_products": 0,
        "multiplicity_max": 1,
        "components": [
            _component("R0", (6, 6), (c1, c2)),
            _component("R1", (8, 8), (o3, o4)),
            _component("PA", (6,), (c1,)),
            _component("PB", (6,), (c2,)),
            _component("PC", (8,), (o3,)),
            _component("PD", (8,), (o4,)),
        ],
    }


def test_assemble_disconnected_bipartite_groups_all_get_group_records() -> None:
    doc = _disconnected_doc(spectator_offset=(0.0, 0.0, 0.0))
    row = _disconnected_row(spectator_offset=(0.0, 0.0, 0.0))
    assembly = assemble_endpoints(doc, row)
    groups = assembly.record["groups"]
    assert [group["group_id"] for group in groups] == ["G0", "G1"]
    assert sum(len(group["nodes"]) for group in groups) == 6
    anchored = next(group for group in groups if group["group_id"] == "G0")
    spectator = next(group for group in groups if group["group_id"] == "G1")
    assert anchored["assembly_basis"] == "candidate_anchor_stored_frame"
    assert anchored["frame_ambiguous"] is False
    assert spectator["assembly_basis"] == "stored_frame_per_group"
    assert spectator["frame_ambiguous"] is True
    for map_ in (1, 2, 3, 4):
        assert map_ in assembly.reactant_coords and map_ in assembly.product_coords


DEGENERATE_SMILES = "[F:1][C:2]([H:3])[H:4]>>[F:1].[C:2]([H:3])[H:4]"


def _degenerate_doc() -> dict[str, Any]:
    return {
        "reaction_id": "RXN_TEST_0004",
        "reaction_smiles": DEGENERATE_SMILES,
        "reactants": [{"tag": "R0", "rows": _rows(4, ("F", "C", "H", "H"))}],
        "products": [
            {"tag": "P0", "rows": [{"local_index": 0, "map": 1, "element": "F", "global_index": 0}]},
            {"tag": "P1", "rows": [
                {"local_index": 0, "map": 2, "element": "C", "global_index": 1},
                {"local_index": 1, "map": 3, "element": "H", "global_index": 2},
                {"local_index": 2, "map": 4, "element": "H", "global_index": 3},
            ]},
        ],
        "reaction_center": {"core": [1, 2], "with_shell": [1, 2, 3, 4]},
        "validation": {"status": "valid"},
    }


def _degenerate_row() -> dict[str, Any]:
    r0 = ((0.0, 0.0, 0.0), (3.0, 0.0, 0.0), (3.8, 0.8, 0.0), (3.8, -0.8, 0.0))
    p1 = ((5.0, 0.0, 0.0), (6.0, 0.0, 0.0), (7.0, 0.0, 0.0))
    return {
        "reaction_id": "RXN_TEST_0004",
        "reaction_smiles": DEGENERATE_SMILES,
        "charge_total_reactants": 0,
        "charge_total_products": 0,
        "multiplicity_max": 1,
        "components": [
            _component("R0", (9, 6, 1, 1), r0),
            _component("P0", (9,), ((50.0, 0.0, 0.0),)),
            _component("P1", (6, 1, 1), p1),
        ],
    }


def test_assemble_collinear_anchors_fall_back_to_degenerate_translation() -> None:
    assembly = assemble_endpoints(_degenerate_doc(), _degenerate_row())
    assert assembly.winner_index == 0
    placements = assembly.record["placements"]
    degenerate = [p for p in placements if p["basis"] == "degenerate_translation"]
    assert degenerate and all(p["anchor_rank_ok"] is False for p in degenerate)
    assert not any(p["basis"] in ("anchor_maps", "all_shared_maps") for p in placements)
    moved = next(p for p in degenerate if p["component"] == "P:P1")
    p1_stored = {2: (5.0, 0.0, 0.0), 3: (6.0, 0.0, 0.0), 4: (7.0, 0.0, 0.0)}
    deltas = {
        (map_, i): assembly.product_coords[map_][i] - stored[i]
        for map_, stored in p1_stored.items()
        for i in range(3)
    }
    assert moved["n_shared"] == 3
    per_atom = [
        (deltas[(map_, 0)], deltas[(map_, 1)], deltas[(map_, 2)])
        for map_ in (2, 3, 4)
    ]
    assert all(np.allclose(delta, per_atom[0], atol=1e-9) for delta in per_atom)


def test_separate_changed_pairs_uses_element_aware_thresholds() -> None:
    coords = {1: (0.0, 0.0, 0.0), 2: (2.2, 0.0, 0.0)}
    moved, records = separate_changed_pairs(
        [(1, 2)], side="P", kind="broken", coords=coords,
        component_of_map={1: "A", 2: "B"}, group_seed_of_component={"A": "A", "B": "A"},
        elements={1: "S", 2: "S"},
    )
    ss_sum = COVALENT_RADII[16] * 2
    entry = records[0]
    assert entry["triggered"] is True
    assert entry["threshold"] == pytest.approx(max(2.0, ss_sum + 0.45))
    assert entry["target"] == pytest.approx(max(3.0, ss_sum + 0.45 + 0.5))
    assert _pair_distance(moved, 1, 2) == pytest.approx(entry["target"])
    assert set(entry) == {
        "side", "kind", "atoms", "separation_evaluated", "triggered",
        "threshold", "target", "d_before", "d_after",
    }
    moved_cp, records_cp = separate_changed_pairs(
        [(1, 2)], side="P", kind="broken", coords=dict(coords),
        component_of_map={1: "A", 2: "B"}, group_seed_of_component={"A": "A", "B": "A"},
        elements={1: "C", 2: "P"},
    )
    cp_sum = COVALENT_RADII[6] + COVALENT_RADII[15]
    assert records_cp[0]["threshold"] == pytest.approx(max(2.0, cp_sum + 0.45))
    assert records_cp[0]["target"] == pytest.approx(max(3.0, cp_sum + 0.45 + 0.5))
    assert records_cp[0]["triggered"] is True
    assert _pair_distance(moved_cp, 1, 2) == pytest.approx(records_cp[0]["target"])


def test_assemble_disconnected_spectator_group_never_silently_overlaps() -> None:
    doc = _disconnected_doc(spectator_offset=(0.0, 0.0, 0.0))
    row = _disconnected_row(spectator_offset=(-9.7, 0.0, 0.0))
    try:
        assembly = assemble_endpoints(doc, row)
    except EndpointError as exc:
        assert exc.code is RejectionCode.G2_ASSEMBLY_COLLISION
        return
    severe = assembly.record["metrics"]["n_severe_contacts"]
    assert severe == 0


def test_assemble_malformed_document_raises_assembly_failed() -> None:
    doc = _dissoc_doc()
    del doc["reaction_center"]
    with pytest.raises(EndpointError) as exc:
        assemble_endpoints(doc, _row())
    assert exc.value.code is RejectionCode.G2_ASSEMBLY_FAILED

    doc = _dissoc_doc()
    del doc["reactants"][0]["rows"]
    with pytest.raises(EndpointError) as exc:
        assemble_endpoints(doc, _row())
    assert exc.value.code is RejectionCode.G2_ASSEMBLY_FAILED

    doc = _dissoc_doc()
    row = _row()
    row["reaction_smiles"] = "this is not a reaction"
    with pytest.raises(EndpointError) as exc:
        assemble_endpoints(doc, row)
    assert exc.value.code is RejectionCode.G2_ASSEMBLY_FAILED

    row = _row()
    big = 1.7e308
    row["components"][1] = _component(
        "P0", R0_Z[:5],
        ((big, 0.0, 0.0), (-big, 0.0, 0.0), (0.0, big, 0.0), (0.0, -big, 0.0), (0.0, 0.0, big)),
    )
    with pytest.raises(EndpointError) as exc:
        assemble_endpoints(_dissoc_doc(), row)
    assert exc.value.code is RejectionCode.G2_ASSEMBLY_FAILED


def test_assemble_overlap_fixture_raises_collision_with_evidence() -> None:
    doc = _dissoc_doc()
    row = _row()
    row["components"][0]["coordinates"][3] = [2.4, 0.6, 0.3]
    with pytest.raises(EndpointError) as exc:
        assemble_endpoints(doc, row)
    assert exc.value.code is RejectionCode.G2_ASSEMBLY_COLLISION
    evidence = exc.value.evidence
    assert evidence is not None
    assert evidence["metrics"]["n_severe_contacts"] >= 1
    assert len(evidence["candidates"]) >= 1


def test_assemble_honors_config_overrides() -> None:
    config = {"g2": {"assembly": {"forming_target_distance": 7.0}}}
    assembly = assemble_endpoints(_assoc_doc(), _assoc_row(), config=config)
    assert _pair_distance(assembly.reactant_coords, 5, 6) == pytest.approx(7.0)
    config = {"g2": {"validity": {"collision_min_distance": 5.0}}}
    with pytest.raises(EndpointError) as exc:
        assemble_endpoints(_assoc_doc(), _assoc_row(), config=config)
    assert exc.value.code is RejectionCode.G2_ASSEMBLY_COLLISION
