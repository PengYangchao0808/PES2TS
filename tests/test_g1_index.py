"""Tests for the G1 skeleton matcher and the unified index tables.

The fixtures are small hand-built inventory component records whose SMILES map
labels obey the stored invariant (the atom carrying map ``k`` sits at XYZ row
``k - 1``), so the tests lock the two decisive semantics: matches are enumerated
in flattened skeleton space and reported in XYZ-row space, and the index tables
translate map label -> local row -> side-global row deterministically while
recording every symmetry-ambiguity candidate.
"""

from __future__ import annotations

from typing import Any

import pytest
from rdkit import Chem

from pes2ts_core.g1.index_map import (
    COVALENT_RADII,
    ComponentIndex,
    bond_geometry_check,
    build_side_indexes,
)
from pes2ts_core.g1.skeleton import (
    ComponentMatch,
    ComponentMismatchError,
    flatten_skeleton,
    match_side_components,
)

TETRAHEDRAL = 0.63  # 1.09 / sqrt(3): a C-H bond at 1.09 Angstrom.

#: Methane: inventory maps are identity, so rows are C, H, H, H, H.
METHANE_INVENTORY: dict[str, Any] = {
    "tag": "R0",
    "smiles": "[C:1]([H:2])([H:3])([H:4])[H:5]",
    "atomic_numbers": [6, 1, 1, 1, 1],
    "coordinates": [
        [0.0, 0.0, 0.0],
        [TETRAHEDRAL, TETRAHEDRAL, TETRAHEDRAL],
        [TETRAHEDRAL, -TETRAHEDRAL, -TETRAHEDRAL],
        [-TETRAHEDRAL, TETRAHEDRAL, -TETRAHEDRAL],
        [-TETRAHEDRAL, -TETRAHEDRAL, TETRAHEDRAL],
    ],
    "charge": 0,
    "multiplicity": 1,
    "EHG": [],
}
METHANE_RXN_MAPS = (11, 7, 8, 9, 10)

#: Water: identity maps, rows are O, H, H.
WATER_INVENTORY: dict[str, Any] = {
    "tag": "R0",
    "smiles": "[O:1]([H:2])[H:3]",
    "atomic_numbers": [8, 1, 1],
    "coordinates": [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
    "charge": 0,
    "multiplicity": 1,
    "EHG": [],
}

#: Water whose SMILES atom order is NOT the row order: map 3 sits on atom 0,
#: so the two hydrogens live at rows 1 and 2 but are written out of order.
WATER_PERMUTED_INVENTORY: dict[str, Any] = {
    "tag": "R0",
    "smiles": "[H:3][O:1][H:2]",
    "atomic_numbers": [8, 1, 1],
    "coordinates": [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
    "charge": 0,
    "multiplicity": 1,
    "EHG": [],
}

#: A molecule with a trivial graph automorphism group (every center has
#: distinct substituents), so exactly one isomorphism must be enumerated.
ASYMMETRIC_INVENTORY: dict[str, Any] = {
    "tag": "R0",
    "smiles": "[F:1][C:2]([Cl:3])([H:4])[C:5]([Br:6])([H:7])[O:8][H:9]",
    "atomic_numbers": [9, 6, 17, 1, 6, 35, 1, 8, 1],
    "coordinates": [[0.0, 0.0, 0.0]] * 9,
    "charge": 0,
    "multiplicity": 1,
    "EHG": [],
}

#: Aromatic benzene (inventory) against a Kekule depiction (reaction side).
BENZENE_INVENTORY: dict[str, Any] = {
    "tag": "R0",
    "smiles": "[cH:1]1[cH:2][cH:3][cH:4][cH:5][cH:6]1",
    "atomic_numbers": [6, 6, 6, 6, 6, 6],
    "coordinates": [[0.0, 0.0, 0.0]] * 6,
    "charge": 0,
    "multiplicity": 1,
    "EHG": [],
}

#: Charge-separated nitrous acid (inventory) against its neutral hypervalent
#: depiction (reaction side): same atoms, same H connectivity, different
#: charges and bond orders.
NITRO_INVENTORY: dict[str, Any] = {
    "tag": "R0",
    "smiles": "[O-:1][N+:2](=[O:3])[H:4]",
    "atomic_numbers": [8, 7, 8, 1],
    "coordinates": [[0.0, 0.0, 0.0]] * 4,
    "charge": 0,
    "multiplicity": 1,
    "EHG": [],
}

#: Chiral inventory depiction against a stereo-free reaction depiction.
CHIRAL_INVENTORY: dict[str, Any] = {
    "tag": "R0",
    "smiles": "[C@H:1]([F:2])([Cl:3])[Br:4]",
    "atomic_numbers": [6, 9, 17, 35],
    "coordinates": [[0.0, 0.0, 0.0]] * 4,
    "charge": 0,
    "multiplicity": 1,
    "EHG": [],
}


def _mol(smiles: str) -> Chem.Mol:
    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(smiles, params)
    assert mol is not None
    return mol


def test_flatten_skeleton_erases_bond_charge_and_stereo_representation() -> None:
    # Given: an aromatic molecule, a charged molecule, and a chiral molecule
    aromatic = _mol("[cH:1]1[cH:2][cH:3][cH:4][cH:5][cH:6]1")
    charged = _mol("[O-:1][N+:2](=[O:3])[H:4]")
    chiral = _mol("[C@H:1]([F:2])([Cl:3])[Br:4]")

    # When: each is flattened with the only permitted skeleton recipe
    flat_aromatic = flatten_skeleton(aromatic)
    flat_charged = flatten_skeleton(charged)
    flat_chiral = flatten_skeleton(chiral)

    # Then: every bond is a single bond, no atom is aromatic or charged, the
    # chiral tag is gone, and atom count plus map numbers are preserved
    for flat in (flat_aromatic, flat_charged, flat_chiral):
        assert all(
            bond.GetBondType() == Chem.BondType.SINGLE
            and not bond.GetIsAromatic()
            for bond in flat.GetBonds()
        )
        assert all(
            atom.GetFormalCharge() == 0
            and not atom.GetIsAromatic()
            and atom.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED
            for atom in flat.GetAtoms()
        )
    assert flat_aromatic.GetNumAtoms() == aromatic.GetNumAtoms()
    assert [
        atom.GetAtomMapNum() for atom in flat_charged.GetAtoms()
    ] == [atom.GetAtomMapNum() for atom in charged.GetAtoms()]


def test_methane_enumerates_all_24_symmetries() -> None:
    # Given: a methane reaction component against the inventory methane record
    # (both sides flatten to a carbon bonded to four explicit hydrogens)
    # When
    matches = match_side_components(
        [_mol("[C:11]([H:7])([H:8])([H:9])[H:10]")],
        [METHANE_RXN_MAPS],
        [METHANE_INVENTORY],
        match_cap=10000,
        max_candidates=64,
    )

    # Then: the carbon is unique (query index 0 -> row 0) and the four
    # equivalent hydrogens give the full S4 automorphism group, 4! = 24
    assert len(matches) == 1
    match = matches[0]
    assert match.rxn_index == 0
    assert match.tag == "R0"
    assert match.tag_candidates == ("R0",)
    assert match.n_isomorphisms == 24
    assert match.truncated is False
    assert len(match.isomorphisms) == 24
    assert match.isomorphisms[0] == (0, 1, 2, 3, 4)
    assert all(iso[0] == 0 and sorted(iso) == [0, 1, 2, 3, 4] for iso in match.isomorphisms)

    # And: the index table reports the ambiguity without losing solutions
    indexes = build_side_indexes(matches, [METHANE_INVENTORY], max_candidates=64)
    assert len(indexes) == 1
    assert indexes[0].status == "symmetric_ambiguous"
    assert indexes[0].n_candidates == 24


def test_asymmetric_component_yields_exactly_one_isomorphism() -> None:
    # Given: a component whose graph automorphism group is trivial
    # When
    matches = match_side_components(
        [_mol("[F:21][C:22]([Cl:23])([H:24])[C:25]([Br:26])([H:27])[O:28][H:29]")],
        [(21, 22, 23, 24, 25, 26, 27, 28, 29)],
        [ASYMMETRIC_INVENTORY],
        match_cap=10000,
        max_candidates=64,
    )

    # Then: exactly one solution, no truncation, unique status
    assert matches[0].n_isomorphisms == 1
    assert matches[0].truncated is False
    assert matches[0].isomorphisms == ((0, 1, 2, 3, 4, 5, 6, 7, 8),)
    indexes = build_side_indexes(matches, [ASYMMETRIC_INVENTORY], max_candidates=64)
    assert indexes[0].status == "unique"


def test_match_cap_truncates_and_marks_methane() -> None:
    # Given: the 24-fold symmetric methane with a match cap of four
    # When
    matches = match_side_components(
        [_mol("[C:11]([H:7])([H:8])([H:9])[H:10]")],
        [METHANE_RXN_MAPS],
        [METHANE_INVENTORY],
        match_cap=4,
        max_candidates=64,
    )

    # Then: the enumeration stops at the cap and says so; every stored
    # solution is a complete bijection over the five rows
    assert matches[0].n_isomorphisms == 4
    assert matches[0].truncated is True
    assert len(matches[0].isomorphisms) == 4
    assert all(sorted(iso) == [0, 1, 2, 3, 4] for iso in matches[0].isomorphisms)
    indexes = build_side_indexes(matches, [METHANE_INVENTORY], max_candidates=64)
    assert indexes[0].status == "truncated"


def test_exact_cap_hit_is_conservatively_truncated() -> None:
    # Given: methane with the cap set to its exact solution count
    # When
    matches = match_side_components(
        [_mol("[C:11]([H:7])([H:8])([H:9])[H:10]")],
        [METHANE_RXN_MAPS],
        [METHANE_INVENTORY],
        match_cap=24,
        max_candidates=64,
    )

    # Then: enumeration reaching the cap is reported as truncated even when it
    # may have been exact (the documented conservative semantics)
    assert matches[0].n_isomorphisms == 24
    assert matches[0].truncated is True


def test_max_candidates_limits_stored_solutions_only() -> None:
    # Given: methane with full enumeration but room for only two stored solutions
    # When
    matches = match_side_components(
        [_mol("[C:11]([H:7])([H:8])([H:9])[H:10]")],
        [METHANE_RXN_MAPS],
        [METHANE_INVENTORY],
        match_cap=10000,
        max_candidates=2,
    )

    # Then: the count is the full 24, only two solutions are kept, and the
    # component is still ambiguous (not truncated)
    assert matches[0].n_isomorphisms == 24
    assert matches[0].truncated is False
    assert len(matches[0].isomorphisms) == 2
    indexes = build_side_indexes(matches, [METHANE_INVENTORY], max_candidates=2)
    assert indexes[0].n_candidates == 2
    assert indexes[0].status == "symmetric_ambiguous"


def test_pairing_prefers_the_lexicographically_smallest_unused_tag() -> None:
    # Given: two identical water inventory records whose input order is
    # reversed (R1 before R0) and a side with two identical water components
    inventory = [dict(WATER_INVENTORY, tag="R1"), dict(WATER_INVENTORY, tag="R0")]
    water_maps = (1, 2, 3)

    # When
    matches = match_side_components(
        [_mol("[H:5][O:4][H:6]"), _mol("[H:8][O:7][H:9]")],
        [water_maps, water_maps],
        inventory,
        match_cap=10000,
        max_candidates=8,
    )

    # Then: the first component records both candidate tags and takes R0; the
    # second can no longer see R0 and pairs with R1
    assert [match.tag for match in matches] == ["R0", "R1"]
    assert matches[0].tag_candidates == ("R0", "R1")
    assert matches[1].tag_candidates == ("R1",)
    assert all(match.n_isomorphisms == 2 for match in matches)


def test_substructure_lookalike_is_not_a_pairing_candidate() -> None:
    # Given: inventory [R0 hydrazine (6 atoms), R1 dinitrogen (2 atoms)] and a
    # side [N2, hydrazine]; the N2 skeleton appears inside hydrazine's N-N
    # bond, so a raw substructure search would offer R0 to the smaller
    # component and then strand the larger one
    hydrazine: dict[str, Any] = {
        "tag": "R0",
        "smiles": "[N:1]([H:2])([H:3])[N:4]([H:5])[H:6]",
        "atomic_numbers": [7, 1, 1, 7, 1, 1],
        "coordinates": [[0.0, 0.0, 0.0]] * 6,
    }
    dinitrogen: dict[str, Any] = {
        "tag": "R1",
        "smiles": "[N:1]#[N:2]",
        "atomic_numbers": [7, 7],
        "coordinates": [[0.0, 0.0, 0.0]] * 2,
    }

    # When
    matches = match_side_components(
        [_mol("[N:5]#[N:6]"), _mol("[N:11]([H:12])([H:13])[N:14]([H:15])[H:16]")],
        [(5, 6), (11, 12, 13, 14, 15, 16)],
        [hydrazine, dinitrogen],
        match_cap=10000,
        max_candidates=4,
    )

    # Then: pairing keeps the same-size bijections -- N2 -> R1, hydrazine ->
    # R0 -- even though R0 contains the smaller skeleton as a substructure
    assert [match.tag for match in matches] == ["R1", "R0"]
    assert matches[0].tag_candidates == ("R1",)
    assert matches[0].n_isomorphisms == 2
    raw = flatten_skeleton(_mol(hydrazine["smiles"])).GetSubstructMatches(
        flatten_skeleton(_mol(dinitrogen["smiles"])),
        uniquify=False,
        useChirality=False,
        maxMatches=100,
    )
    assert len(raw) == 2


def test_component_mismatch_error_carries_the_rxn_index() -> None:
    # Given: inventory water and methane, and a side of water plus ethanol
    inventory = [dict(WATER_INVENTORY, tag="R0"), dict(METHANE_INVENTORY, tag="R1")]
    ethanol = "[C:11]([H:12])([H:13])([O:14][H:15])[H:16]"

    # When / Then: the second component cannot match and names itself
    with pytest.raises(ComponentMismatchError) as excinfo:
        match_side_components(
            [_mol("[H:5][O:4][H:6]"), _mol(ethanol)],
            [(1, 2, 3), (11, 12, 13, 14, 15, 16)],
            inventory,
            match_cap=10000,
            max_candidates=8,
        )
    assert excinfo.value.rxn_index == 1
    assert excinfo.value.args[0].startswith("reaction-side component 1:")

    # And: a single non-matching component is reported at index 0
    with pytest.raises(ComponentMismatchError) as first:
        match_side_components(
            [_mol(ethanol)],
            [(11, 12, 13, 14, 15, 16)],
            inventory,
            match_cap=10000,
            max_candidates=8,
        )
    assert first.value.rxn_index == 0
    assert isinstance(first.value, ValueError)


def test_map_labels_select_the_coordinate_rows() -> None:
    # Given: water whose SMILES writes map 3 on its first atom, so atom order
    # and row order differ (map k lives at row k - 1)
    inventory = WATER_PERMUTED_INVENTORY

    # When
    matches = match_side_components(
        [_mol("[H:5][O:4][H:6]")],
        [(5, 4, 6)],
        [inventory],
        match_cap=10000,
        max_candidates=8,
    )
    again = match_side_components(
        [_mol("[H:5][O:4][H:6]")],
        [(5, 4, 6)],
        [inventory],
        match_cap=10000,
        max_candidates=8,
    )

    # Then: the matching oxygen is row 0 and each hydrogen lands on its own
    # row, not on its SMILES atom index; enumeration order is deterministic
    assert matches[0].isomorphisms == ((2, 0, 1), (1, 0, 2))
    assert again[0].isomorphisms == matches[0].isomorphisms

    indexes = build_side_indexes(matches, [inventory], max_candidates=8)
    assert indexes[0].element_check is True
    assert [row["map"] for row in indexes[0].rows] == [1, 2, 3]
    assert [row["local_index"] for row in indexes[0].rows] == [0, 1, 2]
    assert [row["element"] for row in indexes[0].rows] == ["O", "H", "H"]
    assert indexes[0].candidates == (
        ((3, 2), (1, 0), (2, 1)),
        ((2, 1), (1, 0), (3, 2)),
    )


def test_global_index_accumulates_in_tag_order() -> None:
    # Given: a water and a methane inventory component on one side
    inventory = [dict(WATER_INVENTORY, tag="R0"), dict(METHANE_INVENTORY, tag="R1")]
    matches = match_side_components(
        [_mol("[H:5][O:4][H:6]"), _mol("[C:11]([H:7])([H:8])([H:9])[H:10]")],
        [(1, 2, 3), (11, 7, 8, 9, 10)],
        inventory,
        match_cap=10000,
        max_candidates=8,
    )

    # When: the index tables are built (R0 = 3 atoms, R1 = 5 atoms)
    indexes = build_side_indexes(matches, inventory, max_candidates=8)
    reversed_indexes = build_side_indexes(
        list(reversed(matches)), inventory, max_candidates=8
    )

    # Then: bases accumulate in tag order regardless of the match list order
    assert [index.tag for index in indexes] == ["R0", "R1"]
    assert [index.tag for index in reversed_indexes] == ["R0", "R1"]
    assert (indexes[0].index_base, indexes[0].n_atoms) == (0, 3)
    assert (indexes[1].index_base, indexes[1].n_atoms) == (3, 5)
    assert [row["global_index"] for row in indexes[0].rows] == [0, 1, 2]
    assert [row["global_index"] for row in indexes[1].rows] == [3, 4, 5, 6, 7]
    assert [row["element"] for row in indexes[1].rows] == ["C", "H", "H", "H", "H"]
    assert indexes == reversed_indexes


def test_candidates_are_full_bijections_in_query_order() -> None:
    # Given: methane with room for three stored solutions
    matches = match_side_components(
        [_mol("[C:11]([H:7])([H:8])([H:9])[H:10]")],
        [METHANE_RXN_MAPS],
        [METHANE_INVENTORY],
        match_cap=10000,
        max_candidates=3,
    )

    # When
    indexes = build_side_indexes(matches, [METHANE_INVENTORY], max_candidates=3)
    index = indexes[0]

    # Then: each candidate is a complete map <-> local bijection in query
    # order, transcribed from the enumeration order without re-sorting
    assert len(index.candidates) == 3
    for position, candidate in enumerate(index.candidates):
        assert len(candidate) == 5
        assert sorted(pair[0] for pair in candidate) == [1, 2, 3, 4, 5]
        assert sorted(pair[1] for pair in candidate) == [0, 1, 2, 3, 4]
        assert candidate == tuple(
            (row + 1, row) for row in matches[0].isomorphisms[position]
        )
    assert index.candidates[0] == ((1, 0), (2, 1), (3, 2), (4, 3), (5, 4))


def test_element_check_detects_reordered_atomic_numbers() -> None:
    # Given: the methane record with its atomic-number sequence reordered so
    # the atom carrying map k no longer matches atomic_numbers[k - 1]
    reordered = dict(METHANE_INVENTORY, atomic_numbers=[1, 1, 6, 1, 1])
    matches = match_side_components(
        [_mol("[C:11]([H:7])([H:8])([H:9])[H:10]")],
        [METHANE_RXN_MAPS],
        [reordered],
        match_cap=10000,
        max_candidates=8,
    )

    # When
    bad = build_side_indexes(matches, [reordered], max_candidates=8)
    good = build_side_indexes(matches, [METHANE_INVENTORY], max_candidates=8)

    # Then: the cross-check flags the bad record and passes the good one, and
    # a missing map number (not a 1..n permutation) is flagged as well
    assert bad[0].element_check is False
    assert good[0].element_check is True
    incomplete = dict(METHANE_INVENTORY, smiles="[C:1]([H:2])([H:3])([H:4])[H:6]")
    assert build_side_indexes(
        matches, [incomplete], max_candidates=8
    )[0].element_check is False


def test_bond_geometry_check_accepts_tetrahedral_methane() -> None:
    # Given: the tetrahedral methane record (all C-H bonds 1.09 Angstrom)
    # When
    ok, detail = bond_geometry_check(METHANE_INVENTORY, tolerance=0.45)

    # Then
    assert ok is True
    assert detail == ""


def test_bond_geometry_check_reports_the_worst_offender() -> None:
    # Given: methane with two hydrogens dragged far away (row1 at 3 A, row2
    # at 5 A), so both (1, 2) and (1, 3) exceed the radius-sum limit
    stretched = dict(
        METHANE_INVENTORY,
        coordinates=[
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [-TETRAHEDRAL, TETRAHEDRAL, -TETRAHEDRAL],
            [-TETRAHEDRAL, -TETRAHEDRAL, TETRAHEDRAL],
        ],
    )

    # When
    ok, detail = bond_geometry_check(stretched, tolerance=0.45)

    # Then: the larger excess (the 5 A pair) is the one reported
    assert ok is False
    assert "(1, 3)" in detail
    assert "(1, 2)" not in detail
    assert "5.000" in detail

    # And: a generous tolerance accepts the same coordinates
    relaxed, relaxed_detail = bond_geometry_check(stretched, tolerance=5.0)
    assert relaxed is True
    assert relaxed_detail == ""


def test_bond_geometry_check_rejects_unmapped_bond_maps() -> None:
    # Given: a component whose hydrogen carries no map number
    unmapped = {
        "tag": "R0",
        "smiles": "[C:1][H]",
        "atomic_numbers": [6, 1],
        "coordinates": [[0.0, 0.0, 0.0], [1.09, 0.0, 0.0]],
    }

    # When
    ok, detail = bond_geometry_check(unmapped, tolerance=0.45)

    # Then
    assert ok is False
    assert "outside" in detail


def test_matching_is_representation_insensitive() -> None:
    # Given: stereo, aromaticity, and charge/bond-order depictions that differ
    # between the inventory record and the reaction-side component
    cases: list[tuple[dict[str, Any], str, tuple[int, ...], int]] = [
        (CHIRAL_INVENTORY, "[CH:5]([F:6])([Cl:7])[Br:8]", (5, 6, 7, 8), 1),
        (
            BENZENE_INVENTORY,
            "[CH:7]1=[CH:8][CH:9]=[CH:10][CH:11]=[CH:12]1",
            (7, 8, 9, 10, 11, 12),
            12,
        ),
        (
            NITRO_INVENTORY,
            "[O:5]=[N:6](=[O:7])[H:8]",
            (5, 6, 7, 8),
            2,
        ),
    ]

    # When / Then: flattening both sides matches every pair despite the
    # chirality tag, the aromatic/Kekule bond orders, and the charges
    for record, smiles, maps, expected in cases:
        matches = match_side_components(
            [_mol(smiles)],
            [maps],
            [record],
            match_cap=10000,
            max_candidates=64,
        )
        assert matches[0].tag == "R0"
        assert matches[0].n_isomorphisms == expected


def test_covalent_radii_cover_the_expected_elements() -> None:
    # Given / When / Then: the Cordero table covers Z = 1..86 with the
    # light-element anchors the geometry gate depends on
    assert sorted(COVALENT_RADII) == list(range(1, 87))
    assert COVALENT_RADII[1] == 0.31
    assert COVALENT_RADII[6] == 0.76
    assert COVALENT_RADII[8] == 0.66
    assert COVALENT_RADII[16] == 1.05


def test_matches_and_indexes_are_deterministic() -> None:
    # Given: the same side matched twice
    inventory = [dict(WATER_INVENTORY, tag="R0"), dict(METHANE_INVENTORY, tag="R1")]
    args = (
        [_mol("[H:5][O:4][H:6]"), _mol("[C:11]([H:7])([H:8])([H:9])[H:10]")],
        [(1, 2, 3), (11, 7, 8, 9, 10)],
        inventory,
    )

    # When
    first = match_side_components(*args, match_cap=64, max_candidates=8)
    second = match_side_components(*args, match_cap=64, max_candidates=8)

    # Then: both the matches and the derived index tables are byte-equal
    assert first == second
    assert build_side_indexes(first, inventory, max_candidates=8) == (
        build_side_indexes(second, inventory, max_candidates=8)
    )
    assert all(isinstance(match, ComponentMatch) for match in first)
    assert all(
        isinstance(index, ComponentIndex)
        for index in build_side_indexes(first, inventory, max_candidates=8)
    )
