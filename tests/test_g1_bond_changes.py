"""Offline tests for the G1 atom-map-space bond-change detail.

Every fixture is a synthetic mapped reaction SMILES.  The tests parse both
sides exactly as the parse stage does (``removeHs=False`` on
:class:`rdkit.Chem.SmilesParserParams`), run :func:`compute_bond_changes`, and
cross-check every fixture against the real imported G0 preview function, so
any semantic drift between the authoritative detail and the documented preview
semantics fails the suite.
"""

from __future__ import annotations

from collections.abc import Mapping

import pytest
from rdkit import Chem

from pes2ts_core.g0.strata import compute_bond_changes_preview
from pes2ts_core.g1.bond_changes import (
    BondChanges,
    ChangedBond,
    compute_bond_changes,
    crosscheck_against_preview,
    reaction_center,
)

#: Two methyl radicals combine; nothing breaks.
PURE_FORMATION = "[CH3:1].[CH3:2]>>[CH3:1][CH3:2]"
#: The written reverse: only the C1-C2 bond breaks.
PURE_BREAKING = "[CH3:1][CH3:2]>>[CH3:1].[CH3:2]"
#: Ethane -> ethene: one map pair moves 1 -> 2, so all three lists are non-empty.
ORDER_CHANGE = "[CH3:1][CH3:2]>>[CH2:1]=[CH2:2]"
#: Ethane-style order move 1 -> 3: still one broken and one formed key.
TRIPLE_BOND = "[CH3:1][CH3:2]>>[C:1]#[C:2]"
#: Methane + water -> methanol + H2: H3 moves C1 -> O2 and H7/H8 recombine.
H_MIGRATION = (
    "[C:1]([H:3])([H:4])([H:5])[H:6].[O:2]([H:7])[H:8]"
    ">>[C:1]([H:4])([H:5])([H:6])[O:2][H:3].[H:7][H:8]"
)
#: Benzene + H2 -> 1,3-cyclohexadiene: every ring bond changes (AROMATIC 1.5).
AROMATIC = (
    "[cH:1]1[cH:2][cH:3][cH:4][cH:5][cH:6]1.[H:7][H:8]"
    ">>[C:1]([H:7])([H:8])1[CH:2]=[CH:3][CH:4]=[CH:5][CH2:6]1"
)
#: Four-carbon chain whose terminal bond changes order (shell expansion test).
CHAIN = "[CH3:1][CH2:2][CH2:3][CH3:4]>>[CH2:1]=[CH:2][CH2:3][CH3:4]"
#: Radical recombination: the C2-C3 edge exists only on the product side.
CHAIN_EXTENSION = "[CH3:1].[CH2:2][CH3:3]>>[CH3:1][CH2:2][CH3:3]"
#: The written reverse: the C2-C3 edge exists only on the reactant side.
CHAIN_BREAKAGE = "[CH3:1][CH2:2][CH3:3]>>[CH3:1].[CH2:2][CH3:3]"
#: H-H formation and dissociation (isolated H on one side).
H2_FORMATION = "[H:1].[H:2]>>[H:1][H:2]"
H2_DISSOCIATION = "[H:1][H:2]>>[H:1].[H:2]"
#: Fully mapped but unchanged reaction (zero changes on every axis).
UNCHANGED = "[CH3:1][OH:2]>>[CH3:1][OH:2]"
#: SN2 step: C-Br breaks, C-O forms; both sides hold two components.
SN2 = "[CH3:1][Br:2].[O-:3]>>[CH3:1][O-:3].[Br-:2]"
#: The aromatic ring map pairs of :data:`AROMATIC`.
AROMATIC_RING_PAIRS = {(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (1, 6)}


def _parse_side(side_text: str) -> tuple[list[Chem.Mol], list[list[int]]]:
    """Parse one dot-separated side the way the parse stage does."""
    params = Chem.SmilesParserParams()
    params.removeHs = False
    mols: list[Chem.Mol] = []
    map_lists: list[list[int]] = []
    for component in side_text.split("."):
        mol = Chem.MolFromSmiles(component, params)  # pyright: ignore[reportUnknownMemberType]
        assert mol is not None, component
        mols.append(mol)
        map_lists.append(
            [atom.GetAtomMapNum() for atom in mol.GetAtoms()]  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        )
    return mols, map_lists


def _compute(reaction_smiles: str) -> BondChanges:
    """Compute the bond-change detail of a mapped reaction SMILES."""
    reactant_text, product_text = reaction_smiles.split(">>")
    r_mols, r_maps = _parse_side(reactant_text)
    p_mols, p_maps = _parse_side(product_text)
    return compute_bond_changes(r_mols, p_mols, r_maps, p_maps)


def _assert_preview_agrees(reaction_smiles: str, changes: BondChanges) -> None:
    """Assert the detail counts equal the real imported preview counters."""
    row = {"reaction_smiles": reaction_smiles}
    preview = compute_bond_changes_preview(row)
    assert preview == {
        "n_bonds_formed": len(changes["formed"]),
        "n_bonds_broken": len(changes["broken"]),
        "n_bond_order_changed": len(changes["order_changed"]),
    }
    assert crosscheck_against_preview(row, changes) is True


def test_pure_formation_reports_only_formed_bonds() -> None:
    # Given / When
    changes = _compute(PURE_FORMATION)

    # Then: exactly the new C1-C2 bond; nothing broke and nothing reordered
    assert changes["formed"] == [ChangedBond((1, 2), None, 1.0)]
    assert changes["broken"] == []
    assert changes["order_changed"] == []
    assert changes["hydrogen_migration"] == []
    _assert_preview_agrees(PURE_FORMATION, changes)


def test_pure_breaking_reports_only_broken_bonds() -> None:
    # Given / When
    changes = _compute(PURE_BREAKING)

    # Then: exactly the C1-C2 bond break
    assert changes["formed"] == []
    assert changes["broken"] == [ChangedBond((1, 2), 1.0, None)]
    assert changes["order_changed"] == []
    assert changes["hydrogen_migration"] == []
    _assert_preview_agrees(PURE_BREAKING, changes)


def test_order_change_appears_in_all_three_lists() -> None:
    # Given: ethane -> ethene (same map pair, order 1 -> 2)
    changes = _compute(ORDER_CHANGE)

    # Then: the documented non-exclusive semantics -- one broken key, one formed
    # key, one order-changed pair for the same map pair
    assert changes["formed"] == [ChangedBond((1, 2), None, 2.0)]
    assert changes["broken"] == [ChangedBond((1, 2), 1.0, None)]
    assert changes["order_changed"] == [ChangedBond((1, 2), 1.0, 2.0)]
    assert changes["hydrogen_migration"] == []
    _assert_preview_agrees(ORDER_CHANGE, changes)


def test_triple_bond_formation_is_an_order_change() -> None:
    # Given / When
    changes = _compute(TRIPLE_BOND)

    # Then
    assert changes["formed"] == [ChangedBond((1, 2), None, 3.0)]
    assert changes["broken"] == [ChangedBond((1, 2), 1.0, None)]
    assert changes["order_changed"] == [ChangedBond((1, 2), 1.0, 3.0)]
    _assert_preview_agrees(TRIPLE_BOND, changes)


def test_hydrogen_migration_moves_h_from_carbon_to_oxygen() -> None:
    # Given: methane + water -> methanol + H2, with two components on each side
    reactant_text, product_text = H_MIGRATION.split(">>")
    r_mols, _ = _parse_side(reactant_text)
    p_mols, _ = _parse_side(product_text)
    assert len(r_mols) == len(p_mols) == 2

    # When
    changes = _compute(H_MIGRATION)

    # Then: H3 leaves the methane carbon (1) for the water oxygen (2); H7/H8
    # leave the oxygen for the new H-H bond; the C-H distances stay put
    assert changes["hydrogen_migration"] == [
        {"h": 3, "from": 1, "to": 2},
        {"h": 7, "from": 2, "to": 8},
        {"h": 8, "from": 2, "to": 7},
    ]
    assert changes["formed"] == [
        ChangedBond((1, 2), None, 1.0),
        ChangedBond((2, 3), None, 1.0),
        ChangedBond((7, 8), None, 1.0),
    ]
    assert changes["broken"] == [
        ChangedBond((1, 3), 1.0, None),
        ChangedBond((2, 7), 1.0, None),
        ChangedBond((2, 8), 1.0, None),
    ]
    assert changes["order_changed"] == []
    _assert_preview_agrees(H_MIGRATION, changes)


def test_hh_bond_changes_mark_isolated_hydrogen_partners() -> None:
    # Given / When
    formed = _compute(H2_FORMATION)
    broken = _compute(H2_DISSOCIATION)

    # Then: an isolated H reports None as its partner instead of raising
    assert formed["hydrogen_migration"] == [
        {"h": 1, "from": None, "to": 2},
        {"h": 2, "from": None, "to": 1},
    ]
    assert broken["hydrogen_migration"] == [
        {"h": 1, "from": 2, "to": None},
        {"h": 2, "from": 1, "to": None},
    ]
    _assert_preview_agrees(H2_FORMATION, formed)
    _assert_preview_agrees(H2_DISSOCIATION, broken)


def test_aromatic_ring_bonds_stay_one_point_five() -> None:
    # Given: benzene + H2 -> 1,3-cyclohexadiene
    changes = _compute(AROMATIC)

    # Then: no side was kekulized -- every broken ring key carries order 1.5,
    # and every order-changed ring pair has the aromatic order on the
    # reactant side
    broken_ring = {
        bond.atoms: bond.order_r
        for bond in changes["broken"]
        if bond.atoms in AROMATIC_RING_PAIRS
    }
    assert broken_ring == dict.fromkeys(AROMATIC_RING_PAIRS, 1.5)
    changed_ring = {bond.atoms: (bond.order_r, bond.order_p) for bond in changes["order_changed"]}
    assert changed_ring == {
        (1, 2): (1.5, 1.0),
        (2, 3): (1.5, 2.0),
        (3, 4): (1.5, 1.0),
        (4, 5): (1.5, 2.0),
        (5, 6): (1.5, 1.0),
        (1, 6): (1.5, 1.0),
    }
    assert len(changes["formed"]) == 8
    assert len(changes["broken"]) == 7
    assert changes["hydrogen_migration"] == [
        {"h": 7, "from": 8, "to": 1},
        {"h": 8, "from": 7, "to": 1},
    ]
    _assert_preview_agrees(AROMATIC, changes)


def test_multi_component_sn2_pairs_changes_in_global_map_space() -> None:
    # Given / When
    changes = _compute(SN2)

    # Then: the C1-Br2 bond breaks and the C1-O3 bond forms across components
    assert changes["formed"] == [ChangedBond((1, 3), None, 1.0)]
    assert changes["broken"] == [ChangedBond((1, 2), 1.0, None)]
    assert changes["order_changed"] == []
    assert changes["hydrogen_migration"] == []
    _assert_preview_agrees(SN2, changes)


def test_unchanged_reaction_reports_zero_and_preview_agrees() -> None:
    # Given / When
    changes = _compute(UNCHANGED)

    # Then: every axis is empty and the real preview reports 0, 0, 0
    assert changes["formed"] == []
    assert changes["broken"] == []
    assert changes["order_changed"] == []
    assert changes["hydrogen_migration"] == []
    assert compute_bond_changes_preview({"reaction_smiles": UNCHANGED}) == {
        "n_bonds_formed": 0,
        "n_bonds_broken": 0,
        "n_bond_order_changed": 0,
    }
    _assert_preview_agrees(UNCHANGED, changes)


def test_reaction_center_shell_expands_through_union_graph() -> None:
    # Given: C1=C2-C3-C4, so the changed pair (1, 2) has C3 one bond away
    changes = _compute(CHAIN)

    # When / Then: each shell ring pulls in exactly one more chain atom
    assert reaction_center(changes, 0) == {"core": [1, 2], "with_shell": [1, 2]}
    assert reaction_center(changes, 1) == {"core": [1, 2], "with_shell": [1, 2, 3]}
    assert reaction_center(changes, 2) == {
        "core": [1, 2],
        "with_shell": [1, 2, 3, 4],
    }
    _assert_preview_agrees(CHAIN, changes)


def test_reaction_center_reads_edges_from_either_side() -> None:
    # Given: a C2-C3 edge that exists only on the product side (extension),
    # and its written reverse where it exists only on the reactant side
    extension = _compute(CHAIN_EXTENSION)
    breakage = _compute(CHAIN_BREAKAGE)

    # When / Then: the union graph supplies the unchanged edge either way
    assert reaction_center(extension, 0) == {"core": [1, 2], "with_shell": [1, 2]}
    assert reaction_center(extension, 1) == {"core": [1, 2], "with_shell": [1, 2, 3]}
    assert reaction_center(breakage, 1) == {"core": [1, 2], "with_shell": [1, 2, 3]}
    _assert_preview_agrees(CHAIN_EXTENSION, extension)
    _assert_preview_agrees(CHAIN_BREAKAGE, breakage)


def test_side_components_and_map_lists_must_align() -> None:
    # Given: two components but only one map list for the reactant side
    mols, maps = _parse_side(PURE_FORMATION.split(">>")[0])

    # When / Then: the misaligned input is a programming error, not a silent diff
    with pytest.raises(ValueError, match="zip"):
        _ = compute_bond_changes(mols, mols, maps[:1], maps)


def test_crosscheck_rejects_the_unavailable_sentinel() -> None:
    # Given: an unmapped reaction (preview sentinel) and a zero-change detail
    row = {"reaction_smiles": "CCO>>CC=O"}
    changes = _compute(UNCHANGED)

    # When / Then: the preview cannot be compared and the crosscheck refuses
    assert compute_bond_changes_preview(row) == {
        "n_bonds_formed": -1,
        "n_bonds_broken": -1,
        "n_bond_order_changed": -1,
    }
    assert crosscheck_against_preview(row, changes) is False


def test_crosscheck_reports_false_when_detail_counts_drift() -> None:
    # Given: a genuine change and one extra formed entry not backed by the graph
    changes = _compute(ORDER_CHANGE)
    changes["formed"].append(ChangedBond((7, 8), None, 1.0))

    # When / Then
    assert crosscheck_against_preview({"reaction_smiles": ORDER_CHANGE}, changes) is False


def test_crosscheck_uses_the_imported_preview_function(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a recording stub installed where the module imported the preview
    changes = _compute(ORDER_CHANGE)
    seen: list[Mapping[str, object]] = []

    def _stub(record: Mapping[str, object]) -> dict[str, int]:
        seen.append(record)
        return {"n_bonds_formed": 1, "n_bonds_broken": 1, "n_bond_order_changed": 1}

    monkeypatch.setattr(
        "pes2ts_core.g1.bond_changes.compute_bond_changes_preview", _stub
    )
    row = {"reaction_smiles": ORDER_CHANGE}

    # When
    agreed = crosscheck_against_preview(row, changes)

    # Then: the counters came from the imported function, not a local reimplementation
    assert agreed is True
    assert seen == [row]


def test_output_lists_are_sorted_and_repeatable() -> None:
    # Given / When: the many-entry aromatic fixture is computed twice
    first = _compute(AROMATIC)
    second = _compute(AROMATIC)

    # Then: byte-stable output with every list in documented order
    assert first == second
    for key in ("formed", "broken", "order_changed"):
        pairs = [bond.atoms for bond in first[key]]
        assert pairs == sorted(pairs)
    hydrogen_order = [entry["h"] for entry in first["hydrogen_migration"]]
    assert hydrogen_order == sorted(hydrogen_order)
