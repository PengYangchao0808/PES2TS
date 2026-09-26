"""Offline tests for the G1 mapped-SMILES parser and consistency checks.

Fixtures are hand-written mapped reactions: a multi-component esterification,
an aromatic ring, and the explicit-hydrogen ammonium fragment RDKit would
strip under default sanitization.  Every failure-taxonomy branch of
:mod:`pes2ts_core.g1.parse` is exercised, and charge/spin row re-checks and
map-set mismatch detection are locked.  No network and no real data are
needed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import pytest
from rdkit import Chem

from pes2ts_core.g1.parse import (
    REASON_DUPLICATE_MAP,
    REASON_EMPTY_COMPONENT,
    REASON_EMPTY_SIDE,
    REASON_MULTIPLE_ARROWS,
    REASON_NO_ARROW,
    REASON_UNMAPPED_ATOM,
    REASON_UNPARSEABLE_COMPONENT,
    REASON_ZERO_ATOM_COMPONENT,
    ReactionParseError,
    _parse_component,
    parse_reaction,
    parse_side,
    row_consistency_checks,
    side_map_checks,
)

#: Acetic acid + methanol -> methyl acetate + water, fully mapped; a two
#: component side on the reactants and a five-plus-one atom split on products.
ESTER_REACTANTS = "[CH3:1][C:2](=[O:3])[OH:4].[CH3:5][OH:6]"
ESTER_PRODUCTS = "[CH3:1][C:2](=[O:3])[O:4][CH3:5].[OH2:6]"
ESTER_SMILES = f"{ESTER_REACTANTS}>>{ESTER_PRODUCTS}"
#: Benzene with every ring atom mapped; aromatic bonds must stay 1.5.
BENZENE_MAPPED = "[c:1]1[cH:2][cH:3][cH:4][cH:5][cH:6]1"
#: Ammonium fragment with explicit mapped hydrogens: 3 atoms, not 1.
MAPPED_H_FRAGMENT = "[N+:4]([H:8])[H:9]"
#: Neutral singlet inventory row fields.
CONSISTENT_ROW: Mapping[str, Any] = {
    "charge_total_reactants": 0,
    "charge_total_products": 0,
    "multiplicity_max": 1,
}


def _side_reason(side_text: str) -> str:
    """Return the :class:`ReactionParseError` reason for a bad side."""
    with pytest.raises(ReactionParseError) as excinfo:
        parse_side(side_text)
    return excinfo.value.reason


def test_reaction_parse_error_is_a_value_error_with_a_reason() -> None:
    # Given / When
    error = ReactionParseError("some_reason")

    # Then: the contract is a ValueError whose message is the reason token
    assert isinstance(error, ValueError)
    assert error.reason == "some_reason"
    assert str(error) == "some_reason"


def test_parse_side_returns_components_mols_and_map_lists() -> None:
    # Given / When: surrounding whitespace is not part of the component
    side = parse_side(" [CH3:1][OH:2] ")

    # Then
    assert side.components == ("[CH3:1][OH:2]",)
    assert len(side.mols) == 1
    assert side.mols[0].GetNumAtoms() == 2
    assert side.map_lists == ((1, 2),)


def test_parse_side_preserves_mapped_explicit_hydrogens() -> None:
    # Given: an N-H fragment whose Hs would be dropped by default sanitization
    # When
    side = parse_side(MAPPED_H_FRAGMENT)

    # Then: all three atoms survive with their map numbers in atom order
    assert side.mols[0].GetNumAtoms() == 3
    assert side.map_lists == ((4, 8, 9),)


def test_parse_side_preserves_aromatic_bond_orders() -> None:
    # Given: a fully mapped aromatic ring
    # When
    side = parse_side(BENZENE_MAPPED)

    # Then: no Kekulization -- every ring bond is still aromatic (1.5)
    mol = side.mols[0]
    assert mol.GetNumAtoms() == 6
    assert side.map_lists == ((1, 2, 3, 4, 5, 6),)
    assert [bond.GetBondTypeAsDouble() for bond in mol.GetBonds()] == [1.5] * 6


def test_parse_side_keeps_component_order() -> None:
    # Given / When
    side = parse_side(ESTER_REACTANTS)

    # Then
    assert side.components == (
        "[CH3:1][C:2](=[O:3])[OH:4]",
        "[CH3:5][OH:6]",
    )
    assert [mol.GetNumAtoms() for mol in side.mols] == [4, 2]
    assert side.map_lists == ((1, 2, 3, 4), (5, 6))


@pytest.mark.parametrize(
    ("side_text", "expected_reason"),
    [
        pytest.param("", REASON_EMPTY_SIDE, id="empty"),
        pytest.param("   ", REASON_EMPTY_SIDE, id="blank"),
        pytest.param("[CH3:1].", REASON_EMPTY_COMPONENT, id="trailing-dot"),
        pytest.param(".[CH3:1]", REASON_EMPTY_COMPONENT, id="leading-dot"),
        pytest.param("[CH3:1]..[OH:2]", REASON_EMPTY_COMPONENT, id="double-dot"),
        pytest.param("C%^&", REASON_UNPARSEABLE_COMPONENT, id="rdkit-none"),
        pytest.param(
            "[CH3:1]C%^&", REASON_UNPARSEABLE_COMPONENT, id="rdkit-none-mapped"
        ),
        pytest.param("CC", REASON_UNMAPPED_ATOM, id="unmapped"),
        pytest.param("[CH3:1]CO", REASON_UNMAPPED_ATOM, id="partially-mapped"),
        pytest.param(
            "[CH3:1][OH:1]", REASON_DUPLICATE_MAP, id="duplicate-in-one-component"
        ),
        pytest.param(
            "[CH3:1].[OH:1]", REASON_DUPLICATE_MAP, id="duplicate-across-components"
        ),
    ],
)
def test_parse_side_failure_taxonomy(side_text: str, expected_reason: str) -> None:
    # Given / When / Then: each branch has its own stable reason token
    assert _side_reason(side_text) == expected_reason


def test_parse_component_rejects_a_zero_atom_mol() -> None:
    # Given: RDKit parses the empty string into a 0-atom Mol, never None
    params = Chem.SmilesParserParams()
    params.removeHs = False
    empty = Chem.MolFromSmiles("", params)
    assert empty is not None
    assert empty.GetNumAtoms() == 0

    # When / Then: the atom-count check rejects it (parse_side intercepts empty
    # component strings earlier, so the private parser locks this branch)
    with pytest.raises(ReactionParseError) as excinfo:
        _parse_component("")
    assert excinfo.value.reason == REASON_ZERO_ATOM_COMPONENT


def test_parse_reaction_splits_both_sides_and_checks_consistent() -> None:
    # Given: a fully mapped two-component esterification
    # When
    reactants, products = parse_reaction(ESTER_SMILES)

    # Then: each side keeps its own component split and atom order
    assert reactants.components == (
        "[CH3:1][C:2](=[O:3])[OH:4]",
        "[CH3:5][OH:6]",
    )
    assert reactants.map_lists == ((1, 2, 3, 4), (5, 6))
    assert products.components == (
        "[CH3:1][C:2](=[O:3])[O:4][CH3:5]",
        "[OH2:6]",
    )
    assert products.map_lists == ((1, 2, 3, 4, 5), (6,))

    # And: the map-number sets agree on {1..6} over 12 atoms
    assert side_map_checks(reactants, products) == {
        "maps_unique": True,
        "map_sets_equal": True,
        "n_atoms": 12,
    }

    # And: the neutral singlet inventory row passes its re-check
    assert row_consistency_checks(CONSISTENT_ROW) == {
        "charge_consistent": True,
        "spin_consistent": True,
    }


@pytest.mark.parametrize(
    ("reaction_smiles", "expected_reason"),
    [
        pytest.param("[CH3:1][OH:2]", REASON_NO_ARROW, id="no-arrow"),
        pytest.param(
            "[CH3:1]>>[CH3:1]>>[CH3:1]", REASON_MULTIPLE_ARROWS, id="two-arrows"
        ),
        pytest.param(">>[CH3:1]", REASON_EMPTY_SIDE, id="empty-reactant-side"),
        pytest.param("[CH3:1]>>", REASON_EMPTY_SIDE, id="empty-product-side"),
        pytest.param(
            "[CH3:1]>>C%^&", REASON_UNPARSEABLE_COMPONENT, id="bad-product-component"
        ),
        pytest.param(
            "[CH3:1][OH:1]>>[CH3:1][OH:2]",
            REASON_DUPLICATE_MAP,
            id="duplicate-reactant-map",
        ),
        pytest.param("CC>>CC", REASON_UNMAPPED_ATOM, id="unmapped-both-sides"),
    ],
)
def test_parse_reaction_failure_taxonomy(
    reaction_smiles: str, expected_reason: str
) -> None:
    # Given / When / Then
    with pytest.raises(ReactionParseError) as excinfo:
        parse_reaction(reaction_smiles)
    assert excinfo.value.reason == expected_reason


def test_side_map_checks_detects_a_map_set_mismatch() -> None:
    # Given: the same skeleton with product map 3 instead of 2
    r = parse_side("[CH3:1][OH:2]")
    p = parse_side("[CH3:1][OH:3]")

    # When
    checks = side_map_checks(r, p)

    # Then
    assert checks == {
        "maps_unique": True,
        "map_sets_equal": False,
        "n_atoms": 4,
    }


def test_side_map_checks_reports_duplicate_maps_non_unique() -> None:
    # Given: incoming sides whose map_lists already carry a duplicate
    side = parse_side("[CH3:1][OH:2]")
    duplicated = replace(side, map_lists=((1, 1),))

    # When
    checks = side_map_checks(duplicated, side)

    # Then
    assert checks == {
        "maps_unique": False,
        "map_sets_equal": False,
        "n_atoms": 4,
    }


@pytest.mark.parametrize(
    ("row", "charge_consistent", "spin_consistent"),
    [
        pytest.param({**CONSISTENT_ROW}, True, True, id="neutral-singlet"),
        pytest.param(
            {
                "charge_total_reactants": 1,
                "charge_total_products": 0,
                "multiplicity_max": 1,
            },
            False,
            True,
            id="charged-reactants",
        ),
        pytest.param(
            {
                "charge_total_reactants": 0,
                "charge_total_products": -1,
                "multiplicity_max": 1,
            },
            False,
            True,
            id="charged-products",
        ),
        pytest.param(
            {
                "charge_total_reactants": 0,
                "charge_total_products": 0,
                "multiplicity_max": 3,
            },
            True,
            False,
            id="triplet",
        ),
    ],
)
def test_row_consistency_checks(
    row: Mapping[str, Any], charge_consistent: bool, spin_consistent: bool
) -> None:
    # Given / When
    checks = row_consistency_checks(row)

    # Then
    assert checks == {
        "charge_consistent": charge_consistent,
        "spin_consistent": spin_consistent,
    }


def test_parse_reaction_is_deterministic() -> None:
    # Given: the same mapped reaction parsed twice
    # When
    first = parse_reaction(ESTER_SMILES)
    second = parse_reaction(ESTER_SMILES)

    # Then: components, map lists, and atom counts are identical
    for first_side, second_side in zip(first, second, strict=True):
        assert first_side.components == second_side.components
        assert first_side.map_lists == second_side.map_lists
        assert [mol.GetNumAtoms() for mol in first_side.mols] == [
            mol.GetNumAtoms() for mol in second_side.mols
        ]
