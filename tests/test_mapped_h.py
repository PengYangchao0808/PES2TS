"""Regression tests for mapped explicit-hydrogen preservation (F3 fix).

RDKit 2026.3.6 removes explicit hydrogens during default sanitization, so a
reaction SMILES whose hydrogens are all explicitly mapped (the Reaction-QM
storage pattern) loses every H atom map bonded to a heavy atom.  Both the
atom-map set check and the bond-diff preview must therefore parse with
``removeHs=False`` (a :class:`rdkit.Chem.SmilesParserParams` field; the
``MolFromSmiles`` constructor has no ``removeHs`` keyword argument).
"""

from __future__ import annotations

import numpy as np

from pes2ts_core.g0.reader import ReactionInfoRecord, SpeciesRecord
from pes2ts_core.g0.rejections import RejectionCode
from pes2ts_core.g0.rp_checks import validate_rp_reaction
from pes2ts_core.g0.strata_preview import compute_bond_changes_preview

#: Fully explicit-H reaction mirroring the real H2-dissociation pattern: the
#: H2 molecule H8-H9 breaks and both Hs bind to N4 while the C1#C2 triple bond
#: relaxes into the ring.  Every hydrogen carries an atom map.
EXPLICIT_H_REACTANTS = (
    "[C:1](#[C:2][H:6])/[C:3](=[N:4]/[O:5][H:10])[H:7].[H:8][H:9]"
)
EXPLICIT_H_PRODUCTS = (
    "[C-:1]1=[C:2]([H:6])[C:3]1([N+:4]([O:5][H:10])([H:8])[H:9])[H:7]"
)
#: Element inventory of either side: C3 N1 O1 H5 (balanced heavy atoms and Hs).
EXPLICIT_H_ATOMIC_NUMBERS: tuple[int, ...] = (6, 6, 6, 7, 8, 1, 1, 1, 1, 1)


def _record(reaction_smiles: str) -> ReactionInfoRecord:
    return {
        "reaction_id": "RXN_0000000001",
        "reaction_smiles": reaction_smiles,
        "dE": None,
        "dE_dagger": None,
        "dH": None,
        "dH_dagger": None,
        "dG": None,
        "dG_dagger": None,
    }


def _species(tag: str, atomic_numbers: tuple[int, ...]) -> SpeciesRecord:
    numbers = np.asarray(atomic_numbers, dtype=np.int64)
    return SpeciesRecord(
        tag=tag,
        smiles="O",
        atomic_numbers=numbers,
        coordinates=np.zeros((numbers.size, 3), dtype=np.float64),
        charge=0,
        multiplicity=1,
        EHG=np.zeros(3, dtype=np.float64),
    )


def _validate(
    reaction_smiles: str, atomic_numbers: tuple[int, ...]
) -> tuple[RejectionCode, str] | None:
    species = [
        _species("R0", atomic_numbers),
        _species("P0", atomic_numbers),
    ]
    return validate_rp_reaction(_record(reaction_smiles), species)


def test_map_check_accepts_a_fully_explicit_h_reaction() -> None:
    # Given: a balanced reaction whose hydrogens are explicit and mapped on
    # both sides (previously rejected because RDKit dropped the H maps)
    reaction_smiles = f"{EXPLICIT_H_REACTANTS}>>{EXPLICIT_H_PRODUCTS}"

    # When
    failure = _validate(reaction_smiles, EXPLICIT_H_ATOMIC_NUMBERS)

    # Then: the map-set check agrees on {1..10} and reports no failure
    assert failure is None


def test_map_check_still_rejects_an_h_only_map_mismatch() -> None:
    # Given: the same N-H fragment with one product H carrying map 10 instead
    # of the reactant's map 9 (heavy atoms and map 4/8 identical)
    reaction_smiles = "[N+:4]([H:8])[H:9]>>[N+:4]([H:8])[H:10]"

    # When
    failure = _validate(reaction_smiles, (7, 1, 1))

    # Then: the genuine mismatch is still a BAD_SMILES rejection
    assert failure is not None
    code, detail = failure
    assert code == RejectionCode.BAD_SMILES
    assert "atom map number sets differ" in detail


def test_bond_preview_counts_an_hh_bond_formed_and_broken_in_reverse() -> None:
    # Given: H2 formation and its written reverse
    formation = {"reaction_smiles": "[H:1].[H:2]>>[H:1][H:2]"}
    dissociation = {"reaction_smiles": "[H:1][H:2]>>[H:1].[H:2]"}

    # When
    formed = compute_bond_changes_preview(formation)
    broken = compute_bond_changes_preview(dissociation)

    # Then: the only possible bond key is H1-H2 and it is the one that moves
    assert formed == {
        "n_bonds_formed": 1,
        "n_bonds_broken": 0,
        "n_bond_order_changed": 0,
    }
    assert broken == {
        "n_bonds_formed": 0,
        "n_bonds_broken": 1,
        "n_bond_order_changed": 0,
    }


def test_bond_preview_counts_a_ch_bond_break() -> None:
    # Given: methane -> methyl + H, every H explicit and mapped
    reaction_smiles = (
        "[C:1]([H:2])([H:3])([H:4])[H:5]>>[C:1]([H:2])([H:3])[H:4].[H:5]"
    )

    # When
    preview = compute_bond_changes_preview({"reaction_smiles": reaction_smiles})

    # Then: exactly the C1-H5 bond breaks (it was invisible before the fix)
    assert preview == {
        "n_bonds_formed": 0,
        "n_bonds_broken": 1,
        "n_bond_order_changed": 0,
    }


def test_bond_preview_sees_the_real_pattern_hydrogen_bonds() -> None:
    # Given: the explicit-H pattern in both directions
    forward = {
        "reaction_smiles": f"{EXPLICIT_H_REACTANTS}>>{EXPLICIT_H_PRODUCTS}"
    }
    reverse = {
        "reaction_smiles": f"{EXPLICIT_H_PRODUCTS}>>{EXPLICIT_H_REACTANTS}"
    }

    # When
    forward_preview = compute_bond_changes_preview(forward)
    reverse_preview = compute_bond_changes_preview(reverse)

    # Then: forward breaks the H8-H9 bond and forms two N4-H bonds; the reverse
    # mirrors the counts exactly (including the two order-changed C-C pairs)
    assert forward_preview == {
        "n_bonds_formed": 5,
        "n_bonds_broken": 3,
        "n_bond_order_changed": 2,
    }
    assert reverse_preview == {
        "n_bonds_formed": 3,
        "n_bonds_broken": 5,
        "n_bond_order_changed": 2,
    }
