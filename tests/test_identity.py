"""Tests for map- and direction-invariant reaction identity.

The suite locks the identity contract: atom-map renumbering, component
permutation within a side, and reaction direction must not change the hash,
while stereochemistry must. Typed :class:`~pes2ts_core.g0.identity.IdentityError`
failures cover malformed reaction SMILES, and a scale sanity check keeps the
199,890-reaction pipeline budget in view.
"""

from __future__ import annotations

import re
import time

import pytest

from pes2ts_core.g0 import identity
from pes2ts_core.g0.identity import (
    CANONICALIZATION_PATH,
    IGNORE_ATOM_MAP_KWARG,
    IdentityError,
    canonical_reaction_identity,
    directional_preimage,
    is_reverse_of,
    reverse_pair_key,
)

#: The same reaction written in the reverse direction, atom maps included.
ESTERIFICATION_MAPPED_REVERSE = (
    "[OH2:4].[CH3:1][C:2](=[O:3])[O:7][CH2:6][CH3:5]"
    ">>[CH3:1][C:2](=[O:3])[OH:4].[CH3:5][CH2:6][OH:7]"
)

#: Fischer esterification with Reaction-QM-style atom maps (1..7).
ESTERIFICATION_MAPPED = (
    "[CH3:1][C:2](=[O:3])[OH:4].[CH3:5][CH2:6][OH:7]"
    ">>[CH3:1][C:2](=[O:3])[O:7][CH2:6][CH3:5].[OH2:4]"
)

#: The same reaction with maps renumbered (1->17, 2->4, 3->12, 4->9, 5->20,
#: 6->2, 7->13) and both sides written with their components in reverse order.
ESTERIFICATION_RENUMBERED = (
    "[CH3:20][CH2:2][OH:13].[CH3:17][C:4](=[O:12])[OH:9]"
    ">>[OH2:9].[CH3:17][C:4](=[O:12])[O:13][CH2:2][CH3:20]"
)

#: The same reaction with no atom maps at all.
ESTERIFICATION_UNMAPPED = "CC(=O)O.CCO>>CCOC(C)=O.O"

#: A distinct reaction (the esterification product side is missing water).
ESTERIFICATION_WRONG = "CC(=O)O.CCO>>CC(=O)OCC"


def test_component_order_within_a_side_is_invariant() -> None:
    # Given: the same reaction with both sides' components in opposite order
    first = "CCO.CC(=O)O>>CCOC(C)=O.O"
    second = "CC(=O)O.CCO>>O.CCOC(C)=O"

    # When / Then: the identities match
    assert canonical_reaction_identity(first) == canonical_reaction_identity(second)


def test_reaction_direction_is_invariant() -> None:
    # Given: a reaction written in both directions
    forward = "CCO>>CC=O"
    reverse = "CC=O>>CCO"

    # When / Then: the direction-agnostic identity is shared
    assert canonical_reaction_identity(forward) == canonical_reaction_identity(
        reverse
    )
    assert is_reverse_of(forward, reverse)
    assert is_reverse_of(reverse, forward)


def test_directional_preimage_preserves_written_side_order() -> None:
    # Given: a reaction with several components on both sides
    reactants, products = directional_preimage("CCO.CC(=O)O>>CCOC(C)=O.O")

    # Then: components are canonically sorted WITHIN each side, but the
    # reactant/product assignment follows the written input
    assert reactants == "CC(=O)O+CCO"
    assert products == "CCOC(C)=O+O"


def test_directional_preimage_swaps_for_a_written_reverse() -> None:
    # Given: a reaction and its written reverse
    forward = directional_preimage("CCO.CC(=O)O>>CCOC(C)=O.O")
    reverse = directional_preimage("CCOC(C)=O.O>>CCO.CC(=O)O")

    # Then: the directional tuple is exactly swapped
    assert reverse == (forward[1], forward[0])
    # And: the direction-agnostic identity is still shared
    assert canonical_reaction_identity(
        "CCO.CC(=O)O>>CCOC(C)=O.O"
    ) == canonical_reaction_identity("CCOC(C)=O.O>>CCO.CC(=O)O")


def test_directional_preimage_is_map_and_component_order_invariant() -> None:
    # Given: unmapped, mapped, and renumbered/permuted renderings of one
    # reaction, all written in the same direction
    unmapped = directional_preimage(ESTERIFICATION_UNMAPPED)

    # When / Then: map numbers and intra-side component order never change the
    # directional tuple
    assert directional_preimage(ESTERIFICATION_MAPPED) == unmapped
    assert directional_preimage(ESTERIFICATION_RENUMBERED) == unmapped


def test_map_renumbering_is_invariant() -> None:
    # Given: a mapped reaction and the same reaction with every map renumbered
    # When / Then
    assert canonical_reaction_identity(
        ESTERIFICATION_MAPPED
    ) == canonical_reaction_identity(ESTERIFICATION_RENUMBERED)


def test_mapped_and_unmapped_reactions_share_an_identity() -> None:
    # Given: the mapped and the unmapped representation of one reaction
    mapped = canonical_reaction_identity(ESTERIFICATION_MAPPED)

    # When / Then: atom maps are fully ignored
    assert mapped == canonical_reaction_identity(ESTERIFICATION_UNMAPPED)
    assert canonical_reaction_identity(
        ESTERIFICATION_RENUMBERED
    ) == canonical_reaction_identity(ESTERIFICATION_UNMAPPED)


def test_stereoisomers_have_distinct_identities() -> None:
    # Given: E/Z isomers and an enantiomer pair sharing everything but stereo
    e_isomer = "C/C=C/C>>CC(C)C"
    z_isomer = "C/C=C\\C>>CC(C)C"
    no_stereo = "CC=CC>>CC(C)C"
    r_enantiomer = "C[C@H](O)CC>>CC(C)C"
    s_enantiomer = "C[C@@H](O)CC>>CC(C)C"

    # When / Then: stereo is preserved, so all of these differ
    assert canonical_reaction_identity(e_isomer) != canonical_reaction_identity(
        z_isomer
    )
    assert canonical_reaction_identity(e_isomer) != canonical_reaction_identity(
        no_stereo
    )
    assert canonical_reaction_identity(r_enantiomer) != canonical_reaction_identity(
        s_enantiomer
    )


def test_unparseable_component_raises_identity_error() -> None:
    # Given / When / Then: no bogus hash is produced for either side
    with pytest.raises(IdentityError, match="cannot parse reactant component"):
        canonical_reaction_identity("C1CC>>CC")
    with pytest.raises(IdentityError, match="cannot parse product component"):
        canonical_reaction_identity("CC>>not_a_smiles")


def test_arrow_count_must_be_exactly_one() -> None:
    # Given / When / Then
    with pytest.raises(IdentityError, match="expected exactly one"):
        canonical_reaction_identity("CC")
    with pytest.raises(IdentityError, match="found 0"):
        canonical_reaction_identity("")
    with pytest.raises(IdentityError, match="found 2"):
        canonical_reaction_identity("C>>C>>C")


def test_empty_sides_and_components_raise_identity_error() -> None:
    # Given / When / Then
    with pytest.raises(IdentityError, match="reactant side is empty"):
        canonical_reaction_identity(">>C")
    with pytest.raises(IdentityError, match="product side is empty"):
        canonical_reaction_identity("C>>")
    with pytest.raises(IdentityError, match="empty reactant component"):
        canonical_reaction_identity("C.>>C")
    with pytest.raises(IdentityError, match="empty product component"):
        canonical_reaction_identity("C>>C..C")


def test_non_string_input_raises_identity_error() -> None:
    # Given / When / Then
    with pytest.raises(IdentityError, match="must be a str"):
        canonical_reaction_identity(None)  # pyright: ignore[reportArgumentType]


def test_canonicalization_path_is_probed_at_import() -> None:
    # Given: the pinned rdkit 2026.3.6 accepts the map kwarg
    # Then: the module recorded that path at import instead of picking silently
    assert IGNORE_ATOM_MAP_KWARG is True
    assert CANONICALIZATION_PATH == identity.KWARG_PATH


def test_identity_is_deterministic_and_hex_encoded() -> None:
    # Given / When: the same input is canonicalized twice
    first = canonical_reaction_identity(ESTERIFICATION_MAPPED)
    second = canonical_reaction_identity(ESTERIFICATION_MAPPED)

    # Then: identical 64-char lower-case hex digests
    assert first == second
    assert re.fullmatch(r"[0-9a-f]{64}", first) is not None


def test_reverse_pair_key_groups_a_reaction_with_its_reverse() -> None:
    # Given: a reaction, its written reverse, and an unrelated reaction
    reaction = "CCO.CC(=O)O>>CCOC(C)=O.O"
    reverse = "CCOC(C)=O.O>>CCO.CC(=O)O"
    unrelated = "c1ccccc1>>c1ccc(O)cc1"

    # When / Then
    assert reverse_pair_key(reaction) == reverse_pair_key(reverse)
    assert reverse_pair_key(reaction) == canonical_reaction_identity(reaction)
    assert is_reverse_of(reaction, reverse)
    # Identical reactions share the direction-agnostic key by construction.
    assert is_reverse_of(reaction, reaction)
    assert not is_reverse_of(reaction, unrelated)


def test_realistic_mapped_bimolecular_esterification() -> None:
    # Given: Reaction-QM-style mapped bimolecular esterification plus its
    # reverse and an unmapped rendering
    mapped = canonical_reaction_identity(ESTERIFICATION_MAPPED)
    mapped_reverse = canonical_reaction_identity(ESTERIFICATION_MAPPED_REVERSE)

    # When / Then: map/per-component/direction permutations collapse
    assert mapped == mapped_reverse
    assert mapped == canonical_reaction_identity(ESTERIFICATION_RENUMBERED)
    assert mapped == canonical_reaction_identity(ESTERIFICATION_UNMAPPED)
    # And: a genuinely different reaction keeps its own identity
    assert mapped != canonical_reaction_identity(ESTERIFICATION_WRONG)


def test_surrounding_whitespace_is_tolerated() -> None:
    # Given: the same reaction padded with whitespace around sides/components
    padded = "  CCO . CC(=O)O >> O . CCOC(C)=O  "
    canonical = "CC(=O)O.CCO>>CCOC(C)=O.O"

    # When / Then
    assert canonical_reaction_identity(padded) == canonical_reaction_identity(
        canonical
    )


def test_thousand_identities_complete_under_five_seconds() -> None:
    # Given: a pool spanning mapped, unmapped, reverse, stereo, and aromatic
    pool = (
        ESTERIFICATION_MAPPED,
        ESTERIFICATION_RENUMBERED,
        ESTERIFICATION_UNMAPPED,
        "CCO>>CC=O",
        "CC=O>>CCO",
        "c1ccccc1>>c1ccc(O)cc1",
        "C[C@H](O)CC>>CC(C)C",
        "C[C@@H](O)CC>>CC(C)C",
    )

    # When: 1,000 identities are computed
    start = time.perf_counter()
    for index in range(1000):
        canonical_reaction_identity(pool[index % len(pool)])
    elapsed = time.perf_counter() - start

    # Then: the per-row cost stays far below the 199,890-row pipeline budget
    assert elapsed < 5.0
