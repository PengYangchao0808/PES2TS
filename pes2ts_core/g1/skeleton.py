"""Skeleton-level component matching for G1 (representation-insensitive).

The inventory stores each component's conformer in XYZ-row order while the
component SMILES atom order is independent: only the atom map label ties the
two together -- the atom carrying local map ``k`` occupies row ``k - 1``
(verified on every stored component; reaction SMILES map numbers are a separate
reaction-global space).  Every enumerated match is therefore converted into
XYZ-row space before it leaves the matcher, so each tuple position (the query
atom index of the reaction component) points at the inventory conformer's local
XYZ row directly.

Matching must survive stereochemistry, resonance/bond-order, and formal-charge
representation differences between the stored reaction SMILES and the inventory
species SMILES.  The only permitted recipe flattens both mols (every bond to
``SINGLE``, aromaticity off, formal charges zero, chirality cleared,
``UpdatePropertyCache(strict=False)``, ``Chem.FastFindRings``) and matches
mol-vs-mol with :meth:`rdkit.Chem.Mol.GetSubstructMatches`
(``uniquify=False, useChirality=False, maxMatches=cap``).  Round-tripping a
flattened mol through ``MolToSmiles`` is forbidden: implicit-hydrogen rendering
differences produce false negatives.  The residual mismatches are genuine
constitutional differences and surface as :class:`ComponentMismatchError`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from rdkit import Chem, RDLogger

# RDKit parse failures are surfaced as typed results instead of stderr noise,
# so the C++ logger is muted at import time. (rdkit-stubs omits ``DisableLog``,
# hence the targeted ignore.)
RDLogger.DisableLog("rdApp.*")  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]


def flatten_skeleton(mol: Chem.Mol) -> Chem.Mol:
    """Return a copy of *mol* flattened to a single-bonded, neutral skeleton.

    Every bond becomes ``SINGLE`` with aromaticity off; every atom loses its
    formal charge, aromatic flag, and chiral tag.  Atom map numbers, isotopes,
    and the atom order are preserved.  The result is the representation-free
    matching key: stereochemistry, resonance forms, and charge annotations no
    longer divide two depictions of the same constitution.  Explicit hydrogens
    stay in the graph (they are atoms like any other).
    """
    skeleton = Chem.RWMol(mol)
    for bond in skeleton.GetBonds():  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        bond.SetBondType(Chem.BondType.SINGLE)  # pyright: ignore[reportUnknownMemberType]
        bond.SetIsAromatic(False)  # pyright: ignore[reportUnknownMemberType]
    for atom in skeleton.GetAtoms():  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        atom.SetFormalCharge(0)  # pyright: ignore[reportUnknownMemberType]
        atom.SetIsAromatic(False)  # pyright: ignore[reportUnknownMemberType]
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)  # pyright: ignore[reportUnknownMemberType]
    out = skeleton.GetMol()
    out.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(out)  # pyright: ignore[reportUnknownMemberType]
    return out


class ComponentMismatchError(ValueError):
    """A reaction-side component matches no unused inventory skeleton.

    ``rxn_index`` is the ordinal of the failing component within the reaction
    side (its position in the side's component sequence).  Reaching this error
    means the stored reaction SMILES and the inventory describe different
    constitutions -- a typed rejection, not a matcher bug.
    """

    def __init__(self, rxn_index: int, detail: str) -> None:
        super().__init__(f"reaction-side component {rxn_index}: {detail}")
        self.rxn_index: int = rxn_index


@dataclass(frozen=True, slots=True)
class ComponentMatch:
    """One reaction-side component paired with an inventory component.

    ``tag_candidates`` lists every still-unused tag whose skeleton contains the
    component, in lexicographic tag order; ``tag`` is its first entry, so a
    candidate tuple longer than one records pairing ambiguity deterministically.

    ``isomorphisms`` enumerates the graph isomorphisms of the preferred tag in
    RDKit's deterministic enumeration order and is never re-sorted.  Each tuple
    position is a query atom index of the reaction component and its value is
    the *local XYZ row* of the matched inventory atom (``row == map - 1``).
    ``n_isomorphisms`` is the enumeration count (the raw count, equal to
    ``match_cap`` when truncated) while at most ``max_candidates`` solutions
    are stored.
    """

    rxn_index: int
    tag: str
    tag_candidates: tuple[str, ...]
    n_isomorphisms: int
    truncated: bool
    isomorphisms: tuple[tuple[int, ...], ...]


def _prepare_target(
    record: Mapping[str, Any],
) -> tuple[str, tuple[int, ...], Chem.Mol]:
    """Parse one inventory record into ``(tag, row_of_atom, flat_mol)``.

    ``removeHs=False`` keeps every explicit hydrogen, and ``row_of_atom`` maps
    each parsed atom index to its XYZ row via the local map label
    (``row = map - 1``).  An unparseable or atom-less component is a caller
    contract violation, so it raises :class:`ValueError`.
    """
    tag = str(record["tag"])
    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(str(record["smiles"]), params)  # pyright: ignore[reportUnknownMemberType]
    if mol is None or mol.GetNumAtoms() == 0:  # pyright: ignore[reportUnnecessaryComparison]
        raise ValueError(f"inventory component {tag!r}: SMILES did not parse")
    row_of_atom = tuple(
        int(atom.GetAtomMapNum()) - 1  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
        for atom in mol.GetAtoms()  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    )
    return tag, row_of_atom, flatten_skeleton(mol)


def match_side_components(
    r_mols: Sequence[Chem.Mol],
    r_map_lists: Sequence[Sequence[int]],
    inventory_components: Sequence[Mapping[str, Any]],
    *,
    match_cap: int,
    max_candidates: int,
) -> list[ComponentMatch]:
    """Pair every component of one reaction side with an inventory component.

    *inventory_components* holds only the records of this side (each with a
    unique ``tag``); the caller selects the reactant or product subset.
    Components are matched in side order and the returned list preserves that
    order.  A component's candidate tags are all still-unused tags whose
    flattened mol has the same atom count and contains the component's
    flattened skeleton, tried in lexicographic order; the preferred tag is the
    first candidate and is marked used, so two identical components pair with
    two distinct tags.  Equal atom counts make every accepted substructure
    match a bijection (a graph isomorphism), which the index tables rely on --
    a smaller component whose skeleton merely appears inside a larger
    inventory molecule is not a pairing candidate.

    ``r_map_lists`` holds each reaction component's reaction-level map numbers
    in mol atom order; its length is validated against the mol and it is
    otherwise unused (the matcher works in inventory-constrained graph space).

    Isomorphism enumeration is capped at ``match_cap``; ``truncated`` is true
    exactly when the enumeration reached that cap (so an exact hit at the cap
    is conservatively reported as truncated) and ``n_isomorphisms`` carries the
    enumeration count.  At most ``max_candidates`` solutions are stored.

    Raises
    ------
    ComponentMismatchError
        When a component has no skeleton-compatible unused inventory tag; the
        exception carries the component's ``rxn_index``.
    ValueError
        On invalid caps, mismatched argument lengths, duplicate inventory tags,
        a zero-atom reaction component, or an unparseable inventory component.
    """
    if match_cap < 1:
        raise ValueError(f"match_cap must be positive, got {match_cap}")
    if max_candidates < 1:
        raise ValueError(f"max_candidates must be positive, got {max_candidates}")
    if len(r_mols) != len(r_map_lists):
        raise ValueError(
            f"got {len(r_mols)} reaction mols but {len(r_map_lists)} map lists"
        )
    targets: list[tuple[str, tuple[int, ...], Chem.Mol]] = []
    seen_tags: set[str] = set()
    for record in inventory_components:
        target = _prepare_target(record)
        if target[0] in seen_tags:
            raise ValueError(f"duplicate inventory component tag {target[0]!r}")
        seen_tags.add(target[0])
        targets.append(target)
    targets.sort(key=lambda target: target[0])

    results: list[ComponentMatch] = []
    used_tags: set[str] = set()
    for rxn_index, mol in enumerate(r_mols):
        query = flatten_skeleton(mol)
        if query.GetNumAtoms() == 0:
            raise ValueError(f"reaction-side component {rxn_index} has zero atoms")
        map_count = len(r_map_lists[rxn_index])
        if map_count != query.GetNumAtoms():
            raise ValueError(
                f"reaction-side component {rxn_index}: {query.GetNumAtoms()} atoms but {map_count} map numbers"
            )
        candidates: list[str] = []
        raw_matches: tuple[tuple[int, ...], ...] = ()
        for tag, row_of_atom, flat_target in targets:
            if tag in used_tags:
                continue
            if flat_target.GetNumAtoms() != query.GetNumAtoms():
                continue
            found = flat_target.GetSubstructMatches(
                query, uniquify=False, useChirality=False, maxMatches=match_cap
            )
            if not found:
                continue
            candidates.append(tag)
            if not raw_matches:
                raw_matches = tuple(
                    tuple(row_of_atom[atom] for atom in match) for match in found
                )
        if not candidates:
            raise ComponentMismatchError(
                rxn_index, "no unused inventory component shares its skeleton"
            )
        used_tags.add(candidates[0])
        results.append(
            ComponentMatch(
                rxn_index=rxn_index,
                tag=candidates[0],
                tag_candidates=tuple(candidates),
                n_isomorphisms=len(raw_matches),
                truncated=len(raw_matches) == match_cap,
                isomorphisms=raw_matches[:max_candidates],
            )
        )
    return results


__all__ = [
    "ComponentMatch",
    "ComponentMismatchError",
    "flatten_skeleton",
    "match_side_components",
]
