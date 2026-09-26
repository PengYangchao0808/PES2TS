"""Mapped reaction SMILES parsing and consistency checks for G1.

Every G1 analysis works on atom map numbers, so this module parses reaction
SMILES strictly: both sides of exactly one ``>>`` separator, every component
non-empty and RDKit-parseable, and every atom carrying a nonzero map number
that is unique within its side.  Failure is a :class:`ReactionParseError`
whose ``reason`` is a stable machine-readable token, never a partial result.

Explicit hydrogens written in the SMILES are preserved while parsing
(``removeHs=False`` on :class:`rdkit.Chem.SmilesParserParams`): RDKit's default
sanitization drops mapped explicit Hs bonded to heavy atoms, which would hide
their map numbers and every C-H / H-H bond from the bond-change diff.  Bond
orders are never rewritten -- in particular aromatic bonds stay aromatic
(``GetBondTypeAsDouble() == 1.5``) and are never Kekulized.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

from rdkit import Chem, RDLogger

# RDKit parse failures are surfaced as typed errors instead of stderr noise,
# so the C++ logger is muted at import time. (rdkit-stubs omits
# ``DisableLog``, hence the targeted ignore.)
RDLogger.DisableLog("rdApp.*")  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]

#: Stable machine-readable :class:`ReactionParseError` reason tokens.
REASON_EMPTY_SIDE: Final[str] = "empty_side"
REASON_EMPTY_COMPONENT: Final[str] = "empty_component"
REASON_UNPARSEABLE_COMPONENT: Final[str] = "unparseable_component"
REASON_ZERO_ATOM_COMPONENT: Final[str] = "zero_atom_component"
REASON_UNMAPPED_ATOM: Final[str] = "unmapped_atom"
REASON_DUPLICATE_MAP: Final[str] = "duplicate_map"
REASON_NO_ARROW: Final[str] = "no_arrow"
REASON_MULTIPLE_ARROWS: Final[str] = "multiple_arrows"


class ReactionParseError(ValueError):
    """Raised when a reaction SMILES cannot enter the G1 analysis.

    ``reason`` is the stable machine-readable token recorded as the rejection
    ledger detail; it doubles as the exception message.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason: str = reason


@dataclass(frozen=True, slots=True)
class SideComponents:
    """One parsed side of a mapped reaction SMILES.

    ``components`` holds the component SMILES strings in order of appearance,
    ``mols`` their ``removeHs=False`` parses, and ``map_lists`` each
    component's map numbers in atom order (a tuple position is the atom index
    within that component's mol).
    """

    components: tuple[str, ...]
    mols: tuple[Chem.Mol, ...]
    map_lists: tuple[tuple[int, ...], ...]


def _parse_component(component: str) -> Chem.Mol:
    """Parse one component with explicit Hs kept; reject None and 0 atoms.

    :func:`parse_side` intercepts empty component strings first, but the
    explicit atom-count check stays because RDKit parses the empty string into
    a 0-atom ``Mol`` (not ``None``) -- an ``is None`` check alone would let an
    atom-less component through.
    """
    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(component, params)  # pyright: ignore[reportUnknownMemberType]
    if mol is None:  # pyright: ignore[reportUnnecessaryComparison]
        raise ReactionParseError(REASON_UNPARSEABLE_COMPONENT)
    if mol.GetNumAtoms() == 0:
        raise ReactionParseError(REASON_ZERO_ATOM_COMPONENT)
    return mol


def parse_side(side_text: str) -> SideComponents:
    """Parse one dot-separated side of a mapped reaction SMILES.

    Every atom of every component must carry a nonzero map number, and no map
    number may repeat within the side (map numbers are reaction-global on the
    stored data).  Components keep their order of appearance.  Explicit
    hydrogens are preserved (``removeHs=False``).

    Raises
    ------
    ReactionParseError
        With reason ``empty_side``, ``empty_component``,
        ``unparseable_component``, ``zero_atom_component``, ``unmapped_atom``,
        or ``duplicate_map``.
    """
    if not side_text.strip():
        raise ReactionParseError(REASON_EMPTY_SIDE)
    components: list[str] = []
    mols: list[Chem.Mol] = []
    map_lists: list[tuple[int, ...]] = []
    seen: set[int] = set()
    for raw_component in side_text.split("."):
        component = raw_component.strip()
        if not component:
            raise ReactionParseError(REASON_EMPTY_COMPONENT)
        mol = _parse_component(component)
        numbers: list[int] = []
        for atom in mol.GetAtoms():  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            number = int(atom.GetAtomMapNum())  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
            if number == 0:
                raise ReactionParseError(REASON_UNMAPPED_ATOM)
            if number in seen:
                raise ReactionParseError(REASON_DUPLICATE_MAP)
            seen.add(number)
            numbers.append(number)
        components.append(component)
        mols.append(mol)
        map_lists.append(tuple(numbers))
    return SideComponents(
        components=tuple(components),
        mols=tuple(mols),
        map_lists=tuple(map_lists),
    )


def parse_reaction(reaction_smiles: str) -> tuple[SideComponents, SideComponents]:
    """Parse a mapped reaction SMILES into its reactant and product sides.

    Exactly one ``>>`` separator is required; each side is handed to
    :func:`parse_side`, so the reactants come first and the products second.

    Raises
    ------
    ReactionParseError
        With reason ``no_arrow`` / ``multiple_arrows`` for a malformed
        separator, or any :func:`parse_side` reason for an invalid side.
    """
    arrow_count = reaction_smiles.count(">>")
    if arrow_count == 0:
        raise ReactionParseError(REASON_NO_ARROW)
    if arrow_count > 1:
        raise ReactionParseError(REASON_MULTIPLE_ARROWS)
    reactant_text, product_text = reaction_smiles.split(">>")
    return parse_side(reactant_text), parse_side(product_text)


def side_map_checks(r: SideComponents, p: SideComponents) -> dict[str, bool | int]:
    """Return the map-set consistency summary between the two sides.

    ``maps_unique`` is true when neither side repeats a map number,
    ``map_sets_equal`` compares the reactant and product map-number sets, and
    ``n_atoms`` is the total atom count of both sides.
    """
    reactant_maps = [number for maps in r.map_lists for number in maps]
    product_maps = [number for maps in p.map_lists for number in maps]
    return {
        "maps_unique": len(set(reactant_maps)) == len(reactant_maps)
        and len(set(product_maps)) == len(product_maps),
        "map_sets_equal": set(reactant_maps) == set(product_maps),
        "n_atoms": len(reactant_maps) + len(product_maps),
    }


def row_consistency_checks(row: Mapping[str, Any]) -> dict[str, bool]:
    """Re-check the inventory row's charge and spin fields.

    A consistent row is neutral on both sides (``charge_consistent``, from
    ``charge_total_reactants`` and ``charge_total_products``) and a singlet
    (``spin_consistent``, from ``multiplicity_max``).
    """
    charge_reactants = int(row["charge_total_reactants"])
    charge_products = int(row["charge_total_products"])
    multiplicity = int(row["multiplicity_max"])
    return {
        "charge_consistent": charge_reactants == 0 and charge_products == 0,
        "spin_consistent": multiplicity == 1,
    }


__all__ = [
    "REASON_DUPLICATE_MAP",
    "REASON_EMPTY_COMPONENT",
    "REASON_EMPTY_SIDE",
    "REASON_MULTIPLE_ARROWS",
    "REASON_NO_ARROW",
    "REASON_UNMAPPED_ATOM",
    "REASON_UNPARSEABLE_COMPONENT",
    "REASON_ZERO_ATOM_COMPONENT",
    "ReactionParseError",
    "SideComponents",
    "parse_reaction",
    "parse_side",
    "row_consistency_checks",
    "side_map_checks",
]
