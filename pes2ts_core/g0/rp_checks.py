"""Reactant/product consistency checks for the TS-free inventory.

The checks live apart from :mod:`pes2ts_core.g0.inventory` so the fixed
periodic-table data and the R/P validation rules can be reviewed and tested on
their own.  :func:`validate_rp_reaction` returns the first typed
:class:`~pes2ts_core.g0.rejections.RejectionCode` failure with its detail
string, or ``None`` when the reaction is internally consistent.

None of these checks inspect or accept transition-state data: they operate on
reactant/product species only.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Final

from rdkit import Chem

from pes2ts_core.g0.reader import ARROW, ReactionInfoRecord, SpeciesRecord
from pes2ts_core.g0.rejections import RejectionCode

logger = logging.getLogger(__name__)

#: Element symbols indexed by ``atomic_number - 1`` for Z = 1..86.
ELEMENT_SYMBOLS: Final[tuple[str, ...]] = (
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn",
)


def element_symbols(atomic_numbers: Sequence[int]) -> list[str]:
    """Return the sorted unique element symbols for *atomic_numbers*.

    Raises
    ------
    ValueError
        When an atomic number has no symbol in :data:`ELEMENT_SYMBOLS`.
    """
    symbols: set[str] = set()
    for number in atomic_numbers:
        atomic_number = int(number)
        if not 1 <= atomic_number <= len(ELEMENT_SYMBOLS):
            msg = f"Unknown atomic number {atomic_number}; no element symbol mapped"
            raise ValueError(msg)
        symbols.add(ELEMENT_SYMBOLS[atomic_number - 1])
    return sorted(symbols)


def _mapped_atom_numbers(side: str, label: str, reaction_id: str) -> set[int]:
    """Collect the non-zero atom-map numbers on one side of mapped SMILES.

    Components RDKit cannot parse are skipped with a debug log (the CSV reader
    already rejects unparseable reactions), and ``0`` (the unmapped marker) is
    never collected.
    """
    numbers: set[int] = set()
    for component in side.split("."):
        mol = Chem.MolFromSmiles(component)
        if mol is None:  # pyright: ignore[reportUnnecessaryComparison]
            logger.debug(
                "Reaction %s: skipping unparseable %s component %r in the atom-map check",
                reaction_id,
                label,
                component,
            )
            continue
        numbers.update(
            atom.GetAtomMapNum() for atom in mol.GetAtoms() if atom.GetAtomMapNum() != 0
        )
    return numbers


def _atom_map_mismatch(reaction_smiles: str, reaction_id: str) -> str | None:
    """Return the atom-map mismatch detail, or ``None`` when the check passes.

    A side without any map number is unmapped SMILES: the check is skipped
    (debug log) instead of rejecting, because map numbers are then absent by
    construction.
    """
    reactant_side, product_side = reaction_smiles.split(ARROW)
    reactant_maps = _mapped_atom_numbers(reactant_side, "reactant", reaction_id)
    product_maps = _mapped_atom_numbers(product_side, "product", reaction_id)
    if not reactant_maps or not product_maps:
        logger.debug(
            "Reaction %s: unmapped SMILES side; skipping the atom-map set check",
            reaction_id,
        )
        return None
    if reactant_maps != product_maps:
        return "atom map number sets differ between reactants and products"
    return None


def validate_rp_reaction(
    record: ReactionInfoRecord,
    species: Sequence[SpeciesRecord],
) -> tuple[RejectionCode, str] | None:
    """Return the first R/P consistency failure, or ``None`` when valid.

    Checks run in a fixed order: charge neutrality, singlet spin, atom-count
    balance, element-set balance, then atom-map set balance.  A failing
    reaction yields exactly one failure (the first one).
    """
    reaction_id = record["reaction_id"]

    for item in species:
        if item.charge != 0:
            return (
                RejectionCode.CHARGE_NOT_NEUTRAL,
                f"Species {item.tag} has charge {item.charge}; expected 0",
            )
    for item in species:
        if item.multiplicity != 1:
            return (
                RejectionCode.SPIN_NOT_SINGLET,
                f"Species {item.tag} has multiplicity {item.multiplicity}; expected 1",
            )
    reactants = [item for item in species if item.tag.startswith("R")]
    products = [item for item in species if item.tag.startswith("P")]
    total_reactants = sum(len(item.atomic_numbers) for item in reactants)
    total_products = sum(len(item.atomic_numbers) for item in products)
    if total_reactants != total_products:
        return (
            RejectionCode.ATOM_COUNT_MISMATCH,
            f"Reactant atom count {total_reactants} != product atom count {total_products}",
        )
    reactant_elements = element_symbols(
        [number for item in reactants for number in item.atomic_numbers.tolist()]
    )
    product_elements = element_symbols(
        [number for item in products for number in item.atomic_numbers.tolist()]
    )
    if reactant_elements != product_elements:
        return (
            RejectionCode.ELEMENT_MISMATCH,
            f"Reactant elements {reactant_elements} != product elements {product_elements}",
        )
    map_detail = _atom_map_mismatch(record["reaction_smiles"], reaction_id)
    if map_detail is not None:
        return (RejectionCode.BAD_SMILES, map_detail)
    return None


__all__ = ["ELEMENT_SYMBOLS", "element_symbols", "validate_rp_reaction"]
