"""Stratum computation and the non-authoritative bond-change preview for G0.

The stratification axes come from the TS-free inventory only: the element set,
the component count, and a total-atom bucket, plus the preview bond-change
counters produced by :func:`compute_bond_changes_preview`.  This module owns
that computation; :mod:`pes2ts_core.g0.strata` re-exports the two public
functions and owns cohort selection, so consumers keep the documented
``pes2ts_core.g0.strata`` entry point.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

import pyarrow as pa
from rdkit import Chem, RDLogger

# RDKit parse failures are reported as sentinel previews instead of stderr
# noise, so the C++ logger is muted at import time. (rdkit-stubs omits
# ``DisableLog``, hence the targeted ignore.)
RDLogger.DisableLog("rdApp.*")  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]

#: Sentinel preview counter meaning "unavailable without atom mapping".
PREVIEW_UNAVAILABLE: Final[int] = -1
#: Bucket boundaries over ``total_atoms_reactants`` (inclusive maxima).
SMALL_MAX_ATOMS: Final[int] = 8
MEDIUM_MAX_ATOMS: Final[int] = 16
LARGE_MAX_ATOMS: Final[int] = 28
BUCKET_SMALL: Final[str] = "small"
BUCKET_MEDIUM: Final[str] = "medium"
BUCKET_LARGE: Final[str] = "large"
BUCKET_XL: Final[str] = "xl"

#: One bond key: sorted mapped atom pair plus bond order (as double).
type BondKey = tuple[tuple[int, int], float]
#: Stratification key: ``(element_set, n_components, heavy_atom_bucket)``.
type StratumKey = tuple[str, int, str]

#: The only inventory columns :func:`compute_strata` materializes; the nested
#: per-component coordinate records are never converted to Python objects.
STRATUM_COLUMNS: Final[tuple[str, ...]] = (
    "reaction_id",
    "reaction_smiles",
    "elements",
    "n_reactant_components",
    "n_product_components",
    "total_atoms_reactants",
)


def _component_bond_keys(component: str) -> set[BondKey] | None:
    """Return the mapped bond keys of one component, or ``None`` when unmapped."""
    mol = Chem.MolFromSmiles(component)  # pyright: ignore[reportUnknownMemberType]
    if mol is None or mol.GetNumAtoms() == 0:  # pyright: ignore[reportUnnecessaryComparison]
        return None
    keys: set[BondKey] = set()
    for atom in mol.GetAtoms():  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        if atom.GetAtomMapNum() == 0:  # pyright: ignore[reportUnknownMemberType]
            return None
    for bond in mol.GetBonds():  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        first = bond.GetBeginAtom().GetAtomMapNum()  # pyright: ignore[reportUnknownMemberType]
        second = bond.GetEndAtom().GetAtomMapNum()  # pyright: ignore[reportUnknownMemberType]
        pair = (first, second) if first <= second else (second, first)
        keys.add((pair, bond.GetBondTypeAsDouble()))  # pyright: ignore[reportUnknownMemberType]
    return keys


def _side_bond_keys(side: str) -> set[BondKey] | None:
    """Return the union of mapped bond keys over one dot-separated side."""
    keys: set[BondKey] = set()
    for raw_component in side.split("."):
        component = raw_component.strip()
        if not component:
            return None
        component_keys = _component_bond_keys(component)
        if component_keys is None:
            return None
        keys |= component_keys
    return keys


def _unavailable_preview() -> dict[str, int]:
    """Return the all-sentinel preview for a reaction without usable maps."""
    return {
        "n_bonds_formed": PREVIEW_UNAVAILABLE,
        "n_bonds_broken": PREVIEW_UNAVAILABLE,
        "n_bond_order_changed": PREVIEW_UNAVAILABLE,
    }


def compute_bond_changes_preview(record: Mapping[str, Any]) -> dict[str, int]:
    """Return the mapped-graph bond diff between the reactant and product sides.

    Only the row's ``reaction_smiles`` is read.  Each side is parsed
    component-wise with atom maps preserved (``Chem.MolFromSmiles``), and every
    bond becomes a key ``((map_a, map_b), order)`` with the map pair sorted.
    ``n_bonds_formed`` counts product keys absent from the reactant key set,
    ``n_bonds_broken`` counts reactant keys absent from the product key set,
    and ``n_bond_order_changed`` counts map pairs bonded on both sides with a
    different order.  The key sets are not mutually exclusive by construction,
    so a pure bond-order change contributes to all three counters.

    These values are a clearly-labelled **preview** (``authoritative=false``):
    G1 must import this function rather than reimplementing it, and must
    recompute authoritative values from the real atom mapping.  The function
    is re-exported by :mod:`pes2ts_core.g0.strata`, the documented entry point.

    A reaction whose SMILES is missing, has no single ``>>``, fails to parse,
    has an empty component, or carries any atom without a nonzero map number
    returns the sentinel ``-1`` for all three counters ("unavailable without
    mapping"); this function never raises on bad input.
    """
    text = record.get("reaction_smiles")
    if not isinstance(text, str) or text.count(">>") != 1:
        return _unavailable_preview()
    reactant_text, product_text = text.split(">>")
    reactant_keys = _side_bond_keys(reactant_text)
    product_keys = _side_bond_keys(product_text)
    if reactant_keys is None or product_keys is None:
        return _unavailable_preview()
    reactant_orders: dict[tuple[int, int], set[float]] = {}
    for pair, order in reactant_keys:
        reactant_orders.setdefault(pair, set()).add(order)
    product_orders: dict[tuple[int, int], set[float]] = {}
    for pair, order in product_keys:
        product_orders.setdefault(pair, set()).add(order)
    order_changed = sum(
        1
        for pair, orders in reactant_orders.items()
        if pair in product_orders and product_orders[pair] != orders
    )
    return {
        "n_bonds_formed": len(product_keys - reactant_keys),
        "n_bonds_broken": len(reactant_keys - product_keys),
        "n_bond_order_changed": order_changed,
    }


def _heavy_atom_bucket(total_atoms_reactants: int) -> str:
    """Return the small/medium/large/xl bucket for a reactant atom count."""
    if total_atoms_reactants <= SMALL_MAX_ATOMS:
        return BUCKET_SMALL
    if total_atoms_reactants <= MEDIUM_MAX_ATOMS:
        return BUCKET_MEDIUM
    if total_atoms_reactants <= LARGE_MAX_ATOMS:
        return BUCKET_LARGE
    return BUCKET_XL


def compute_strata(inventory_table: pa.Table) -> list[dict[str, object]]:
    """Attach stratification fields to every inventory row.

    Accepts the inventory Parquet table and returns one record per row
    carrying ``reaction_id``, ``element_set`` (the sorted element symbols
    joined with ``,``), ``n_components`` (reactant plus product components),
    ``heavy_atom_bucket``, and the preview bond-change counters from
    :func:`compute_bond_changes_preview`.  The bucket is derived from
    ``total_atoms_reactants`` -- hydrogens included, despite the field name --
    as ``<=8`` small, ``9-16`` medium, ``17-28`` large, ``>28`` xl.  The
    inventory is TS-free by construction, so no transition-state or IRC data
    can enter a stratum.
    """
    records: list[dict[str, object]] = []
    for row in inventory_table.select(STRATUM_COLUMNS).to_pylist():
        record: dict[str, object] = {
            "reaction_id": row["reaction_id"],
            "element_set": ",".join(str(symbol) for symbol in row["elements"]),
            "n_components": int(row["n_reactant_components"])
            + int(row["n_product_components"]),
            "heavy_atom_bucket": _heavy_atom_bucket(int(row["total_atoms_reactants"])),
        }
        record.update(compute_bond_changes_preview(row))
        records.append(record)
    return records


__all__ = [
    "PREVIEW_UNAVAILABLE",
    "STRATUM_COLUMNS",
    "StratumKey",
    "compute_bond_changes_preview",
    "compute_strata",
]
