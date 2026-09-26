"""Atom-map-space bond changes, hydrogen migration, and the reaction center.

This is the authoritative G1 bond-change detail.  The module receives
already-parsed reactant/product molecules together with the global map number
of every atom (the parse stage owns SMILES splitting, explicit-hydrogen
retention, and map validation) and diffs the two bond graphs in map space.
Both sides are read through the same raw graph view -- ``GetBondTypeAsDouble()``
on the parsed molecules, never a kekulized copy of one side only -- so an
aromatic bond is ``1.5`` on either side.

The diff semantics match the G0 preview exactly
(:func:`pes2ts_core.g0.strata.compute_bond_changes_preview`, imported by
:func:`crosscheck_against_preview`): a bond key is the sorted map-number pair
plus its order, ``formed`` is the product key set minus the reactant key set,
``broken`` is the reactant key set minus the product key set, and
``order_changed`` lists the map pairs bonded on both sides whose order sets
differ.  The three lists are deliberately **not mutually exclusive**: a pure
bond-order change contributes one broken key, one formed key, and one
order-changed pair.

Besides the four documented lists, the returned mapping carries the sorted
edges of the reactant/product union graph under the private key
``_union_edges``: :func:`reaction_center` needs the full union graph
(including bonds that did not change) to expand the core through ``shell``
neighbour rings.  Document assembly copies the four documented keys and never
the private carrier.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, NotRequired, TypedDict

from rdkit import Chem, RDLogger

from pes2ts_core.g0.strata import compute_bond_changes_preview

# RDKit valence complaints become computed failures rather than stderr noise,
# so the C++ logger is muted at import time. (rdkit-stubs omits ``DisableLog``,
# hence the targeted ignore.)
RDLogger.DisableLog("rdApp.*")  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]

#: Atomic number of hydrogen -- the only element tracked for migration.
HYDROGEN_ATOMIC_NUMBER: Final[int] = 1
#: G0-documented preview sentinel: all three counters are ``-1`` when atom
#: mapping is unusable.  G1 aligns with the same value, so an unavailable
#: preview is never mistaken for a genuine zero-change reaction.
PREVIEW_UNAVAILABLE: Final[int] = -1

#: One bond key: sorted map-number pair plus the bond order as a double.
type BondKey = tuple[tuple[int, int], float]
#: A sorted global map-number pair.
type MapPair = tuple[int, int]


@dataclass(frozen=True, slots=True)
class ChangedBond:
    """One mapped-bond difference; ``None`` marks the side where it is absent."""

    atoms: MapPair
    order_r: float | None
    order_p: float | None


MigrationEntry = TypedDict(
    "MigrationEntry", {"h": int, "from": int | None, "to": int | None}
)


class BondChanges(TypedDict):
    """The four documented detail lists plus the private union-edge carrier."""

    formed: list[ChangedBond]
    broken: list[ChangedBond]
    order_changed: list[ChangedBond]
    hydrogen_migration: list[MigrationEntry]
    _union_edges: NotRequired[tuple[MapPair, ...]]


class ReactionCenter(TypedDict):
    """Core map numbers and their ``shell``-ring neighbourhood, both sorted."""

    core: list[int]
    with_shell: list[int]


def _side_bond_keys(
    mols: Sequence[Chem.Mol], map_lists: Sequence[Sequence[int]]
) -> set[BondKey]:
    """Return every mapped bond key of one already-parsed side."""
    keys: set[BondKey] = set()
    for mol, maps in zip(mols, map_lists, strict=True):
        for bond in mol.GetBonds():  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            first = maps[bond.GetBeginAtomIdx()]  # pyright: ignore[reportUnknownMemberType]
            second = maps[bond.GetEndAtomIdx()]  # pyright: ignore[reportUnknownMemberType]
            pair: MapPair = (first, second) if first <= second else (second, first)
            keys.add((pair, bond.GetBondTypeAsDouble()))  # pyright: ignore[reportUnknownMemberType]
    return keys


def _pair_orders(keys: set[BondKey]) -> dict[MapPair, set[float]]:
    """Group bond orders by sorted map pair."""
    orders: dict[MapPair, set[float]] = {}
    for pair, order in keys:
        orders.setdefault(pair, set()).add(order)
    return orders


def _single_order(orders: set[float]) -> float:
    """Return the deterministic representative of one pair's order set.

    Valid input has at most one bond per map pair and side (map numbers are
    unique within a side), so the set is a singleton in practice; ``min`` only
    pins a deterministic value for malformed multi-bond input.
    """
    return min(orders)


def _detail_order(bond: ChangedBond) -> tuple[int, int, float, float]:
    """Return the documented atoms-pair ordering key (``None`` orders first)."""
    return (
        bond.atoms[0],
        bond.atoms[1],
        -1.0 if bond.order_r is None else bond.order_r,
        -1.0 if bond.order_p is None else bond.order_p,
    )


def _union_edges(r_keys: set[BondKey], p_keys: set[BondKey]) -> tuple[MapPair, ...]:
    """Return the sorted map-space edges of the reactant/product union graph."""
    return tuple(sorted({pair for pair, _ in r_keys | p_keys}))


def _hydrogen_partners(
    mols: Sequence[Chem.Mol], map_lists: Sequence[Sequence[int]]
) -> dict[int, int | None]:
    """Map every hydrogen's map number to its bonded partner on one side.

    A hydrogen with no bond at all (for example a proton about to recombine
    into H2) reports ``None``.  H-H bonds are read from the same raw graph as
    the diff, so both partners are reported like any other bond.
    """
    partners: dict[int, set[int]] = {}
    hydrogens: set[int] = set()
    for mol, maps in zip(mols, map_lists, strict=True):
        for atom in mol.GetAtoms():  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            if atom.GetAtomicNum() == HYDROGEN_ATOMIC_NUMBER:  # pyright: ignore[reportUnknownMemberType]
                hydrogens.add(maps[atom.GetIdx()])  # pyright: ignore[reportUnknownMemberType]
        for bond in mol.GetBonds():  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            begin = bond.GetBeginAtomIdx()  # pyright: ignore[reportUnknownMemberType]
            end = bond.GetEndAtomIdx()  # pyright: ignore[reportUnknownMemberType]
            if mol.GetAtomWithIdx(begin).GetAtomicNum() == HYDROGEN_ATOMIC_NUMBER:  # pyright: ignore[reportUnknownMemberType]
                partners.setdefault(maps[begin], set()).add(maps[end])
            if mol.GetAtomWithIdx(end).GetAtomicNum() == HYDROGEN_ATOMIC_NUMBER:  # pyright: ignore[reportUnknownMemberType]
                partners.setdefault(maps[end], set()).add(maps[begin])
    return {h: (min(partners[h]) if h in partners else None) for h in hydrogens}


def compute_bond_changes(
    r_mols: Sequence[Chem.Mol],
    p_mols: Sequence[Chem.Mol],
    r_map_lists: Sequence[Sequence[int]],
    p_map_lists: Sequence[Sequence[int]],
) -> BondChanges:
    """Return the mapped bond diff, hydrogen migrations, and the union edges.

    ``r_mols``/``p_mols`` are the already-parsed components of each side, and
    ``r_map_lists``/``p_map_lists`` carry the global map number of every atom
    in the matching component (component order and atom order aligned).  Both
    sides must have been parsed identically with explicit hydrogens retained;
    map validation is the parse stage's job, so this function never re-parses
    a SMILES and never kekulizes.

    The three bond lists follow the documented preview semantics and are not
    mutually exclusive (see the module docstring).  ``hydrogen_migration``
    reports one entry per mapped hydrogen whose bonded partner differs between
    the sides (``None`` = isolated on that side), sorted by hydrogen map
    number.  All lists are sorted; the private ``_union_edges`` entry is the
    carrier consumed by :func:`reaction_center`.
    """
    r_keys = _side_bond_keys(r_mols, r_map_lists)
    p_keys = _side_bond_keys(p_mols, p_map_lists)
    r_orders = _pair_orders(r_keys)
    p_orders = _pair_orders(p_keys)
    formed = sorted(
        (ChangedBond(pair, None, order) for pair, order in p_keys - r_keys),
        key=_detail_order,
    )
    broken = sorted(
        (ChangedBond(pair, order, None) for pair, order in r_keys - p_keys),
        key=_detail_order,
    )
    order_changed = sorted(
        (
            ChangedBond(
                pair, _single_order(r_orders[pair]), _single_order(p_orders[pair])
            )
            for pair in r_orders.keys() & p_orders.keys()
            if r_orders[pair] != p_orders[pair]
        ),
        key=_detail_order,
    )
    r_hydrogens = _hydrogen_partners(r_mols, r_map_lists)
    p_hydrogens = _hydrogen_partners(p_mols, p_map_lists)
    hydrogen_migration: list[MigrationEntry] = [
        {"h": h, "from": r_hydrogens.get(h), "to": p_hydrogens.get(h)}
        for h in sorted(set(r_hydrogens) | set(p_hydrogens))
        if r_hydrogens.get(h) != p_hydrogens.get(h)
    ]
    return {
        "formed": formed,
        "broken": broken,
        "order_changed": order_changed,
        "hydrogen_migration": hydrogen_migration,
        "_union_edges": _union_edges(r_keys, p_keys),
    }


def reaction_center(changes: BondChanges, shell: int) -> ReactionCenter:
    """Return the core atoms and their ``shell``-ring neighbourhood.

    ``core`` is every endpoint of ``formed`` union ``broken`` union
    ``order_changed``.  ``with_shell`` additionally contains every atom within
    ``shell`` bonds of the core on the reactant/product union graph carried by
    the ``_union_edges`` key; ``shell=0`` therefore reproduces ``core``.  Both
    lists are sorted map numbers.
    """
    core = sorted(
        {atom for bond in changes["formed"] for atom in bond.atoms}
        | {atom for bond in changes["broken"] for atom in bond.atoms}
        | {atom for bond in changes["order_changed"] for atom in bond.atoms}
    )
    adjacency: dict[int, set[int]] = {}
    for left, right in changes.get("_union_edges", ()):
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
    expanded = set(core)
    for _ in range(max(shell, 0)):
        neighbours = {n for atom in expanded for n in adjacency.get(atom, ())}
        expanded |= neighbours
    return {"core": core, "with_shell": sorted(expanded)}


def crosscheck_against_preview(row: Mapping[str, object], changes: BondChanges) -> bool:
    """Return whether the detail counts equal the imported preview counters.

    The counters come from :func:`pes2ts_core.g0.strata.compute_bond_changes_preview`
    (imported, never reimplemented): the G1 obligation requires the
    authoritative detail to agree with the documented preview.  When the
    preview is the unavailable sentinel the comparison is impossible and the
    result is ``False``; the caller rejects with ``G1_PREVIEW_CONFLICT``.
    """
    preview = compute_bond_changes_preview(row)
    if any(counter == PREVIEW_UNAVAILABLE for counter in preview.values()):
        return False
    return (
        len(changes["formed"]) == preview["n_bonds_formed"]
        and len(changes["broken"]) == preview["n_bonds_broken"]
        and len(changes["order_changed"]) == preview["n_bond_order_changed"]
    )


__all__ = [
    "PREVIEW_UNAVAILABLE",
    "BondChanges",
    "ChangedBond",
    "ReactionCenter",
    "compute_bond_changes",
    "crosscheck_against_preview",
    "reaction_center",
]
