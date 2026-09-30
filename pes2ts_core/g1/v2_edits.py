"""G1 v2 bond edits: mutually exclusive per-pair semantics on the mapped graph.

The v1 diff (``bond_changes.compute_bond_changes`` / the G0 preview) reports
*key-set* differences, so one single->double bond change produced a broken
key, a formed key, **and** an order-changed pair for the same atom pair.  v2
instead keys every edit by the **unordered map pair** alone and assigns
exactly one ``edit_kind``:

- ``formed``          -- no bond on the reactant side, a bond on the product;
- ``broken``          -- a bond on the reactant side, none on the product;
- ``order_changed``   -- bonded on both sides with different bond order.

Original bond orders are preserved verbatim (aromatic bonds stay ``1.5``),
edited aromatic bonds are annotated with their conjugated-region id, and
hydrogen partner changes are re-derived from the same graphs with an
exhaustive partner-state taxonomy (``transfer`` / ``release`` / ``capture`` /
``to_hh`` / ``from_hh`` / ``hh_release`` / ``hh_form_free`` / ``hh_swap``) so
H2 events are never mislabelled as ordinary transfers.

Everything here is pure: no file I/O, no RDKit, no truth access.  Inputs are
the map-space graph payload blocks (``elements``, ``r_bonds``, ``p_bonds`` as
``[a, b, order]`` triples) exactly as persisted in the P1 documents.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from typing import Final, TypedDict

from pes2ts_core.g1.v2_schema import (
    AROMATIC_ORDER,
    EDIT_BROKEN,
    EDIT_FORMED,
    EDIT_ORDER_CHANGED,
    H_CHANGE_CAPTURE,
    H_CHANGE_FROM_HH,
    H_CHANGE_HH_FORM_FREE,
    H_CHANGE_HH_RELEASE,
    H_CHANGE_HH_SWAP,
    H_CHANGE_RELEASE,
    H_CHANGE_TO_HH,
    H_CHANGE_TRANSFER,
    ISSUE_EDIT_CONTRACT_VIOLATION,
    ISSUE_MULTI_BOND_PAIR,
)

#: Sorted map-number pair type.
type MapPair = tuple[int, int]
#: Bond order per pair on one side (``None`` = no bond).
type SideOrders = dict[MapPair, float | None]

_HYDROGEN: Final[str] = "H"
_STATE_FREE: Final[str] = "free"
_STATE_HH: Final[str] = "hh"
_STATE_HEAVY: Final[str] = "heavy"
#: Sort-time replacement for an absent bond order (real orders are >= 0.5).
_ORDER_SORT_KEY: Final[dict[None, float]] = {None: -1.0}

_KIND_BY_TRANSITION: Final[dict[tuple[str, str], str]] = {
    (_STATE_HEAVY, _STATE_HEAVY): H_CHANGE_TRANSFER,
    (_STATE_HEAVY, _STATE_FREE): H_CHANGE_RELEASE,
    (_STATE_FREE, _STATE_HEAVY): H_CHANGE_CAPTURE,
    (_STATE_HEAVY, _STATE_HH): H_CHANGE_TO_HH,
    (_STATE_HH, _STATE_HEAVY): H_CHANGE_FROM_HH,
    (_STATE_HH, _STATE_FREE): H_CHANGE_HH_RELEASE,
    (_STATE_FREE, _STATE_HH): H_CHANGE_HH_FORM_FREE,
    (_STATE_HH, _STATE_HH): H_CHANGE_HH_SWAP,
}


class EditRecord(TypedDict):
    """One mutually exclusive bond edit on an unordered map pair."""

    pair: list[int]
    elements: list[str]
    r_bond_order: float | None
    p_bond_order: float | None
    edit_kind: str
    aromatic_region: str | None


HydrogenChangeRecord = TypedDict(
    "HydrogenChangeRecord",
    {
        "h": int,
        "from": int | None,
        "to": int | None,
        "from_state": str,
        "to_state": str,
        "kind": str,
    },
)


def _sorted_pair(first: int, second: int) -> MapPair:
    return (first, second) if first <= second else (second, first)


def side_pair_orders(
    bonds: Sequence[Sequence[float]],
) -> tuple[SideOrders, list[MapPair]]:
    """Return ``(pair -> order, multi-bond pairs)`` of one side's bond list.

    A pair with more than one bond on the same side is malformed source
    input; the minimum order keeps the downstream computation deterministic
    while the caller turns the returned list into a typed issue code.
    """
    orders_by_pair: dict[MapPair, set[float]] = {}
    for bond in bonds:
        first, second, order = int(bond[0]), int(bond[1]), float(bond[2])
        orders_by_pair.setdefault(_sorted_pair(first, second), set()).add(order)
    multi = sorted(pair for pair, orders in orders_by_pair.items() if len(orders) > 1)
    orders: SideOrders = {pair: min(items) for pair, items in orders_by_pair.items()}
    return orders, multi


def _hydrogen_partners(
    elements: Mapping[int, str], orders: SideOrders
) -> dict[int, int | None]:
    """Return every hydrogen's bonded partner (``None`` when isolated).

    An H-H bond records each hydrogen with the other as its partner, so a
    forming or breaking H2 is visible on both members.
    """
    partners: dict[int, int] = {}
    for (first, second) in orders:
        first_is_h = elements.get(first) == _HYDROGEN
        second_is_h = elements.get(second) == _HYDROGEN
        if first_is_h:
            partners[first] = second
        if second_is_h:
            partners[second] = first
    return {
        h: partners.get(h)
        for h in sorted(int(m) for m, symbol in elements.items() if symbol == _HYDROGEN)
    }


def _partner_state(
    elements: Mapping[int, str], partner: int | None
) -> str:
    if partner is None:
        return _STATE_FREE
    return _STATE_HH if elements.get(partner) == _HYDROGEN else _STATE_HEAVY


def hydrogen_partner_changes(
    elements: Mapping[int, str],
    r_orders: SideOrders,
    p_orders: SideOrders,
) -> list[HydrogenChangeRecord]:
    """Return every hydrogen partner change with its exhaustive kind."""
    r_partners = _hydrogen_partners(elements, r_orders)
    p_partners = _hydrogen_partners(elements, p_orders)
    records: list[HydrogenChangeRecord] = []
    for h in sorted(set(r_partners) | set(p_partners)):
        source = r_partners.get(h)
        target = p_partners.get(h)
        if source == target:
            continue
        from_state = _partner_state(elements, source)
        to_state = _partner_state(elements, target)
        records.append({
            "h": h,
            "from": source,
            "to": target,
            "from_state": from_state,
            "to_state": to_state,
            "kind": _KIND_BY_TRANSITION[(from_state, to_state)],
        })
    return records


def _aromatic_regions(
    r_orders: SideOrders, p_orders: SideOrders
) -> tuple[dict[MapPair, str], dict[str, list[MapPair]]]:
    """Return ``(edited pair -> region id, region id -> member pairs)``.

    A region is a connected component of the union graph restricted to bonds
    that are aromatic (order ``1.5``) on at least one side, so a benzene ring
    that re-kekulizes stays one region instead of six independent edits.
    """
    aromatic_pairs = {
        pair
        for pair, order in r_orders.items()
        if order == AROMATIC_ORDER
    } | {
        pair
        for pair, order in p_orders.items()
        if order == AROMATIC_ORDER
    }
    adjacency: dict[int, set[int]] = {}
    for first, second in aromatic_pairs:
        adjacency.setdefault(first, set()).add(second)
        adjacency.setdefault(second, set()).add(first)
    visited: set[int] = set()
    regions: dict[str, list[MapPair]] = {}
    pair_region: dict[MapPair, str] = {}
    for start in sorted(adjacency):
        if start in visited:
            continue
        component: set[int] = {start}
        stack = [start]
        while stack:
            node = stack.pop()
            for neighbour in adjacency[node]:
                if neighbour not in component:
                    component.add(neighbour)
                    stack.append(neighbour)
        visited |= component
        members = sorted(
            pair for pair in aromatic_pairs if pair[0] in component and pair[1] in component
        )
        region_id = hashlib.sha256(
            repr(members).encode("utf-8")
        ).hexdigest()[:12]
        regions[region_id] = members
        for pair in members:
            pair_region[pair] = region_id
    return pair_region, regions


def _edit_kind(r_order: float | None, p_order: float | None) -> str:
    if r_order is None and p_order is not None:
        return EDIT_FORMED
    if r_order is not None and p_order is None:
        return EDIT_BROKEN
    return EDIT_ORDER_CHANGED


def exclusive_edits(
    elements: Mapping[int, str],
    r_bonds: Sequence[Sequence[float]],
    p_bonds: Sequence[Sequence[float]],
) -> dict[str, object]:
    """Return the complete v2 edit block of one mapped reaction graph.

    The returned mapping carries ``edits`` (one record per changed unordered
    pair, sorted), ``hydrogen_partner_changes`` (one record per hydrogen
    whose partner differs, sorted), ``aromatic_regions`` (region id ->
    member pairs of every conjugated region containing an edited aromatic
    bond), ``multi_bond_pairs``, ``unchanged_pairs`` count, and the union
    adjacency (private carrier for shell expansion, stripped before
    serialization by the caller).
    """
    r_orders, r_multi = side_pair_orders(r_bonds)
    p_orders, p_multi = side_pair_orders(p_bonds)
    all_pairs = sorted(set(r_orders) | set(p_orders))
    pair_region, regions = _aromatic_regions(r_orders, p_orders)
    edits: list[EditRecord] = []
    unchanged = 0
    for pair in all_pairs:
        r_order = r_orders.get(pair)
        p_order = p_orders.get(pair)
        if r_order == p_order:
            unchanged += 1
            continue
        first, second = pair
        aromatic = pair_region.get(pair) if (
            r_order == AROMATIC_ORDER or p_order == AROMATIC_ORDER
        ) else None
        edits.append({
            "pair": [first, second],
            "elements": [elements.get(first, "?"), elements.get(second, "?")],
            "r_bond_order": r_order,
            "p_bond_order": p_order,
            "edit_kind": _edit_kind(r_order, p_order),
            "aromatic_region": aromatic,
        })
    regions_with_edits = {
        region_id: [[first, second] for first, second in members]
        for region_id, members in regions.items()
        if any(edit["aromatic_region"] == region_id for edit in edits)
    }
    adjacency: dict[int, set[int]] = {}
    for first, second in all_pairs:
        adjacency.setdefault(first, set()).add(second)
        adjacency.setdefault(second, set()).add(first)
    return {
        "edits": edits,
        "hydrogen_partner_changes": hydrogen_partner_changes(elements, r_orders, p_orders),
        "aromatic_regions": regions_with_edits,
        "multi_bond_pairs": sorted([*r_multi, *p_multi]),
        "n_unchanged_pairs": unchanged,
        "_adjacency": {atom: sorted(neighbours) for atom, neighbours in sorted(adjacency.items())},
    }


def reaction_center(
    edits: Iterable[Mapping[str, object]],
    hydrogen_changes: Iterable[Mapping[str, object]],
    adjacency: Mapping[int, Sequence[int]],
    shell: int,
) -> dict[str, list[int]]:
    """Return the v2 reaction center (edit endpoints + H-change atoms)."""
    core: set[int] = set()
    for edit in edits:
        first, second = (int(value) for value in edit["pair"])  # type: ignore[index]
        core.update((first, second))
    for change in hydrogen_changes:
        core.add(int(change["h"]))  # type: ignore[index]
        for key in ("from", "to"):
            value = change.get(key)  # type: ignore[typeddict-item]
            if value is not None:
                core.add(int(value))  # type: ignore[arg-type]
    expanded = set(core)
    neighbour_map = {int(atom): set(int(n) for n in neighbours) for atom, neighbours in adjacency.items()}
    for _ in range(max(shell, 0)):
        neighbours = {n for atom in expanded for n in neighbour_map.get(atom, ())}
        expanded |= neighbours
    return {"core": sorted(core), "with_shell": sorted(expanded)}


def validate_edit_contract(
    edits: Sequence[Mapping[str, object]],
    hydrogen_changes: Sequence[Mapping[str, object]],
    center: Mapping[str, Sequence[int]],
    multi_bond_pairs: Sequence[tuple[int, int]],
) -> list[str]:
    """Return every internal-contract violation of one v2 edit block.

    Checks: pair uniqueness across all edits; per-edit exclusivity and order
    consistency with the declared kind; hydrogen records have differing
    from/to partners with a kind matching their states; and the center core
    covers every edit endpoint and every hydrogen-change atom/partner.
    """
    problems: list[str] = []
    if multi_bond_pairs:
        problems.append(ISSUE_MULTI_BOND_PAIR)
    seen_pairs: set[MapPair] = set()
    for edit in edits:
        pair = tuple(int(value) for value in edit["pair"])  # type: ignore[index]
        if pair in seen_pairs:
            problems.append(ISSUE_EDIT_CONTRACT_VIOLATION)
            continue
        seen_pairs.add(pair)
        r_order = edit["r_bond_order"]
        p_order = edit["p_bond_order"]
        kind = str(edit["edit_kind"])
        expected = _edit_kind(
            None if r_order is None else float(r_order),  # type: ignore[arg-type]
            None if p_order is None else float(p_order),  # type: ignore[arg-type]
        )
        if kind != expected:
            problems.append(ISSUE_EDIT_CONTRACT_VIOLATION)
    core = {int(value) for value in center["core"]}
    for change in hydrogen_changes:
        if change["from"] == change["to"]:
            problems.append(ISSUE_EDIT_CONTRACT_VIOLATION)
        for key in ("h", "from", "to"):
            value = change.get(key)
            if value is not None and int(value) not in core:  # type: ignore[arg-type]
                problems.append(ISSUE_EDIT_CONTRACT_VIOLATION)
    return sorted(set(problems))


def legacy_events_from_graph(
    elements: Mapping[int, str],
    r_bonds: Sequence[Sequence[float]],
    p_bonds: Sequence[Sequence[float]],
) -> dict[str, list[dict[str, object]]]:
    """Return the v1-semantics event lists recomputed from the graph.

    ``formed``/``broken`` are key-set differences (pair **plus order**), and
    ``order_changed`` lists pairs bonded on both sides whose order sets
    differ -- exactly the :mod:`pes2ts_core.g1.bond_changes` semantics.  The
    v2 build compares these against the events persisted in the P1 document
    to prove the v1 record is derivable from the same source graph.
    """
    r_orders, _ = side_pair_orders(r_bonds)
    p_orders, _ = side_pair_orders(p_bonds)
    r_keys = {(pair, order) for pair, order in r_orders.items() if order is not None}
    p_keys = {(pair, order) for pair, order in p_orders.items() if order is not None}
    formed = [
        {"atoms": [pair[0], pair[1]], "order_r": None, "order_p": order}
        for pair, order in sorted(p_keys - r_keys)
    ]
    broken = [
        {"atoms": [pair[0], pair[1]], "order_r": order, "order_p": None}
        for pair, order in sorted(r_keys - p_keys)
    ]
    order_changed = [
        {
            "atoms": [pair[0], pair[1]],
            "order_r": r_orders[pair],
            "order_p": p_orders[pair],
        }
        for pair in sorted(set(r_orders) & set(p_orders))
        if r_orders[pair] != p_orders[pair]
    ]
    r_partners = _hydrogen_partners(elements, r_orders)
    p_partners = _hydrogen_partners(elements, p_orders)
    migration = [
        {"h": h, "from": r_partners.get(h), "to": p_partners.get(h)}
        for h in sorted(set(r_partners) | set(p_partners))
        if r_partners.get(h) != p_partners.get(h)
    ]
    return {
        "formed": formed,
        "broken": broken,
        "order_changed": order_changed,
        "hydrogen_migration": migration,
    }


def legacy_reconciles(
    persisted: Mapping[str, Sequence[Mapping[str, object]]] | None,
    recomputed: Mapping[str, Sequence[Mapping[str, object]]],
) -> bool:
    """Return whether the persisted v1 events equal the recomputed ones."""
    if not isinstance(persisted, Mapping):
        return False
    for key, expected in recomputed.items():
        actual = persisted.get(key)
        if not isinstance(actual, list) or len(actual) != len(expected):
            return False
        for entry, wanted in zip(actual, expected, strict=True):
            if not isinstance(entry, Mapping):
                return False
            if sorted(entry.keys()) != sorted(wanted.keys()):
                return False
            if entry.get("atoms") != wanted.get("atoms"):
                return False
            for order_key in ("order_r", "order_p", "from", "to"):
                if order_key in wanted and entry.get(order_key) != wanted.get(order_key):
                    return False
    return True


def edit_signature(
    elements: Mapping[int, str],
    r_bonds: Sequence[Sequence[float]],
    p_bonds: Sequence[Sequence[float]],
) -> tuple[object, ...]:
    """Return the element-labeled chemical signature of one graph pair.

    The signature is the sorted multiset of element-labelled edits plus the
    per-element multisets of per-atom edit involvements and hydrogen-change
    kinds.  It is invariant under any global map relabeling, so the collapse
    audit can compare the signature of the selected assignment against every
    enumerated alternative: differing signatures prove the collapse changes
    the recorded chemistry (``collapse_edit_ambiguous``).
    """
    block = exclusive_edits(elements, r_bonds, p_bonds)
    edits: list[dict[str, object]] = block["edits"]  # type: ignore[assignment]
    changes: list[dict[str, object]] = block["hydrogen_partner_changes"]  # type: ignore[assignment]
    edit_multiset = sorted(
        (
            edit["elements"][0], edit["elements"][1],  # type: ignore[index]
            _ORDER_SORT_KEY.get(edit["r_bond_order"], edit["r_bond_order"]),  # type: ignore[index]
            _ORDER_SORT_KEY.get(edit["p_bond_order"], edit["p_bond_order"]),  # type: ignore[index]
            edit["edit_kind"],  # type: ignore[index]
        )
        for edit in edits
    )
    atom_roles: dict[int, list[object]] = {}
    for edit in edits:
        first, second = (int(value) for value in edit["pair"])  # type: ignore[index]
        kind = edit["edit_kind"]  # type: ignore[index]
        r_order = edit["r_bond_order"]  # type: ignore[index]
        p_order = edit["p_bond_order"]  # type: ignore[index]
        atom_roles.setdefault(first, []).append(
            (edit["elements"][1], kind, r_order, p_order)  # type: ignore[index]
        )
        atom_roles.setdefault(second, []).append(
            (edit["elements"][0], kind, r_order, p_order)  # type: ignore[index]
        )
    for change in changes:
        h = int(change["h"])
        atom_roles.setdefault(h, []).append(
            ("Hc", change["kind"], change["from_state"], change["to_state"])
        )
    roles_by_element: dict[str, list[tuple[object, ...]]] = {}
    for atom, roles in atom_roles.items():
        roles_by_element.setdefault(elements.get(atom, "?"), []).append(sorted(roles, key=repr))
    role_signature = tuple(
        (element, sorted((repr(role) for role in roles)))
        for element, roles in sorted(roles_by_element.items())
    )
    h_multiset = sorted(
        (
            elements.get(int(change["h"]), "?"),
            elements.get(int(change["from"]), "?") if change["from"] is not None else "",
            elements.get(int(change["to"]), "?") if change["to"] is not None else "",
            change["kind"],
        )
        for change in changes
    )
    return (tuple(edit_multiset), role_signature, tuple(h_multiset))


def edit_counts(block: Mapping[str, object]) -> dict[str, int]:
    """Return per-kind edit and hydrogen-change counts of one v2 block."""
    edits: Sequence[Mapping[str, object]] = block["edits"]  # type: ignore[assignment]
    changes: Sequence[Mapping[str, object]] = block["hydrogen_partner_changes"]  # type: ignore[assignment]
    counts = {
        kind: sum(1 for edit in edits if edit["edit_kind"] == kind)
        for kind in (EDIT_FORMED, EDIT_BROKEN, EDIT_ORDER_CHANGED)
    }
    for change in changes:
        counts[str(change["kind"])] = counts.get(str(change["kind"]), 0) + 1
    return counts


__all__ = [
    "edit_counts",
    "edit_signature",
    "exclusive_edits",
    "hydrogen_partner_changes",
    "legacy_events_from_graph",
    "legacy_reconciles",
    "reaction_center",
    "side_pair_orders",
    "validate_edit_contract",
]
