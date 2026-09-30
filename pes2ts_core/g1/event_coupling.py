"""Typed event coupling graph H over bundle + edit graph + endpoint context.

Design §4.3: merge atom-level edits into chemically meaningful **event nodes**
and record **typed coupling links** between them.  Event node types:

- ``H_TRANSFER``      -- one explicit H partner change (kind ``transfer`` /
  ``release`` / ``capture``; v2_edits taxonomy), owning its H–X bond edits;
- ``H2_EVENT``        -- one H–H cluster from ``HH_EVENT_KINDS`` changes
  (``to_hh`` / ``from_hh`` / ``hh_release`` / ``hh_form_free`` / ``hh_swap``),
  deduplicated by the unordered H–H pair set so two H records never count as
  two reactions; H2 formation is never labelled an H transfer;
- ``CONNECTIVITY_EXCHANGE`` -- formed+broken edits sharing a non-H centre atom
  (substitution pattern A–B broken + B–C formed sharing B);
- ``RING_REORGANIZATION``   -- one todo-8 ring group (edits whose endpoint
  support paths / shared ring regions evidence one ring creation or
  destruction; two disjoint formed bonds closing one ring = ONE event);
- ``DELOCALIZED_REGION``    -- all edits inside one aromatic/conjugated region
  (v2 region map; one region = ONE event, never six per-bond coordinates);
  isolated bond-order rearrangements without any other rule are degenerate
  singletons of the same electronic-rearrangement class;
- ``SHARED_CENTER``         -- leftover edits grouped by a shared
  edit-graph centre atom (design: strong local association candidate);
- ``CONTEXT_NEAR``          -- weak link only (skeleton shells overlap within
  ``context_radius`` but no strong evidence); it NEVER merges events.

Every coupling rule returns a typed :class:`CouplingRecord`
(``rule_id`` / ``support_atom_maps`` / ``support_edges`` / ``evidence_level``).
``strong_components`` are union-find over **strong** links only (shared-edit
membership + SHARED_CENTER); weak ``CONTEXT_NEAR`` links are listed separately
and no transitive closure is ever taken over them — two remote edits in one
molecule are not auto-synchronised.  The membership registry maps every edit
pair to the events that own it (exactly one, or explicitly shared with a
per-membership justification rule such as ``shared_center`` /
``aromatic_region_membership`` / ``ring_group_membership``).

GOLDEN projections (reproduced from ``build_evidence.py`` / v2_edits through
the EndpointGraphBundle alone): ``hydrogen_events`` (``{from, from_state, h,
kind, to, to_state}`` per hydrogen partner change) and ``aromatic_regions``
(``{region_id: [[a, b], ...]}``, ``{}`` when no edited aromatic region).

This module only computes event structure — it never generates scan
coordinates, never reads the data trees, and never touches truth.  Everything
is pure map-space graph math; inputs are read-only imports of the todo-5/7/8
contracts (adapters live here; those modules are not modified).
"""

# noqa: SIZE_OK — task-specified single §4.3 event-coupling contract module
# (typed events + coupling rules + strong/weak components + membership
# registry + golden H/aromatic projections); same precedent as
# endpoint_graph.py (todo 5), reaction_edit_graph.py (todo 7) and
# endpoint_context.py (todo 8).

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from pes2ts_core.g1.endpoint_context import EndpointContext
from pes2ts_core.g1.endpoint_graph import EndpointGraphBundle
from pes2ts_core.g1.reaction_edit_graph import ReactionEditGraph
from pes2ts_core.g1.v2_edits import exclusive_edits, hydrogen_partner_changes
from pes2ts_core.g1.v2_schema import (
    AROMATIC_ORDER,
    EDIT_BROKEN,
    EDIT_FORMED,
    EDIT_ORDER_CHANGED,
    H_CHANGE_CAPTURE,
    H_CHANGE_RELEASE,
    H_CHANGE_TRANSFER,
    HH_EVENT_KINDS,
)

SCHEMA_VERSION: Final[str] = "g1_event_coupling_v1"

# ---------------------------------------------------------------------------
# Event node types (design §4.3 vocabulary; CONTEXT_NEAR is link-only).
# ---------------------------------------------------------------------------
EVENT_H_TRANSFER: Final[str] = "H_TRANSFER"
EVENT_H2: Final[str] = "H2_EVENT"
EVENT_CONNECTIVITY_EXCHANGE: Final[str] = "CONNECTIVITY_EXCHANGE"
EVENT_RING: Final[str] = "RING_REORGANIZATION"
EVENT_DELOCALIZED: Final[str] = "DELOCALIZED_REGION"
EVENT_SHARED_CENTER: Final[str] = "SHARED_CENTER"
EVENT_CONTEXT_NEAR: Final[str] = "CONTEXT_NEAR"
#: Full §4.3 vocabulary, in documentation order.  Event *nodes* use the first
#: six; ``CONTEXT_NEAR`` materialises only as weak link records.
EVENT_TYPES: Final[tuple[str, ...]] = (
    EVENT_H_TRANSFER,
    EVENT_H2,
    EVENT_CONNECTIVITY_EXCHANGE,
    EVENT_RING,
    EVENT_DELOCALIZED,
    EVENT_SHARED_CENTER,
    EVENT_CONTEXT_NEAR,
)
_EVENT_TYPE_RANK: Final[dict[str, int]] = {name: i for i, name in enumerate(EVENT_TYPES)}

# ---------------------------------------------------------------------------
# Evidence levels + rule ids.
# ---------------------------------------------------------------------------
EVIDENCE_STRONG: Final[str] = "strong"
EVIDENCE_WEAK: Final[str] = "weak"

RULE_H_TRANSFER: Final[str] = "h_transfer"
RULE_H2_EVENT: Final[str] = "h2_event"
RULE_CONNECTIVITY_EXCHANGE: Final[str] = "connectivity_exchange"
RULE_CONNECTIVITY_SINGLE: Final[str] = "connectivity_single"
RULE_RING_REORGANIZATION: Final[str] = "ring_reorganization"
RULE_DELOCALIZED_REGION: Final[str] = "delocalized_region"
RULE_ORDER_CHANGE_SINGLETON: Final[str] = "order_change_singleton"
RULE_SHARED_CENTER: Final[str] = "shared_center"
RULE_SHARED_EDIT: Final[str] = "shared_edit_membership"
RULE_CONTEXT_NEAR: Final[str] = "context_near"

# Multi-membership justification rules (registry, design §4.3 membership).
JUST_AROMATIC_REGION: Final[str] = "aromatic_region_membership"
JUST_RING_GROUP: Final[str] = "ring_group_membership"
JUST_SHARED_CENTER: Final[str] = "shared_center"
JUST_H_PARTNER: Final[str] = "h_partner_change"

_FREE_TEXT: Final[str] = "free"
_HYDROGEN: Final[str] = "H"

#: Sorted unordered map pair.
type MapPair = tuple[int, int]


@dataclass(frozen=True, slots=True)
class CouplingRecord:
    """Typed output of one coupling rule (design §4.3 required fields)."""

    rule_id: str
    evidence_level: str
    event_ids: tuple[str, ...]
    support_atom_maps: tuple[int, ...]
    support_edges: tuple[MapPair, ...]

    def to_record(self) -> dict[str, Any]:
        """JSON-safe projection; stable field set."""
        return {
            "rule_id": self.rule_id,
            "evidence_level": self.evidence_level,
            "event_ids": list(self.event_ids),
            "support_atom_maps": list(self.support_atom_maps),
            "support_edges": [list(edge) for edge in self.support_edges],
        }


@dataclass(frozen=True, slots=True)
class MembershipJustification:
    """Why one edit is additionally claimed by one named event."""

    rule_id: str
    event_id: str
    detail: str

    def to_record(self) -> dict[str, Any]:
        """JSON-safe projection."""
        return {
            "rule_id": self.rule_id,
            "event_id": self.event_id,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class MembershipRecord:
    """Registry row: edit key -> owning event ids (+ justifications)."""

    edit_pair: MapPair
    event_ids: tuple[str, ...]
    justifications: tuple[MembershipJustification, ...]

    def to_record(self) -> dict[str, Any]:
        """JSON-safe projection."""
        return {
            "edit_pair": [self.edit_pair[0], self.edit_pair[1]],
            "event_ids": list(self.event_ids),
            "justifications": [j.to_record() for j in self.justifications],
        }


@dataclass(frozen=True, slots=True)
class EventNode:
    """One chemically meaningful event node of graph H."""

    event_id: str
    event_type: str
    rule_id: str
    edit_pairs: tuple[MapPair, ...]
    support_atom_maps: tuple[int, ...]
    support_edges: tuple[MapPair, ...]
    evidence_level: str
    metadata: tuple[tuple[str, str], ...]

    def to_record(self) -> dict[str, Any]:
        """JSON-safe projection; ``metadata`` keys sorted via stable dumps."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "rule_id": self.rule_id,
            "edit_pairs": [list(pair) for pair in self.edit_pairs],
            "support_atom_maps": list(self.support_atom_maps),
            "support_edges": [list(edge) for edge in self.support_edges],
            "evidence_level": self.evidence_level,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class EventCouplingGraph:
    """Complete typed event coupling graph H of one mapped reaction."""

    schema_version: str
    context_radius: int
    events: tuple[EventNode, ...]
    coupling_records: tuple[CouplingRecord, ...]
    strong_components: tuple[tuple[str, ...], ...]
    weak_links: tuple[CouplingRecord, ...]
    membership: tuple[MembershipRecord, ...]
    hydrogen_events: tuple[dict[str, Any], ...]
    aromatic_regions: dict[str, list[list[int]]]

    def events_by_type(self, event_type: str) -> tuple[EventNode, ...]:
        """Return every event node of one type (sorted by event_id)."""
        return tuple(e for e in self.events if e.event_type == event_type)

    def component_of(self, event_id: str) -> tuple[str, ...]:
        """Return the strong component containing ``event_id``."""
        for component in self.strong_components:
            if event_id in component:
                return component
        raise KeyError(event_id)

    def membership_map(self) -> dict[MapPair, tuple[str, ...]]:
        """Return ``edit pair -> owning event ids`` registry mapping."""
        return {row.edit_pair: row.event_ids for row in self.membership}

    def to_doc(self) -> dict[str, Any]:
        """JSON-safe projection; never contains truth-derived keys."""
        return {
            "schema_version": self.schema_version,
            "context_radius": self.context_radius,
            "events": [event.to_record() for event in self.events],
            "strong_components": [list(c) for c in self.strong_components],
            "weak_links": [link.to_record() for link in self.weak_links],
            "coupling_records": [rec.to_record() for rec in self.coupling_records],
            "membership": [row.to_record() for row in self.membership],
            "hydrogen_events": [dict(item) for item in self.hydrogen_events],
            "aromatic_regions": {
                region_id: [list(pair) for pair in members]
                for region_id, members in sorted(self.aromatic_regions.items())
            },
        }


# ---------------------------------------------------------------------------
# Small helpers.
# ---------------------------------------------------------------------------


def _sorted_pair(first: int, second: int) -> MapPair:
    return (first, second) if first <= second else (second, first)


def _field(record: Any, name: str) -> Any:
    """Read a field from an edit/change record (attribute or mapping)."""
    if isinstance(record, Mapping):
        return record[name]
    return getattr(record, name)


def _partner_text(partner: int | None) -> str:
    return _FREE_TEXT if partner is None else str(int(partner))


def _side_orders(graph: Any) -> dict[MapPair, float]:
    """Return ``pair -> bond order`` of one side; aromatic stays ``1.5``."""
    orders: dict[MapPair, float] = {}
    for edge in graph.edges:
        pair = _sorted_pair(int(edge.map_a), int(edge.map_b))
        order = AROMATIC_ORDER if edge.aromatic else float(edge.bond_order)
        orders[pair] = min(orders.get(pair, order), order)
    return orders


def _bundle_elements(bundle: EndpointGraphBundle) -> dict[int, str]:
    """Return ``map_id -> element`` from the R graph (conservation-checked)."""
    elements = {int(node.map_id): str(node.element) for node in bundle.r_graph.nodes}
    for node in bundle.p_graph.nodes:
        elements.setdefault(int(node.map_id), str(node.element))
    return elements


def _adjacency(map_ids: Sequence[int], edges: Sequence[MapPair]) -> dict[int, list[int]]:
    """Sorted adjacency over an undirected map-pair edge list."""
    adjacency: dict[int, list[int]] = {m: [] for m in map_ids}
    for left, right in edges:
        adjacency.setdefault(left, []).append(right)
        adjacency.setdefault(right, []).append(left)
    for members in adjacency.values():
        members.sort()
    return adjacency


def _union_find_groups(
    items: Sequence[Any],
    share_keys: Any,
    *,
    sort_key: Any,
) -> list[list[Any]]:
    """Union-find groups of ``items`` linked when ``share_keys`` intersect.

    Deterministic: seeds processed in order; each group sorted by
    ``sort_key``; groups ordered by their minimum ``sort_key``.
    """
    parent = list(range(len(items)))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a: int, b: int) -> None:
        root_a, root_b = find(a), find(b)
        if root_a == root_b:
            return
        if root_a < root_b:
            parent[root_b] = root_a
        else:
            parent[root_a] = root_b

    key_sets = [set(share_keys(item)) for item in items]
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if key_sets[i] & key_sets[j]:
                union(i, j)
    groups: dict[int, list[Any]] = {}
    for index, item in enumerate(items):
        groups.setdefault(find(index), []).append(item)
    ordered = sorted(groups.values(), key=lambda g: min(sort_key(x) for x in g))
    return [sorted(group, key=sort_key) for group in ordered]


# ---------------------------------------------------------------------------
# Golden projections (bundle-only; build_evidence.py / v2_edits port).
# ---------------------------------------------------------------------------


def hydrogen_events_from_bundle(bundle: EndpointGraphBundle) -> tuple[dict[str, Any], ...]:
    """Return evidence ``hydrogen_events`` from the endpoint graphs alone.

    Port path: map-space orders (aromatic ``1.5``) + elements ->
    :func:`pes2ts_core.g1.v2_edits.hydrogen_partner_changes` (the exact
    producer of the Demo24 evidence ``hydrogen_partner_changes`` block that
    ``build_evidence.py`` copies into ``features.hydrogen_events``).
    """
    elements = _bundle_elements(bundle)
    changes = hydrogen_partner_changes(
        elements, _side_orders(bundle.r_graph), _side_orders(bundle.p_graph)
    )
    return tuple(dict(change) for change in changes)


def aromatic_regions_from_bundle(bundle: EndpointGraphBundle) -> dict[str, list[list[int]]]:
    """Return evidence ``aromatic_regions`` (``{}`` or region map) from graphs.

    Port path: same orders/elements -> :func:`pes2ts_core.g1.v2_edits.exclusive_edits`
    ``aromatic_regions`` (regions containing at least one edited aromatic bond;
    region id = SHA256 over member pairs, first 12 hex chars).
    """
    elements = _bundle_elements(bundle)
    r_orders = _side_orders(bundle.r_graph)
    p_orders = _side_orders(bundle.p_graph)
    r_bonds = [[a, b, order] for (a, b), order in sorted(r_orders.items())]
    p_bonds = [[a, b, order] for (a, b), order in sorted(p_orders.items())]
    block = exclusive_edits(elements, r_bonds, p_bonds)
    regions = block["aromatic_regions"]
    return {str(rid): [list(pair) for pair in members] for rid, members in regions.items()}


# ---------------------------------------------------------------------------
# Builder.
# ---------------------------------------------------------------------------


@dataclass
class _EditInfo:
    """Internal per-edit carrier used during event assembly."""

    pair: MapPair
    kind: str
    aromatic_region: str | None


@dataclass
class _EventDraft:
    """Internal event under construction (pre-sort)."""

    event_id: str
    event_type: str
    rule_id: str
    edits: list[MapPair]
    support_atoms: set[int]
    support_edges: set[MapPair]
    metadata: dict[str, str]


def _finalize_event(draft: _EventDraft) -> EventNode:
    """Freeze one draft into a deterministically ordered EventNode."""
    return EventNode(
        event_id=draft.event_id,
        event_type=draft.event_type,
        rule_id=draft.rule_id,
        edit_pairs=tuple(sorted(set(draft.edits))),
        support_atom_maps=tuple(sorted(draft.support_atoms)),
        support_edges=tuple(sorted(set(draft.support_edges))),
        evidence_level=EVIDENCE_STRONG,
        metadata=tuple(sorted(draft.metadata.items())),
    )


def _justification_for(
    edit: _EditInfo,
    primary_id: str,
    other_id: str,
    other_type: str,
    other_meta: Mapping[str, str],
) -> MembershipJustification:
    """Return the typed justification for one additional event membership."""
    if other_type == EVENT_DELOCALIZED and edit.aromatic_region:
        detail = f"aromatic_region:{edit.aromatic_region}"
        return MembershipJustification(JUST_AROMATIC_REGION, other_id, detail)
    if other_type == EVENT_RING and "ring_group_id" in other_meta:
        detail = f"ring_group:{other_meta['ring_group_id']}"
        return MembershipJustification(JUST_RING_GROUP, other_id, detail)
    if other_type in (EVENT_H_TRANSFER, EVENT_H2):
        detail = f"primary:{primary_id}"
        return MembershipJustification(JUST_H_PARTNER, other_id, detail)
    detail = f"primary:{primary_id}"
    return MembershipJustification(JUST_SHARED_CENTER, other_id, detail)


def build_event_coupling_graph(
    bundle: EndpointGraphBundle,
    edit_graph: ReactionEditGraph,
    context: EndpointContext,
) -> EventCouplingGraph:
    """Build the typed event coupling graph H of one mapped reaction.

    Inputs are the frozen todo-5 bundle, todo-7 edit graph and todo-8 context
    (read-only).  All seven §4.3 rule families are applied deterministically;
    strong components union-find over strong links only; every edit pair
    appears in the membership registry exactly once or is explicitly shared
    with a per-additional-membership justification.
    """
    radius = int(context.context_radius)
    hydrogen_events = hydrogen_events_from_bundle(bundle)
    aromatic_regions = aromatic_regions_from_bundle(bundle)

    edits: list[_EditInfo] = []
    edits_by_pair: dict[MapPair, _EditInfo] = {}
    for record in edit_graph.edits:
        pair_raw = _field(record, "pair")
        pair = _sorted_pair(int(pair_raw[0]), int(pair_raw[1]))
        info = _EditInfo(
            pair=pair,
            kind=str(_field(record, "edit_kind")),
            aromatic_region=(
                None
                if _field(record, "aromatic_region") is None
                else str(_field(record, "aromatic_region"))
            ),
        )
        edits.append(info)
        edits_by_pair[pair] = info
    edits.sort(key=lambda e: e.pair)
    edit_atoms = sorted({atom for e in edits for atom in e.pair})

    # edit-graph centre atoms (todo 7) reused for SHARED_CENTER evidence.
    center_atoms = set(int(a) for a in getattr(edit_graph, "center_atoms", ()))
    edit_degree = {int(k): int(v) for k, v in getattr(edit_graph, "edit_degree", {}).items()}

    elements = _bundle_elements(bundle)
    union_edges: list[MapPair] = []
    for edge in bundle.r_graph.edges:
        union_edges.append(_sorted_pair(int(edge.map_a), int(edge.map_b)))
    for edge in bundle.p_graph.edges:
        union_edges.append(_sorted_pair(int(edge.map_a), int(edge.map_b)))
    union_edges = sorted(set(union_edges))
    union_adj = _adjacency(edit_atoms, union_edges)

    def edit_shell(pair: MapPair) -> frozenset[int]:
        """Hop-radius shell of one edit on the R∪P invariant skeleton."""
        distances: dict[int, int] = {atom: 0 for atom in pair if atom in union_adj}
        queue = sorted(distances)
        head = 0
        while head < len(queue):
            node = queue[head]
            head += 1
            if distances[node] >= radius:
                continue
            for nxt in union_adj.get(node, ()):
                if nxt not in distances:
                    distances[nxt] = distances[node] + 1
                    queue.append(nxt)
        return frozenset(distances)

    shells = {e.pair: edit_shell(e.pair) for e in edits}

    # ------------------------------------------------------------------
    # Rule 1/2: H_TRANSFER + H2_EVENT (v2_edits taxonomy, H2 dedup).
    # ------------------------------------------------------------------
    events: list[_EventDraft] = []
    primary_owner: dict[MapPair, str] = {}
    hh_cluster_members: dict[MapPair, int] = {}  # H-H pair -> cluster index
    hh_clusters: list[set[int]] = []

    def _hh_cluster(h_pair: MapPair) -> int:
        for index, cluster in enumerate(hh_clusters):
            if set(h_pair) & cluster:
                cluster |= set(h_pair)
                return index
        hh_clusters.append(set(h_pair))
        return len(hh_clusters) - 1

    transfer_changes: list[dict[str, Any]] = []
    hh_changes: list[dict[str, Any]] = []
    for change in hydrogen_events:
        kind = str(change["kind"])
        if kind in HH_EVENT_KINDS:
            hh_changes.append(dict(change))
            from_partner = change["from"]
            to_partner = change["to"]
            if from_partner is not None and change["from_state"] == "hh":
                idx = _hh_cluster(_sorted_pair(int(change["h"]), int(from_partner)))
                hh_cluster_members[_sorted_pair(int(change["h"]), int(from_partner))] = idx
            if to_partner is not None and change["to_state"] == "hh":
                idx = _hh_cluster(_sorted_pair(int(change["h"]), int(to_partner)))
                hh_cluster_members[_sorted_pair(int(change["h"]), int(to_partner))] = idx
        else:
            transfer_changes.append(dict(change))

    for cluster_index, cluster in enumerate(sorted(hh_clusters, key=lambda s: sorted(s))):
        cluster_pairs = sorted(
            pair for pair, idx in hh_cluster_members.items() if idx == cluster_index
        )
        atoms = sorted(cluster)
        event_id = EVENT_H2 + ":hh" + "_".join(str(a) for a in atoms)
        draft = _EventDraft(
            event_id=event_id,
            event_type=EVENT_H2,
            rule_id=RULE_H2_EVENT,
            edits=[],
            support_atoms=set(atoms),
            support_edges=set(cluster_pairs),
            metadata={"hh_cluster": "_".join(str(a) for a in atoms)},
        )
        for change in hh_changes:
            h_atom = int(change["h"])
            if h_atom not in cluster:
                continue
            for key, state_key in (("from", "from_state"), ("to", "to_state")):
                partner = change[key]
                if partner is None:
                    continue
                partner = int(partner)
                draft.support_atoms.add(partner)
                if change[state_key] == _HYDROGEN:
                    ep = _sorted_pair(h_atom, partner)
                    if ep in edits_by_pair:
                        draft.support_edges.add(ep)
                else:
                    ep = _sorted_pair(h_atom, partner)
                    if ep in edits_by_pair and ep not in primary_owner:
                        draft.edits.append(ep)
                        primary_owner[ep] = event_id
        for pair in cluster_pairs:
            if pair in edits_by_pair and pair not in primary_owner:
                draft.edits.append(pair)
                primary_owner[pair] = event_id
        events.append(draft)

    for change in sorted(
        transfer_changes,
        key=lambda c: (int(c["h"]), _partner_text(c["from"]), _partner_text(c["to"])),
    ):
        h_atom = int(change["h"])
        src = None if change["from"] is None else int(change["from"])
        dst = None if change["to"] is None else int(change["to"])
        event_id = (
            f"{EVENT_H_TRANSFER}:h{h_atom}:{_partner_text(src)}-{_partner_text(dst)}"
        )
        draft = _EventDraft(
            event_id=event_id,
            event_type=EVENT_H_TRANSFER,
            rule_id=RULE_H_TRANSFER,
            edits=[],
            support_atoms={h_atom},
            support_edges=set(),
            metadata={
                "h": str(h_atom),
                "kind": str(change["kind"]),
                "from": _partner_text(src),
                "from_state": str(change["from_state"]),
                "to": _partner_text(dst),
                "to_state": str(change["to_state"]),
            },
        )
        for partner in (src, dst):
            if partner is None:
                continue
            draft.support_atoms.add(partner)
            ep = _sorted_pair(h_atom, partner)
            if ep in edits_by_pair and ep not in primary_owner:
                draft.edits.append(ep)
                primary_owner[ep] = event_id
                draft.support_edges.add(ep)
        events.append(draft)

    # ------------------------------------------------------------------
    # Rule 3: CONNECTIVITY_EXCHANGE on remaining formed+broken edits.
    # ------------------------------------------------------------------
    remaining_fb = [
        e
        for e in edits
        if e.kind in (EDIT_FORMED, EDIT_BROKEN) and e.pair not in primary_owner
    ]
    exchange_groups = _union_find_groups(
        remaining_fb,
        lambda e: {a for a in e.pair if elements.get(a) != _HYDROGEN},
        sort_key=lambda e: e.pair,
    )
    for group in exchange_groups:
        shared_centers = sorted(
            atom
            for atom in {a for e in group for a in e.pair}
            if sum(1 for e in group if atom in e.pair) >= 2
            and elements.get(atom) != _HYDROGEN
        )
        if len(group) < 2 or not shared_centers:
            continue
        event_id = (
            EVENT_CONNECTIVITY_EXCHANGE
            + ":c"
            + "_".join(str(a) for a in shared_centers)
        )
        draft = _EventDraft(
            event_id=event_id,
            event_type=EVENT_CONNECTIVITY_EXCHANGE,
            rule_id=RULE_CONNECTIVITY_EXCHANGE,
            edits=[e.pair for e in group],
            support_atoms={a for e in group for a in e.pair},
            support_edges={e.pair for e in group},
            metadata={"centers": "_".join(str(a) for a in shared_centers)},
        )
        for e in group:
            primary_owner.setdefault(e.pair, event_id)
        events.append(draft)

    # ------------------------------------------------------------------
    # Rule 4: RING_REORGANIZATION from todo-8 ring groups.
    # ------------------------------------------------------------------
    for group in context.ring_groups:
        group_id = int(group.group_id)
        event_id = f"{EVENT_RING}:g{group_id}"
        group_pairs = [
            _sorted_pair(int(pair[0]), int(pair[1])) for pair in group.edit_pairs
        ]
        owned = [p for p in group_pairs if p in edits_by_pair]
        if not owned:
            continue
        draft = _EventDraft(
            event_id=event_id,
            event_type=EVENT_RING,
            rule_id=RULE_RING_REORGANIZATION,
            edits=[],
            support_atoms=set(int(a) for a in group.evidence_atoms),
            support_edges=set(owned),
            metadata={
                "ring_group_id": str(group_id),
                "basis": str(group.basis),
            },
        )
        for pair in owned:
            draft.support_atoms.update(pair)
            if pair not in primary_owner:
                draft.edits.append(pair)
                primary_owner[pair] = event_id
        events.append(draft)

    # ------------------------------------------------------------------
    # Rule 5: DELOCALIZED_REGION, one event per aromatic region with edits.
    # ------------------------------------------------------------------
    pair_region: dict[MapPair, str] = {}
    for region_id, members in aromatic_regions.items():
        for raw in members:
            pair_region[_sorted_pair(int(raw[0]), int(raw[1]))] = region_id
    for region_id in sorted(aromatic_regions):
        region_members = [
            _sorted_pair(int(raw[0]), int(raw[1]))
            for raw in aromatic_regions[region_id]
        ]
        owned = [p for p in region_members if p in edits_by_pair]
        if not owned:
            continue
        event_id = f"{EVENT_DELOCALIZED}:{region_id}"
        draft = _EventDraft(
            event_id=event_id,
            event_type=EVENT_DELOCALIZED,
            rule_id=RULE_DELOCALIZED_REGION,
            edits=[],
            support_atoms={a for p in region_members for a in p},
            support_edges=set(region_members),
            metadata={"region_id": region_id},
        )
        for pair in owned:
            if pair not in primary_owner:
                draft.edits.append(pair)
                primary_owner[pair] = event_id
        events.append(draft)

    # ------------------------------------------------------------------
    # Rule 6: leftovers -> SHARED_CENTER groups or kind-based singletons.
    # ------------------------------------------------------------------
    leftovers = [e for e in edits if e.pair not in primary_owner]
    leftover_groups = _union_find_groups(
        leftovers,
        lambda e: {a for a in e.pair if edit_degree.get(a, 0) >= 2},
        sort_key=lambda e: e.pair,
    )
    for group in leftover_groups:
        if len(group) >= 2:
            shared = sorted(
                atom
                for atom in {a for e in group for a in e.pair}
                if sum(1 for e in group if atom in e.pair) >= 2
            )
            event_id = EVENT_SHARED_CENTER + ":c" + "_".join(str(a) for a in shared)
            draft = _EventDraft(
                event_id=event_id,
                event_type=EVENT_SHARED_CENTER,
                rule_id=RULE_SHARED_CENTER,
                edits=[e.pair for e in group],
                support_atoms={a for e in group for a in e.pair},
                support_edges={e.pair for e in group},
                metadata={"centers": "_".join(str(a) for a in shared)},
            )
            for e in group:
                primary_owner.setdefault(e.pair, event_id)
            events.append(draft)
            continue
        e = group[0]
        if e.kind == EDIT_ORDER_CHANGED:
            event_id = f"{EVENT_DELOCALIZED}:edit{e.pair[0]}-{e.pair[1]}"
            rule_id = RULE_ORDER_CHANGE_SINGLETON
            event_type = EVENT_DELOCALIZED
        else:
            event_id = f"{EVENT_CONNECTIVITY_EXCHANGE}:edit{e.pair[0]}-{e.pair[1]}"
            rule_id = RULE_CONNECTIVITY_SINGLE
            event_type = EVENT_CONNECTIVITY_EXCHANGE
        draft = _EventDraft(
            event_id=event_id,
            event_type=event_type,
            rule_id=rule_id,
            edits=[e.pair],
            support_atoms=set(e.pair),
            support_edges={e.pair},
            metadata={"edit_kind": e.kind},
        )
        primary_owner[e.pair] = event_id
        events.append(draft)

    # ------------------------------------------------------------------
    # Additional memberships (multi-membership registry rows).
    # ------------------------------------------------------------------
    extra_members: dict[MapPair, list[str]] = {}
    event_meta_by_id: dict[str, dict[str, str]] = {}
    event_type_by_id: dict[str, str] = {}
    for draft in events:
        event_meta_by_id[draft.event_id] = dict(draft.metadata)
        event_type_by_id[draft.event_id] = draft.event_type

    for group in context.ring_groups:
        group_id = int(group.group_id)
        event_id = f"{EVENT_RING}:g{group_id}"
        if event_id not in event_type_by_id:
            continue
        for raw in group.edit_pairs:
            pair = _sorted_pair(int(raw[0]), int(raw[1]))
            if pair not in edits_by_pair:
                continue
            primary = primary_owner.get(pair)
            if primary is None or primary == event_id:
                continue
            extra_members.setdefault(pair, []).append(event_id)

    for pair, region_id in pair_region.items():
        if pair not in edits_by_pair:
            continue
        event_id = f"{EVENT_DELOCALIZED}:{region_id}"
        if event_id not in event_type_by_id:
            continue
        primary = primary_owner.get(pair)
        if primary is None or primary == event_id:
            continue
        extra_members.setdefault(pair, []).append(event_id)

    membership_rows: list[MembershipRecord] = []
    for e in edits:
        primary = primary_owner[e.pair]
        extras = sorted(set(extra_members.get(e.pair, ())))
        event_ids = tuple(sorted({primary, *extras}))
        justifications = tuple(
            sorted(
                (
                    _justification_for(
                        e, primary, other, event_type_by_id[other], event_meta_by_id[other]
                    )
                    for other in extras
                ),
                key=lambda j: (j.event_id, j.rule_id),
            )
        )
        membership_rows.append(
            MembershipRecord(
                edit_pair=e.pair, event_ids=event_ids, justifications=justifications
            )
        )

    # ------------------------------------------------------------------
    # Final event nodes (deterministic order) + per-event edit lists.
    # ------------------------------------------------------------------
    draft_by_id = {draft.event_id: draft for draft in events}
    for row in membership_rows:
        for event_id in row.event_ids:
            draft = draft_by_id[event_id]
            if row.edit_pair in draft.edits:
                continue
            draft.edits.append(row.edit_pair)
            draft.support_atoms.update(row.edit_pair)
            draft.support_edges.add(row.edit_pair)
    for draft in events:
        draft.edits = sorted(set(draft.edits))
        draft.support_edges = set(draft.support_edges) | set(draft.edits)

    final_events = sorted(
        (_finalize_event(draft) for draft in events),
        key=lambda e: (_EVENT_TYPE_RANK[e.event_type], e.event_id),
    )
    event_ids_sorted = [e.event_id for e in final_events]

    # ------------------------------------------------------------------
    # Coupling records: event evidence + strong links + weak CONTEXT_NEAR.
    # ------------------------------------------------------------------
    coupling: list[CouplingRecord] = []
    for event in final_events:
        coupling.append(
            CouplingRecord(
                rule_id=event.rule_id,
                evidence_level=EVIDENCE_STRONG,
                event_ids=(event.event_id,),
                support_atom_maps=event.support_atom_maps,
                support_edges=event.support_edges,
            )
        )

    edits_in_event: dict[str, set[MapPair]] = {e.event_id: set(e.edit_pairs) for e in final_events}
    atoms_in_event: dict[str, set[int]] = {
        e.event_id: set(e.support_atom_maps) for e in final_events
    }

    def _is_center(atom: int) -> bool:
        return atom in center_atoms or edit_degree.get(atom, 0) >= 2

    strong_links: list[CouplingRecord] = []
    weak_links: list[CouplingRecord] = []
    linked_pairs: set[tuple[str, str]] = set()

    # Shared-edit membership -> strong link.
    for row in membership_rows:
        if len(row.event_ids) < 2:
            continue
        ids = list(row.event_ids)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                key = (ids[i], ids[j]) if ids[i] <= ids[j] else (ids[j], ids[i])
                if key in linked_pairs:
                    continue
                linked_pairs.add(key)
                strong_links.append(
                    CouplingRecord(
                        rule_id=RULE_SHARED_EDIT,
                        evidence_level=EVIDENCE_STRONG,
                        event_ids=key,
                        support_atom_maps=tuple(row.edit_pair),
                        support_edges=(row.edit_pair,),
                    )
                )

    # SHARED_CENTER: events sharing a non-spectator centre atom.
    for i, id_a in enumerate(event_ids_sorted):
        for id_b in event_ids_sorted[i + 1 :]:
            key = (id_a, id_b)
            if key in linked_pairs:
                continue
            shared_centers = sorted(
                atom
                for atom in atoms_in_event[id_a] & atoms_in_event[id_b]
                if _is_center(atom)
            )
            if not shared_centers:
                continue
            linked_pairs.add(key)
            strong_links.append(
                CouplingRecord(
                    rule_id=RULE_SHARED_CENTER,
                    evidence_level=EVIDENCE_STRONG,
                    event_ids=key,
                    support_atom_maps=tuple(shared_centers),
                    support_edges=(),
                )
            )

    # CONTEXT_NEAR: shell overlap without strong evidence (weak by definition).
    for i, id_a in enumerate(event_ids_sorted):
        for id_b in event_ids_sorted[i + 1 :]:
            key = (id_a, id_b)
            if key in linked_pairs:
                continue
            near_atoms: set[int] = set()
            for pair_a in edits_in_event[id_a]:
                for pair_b in edits_in_event[id_b]:
                    overlap = shells[pair_a] & shells[pair_b]
                    if overlap:
                        near_atoms |= set(overlap)
            if not near_atoms:
                continue
            weak_links.append(
                CouplingRecord(
                    rule_id=RULE_CONTEXT_NEAR,
                    evidence_level=EVIDENCE_WEAK,
                    event_ids=key,
                    support_atom_maps=tuple(sorted(near_atoms)),
                    support_edges=(),
                )
            )

    strong_links.sort(key=lambda r: (r.event_ids, r.rule_id, r.support_atom_maps))
    weak_links.sort(key=lambda r: (r.event_ids, r.rule_id, r.support_atom_maps))

    # ------------------------------------------------------------------
    # Strong components: union-find over strong links ONLY.
    # ------------------------------------------------------------------
    parent = {event_id: event_id for event_id in event_ids_sorted}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a: str, b: str) -> None:
        root_a, root_b = find(a), find(b)
        if root_a == root_b:
            return
        if root_a < root_b:
            parent[root_b] = root_a
        else:
            parent[root_a] = root_b

    for link in strong_links:
        a, b = link.event_ids
        union(a, b)

    components_map: dict[str, list[str]] = {}
    for event_id in event_ids_sorted:
        components_map.setdefault(find(event_id), []).append(event_id)
    strong_components = tuple(
        sorted((tuple(sorted(members)) for members in components_map.values()), key=lambda c: c)
    )

    membership_rows.sort(key=lambda row: row.edit_pair)
    coupling.extend(strong_links)
    coupling.extend(weak_links)
    coupling.sort(key=lambda r: (r.rule_id, r.event_ids, r.support_atom_maps))

    return EventCouplingGraph(
        schema_version=SCHEMA_VERSION,
        context_radius=radius,
        events=tuple(final_events),
        coupling_records=tuple(coupling),
        strong_components=strong_components,
        weak_links=tuple(weak_links),
        membership=tuple(membership_rows),
        hydrogen_events=hydrogen_events,
        aromatic_regions=aromatic_regions,
    )


__all__ = [
    "SCHEMA_VERSION",
    "EVIDENCE_STRONG",
    "EVIDENCE_WEAK",
    "EVENT_CONTEXT_NEAR",
    "EVENT_CONNECTIVITY_EXCHANGE",
    "EVENT_DELOCALIZED",
    "EVENT_H2",
    "EVENT_H_TRANSFER",
    "EVENT_RING",
    "EVENT_SHARED_CENTER",
    "EVENT_TYPES",
    "JUST_AROMATIC_REGION",
    "JUST_H_PARTNER",
    "JUST_RING_GROUP",
    "JUST_SHARED_CENTER",
    "RULE_CONNECTIVITY_EXCHANGE",
    "RULE_CONNECTIVITY_SINGLE",
    "RULE_CONTEXT_NEAR",
    "RULE_DELOCALIZED_REGION",
    "RULE_H2_EVENT",
    "RULE_H_TRANSFER",
    "RULE_ORDER_CHANGE_SINGLETON",
    "RULE_RING_REORGANIZATION",
    "RULE_SHARED_CENTER",
    "RULE_SHARED_EDIT",
    "CouplingRecord",
    "EventCouplingGraph",
    "EventNode",
    "MapPair",
    "MembershipJustification",
    "MembershipRecord",
    "aromatic_regions_from_bundle",
    "build_event_coupling_graph",
    "hydrogen_events_from_bundle",
]
