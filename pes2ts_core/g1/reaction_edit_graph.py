"""Atom-level reaction edit graph ΔG from an EndpointGraphBundle (design §4.1).

The edit graph is pure map-space graph math over the complete R/P bond graphs
of a frozen :class:`pes2ts_core.g1.endpoint_graph.EndpointGraphBundle`:

- **F/B/O mutually exclusive attribution.**  Every unordered map pair whose
  bond order differs between R and P gets exactly one ``edit_kind`` —
  ``formed`` (R-absent/P-present), ``broken`` (R-present/P-absent), or
  ``order_changed`` (bonded on both sides, different order).  A raised-order
  pair is ONE ``order_changed`` record, never a simultaneous F and B (the v2
  contract, same tie-breaks as :mod:`pes2ts_core.g1.v2_edits` and the Demo24
  evidence builder ``outputs/pes_generation_strategy_design_v1/build_evidence.py``).
  Aromatic bonds stay ``1.5`` on both sides.  Hydrogen partner changes surface
  naturally as bond edits (H–X broken + H–Y formed → broken/formed pairs on
  ``(H,X)``/``(H,Y)``); no separate H path exists here.
- **atom_events / atom_attribute_changes.**  Every atom whose bond-partner
  set *or* attributes changed is listed independently of the edge edits.
  Attribute changes (formal charge, radical electrons, aromatic flag, chiral
  tag) reproduce the evidence ``atom_attribute_changes`` records byte-for-byte;
  element/isotope changes cannot occur in a valid bundle (conservation
  raises upstream).
- **Components and cycle rank.**  ``edit_components`` over all F/B/O edit
  edges, ``connectivity_edit_components`` over F/B-only edges, and
  ``edit_cycle_rank = |E| − |V| + C`` over the F/B/O edit graph.  Edit-graph
  connectivity is NOT mechanistic synchrony — that is the todo-9 coupling
  layer's job.
- **Determinism.**  Pairs sorted, components ordered by their minimum map
  (DFS port of the evidence builder), atoms map-ascending; the same input
  always serializes identically.

``aromatic_region`` ids are carried through from caller-supplied region
mappings (e.g. a bundle/document ``aromatic_regions`` block); region GROUPING
itself is todo-9 / ``v2_edits`` territory and is never computed here.  Fields
that belong to the todo-8 context layer (``distance_R_A``/``distance_P_A``/
``cross_component_*``/``support_path_*``) are deliberately absent; per-edit
records are plain dicts via :meth:`ReactionEdit.to_record` so todo 8 can
extend them additively.

Everything here is pure: no file I/O, no RDKit, no truth access.
"""

# noqa: SIZE_OK — task-specified single edit-graph contract module (design §4.1
# frozen schema + exclusive-edit builder + component math); same precedent as
# endpoint_graph.py (todo 5) and graph_rebuild.py (todo 6).

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from pes2ts_core.g1.endpoint_graph import (
    AtomNode,
    EndpointGraphBundle,
    SideGraph,
)
from pes2ts_core.g1.v2_schema import (
    AROMATIC_ORDER,
    EDIT_BROKEN,
    EDIT_FORMED,
    EDIT_ORDER_CHANGED,
)

SCHEMA_VERSION: Final[str] = "g1_reaction_edit_graph_v1"
#: Multi-edit centre threshold: atoms shared by at least this many F/B/O edits.
CENTER_DEGREE: Final[int] = 2

#: Sorted map-number pair.
type MapPair = tuple[int, int]
#: Aromatic-region input: either ``{region_id: [[a, b], ...]}`` or ``{(a, b): region_id}``.
type AromaticRegionsInput = Mapping[object, object]


@dataclass(frozen=True, slots=True)
class ReactionEdit:
    """One mutually exclusive bond edit on an unordered map pair."""

    pair: MapPair
    elements: tuple[str, str]
    r_bond_order: float | None
    p_bond_order: float | None
    edit_kind: str
    aromatic_region: str | None

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-safe evidence-schema record.

        Exactly the six golden fields.  Todo 8 extends per-edit records by
        copying this dict and adding context keys (``distance_R_A`` etc.).
        """
        first, second = self.pair
        return {
            "pair": [first, second],
            "elements": [self.elements[0], self.elements[1]],
            "r_bond_order": self.r_bond_order,
            "p_bond_order": self.p_bond_order,
            "edit_kind": self.edit_kind,
            "aromatic_region": self.aromatic_region,
        }


@dataclass(frozen=True, slots=True)
class ReactionEditGraph:
    """Complete atom-level edit graph ΔG of one mapped reaction."""

    edits: tuple[ReactionEdit, ...]
    edit_counts: dict[str, int]
    edit_components: tuple[tuple[int, ...], ...]
    connectivity_edit_components: tuple[tuple[int, ...], ...]
    edit_cycle_rank: int
    edit_degree: dict[int, int]
    center_atoms: tuple[int, ...]
    atom_events: tuple[dict[str, Any], ...]
    atom_attribute_changes: tuple[dict[str, Any], ...]

    def edit_records(self) -> list[dict[str, Any]]:
        """Return per-edit records (todo-8 extension seam)."""
        return [edit.to_record() for edit in self.edits]

    def to_doc(self) -> dict[str, Any]:
        """Return the JSON-safe document; no truth-derived keys ever."""
        return {
            "schema_version": SCHEMA_VERSION,
            "edits": self.edit_records(),
            "edit_counts": dict(self.edit_counts),
            "edit_components": [list(comp) for comp in self.edit_components],
            "connectivity_edit_components": [
                list(comp) for comp in self.connectivity_edit_components
            ],
            "edit_cycle_rank": self.edit_cycle_rank,
            "edit_degree": {str(k): v for k, v in sorted(self.edit_degree.items())},
            "center_atoms": list(self.center_atoms),
            "atom_events": [dict(event) for event in self.atom_events],
            "atom_attribute_changes": [
                dict(record) for record in self.atom_attribute_changes
            ],
        }


def _side_pair_orders(graph: SideGraph) -> dict[MapPair, float]:
    """Return ``pair -> bond order`` of one side; aromatic stays ``1.5``.

    Mirrors the evidence builder's ``graph()``: ``1.5 if aromatic else
    bond_order``.  A multi-bond pair (malformed source) keeps the minimum
    order for determinism, same as :func:`pes2ts_core.g1.v2_edits.side_pair_orders`.
    """
    orders: dict[MapPair, float] = {}
    for edge in graph.edges:
        pair: MapPair = (edge.map_a, edge.map_b)
        order = AROMATIC_ORDER if edge.aromatic else edge.bond_order
        if pair in orders:
            orders[pair] = min(orders[pair], order)
        else:
            orders[pair] = order
    return orders


def _normalize_aromatic_regions(
    regions: AromaticRegionsInput | None,
) -> dict[MapPair, str]:
    """Return ``pair -> region id`` from either accepted region mapping shape."""
    if regions is None:
        return {}
    pair_region: dict[MapPair, str] = {}
    for key, value in regions.items():
        if isinstance(key, str):
            for raw_pair in value:  # type: ignore[union-attr]
                first, second = int(raw_pair[0]), int(raw_pair[1])
                pair = (first, second) if first <= second else (second, first)
                pair_region[pair] = key
        else:
            first, second = int(key[0]), int(key[1])  # type: ignore[index]
            pair = (first, second) if first <= second else (second, first)
            pair_region[pair] = str(value)
    return pair_region


def _connected_components(
    vertices: Sequence[int], edges: Sequence[MapPair]
) -> list[list[int]]:
    """Return connected components ordered by minimum map (evidence port).

    Faithful port of ``build_evidence.py::components``: DFS from the smallest
    unseen vertex, neighbours explored in sorted order, each component sorted.
    """
    adjacency: dict[int, set[int]] = {v: set() for v in vertices}
    for first, second in edges:
        adjacency[first].add(second)
        adjacency[second].add(first)
    result: list[list[int]] = []
    unseen = set(vertices)
    while unseen:
        seed = min(unseen)
        seen = {seed}
        stack = [seed]
        unseen.remove(seed)
        while stack:
            for neighbour in sorted(adjacency[stack.pop()] & unseen):
                unseen.remove(neighbour)
                seen.add(neighbour)
                stack.append(neighbour)
        result.append(sorted(seen))
    return result


def _partner_maps(graph: SideGraph) -> dict[int, tuple[int, ...]]:
    """Return ``atom -> sorted bonded partners`` of one side."""
    partners: dict[int, set[int]] = {}
    for edge in graph.edges:
        partners.setdefault(edge.map_a, set()).add(edge.map_b)
        partners.setdefault(edge.map_b, set()).add(edge.map_a)
    return {atom: tuple(sorted(neighbours)) for atom, neighbours in partners.items()}


def _attribute_block(node: AtomNode) -> dict[str, Any]:
    """Return the evidence-schema attribute dict of one atom node."""
    return {
        "formal_charge": node.formal_charge,
        "radical_electrons": node.radical_electrons,
        "aromatic": node.aromatic,
        "chiral_tag": node.stereo,
    }


def build_reaction_edit_graph(
    bundle: EndpointGraphBundle,
    aromatic_regions: AromaticRegionsInput | None = None,
) -> ReactionEditGraph:
    """Build the atom-level reaction edit graph ΔG of one endpoint bundle.

    ``aromatic_regions`` optionally carries precomputed conjugated-region ids
    in either ``{region_id: [[a, b], ...]}`` or ``{(a, b): region_id}`` shape;
    when omitted, a ``getattr(bundle, "aromatic_regions", None)`` fallback is
    consulted so an extended bundle can supply them.  Edits outside any
    provided region (and every edit when nothing is provided) get
    ``aromatic_region=None``.  Region grouping is never recomputed here.
    """
    r_orders = _side_pair_orders(bundle.r_graph)
    p_orders = _side_pair_orders(bundle.p_graph)
    r_nodes = {node.map_id: node for node in bundle.r_graph.nodes}
    p_nodes = {node.map_id: node for node in bundle.p_graph.nodes}
    elements = {map_id: node.element for map_id, node in r_nodes.items()}
    if aromatic_regions is None:
        aromatic_regions = getattr(bundle, "aromatic_regions", None)
    pair_region = _normalize_aromatic_regions(aromatic_regions)

    edits: list[ReactionEdit] = []
    for pair in sorted(set(r_orders) | set(p_orders)):
        r_order = r_orders.get(pair)
        p_order = p_orders.get(pair)
        if r_order == p_order:
            continue
        if r_order is None:
            kind = EDIT_FORMED
        elif p_order is None:
            kind = EDIT_BROKEN
        else:
            kind = EDIT_ORDER_CHANGED
        first, second = pair
        edits.append(
            ReactionEdit(
                pair=pair,
                elements=(elements.get(first, "?"), elements.get(second, "?")),
                r_bond_order=r_order,
                p_bond_order=p_order,
                edit_kind=kind,
                aromatic_region=pair_region.get(pair),
            )
        )

    pairs = [edit.pair for edit in edits]
    fb_pairs = [edit.pair for edit in edits if edit.edit_kind != EDIT_ORDER_CHANGED]
    edit_vertices = sorted({atom for pair in pairs for atom in pair})
    fb_vertices = sorted({atom for pair in fb_pairs for atom in pair})
    edit_components = _connected_components(edit_vertices, pairs)
    fb_components = _connected_components(fb_vertices, fb_pairs)
    edit_cycle_rank = len(pairs) - len(edit_vertices) + len(edit_components)

    degree_counter: Counter[int] = Counter()
    for first, second in pairs:
        degree_counter[first] += 1
        degree_counter[second] += 1
    edit_degree = {atom: degree_counter[atom] for atom in sorted(degree_counter)}
    center_atoms = tuple(
        atom for atom, degree in edit_degree.items() if degree >= CENTER_DEGREE
    )

    r_partners = _partner_maps(bundle.r_graph)
    p_partners = _partner_maps(bundle.p_graph)
    atom_events: list[dict[str, Any]] = []
    atom_attribute_changes: list[dict[str, Any]] = []
    for map_id in sorted(set(r_nodes) | set(p_nodes)):
        r_block = _attribute_block(r_nodes[map_id]) if map_id in r_nodes else {}
        p_block = _attribute_block(p_nodes[map_id]) if map_id in p_nodes else {}
        r_set = r_partners.get(map_id, ())
        p_set = p_partners.get(map_id, ())
        partners_changed = r_set != p_set
        attributes_changed = r_block != p_block
        if attributes_changed:
            atom_attribute_changes.append(
                {"atom_map_id": map_id, "R": r_block, "P": p_block}
            )
        if partners_changed or attributes_changed:
            atom_events.append(
                {
                    "atom_map_id": map_id,
                    "partners_r": list(r_set),
                    "partners_p": list(p_set),
                    "partner_set_changed": partners_changed,
                    "attributes_changed": attributes_changed,
                }
            )

    return ReactionEditGraph(
        edits=tuple(edits),
        edit_counts=dict(Counter(edit.edit_kind for edit in edits)),
        edit_components=tuple(tuple(comp) for comp in edit_components),
        connectivity_edit_components=tuple(tuple(comp) for comp in fb_components),
        edit_cycle_rank=edit_cycle_rank,
        edit_degree=edit_degree,
        center_atoms=center_atoms,
        atom_events=tuple(atom_events),
        atom_attribute_changes=tuple(atom_attribute_changes),
    )


__all__ = [
    "CENTER_DEGREE",
    "SCHEMA_VERSION",
    "AromaticRegionsInput",
    "MapPair",
    "ReactionEdit",
    "ReactionEditGraph",
    "build_reaction_edit_graph",
]
