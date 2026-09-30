"""Endpoint context graph over an EndpointGraphBundle + atom-level edit graph.

Design §4.2: support paths, cross-component flags, R/P bridge edges, ring
membership, biconnected blocks, cycle ranks, aromatic regions, context-radius
shells, and spectator components — all deterministic.  Support-path and
cross-component semantics are ported exactly from
``outputs/pes_generation_strategy_design_v1/build_evidence.py`` (BFS shortest
path with sorted neighbour expansion; per-side component inequality on the
edit pair).  Graph algorithms are stdlib-only (union-find, BFS, DFS
low-link) — no networkx.  The edit-graph input is duck-typed against the
todo-7 ``EditGraph`` interface (``.edits`` records with ``pair``,
``edit_kind``, ``r_bond_order``, ``p_bond_order``; ``.atom_events``); when
``pes2ts_core.g1.reaction_edit_graph`` is importable it is re-exported as
``EditGraph`` for downstream typing.

Ring-size multisets come from a deterministic fundamental-cycle basis and are
NEVER the only ring classification: ``in_cycle`` (block membership),
biconnected blocks, bridges, and cycle ranks ship alongside (plan todo 8:
禁止依赖任意 SSSR 作唯一分类).
"""

# noqa: SIZE_OK — design §4.2 single endpoint-context module (topology analysis
# + edit-context assembly + evidence projections); split deferred to todo 9/10
# consumer needs.

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Protocol

from pes2ts_core.g1.endpoint_graph import (
    AtomNode,
    BondEdge,
    EndpointGraphBundle,
    GraphComponent,
    SideGraph,
)

try:  # pragma: no cover - exercised only once todo 7 lands
    from pes2ts_core.g1.reaction_edit_graph import EditGraph
except ImportError:  # pragma: no cover
    EditGraph = None  # type: ignore[assignment]

DEFAULT_CONTEXT_RADIUS: Final[int] = 2
SIDE_REACTANT: Final[str] = "reactant"
SIDE_PRODUCT: Final[str] = "product"
KIND_ORDER_CHANGED: Final[str] = "order_changed"
KIND_FORMED: Final[str] = "formed"
KIND_BROKEN: Final[str] = "broken"
BASIS_RING_REGION: Final[str] = "shared_ring_region"
BASIS_PATH_OVERLAP: Final[str] = "support_path_overlap"


class EditRecordProtocol(Protocol):
    """One atom-level edit record (todo-7 ``EditRecord`` shape)."""

    pair: tuple[int, int]
    edit_kind: str
    r_bond_order: float | None
    p_bond_order: float | None


class EditGraphProtocol(Protocol):
    """Todo-7 ``EditGraph`` shape consumed by this module."""

    edits: Sequence[Any]
    atom_events: Sequence[Any]


@dataclass(frozen=True, slots=True)
class EndpointElectronic:
    """Case-level electronic state of one endpoint (from the review case)."""

    charge: int
    multiplicity: int


@dataclass(frozen=True, slots=True)
class AtomTopology:
    """Per-atom context: component, ring membership, blocks, aromatic region."""

    map_id: int
    component_id: int
    in_cycle: bool
    ring_sizes: tuple[int, ...]
    biconnected_block_ids: tuple[int, ...]
    aromatic: bool
    aromatic_region_id: int | None


@dataclass(frozen=True, slots=True)
class SideTopology:
    """One side's context-graph analysis (deterministic, sorted orders)."""

    n_atoms: int
    n_edges: int
    n_components: int
    cycle_rank: int
    cycle_rank_per_component: tuple[tuple[int, int], ...]
    atoms: tuple[AtomTopology, ...]
    bridges: tuple[tuple[int, int], ...]
    biconnected_blocks: tuple[tuple[int, ...], ...]
    two_edge_components: tuple[tuple[int, ...], ...]
    aromatic_atoms: tuple[int, ...]
    aromatic_regions: tuple[tuple[int, ...], ...]
    aromatic_boundary: tuple[int, ...]
    aromatic_periphery: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class EditContext:
    """One edit enriched with endpoint-context evidence (evidence-field names)."""

    pair: tuple[int, int]
    edit_kind: str
    r_bond_order: float | None
    p_bond_order: float | None
    support_path_R: tuple[int, ...] | None
    support_path_P: tuple[int, ...] | None
    cross_component_R: bool
    cross_component_P: bool
    same_component_R: bool
    same_component_P: bool
    distance_R_A: float | None
    distance_P_A: float | None
    ring_evidence_atoms: tuple[int, ...]
    ring_group_id: int | None
    ring_group_basis: str | None


@dataclass(frozen=True, slots=True)
class RingGroup:
    """Edits sharing ring evidence (support-path overlap / shared ring region)."""

    group_id: int
    edit_indices: tuple[int, ...]
    edit_pairs: tuple[tuple[int, ...], ...]
    basis: str
    evidence_atoms: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class NeighborhoodShell:
    """Hop-distance shell around edit atoms on one side (context_radius N)."""

    radius: int
    seeds: tuple[int, ...]
    map_ids: tuple[int, ...]
    distance_by_map: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class EndpointFeatures:
    """Evidence ``features.endpoints.<side>`` block (build_evidence.py port)."""

    side: str
    n_components: int
    cycle_rank: int
    charge: int | None
    multiplicity: int | None
    rdkit_formal_charge_sum: int
    rdkit_charged_atom_count: int
    rdkit_radical_electron_count: int

    def to_evidence_dict(self) -> dict[str, Any]:
        """Return the exact evidence JSON key set for this endpoint side."""
        return {
            "n_components": self.n_components,
            "cycle_rank": self.cycle_rank,
            "charge": self.charge,
            "multiplicity": self.multiplicity,
            "rdkit_formal_charge_sum": self.rdkit_formal_charge_sum,
            "rdkit_charged_atom_count": self.rdkit_charged_atom_count,
            "rdkit_radical_electron_count": self.rdkit_radical_electron_count,
        }


@dataclass(frozen=True, slots=True)
class EndpointContext:
    """Full endpoint context graph (design §4.2) over bundle + edits."""

    context_radius: int
    r_topology: SideTopology
    p_topology: SideTopology
    edits: tuple[EditContext, ...]
    ring_groups: tuple[RingGroup, ...]
    shells_R: NeighborhoodShell
    shells_P: NeighborhoodShell
    spectator_components_R: tuple[tuple[int, ...], ...]
    spectator_components_P: tuple[tuple[int, ...], ...]
    bridge_edges_R: tuple[tuple[int, int], ...]
    bridge_edges_P: tuple[tuple[int, int], ...]
    endpoints: tuple[EndpointFeatures, EndpointFeatures]
    edit_atoms: tuple[int, ...]
    atom_events: tuple[Any, ...]

    def endpoints_evidence_block(self) -> dict[str, dict[str, Any]]:
        """Return ``{reactant: {...}, product: {...}}`` evidence endpoints."""
        reactant, product = self.endpoints
        return {
            SIDE_REACTANT: reactant.to_evidence_dict(),
            SIDE_PRODUCT: product.to_evidence_dict(),
        }

    def to_doc(self) -> dict[str, Any]:
        """JSON-safe projection; never contains truth-derived keys."""
        return {
            "context_radius": self.context_radius,
            "endpoints": self.endpoints_evidence_block(),
            "edits": [evidence_edit_fields(edit) | _edit_context_extra(edit) for edit in self.edits],
            "ring_groups": [
                {
                    "group_id": group.group_id,
                    "edit_indices": list(group.edit_indices),
                    "edit_pairs": [list(pair) for pair in group.edit_pairs],
                    "basis": group.basis,
                    "evidence_atoms": list(group.evidence_atoms),
                }
                for group in self.ring_groups
            ],
            "shells": {
                "R": _shell_doc(self.shells_R),
                "P": _shell_doc(self.shells_P),
            },
            "spectator_components_R": [list(c) for c in self.spectator_components_R],
            "spectator_components_P": [list(c) for c in self.spectator_components_P],
            "bridge_edges_R": [list(e) for e in self.bridge_edges_R],
            "bridge_edges_P": [list(e) for e in self.bridge_edges_P],
            "edit_atoms": list(self.edit_atoms),
            "topologies": {
                "R": _topology_doc(self.r_topology),
                "P": _topology_doc(self.p_topology),
            },
        }


def _shell_doc(shell: NeighborhoodShell) -> dict[str, Any]:
    """JSON-safe shell projection."""
    return {
        "radius": shell.radius,
        "seeds": list(shell.seeds),
        "map_ids": list(shell.map_ids),
        "distance_by_map": [[m, d] for m, d in shell.distance_by_map],
    }


def _topology_doc(topology: SideTopology) -> dict[str, Any]:
    """JSON-safe side-topology projection."""
    return {
        "n_atoms": topology.n_atoms,
        "n_edges": topology.n_edges,
        "n_components": topology.n_components,
        "cycle_rank": topology.cycle_rank,
        "cycle_rank_per_component": [list(row) for row in topology.cycle_rank_per_component],
        "bridges": [list(e) for e in topology.bridges],
        "biconnected_blocks": [list(b) for b in topology.biconnected_blocks],
        "two_edge_components": [list(c) for c in topology.two_edge_components],
        "aromatic_atoms": list(topology.aromatic_atoms),
        "aromatic_regions": [list(r) for r in topology.aromatic_regions],
        "aromatic_boundary": list(topology.aromatic_boundary),
        "aromatic_periphery": list(topology.aromatic_periphery),
        "atoms": [
            {
                "map_id": a.map_id,
                "component_id": a.component_id,
                "in_cycle": a.in_cycle,
                "ring_sizes": list(a.ring_sizes),
                "biconnected_block_ids": list(a.biconnected_block_ids),
                "aromatic": a.aromatic,
                "aromatic_region_id": a.aromatic_region_id,
            }
            for a in topology.atoms
        ],
    }


def _edit_context_extra(edit: EditContext) -> dict[str, Any]:
    """Extra per-edit context fields beyond the evidence projection."""
    return {
        "same_component_R": edit.same_component_R,
        "same_component_P": edit.same_component_P,
        "ring_evidence_atoms": list(edit.ring_evidence_atoms),
        "ring_group_id": edit.ring_group_id,
        "ring_group_basis": edit.ring_group_basis,
    }


def evidence_edit_fields(edit: EditContext) -> dict[str, Any]:
    """Evidence ``records[].edits[]`` projection (build_evidence.py keys).

    Key presence mirrors ``build_evidence.py``: every edit carries both
    ``cross_component_*`` flags and both bond orders; ``support_path_R`` is
    emitted only for ``formed`` (value may be ``None`` when cross-component),
    ``support_path_P`` only for ``broken``; ``order_changed`` carries neither;
    distances are emitted when geometry was supplied.
    """
    item: dict[str, Any] = {
        "pair": [edit.pair[0], edit.pair[1]],
        "edit_kind": edit.edit_kind,
        "r_bond_order": edit.r_bond_order,
        "p_bond_order": edit.p_bond_order,
        "cross_component_R": edit.cross_component_R,
        "cross_component_P": edit.cross_component_P,
    }
    if edit.distance_R_A is not None:
        item["distance_R_A"] = edit.distance_R_A
    if edit.distance_P_A is not None:
        item["distance_P_A"] = edit.distance_P_A
    if edit.edit_kind == KIND_FORMED:
        item["support_path_R"] = None if edit.support_path_R is None else list(edit.support_path_R)
    elif edit.edit_kind == KIND_BROKEN:
        item["support_path_P"] = None if edit.support_path_P is None else list(edit.support_path_P)
    return item


def shortest_support_path(
    bonds: Iterable[tuple[int, int]], start: int, goal: int
) -> tuple[int, ...] | None:
    """BFS shortest path ported verbatim from build_evidence.py.

    Neighbours expand in sorted order; the returned path includes both
    endpoints; ``None`` when ``goal`` is unreachable from ``start``.
    """
    adjacency: dict[int, list[int]] = {}
    for pair in bonds:
        left, right = pair
        adjacency.setdefault(left, []).append(right)
        adjacency.setdefault(right, []).append(left)
    queue: deque[list[int]] = deque([[start]])
    visited = {start}
    while queue:
        path = queue.popleft()
        if path[-1] == goal:
            return tuple(path)
        for nxt in sorted(adjacency.get(path[-1], [])):
            if nxt not in visited:
                visited.add(nxt)
                queue.append(path + [nxt])
    return None


def context_radius_from_config(config: Mapping[str, Any]) -> int:
    """Read ``scan_strategy.context_radius``; invalid values fall back to 2."""
    section = config.get("scan_strategy")
    if not isinstance(section, Mapping):
        return DEFAULT_CONTEXT_RADIUS
    raw = section.get("context_radius", DEFAULT_CONTEXT_RADIUS)
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        return DEFAULT_CONTEXT_RADIUS
    value = int(raw)
    if value != raw or value < 1:
        return DEFAULT_CONTEXT_RADIUS
    return value


def _edge_pairs(graph: SideGraph) -> list[tuple[int, int]]:
    """Sorted unordered edge pairs of one side graph."""
    return sorted((e.map_a, e.map_b) for e in graph.edges)


def _adjacency(map_ids: Sequence[int], edges: Iterable[tuple[int, int]]) -> dict[int, list[int]]:
    """Sorted adjacency from map ids + unordered edge pairs."""
    adjacency: dict[int, list[int]] = {m: [] for m in map_ids}
    for left, right in edges:
        adjacency[left].append(right)
        adjacency[right].append(left)
    for members in adjacency.values():
        members.sort()
    return adjacency


def _find(parent: dict[int, int], node: int) -> int:
    """Union-find root with path halving."""
    while parent[node] != node:
        parent[node] = parent[parent[node]]
        node = parent[node]
    return node


def _union(parent: dict[int, int], left: int, right: int) -> None:
    """Union two sets; smaller root attaches under the larger."""
    root_left, root_right = _find(parent, left), _find(parent, right)
    if root_left == root_right:
        return
    if root_left < root_right:
        parent[root_right] = root_left
    else:
        parent[root_left] = root_right


def _bridges(map_ids: Sequence[int], edges: Sequence[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    """Classic graph bridges via DFS low-link (sorted, deduplicated)."""
    adjacency = _adjacency(map_ids, edges)
    index: dict[int, int] = {}
    low: dict[int, int] = {}
    found: set[tuple[int, int]] = set()
    counter = 0

    def dfs(node: int, parent: int | None) -> None:
        nonlocal counter
        index[node] = low[node] = counter
        counter += 1
        for nxt in adjacency[node]:
            if nxt not in index:
                dfs(nxt, node)
                low[node] = min(low[node], low[nxt])
                if low[nxt] > index[node]:
                    found.add((node, nxt) if node < nxt else (nxt, node))
            elif nxt != parent:
                low[node] = min(low[node], index[nxt])

    for map_id in map_ids:
        if map_id not in index:
            dfs(map_id, None)
    return tuple(sorted(found))


def _biconnected_blocks(
    map_ids: Sequence[int], edges: Sequence[tuple[int, int]]
) -> tuple[tuple[int, ...], ...]:
    """2-vertex-connected blocks (bridges appear as two-atom blocks)."""
    adjacency = _adjacency(map_ids, edges)
    index: dict[int, int] = {}
    low: dict[int, int] = {}
    edge_stack: list[tuple[int, int]] = []
    blocks: set[frozenset[int]] = set()
    counter = 0

    def pop_block_until(edge: tuple[int, int]) -> None:
        block: set[int] = set()
        while True:
            current = edge_stack.pop()
            block.update(current)
            if current == edge:
                break
        blocks.add(frozenset(block))

    def dfs(node: int, parent: int | None) -> None:
        nonlocal counter
        index[node] = low[node] = counter
        counter += 1
        children = 0
        for nxt in adjacency[node]:
            edge = (node, nxt) if node < nxt else (nxt, node)
            if nxt not in index:
                edge_stack.append(edge)
                children += 1
                dfs(nxt, node)
                low[node] = min(low[node], low[nxt])
                if (parent is None and children > 1) or (
                    parent is not None and low[nxt] >= index[node]
                ):
                    pop_block_until(edge)
            elif nxt != parent and index[nxt] < index[node]:
                edge_stack.append(edge)
                low[node] = min(low[node], index[nxt])
        if parent is None and edge_stack:
            block: set[int] = set()
            while edge_stack:
                block.update(edge_stack.pop())
            blocks.add(frozenset(block))

    for map_id in map_ids:
        if map_id not in index:
            dfs(map_id, None)
    return tuple(sorted(tuple(sorted(block)) for block in blocks))


def _two_edge_components(
    map_ids: Sequence[int],
    edges: Sequence[tuple[int, int]],
    bridges: Sequence[tuple[int, int]],
) -> tuple[tuple[int, ...], ...]:
    """2-edge-connected components (bridge removal + union-find)."""
    bridge_set = set(bridges)
    parent = {m: m for m in map_ids}
    for left, right in edges:
        key = (left, right) if left < right else (right, left)
        if key not in bridge_set:
            _union(parent, left, right)
    groups: dict[int, list[int]] = {}
    for map_id in map_ids:
        groups.setdefault(_find(parent, map_id), []).append(map_id)
    return tuple(sorted(tuple(sorted(group)) for group in groups.values()))


def _fundamental_ring_sizes(
    map_ids: Sequence[int], edges: Sequence[tuple[int, int]]
) -> dict[int, tuple[int, ...]]:
    """Per-atom multiset of fundamental-cycle sizes (deterministic BFS tree).

    Basis-dependent by construction; downstream must NOT treat this as the
    unique SSSR — ``in_cycle``/blocks/bridges/cycle ranks carry the rest.
    """
    adjacency = _adjacency(map_ids, edges)
    edge_set = {(min(a, b), max(a, b)) for a, b in edges}
    sizes: dict[int, list[int]] = {m: [] for m in map_ids}
    seen: set[int] = set()
    for seed in map_ids:
        if seed in seen:
            continue
        parent: dict[int, int | None] = {seed: None}
        tree_edges: set[tuple[int, int]] = set()
        queue: deque[int] = deque([seed])
        seen.add(seed)
        while queue:
            node = queue.popleft()
            for nxt in adjacency[node]:
                if nxt not in parent:
                    parent[nxt] = node
                    tree_edges.add((node, nxt) if node < nxt else (nxt, node))
                    seen.add(nxt)
                    queue.append(nxt)

        def path_to_root(node: int) -> list[int]:
            chain: list[int] = []
            current: int | None = node
            while current is not None:
                chain.append(current)
                current = parent[current]
            return chain

        for left, right in sorted(edge_set):
            if left not in parent or right not in parent:
                continue
            if (left, right) in tree_edges:
                continue
            chain_left = path_to_root(left)
            chain_right = path_to_root(right)
            right_set = set(chain_right)
            lca = next(n for n in chain_left if n in right_set)
            cycle_nodes = set(chain_left[: chain_left.index(lca) + 1]) | set(
                chain_right[: chain_right.index(lca) + 1]
            )
            size = len(cycle_nodes)
            for map_id in cycle_nodes:
                sizes[map_id].append(size)
    return {m: tuple(sorted(values)) for m, values in sizes.items()}


def _aromatic_regions(
    map_ids: Sequence[int],
    nodes_by_map: Mapping[int, AtomNode],
    edges: Sequence[tuple[int, int]],
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...], tuple[int, ...]]:
    """Aromatic atom clusters + external boundary + aromatic periphery."""
    aromatic = {m for m in map_ids if nodes_by_map[m].aromatic}
    adjacency = _adjacency(map_ids, edges)
    regions: list[tuple[int, ...]] = []
    unseen = set(aromatic)
    while unseen:
        seed = min(unseen)
        component = {seed}
        queue: deque[int] = deque([seed])
        unseen.remove(seed)
        while queue:
            node = queue.popleft()
            for nxt in adjacency[node]:
                if nxt in unseen and nxt in aromatic:
                    unseen.remove(nxt)
                    component.add(nxt)
                    queue.append(nxt)
        regions.append(tuple(sorted(component)))
    boundary = {
        atom
        for atom in map_ids
        if atom not in aromatic
        and any(nxt in aromatic for nxt in adjacency[atom])
    }
    periphery = {
        atom
        for atom in aromatic
        if any(nxt not in aromatic for nxt in adjacency[atom])
    }
    return tuple(sorted(regions)), tuple(sorted(boundary)), tuple(sorted(periphery))


def analyze_side_graph(graph: SideGraph) -> SideTopology:
    """Analyze one endpoint side graph into its context topology."""
    map_ids = tuple(sorted(n.map_id for n in graph.nodes))
    nodes_by_map = {n.map_id: n for n in graph.nodes}
    edges = _edge_pairs(graph)
    component_of: dict[int, int] = {}
    for component in graph.components:
        for map_id in component.map_ids:
            component_of[map_id] = component.component_id
    edges_by_component: dict[int, list[tuple[int, int]]] = {
        component.component_id: [] for component in graph.components
    }
    for left, right in edges:
        edges_by_component[component_of[left]].append((left, right))
    rank_rows = tuple(
        (
            component.component_id,
            len(edges_by_component[component.component_id])
            - len(component.map_ids)
            + 1,
        )
        for component in graph.components
    )
    bridges = _bridges(map_ids, edges)
    blocks = _biconnected_blocks(map_ids, edges)
    two_edge = _two_edge_components(map_ids, edges, bridges)
    ring_sizes = _fundamental_ring_sizes(map_ids, edges)
    block_ids_by_atom: dict[int, list[int]] = {m: [] for m in map_ids}
    for block_id, block in enumerate(blocks):
        for map_id in block:
            block_ids_by_atom[map_id].append(block_id)
    cyclic_blocks = {block_id for block_id, block in enumerate(blocks) if len(block) >= 3}
    regions, boundary, periphery = _aromatic_regions(map_ids, nodes_by_map, edges)
    region_of: dict[int, int] = {}
    for region_id, region in enumerate(regions):
        for map_id in region:
            region_of[map_id] = region_id
    atoms = tuple(
        AtomTopology(
            map_id=map_id,
            component_id=component_of[map_id],
            in_cycle=any(
                block_id in cyclic_blocks for block_id in block_ids_by_atom[map_id]
            ),
            ring_sizes=ring_sizes[map_id],
            biconnected_block_ids=tuple(block_ids_by_atom[map_id]),
            aromatic=nodes_by_map[map_id].aromatic,
            aromatic_region_id=region_of.get(map_id),
        )
        for map_id in map_ids
    )
    return SideTopology(
        n_atoms=len(map_ids),
        n_edges=len(edges),
        n_components=len(graph.components),
        cycle_rank=len(edges) - len(map_ids) + len(graph.components),
        cycle_rank_per_component=rank_rows,
        atoms=atoms,
        bridges=bridges,
        biconnected_blocks=blocks,
        two_edge_components=two_edge,
        aromatic_atoms=tuple(sorted(region_of)),
        aromatic_regions=regions,
        aromatic_boundary=boundary,
        aromatic_periphery=periphery,
    )


def _component_of(graph: SideGraph) -> dict[int, int]:
    """Map id → bundle component_id (bundle numbering, smallest-map ordered)."""
    mapping: dict[int, int] = {}
    for component in graph.components:
        for map_id in component.map_ids:
            mapping[map_id] = component.component_id
    return mapping


def _region_atom_set(
    pair: tuple[int, int],
    edge_set: frozenset[tuple[int, int]],
    bridges: frozenset[tuple[int, int]],
    two_edge: Sequence[tuple[int, ...]],
) -> frozenset[int] | None:
    """2-edge-connected component containing ``pair`` when the edge is cyclic."""
    left, right = pair
    key = (left, right) if left < right else (right, left)
    if key not in edge_set or key in bridges:
        return None
    for component in two_edge:
        if left in component and right in component:
            return frozenset(component)
    return None


def _neighborhood_shell(
    map_ids: Sequence[int],
    edges: Sequence[tuple[int, int]],
    seeds: Iterable[int],
    radius: int,
) -> NeighborhoodShell:
    """Multi-source BFS shell of hop-radius ``radius`` around ``seeds``."""
    adjacency = _adjacency(map_ids, edges)
    seed_set = {s for s in seeds if s in adjacency}
    distances: dict[int, int] = {s: 0 for s in seed_set}
    queue: deque[int] = deque(sorted(seed_set))
    while queue:
        node = queue.popleft()
        if distances[node] >= radius:
            continue
        for nxt in adjacency[node]:
            if nxt not in distances:
                distances[nxt] = distances[node] + 1
                queue.append(nxt)
    return NeighborhoodShell(
        radius=radius,
        seeds=tuple(sorted(seed_set)),
        map_ids=tuple(sorted(distances)),
        distance_by_map=tuple(sorted(distances.items())),
    )


def _rec_field(record: Any, name: str) -> Any:
    """Read a field from an edit record (dataclass attribute or mapping)."""
    if isinstance(record, Mapping):
        return record[name]
    return getattr(record, name)


def _endpoint_features(
    side: str, graph: SideGraph, electronic: EndpointElectronic | None
) -> EndpointFeatures:
    """Evidence endpoints block for one side (cycle rank = |E|-|V|+C)."""
    n_atoms = len(graph.nodes)
    n_edges = len(graph.edges)
    n_components = len(graph.components)
    return EndpointFeatures(
        side=side,
        n_components=n_components,
        cycle_rank=n_edges - n_atoms + n_components,
        charge=None if electronic is None else electronic.charge,
        multiplicity=None if electronic is None else electronic.multiplicity,
        rdkit_formal_charge_sum=sum(n.formal_charge for n in graph.nodes),
        rdkit_charged_atom_count=sum(1 for n in graph.nodes if n.formal_charge != 0),
        rdkit_radical_electron_count=sum(n.radical_electrons for n in graph.nodes),
    )


def _spectators(
    components: Sequence[GraphComponent], edit_atoms: frozenset[int]
) -> tuple[tuple[int, ...], ...]:
    """Components with no edit atoms (graph-invariant spectators)."""
    return tuple(
        tuple(sorted(component.map_ids))
        for component in components
        if not (set(component.map_ids) & edit_atoms)
    )


def build_endpoint_context(
    bundle: EndpointGraphBundle,
    edit_graph: EditGraphProtocol | Any,
    *,
    r_coordinates: Mapping[int, tuple[float, float, float]] | None = None,
    p_coordinates: Mapping[int, tuple[float, float, float]] | None = None,
    endpoint_electronic: Mapping[str, EndpointElectronic] | None = None,
    context_radius: int | None = None,
) -> EndpointContext:
    """Build the endpoint context graph from bundle + atom-level edits.

    ``r_coordinates``/``p_coordinates`` (map → xyz, angstrom) supply the
    geometry refs the bundle binds by hash but does not store; distances are
    ``round(math.dist(...), 6)`` exactly as ``build_evidence.py``.  The
    geometry bundle identity still guards representation; coordinates are an
    explicit caller input, never re-read from internal data trees.
    """
    radius = DEFAULT_CONTEXT_RADIUS if context_radius is None else int(context_radius)
    if radius < 1:
        raise ValueError("context_radius must be >= 1")
    r_topology = analyze_side_graph(bundle.r_graph)
    p_topology = analyze_side_graph(bundle.p_graph)
    r_component = _component_of(bundle.r_graph)
    p_component = _component_of(bundle.p_graph)
    r_pairs = frozenset((e.map_a, e.map_b) for e in bundle.r_graph.edges)
    p_pairs = frozenset((e.map_a, e.map_b) for e in bundle.p_graph.edges)
    r_bridges = frozenset(r_topology.bridges)
    p_bridges = frozenset(p_topology.bridges)

    raw_edits: Sequence[Any] = tuple(getattr(edit_graph, "edits", ()))
    atom_events = tuple(getattr(edit_graph, "atom_events", ()))
    pending: list[dict[str, Any]] = []
    edit_atoms: set[int] = set()
    for record in raw_edits:
        pair_raw = _rec_field(record, "pair")
        pair = (int(pair_raw[0]), int(pair_raw[1]))
        kind = str(_rec_field(record, "edit_kind"))
        r_order_raw = _rec_field(record, "r_bond_order")
        p_order_raw = _rec_field(record, "p_bond_order")
        r_order = None if r_order_raw is None else float(r_order_raw)
        p_order = None if p_order_raw is None else float(p_order_raw)
        edit_atoms.update(pair)
        if kind == KIND_FORMED:
            support_r = shortest_support_path(_edge_pairs(bundle.r_graph), pair[0], pair[1])
            support_p = None
            region = _region_atom_set(pair, p_pairs, p_bridges, p_topology.two_edge_components)
            path_atoms = frozenset(support_r or ()) | frozenset(pair)
        elif kind == KIND_BROKEN:
            support_r = None
            support_p = shortest_support_path(_edge_pairs(bundle.p_graph), pair[0], pair[1])
            region = _region_atom_set(pair, r_pairs, r_bridges, r_topology.two_edge_components)
            path_atoms = frozenset(support_p or ()) | frozenset(pair)
        else:
            support_r = None
            support_p = None
            region = None
            path_atoms = frozenset()
        cross_r = r_component[pair[0]] != r_component[pair[1]]
        cross_p = p_component[pair[0]] != p_component[pair[1]]
        distance_r = None
        distance_p = None
        if r_coordinates is not None:
            distance_r = round(
                math.dist(r_coordinates[pair[0]], r_coordinates[pair[1]]), 6
            )
        if p_coordinates is not None:
            distance_p = round(
                math.dist(p_coordinates[pair[0]], p_coordinates[pair[1]]), 6
            )
        evidence_atoms = tuple(sorted((region or frozenset()) | path_atoms))
        pending.append(
            {
                "pair": pair,
                "kind": kind,
                "r_order": r_order,
                "p_order": p_order,
                "support_r": support_r,
                "support_p": support_p,
                "cross_r": cross_r,
                "cross_p": cross_p,
                "distance_r": distance_r,
                "distance_p": distance_p,
                "region": region,
                "path_atoms": path_atoms,
                "evidence_atoms": evidence_atoms,
            }
        )

    # Ring grouping: union-find over formed/broken edits with ring evidence.
    parent = list(range(len(pending)))
    fired: list[set[str]] = [set() for _ in pending]
    for i in range(len(pending)):
        for j in range(i + 1, len(pending)):
            left, right = pending[i], pending[j]
            region_link = bool(
                left["region"] is not None
                and right["region"] is not None
                and set(left["region"]) & set(right["region"])
            )
            path_link = bool(
                left["kind"] == right["kind"]
                and left["kind"] != KIND_ORDER_CHANGED
                and left["path_atoms"]
                and right["path_atoms"]
                and set(left["path_atoms"]) & set(right["path_atoms"])
            )
            if not region_link and not path_link:
                continue
            root_i, root_j = _find(parent, i), _find(parent, j)
            if root_i != root_j:
                parent[root_j] = root_i
                fired[root_i] |= fired[root_j]
            if region_link:
                fired[_find(parent, i)].add(BASIS_RING_REGION)
            if path_link:
                fired[_find(parent, i)].add(BASIS_PATH_OVERLAP)

    groups_map: dict[int, list[int]] = {}
    for index, item in enumerate(pending):
        has_evidence = item["region"] is not None or bool(item["path_atoms"])
        if item["kind"] == KIND_ORDER_CHANGED or not has_evidence:
            continue
        groups_map.setdefault(_find(parent, index), []).append(index)
    ordered_roots = sorted(groups_map, key=lambda root: min(groups_map[root]))
    ring_group_id_by_index: dict[int, int] = {}
    ring_groups: list[RingGroup] = []
    for group_id, root in enumerate(ordered_roots):
        indices = tuple(sorted(groups_map[root]))
        for index in indices:
            ring_group_id_by_index[index] = group_id
        rules = set().union(*(fired[_find(parent, index)] for index in indices))
        if len(indices) == 1:
            basis = BASIS_RING_REGION if pending[indices[0]]["region"] is not None else BASIS_PATH_OVERLAP
        else:
            basis = "+".join(sorted(rules)) if rules else BASIS_RING_REGION
        evidence_atoms: set[int] = set()
        for index in indices:
            evidence_atoms.update(pending[index]["evidence_atoms"])
        ring_groups.append(
            RingGroup(
                group_id=group_id,
                edit_indices=indices,
                edit_pairs=tuple(pending[index]["pair"] for index in indices),
                basis=basis,
                evidence_atoms=tuple(sorted(evidence_atoms)),
            )
        )

    edits: list[EditContext] = []
    for index, item in enumerate(pending):
        group_id = ring_group_id_by_index.get(index)
        basis: str | None = None
        if group_id is not None:
            basis = ring_groups[group_id].basis
        edits.append(
            EditContext(
                pair=item["pair"],
                edit_kind=item["kind"],
                r_bond_order=item["r_order"],
                p_bond_order=item["p_order"],
                support_path_R=item["support_r"],
                support_path_P=item["support_p"],
                cross_component_R=item["cross_r"],
                cross_component_P=item["cross_p"],
                same_component_R=not item["cross_r"],
                same_component_P=not item["cross_p"],
                distance_R_A=item["distance_r"],
                distance_P_A=item["distance_p"],
                ring_evidence_atoms=item["evidence_atoms"],
                ring_group_id=group_id,
                ring_group_basis=basis,
            )
        )

    seed_set = frozenset(edit_atoms)
    bridge_edges_r = tuple(
        sorted(
            (left, right)
            for left, right in _edge_pairs(bundle.r_graph)
            if p_component[left] != p_component[right]
        )
    )
    bridge_edges_p = tuple(
        sorted(
            (left, right)
            for left, right in _edge_pairs(bundle.p_graph)
            if r_component[left] != r_component[right]
        )
    )
    electronic = endpoint_electronic or {}
    reactant_features = _endpoint_features(
        SIDE_REACTANT,
        bundle.r_graph,
        electronic.get(SIDE_REACTANT),
    )
    product_features = _endpoint_features(
        SIDE_PRODUCT,
        bundle.p_graph,
        electronic.get(SIDE_PRODUCT),
    )
    r_node_ids = tuple(n.map_id for n in bundle.r_graph.nodes)
    p_node_ids = tuple(n.map_id for n in bundle.p_graph.nodes)
    return EndpointContext(
        context_radius=radius,
        r_topology=r_topology,
        p_topology=p_topology,
        edits=tuple(edits),
        ring_groups=tuple(ring_groups),
        shells_R=_neighborhood_shell(
            r_node_ids, _edge_pairs(bundle.r_graph), seed_set, radius
        ),
        shells_P=_neighborhood_shell(
            p_node_ids, _edge_pairs(bundle.p_graph), seed_set, radius
        ),
        spectator_components_R=_spectators(bundle.r_graph.components, seed_set),
        spectator_components_P=_spectators(bundle.p_graph.components, seed_set),
        bridge_edges_R=bridge_edges_r,
        bridge_edges_P=bridge_edges_p,
        endpoints=(reactant_features, product_features),
        edit_atoms=tuple(sorted(edit_atoms)),
        atom_events=atom_events,
    )


__all__ = [
    "BASIS_PATH_OVERLAP",
    "BASIS_RING_REGION",
    "DEFAULT_CONTEXT_RADIUS",
    "EditGraph",
    "EditGraphProtocol",
    "EditRecordProtocol",
    "AtomTopology",
    "EditContext",
    "EndpointContext",
    "EndpointElectronic",
    "EndpointFeatures",
    "NeighborhoodShell",
    "RingGroup",
    "SIDE_PRODUCT",
    "SIDE_REACTANT",
    "SideTopology",
    "analyze_side_graph",
    "build_endpoint_context",
    "context_radius_from_config",
    "evidence_edit_fields",
    "shortest_support_path",
]
