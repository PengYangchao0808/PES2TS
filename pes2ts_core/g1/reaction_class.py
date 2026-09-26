"""P2 reaction classification: map-number-independent hierarchical templates.

The classifier turns one P1 document into four hierarchy levels -- ``l0``
event-count family, ``l1`` reaction-center template, ``l2``/``l3`` one- and
two-shell context templates -- plus human-readable rule labels and the
IRC-derived pathway labels kept strictly separate from the structural
clusters.

Every level's cluster id is ``taxonomy_version + level + sha256(canonical
signature)``.  The canonical signature is computed by an
individualization-refinement canonical labeling (WL color refinement with
exhaustive tie branching under a budget) over the labeled center/context
graph, and the *direction-canonical* form is the lexicographic minimum of
the template and its R<->P reversal, so map relabeling, component
reordering, candidate reshuffling, or writing the reaction backwards can
never split one chemical transformation into two clusters.  When the
branching budget is exhausted the WL-stable serialization is used and the
document is flagged -- a deterministic but non-exhaustive fallback, never a
silent one.
"""

from __future__ import annotations

import hashlib
import itertools
import logging
from collections.abc import Mapping, Sequence
from typing import Any, Final

from pes2ts_core.g1.truth_schema import (
    DEFAULT_CANONICAL_BUDGET,
    DEFAULT_TAXONOMY_VERSION,
    FAMILY_LABELS,
)

logger = logging.getLogger(__name__)

#: Hierarchy level names in the P2 document and cluster ids.
LEVEL_L0: Final[str] = "l0_edit_family"
LEVEL_L1: Final[str] = "l1_center_template"
LEVEL_L2: Final[str] = "l2_context_r1"
LEVEL_L3: Final[str] = "l3_context_r2"

#: Canonical-labeling budget-exceeded document flag.
FLAG_CANONICAL_BUDGET: Final[str] = "canonical_budget_exhausted"

#: Bond-order rendering inside signatures ("~" marks an absent side).
_ABSENT_ORDER: Final[str] = "~"


def _stable_hash_int(text: str) -> int:
    """Return a process-independent integer hash of *text*."""
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def _order_text(order: float | None) -> str:
    """Render one bond order (``~`` when the side has no bond)."""
    return _ABSENT_ORDER if order is None else f"{order:.2f}"


def _edge_label(order_r: float | None, order_p: float | None) -> str:
    return f"r{_order_text(order_r)}p{_order_text(order_p)}"


def _order_of_side(entry: Mapping[str, Any], side: str) -> float | None:
    """Return one event entry's bond order on *side* ("order_r"/"order_p")."""
    value = entry.get(f"order_{side}")
    return None if value is None else float(value)


#: Node role tags swapped when the reaction direction is reversed.
_REVERSED_ROLES: Final[dict[str, str]] = {"F": "B", "B": "F", "Hf": "Ht", "Ht": "Hf"}


def _node_label_text(element: str, roles: tuple[str, ...]) -> str:
    """Render one node label (element plus sorted role tags)."""
    return f"{element}{''.join(roles)}"


class _TemplateGraph:
    """Labeled graph over a subset of map numbers (never serialized by map)."""

    def __init__(
        self,
        node_labels: Mapping[int, tuple[str, tuple[str, ...]]],
        edges: Sequence[tuple[int, int, float | None, float | None]],
    ) -> None:
        self.node_labels = dict(node_labels)
        self._label_text = {
            node: _node_label_text(element, roles)
            for node, (element, roles) in self.node_labels.items()
        }
        self.adjacency: dict[int, list[tuple[int, str]]] = {
            node: [] for node in self.node_labels
        }
        self._edge_labels: dict[tuple[int, int], str] = {}
        for first, second, order_r, order_p in edges:
            if first not in self.node_labels or second not in self.node_labels:
                continue
            label = _edge_label(order_r, order_p)
            pair = (first, second) if first <= second else (second, first)
            self._edge_labels[pair] = label
            self.adjacency[first].append((second, label))
            self.adjacency[second].append((first, label))
        for node in self.adjacency:
            self.adjacency[node].sort()

    def reversed_direction(self) -> "_TemplateGraph":
        """Return the template with every R/P semantic swapped.

        Bond orders swap sides and the direction-bearing role tags invert
        (formed endpoint <-> broken endpoint, H donor <-> H acceptor), so a
        reaction written backwards canonicalizes to the same string.
        """
        flipped_nodes = {
            node: (element, tuple(sorted(
                _REVERSED_ROLES.get(role, role) for role in roles
            )))
            for node, (element, roles) in self.node_labels.items()
        }
        flipped_edges = [
            (first, second, order_p, order_r)
            for first, second, order_r, order_p in self._edges_with_labels()
        ]
        return _TemplateGraph(flipped_nodes, flipped_edges)

    def label_text(self, node: int) -> str:
        """Return the rendered text label of *node*."""
        return self._label_text[node]

    @staticmethod
    def label_text_of(label: tuple[str, tuple[str, ...]]) -> str:
        """Render a structured node label as text."""
        return _node_label_text(label[0], label[1])

    def _edges_with_labels(self) -> list[tuple[int, int, float | None, float | None]]:
        edges: list[tuple[int, int, float | None, float | None]] = []
        for (first, second), label in self._edge_labels.items():
            order_r: float | None
            order_p: float | None
            r_text = label[1:label.index("p")]
            p_text = label[label.index("p") + 1:]
            order_r = None if r_text == _ABSENT_ORDER else float(r_text)
            order_p = None if p_text == _ABSENT_ORDER else float(p_text)
            edges.append((first, second, order_r, order_p))
        return sorted(edges)


def _refine_colors(graph: _TemplateGraph, colors: dict[int, int]) -> dict[int, int]:
    """One WL refinement pass sequence until the coloring is stable."""
    while True:
        signatures = {
            node: (
                colors[node],
                tuple(sorted(
                    (colors[neighbor], label)
                    for neighbor, label in graph.adjacency[node]
                )),
            )
            for node in sorted(graph.adjacency)
        }
        order = sorted(set(signatures.values()))
        remap = {value: index for index, value in enumerate(order)}
        refined = {node: remap[signatures[node]] for node in signatures}
        if refined == colors:
            return colors
        colors = refined


def _individualize(colors: dict[int, int], node: int) -> dict[int, int]:
    """Give *node* a fresh unique color, keeping every other color intact."""
    shifted = dict(colors)
    shifted[node] = max(colors.values()) + 1
    return shifted


def _serialize(graph: _TemplateGraph, colors: dict[int, int]) -> str:
    """Serialize the graph under an all-singleton canonical coloring."""
    ordered = sorted(graph.adjacency, key=lambda node: colors[node])
    position = {node: index for index, node in enumerate(ordered)}
    parts: list[str] = [f"n{len(ordered)}"]
    for node in ordered:
        connections = sorted(
            (position[neighbor], label)
            for neighbor, label in graph.adjacency[node]
        )
        parts.append(
            graph.label_text(node) + ":"
            + ",".join(f"{index}#{label}" for index, label in connections)
        )
    return ";".join(parts)


def _canonical_serialization(
    graph: _TemplateGraph, *, budget: int
) -> tuple[str, bool]:
    """Return ``(canonical string, exhausted)`` via individualization search."""
    label_colors = {
        label: _stable_hash_int(graph.label_text_of(label))
        for label in set(graph.node_labels.values())
    }
    colors = {node: label_colors[graph.node_labels[node]] for node in graph.adjacency}
    colors = _refine_colors(graph, colors)
    best: list[str] = []
    budget_state = {"used": 0, "exhausted": True}

    def search(current: dict[int, int]) -> None:
        if len(set(current.values())) == len(current):
            best.append(_serialize(graph, current))
            return
        cells: dict[int, list[int]] = {}
        for node, color in current.items():
            cells.setdefault(color, []).append(node)
        target = min(
            (sorted(members) for members in cells.values() if len(members) > 1),
            key=lambda members: members,
        )
        for candidate in target:
            budget_state["used"] += 1
            if budget_state["used"] > budget:
                budget_state["exhausted"] = False
                continue
            search(_refine_colors(graph, _individualize(current, candidate)))

    search(colors)
    if best:
        return min(best), budget_state["exhausted"]
    # Budget exhausted before any complete labeling: fall back to the WL-stable
    # serialization (deterministic, flagged by the caller).
    fallback = _serialize(graph, colors) if len(set(colors.values())) == len(colors) else _wl_stable_serialization(graph, colors)
    return fallback, False


def _wl_stable_serialization(graph: _TemplateGraph, colors: dict[int, int]) -> str:
    """Deterministic fallback serialization under a non-discrete coloring."""
    ordered = sorted(graph.adjacency, key=lambda node: (colors[node], graph.label_text(node)))
    position = {node: index for index, node in enumerate(ordered)}
    parts = [f"n{len(ordered)}wl"]
    for node in ordered:
        connections = sorted(
            (position[neighbor], label)
            for neighbor, label in graph.adjacency[node]
        )
        parts.append(
            f"c{colors[node]}+{graph.label_text(node)}:"
            + ",".join(f"{index}#{label}" for index, label in connections)
        )
    return ";".join(parts)


def _direction_canonical(graph: _TemplateGraph, *, budget: int) -> tuple[str, bool]:
    """Return the direction-invariant canonical string and budget flag."""
    forward, forward_ok = _canonical_serialization(graph, budget=budget)
    backward, backward_ok = _canonical_serialization(graph.reversed_direction(), budget=budget)
    return min(forward, backward), forward_ok and backward_ok


def _node_roles(bond_events: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[int, set[str]]:
    """Return the event-role set of every map number."""
    roles: dict[int, set[str]] = {}
    for kind, tag in (("formed", "F"), ("broken", "B"), ("order_changed", "O")):
        for entry in bond_events.get(kind, ()):
            for map_number in entry["atoms"]:
                roles.setdefault(int(map_number), set()).add(tag)
    for entry in bond_events.get("hydrogen_migration", ()):
        h_map = int(entry["h"])
        roles.setdefault(h_map, set()).add("Hm")
        if entry.get("from") is not None:
            roles.setdefault(int(entry["from"]), set()).add("Hf")
        if entry.get("to") is not None:
            roles.setdefault(int(entry["to"]), set()).add("Ht")
    return roles


def _shell_expansion(
    center: Sequence[int],
    r_edges: Sequence[Sequence[float]],
    p_edges: Sequence[Sequence[float]],
    shells: int,
) -> set[int]:
    """Return *center* plus *shells* union-graph neighbourhood rings."""
    adjacency: dict[int, set[int]] = {}
    for edge in [*r_edges, *p_edges]:
        first, second = int(edge[0]), int(edge[1])
        adjacency.setdefault(first, set()).add(second)
        adjacency.setdefault(second, set()).add(first)
    expanded = set(int(value) for value in center)
    for _ in range(max(shells, 0)):
        neighbours = {n for atom in expanded for n in adjacency.get(atom, ())}
        expanded |= neighbours
    return expanded


def _template_graph_for(
    nodes: set[int],
    elements: Mapping[int, str],
    roles: Mapping[int, set[str]],
    r_edges: Sequence[Sequence[float]],
    p_edges: Sequence[Sequence[float]],
) -> _TemplateGraph:
    """Build the labeled template graph over *nodes* from one P1 graph block."""
    node_labels = {
        node: (elements.get(node, "?"), tuple(sorted(roles.get(node, ()))))
        for node in sorted(nodes)
    }
    edge_orders: dict[tuple[int, int], list[float | None]] = {}
    for side in ("r", "p"):
        for edge in (r_edges if side == "r" else p_edges):
            first, second = int(edge[0]), int(edge[1])
            if first not in nodes or second not in nodes:
                continue
            pair = (first, second) if first <= second else (second, first)
            edge_orders.setdefault(pair, [None, None])
            edge_orders[pair][0 if side == "r" else 1] = float(edge[2])
    edges = [
        (pair[0], pair[1], orders[0], orders[1])
        for pair, orders in sorted(edge_orders.items())
    ]
    return _TemplateGraph(node_labels, edges)


def _edit_family_signature(bond_events: Mapping[str, Sequence[Mapping[str, Any]]]) -> str:
    """Return the L0 event-composition signature string."""
    return (
        f"F{len(bond_events.get('formed', ()))}"
        f"B{len(bond_events.get('broken', ()))}"
        f"O{len(bond_events.get('order_changed', ()))}"
        f"H{len(bond_events.get('hydrogen_migration', ()))}"
    )


def _side_ring_count(
    edges: Sequence[Sequence[float]], n_nodes: int, n_components: int
) -> int:
    """Cyclomatic number of one side's bond graph."""
    return max(len(edges) - n_nodes + n_components, 0)


def family_labels(
    bond_events: Mapping[str, Sequence[Mapping[str, Any]]],
    graph: Mapping[str, Any],
) -> list[str]:
    """Return the human-readable rule labels (multi-label, directional)."""
    n_formed = len(bond_events.get("formed", ()))
    n_broken = len(bond_events.get("broken", ()))
    r_nodes = set(graph["r_components"].values())
    p_nodes = set(graph["p_components"].values())
    n_r = len(r_nodes)
    n_p = len(p_nodes)
    rings_r = _side_ring_count(graph["r_bonds"], len(graph["elements"]), n_r)
    rings_p = _side_ring_count(graph["p_bonds"], len(graph["elements"]), n_p)
    labels: list[str] = []
    if bond_events.get("hydrogen_migration"):
        labels.append("h_transfer")
    if rings_p > rings_r:
        labels.append("ring_closure")
    if rings_r > rings_p:
        labels.append("ring_opening")
    if n_formed > 0 and n_broken == 0 and n_r > n_p:
        labels.append("addition")
    if n_broken > 0 and n_formed == 0 and n_p > n_r:
        labels.append("fragmentation")
    if n_broken >= 2 and n_formed >= 1 and n_p > n_r:
        labels.append("elimination")
    if n_formed > 0 and n_broken > 0 and n_formed == n_broken and n_r == n_p:
        labels.append("substitution")
    if n_formed > 0 and n_broken > 0 and n_formed != n_broken and n_r == n_p:
        labels.append("rearrangement")
    if not labels:
        labels.append("other")
    return labels


def classify_reaction(
    p1_document: Mapping[str, Any],
    *,
    taxonomy_version: str = DEFAULT_TAXONOMY_VERSION,
    canonical_budget: int = DEFAULT_CANONICAL_BUDGET,
) -> dict[str, Any] | None:
    """Return one P1 document's complete classification block.

    ``None`` marks an unclassifiable document (missing graph or events);
    the P2 builder records the typed ``unclassifiable`` status for those.
    """
    graph = p1_document.get("graph")
    events = p1_document.get("bond_events")
    if not isinstance(graph, Mapping) or not isinstance(events, Mapping):
        return None
    elements = {int(k): str(v) for k, v in graph["elements"].items()}
    roles = _node_roles(events)
    center = sorted(roles)
    if not center:
        return None
    r_edges = graph["r_bonds"]
    p_edges = graph["p_bonds"]
    levels: dict[str, dict[str, Any]] = {}
    l0_signature = _edit_family_signature(events)
    def cluster_id(level: str, signature: str) -> str:
        digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:16]
        return f"{taxonomy_version}:{level}:{digest}"
    levels[LEVEL_L0] = {
        "signature": l0_signature, "cluster_id": cluster_id(LEVEL_L0, l0_signature),
    }
    exhausted_all = True
    shell1 = sorted(_shell_expansion(center, r_edges, p_edges, 1))
    shell2 = sorted(_shell_expansion(center, r_edges, p_edges, 2))
    for level, nodes in (
        (LEVEL_L1, center),
        (LEVEL_L2, shell1),
        (LEVEL_L3, shell2),
    ):
        template = _template_graph_for(set(nodes), elements, roles, r_edges, p_edges)
        signature, exhausted = _direction_canonical(template, budget=canonical_budget)
        levels[level] = {
            "signature_sha256": hashlib.sha256(signature.encode("utf-8")).hexdigest(),
            "cluster_id": cluster_id(level, signature),
            "n_nodes": len(nodes),
        }
        exhausted_all = exhausted_all and exhausted
    irc = p1_document.get("irc_validation")
    irc_block: Mapping[str, Any] = irc if isinstance(irc, Mapping) else {}
    support_histogram = {
        "support": irc_block.get("n_support", 0),
        "weak": irc_block.get("n_weak", 0),
        "mismatch": irc_block.get("n_mismatch", 0),
    }
    return {
        "levels": levels,
        "family_labels": family_labels(events, graph),
        "pathway_labels": {
            "irc_orientation": irc_block.get("orientation"),
            "endpoint_match": irc_block.get("endpoint_match"),
            "synchrony": irc_block.get("synchrony"),
            "event_support_histogram": support_histogram,
        },
        "n_center_atoms": len(center),
        "n_context_r1": len(shell1),
        "n_context_r2": len(shell2),
        "flags": [] if exhausted_all else [FLAG_CANONICAL_BUDGET],
    }


__all__ = [
    "FLAG_CANONICAL_BUDGET",
    "LEVEL_L0",
    "LEVEL_L1",
    "LEVEL_L2",
    "LEVEL_L3",
    "classify_reaction",
    "family_labels",
]
