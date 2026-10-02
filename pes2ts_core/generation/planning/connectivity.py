"""A connectivity-only execution view; the complete edit graph stays archived."""
from __future__ import annotations

from collections import Counter
from dataclasses import replace

from pes2ts_core.g1.reaction_edit_graph import ReactionEditGraph, _connected_components


def connectivity_view(graph: ReactionEditGraph) -> ReactionEditGraph:
    """Recompute centres/components from formed and broken edges only."""
    edits = tuple(e for e in graph.edits if e.edit_kind in {"formed", "broken"})
    pairs = [e.pair for e in edits]
    vertices = sorted({a for pair in pairs for a in pair})
    degree = Counter(a for pair in pairs for a in pair)
    components = tuple(tuple(c) for c in _connected_components(vertices, pairs))
    return replace(
        graph, edits=edits,
        edit_counts={k: sum(e.edit_kind == k for e in edits)
                     for k in ("formed", "broken", "order_changed")},
        edit_components=components, connectivity_edit_components=components,
        edit_cycle_rank=len(pairs) - len(vertices) + len(components),
        edit_degree=dict(degree), center_atoms=tuple(a for a in vertices if degree[a] >= 2),
        atom_events=tuple({**e, "attributes_changed": False} for e in graph.atom_events
                          if e.get("atom_map_id") in vertices and e.get("partner_set_changed")),
        atom_attribute_changes=(),
    )


def required_pairs(bundle) -> set[tuple[int, int]]:
    """Bond existence, rather than a change of bond order, defines scope."""
    sides = [{tuple(sorted((e.map_a, e.map_b))) for e in g.edges}
             for g in (bundle.r_graph, bundle.p_graph)]
    return sides[0] ^ sides[1]


def validate_driver_pairs(drivers, required) -> None:
    pairs = [tuple(sorted(d.get("maps", ()))) for d in drivers]
    if not required:
        raise ValueError("OUT_OF_SCOPE_NO_CONNECTIVITY_EDIT")
    if len(set(pairs)) != len(pairs) or set(pairs) != set(required):
        raise ValueError("CONNECTIVITY_DRIVER_SET_MISMATCH")
    if any(d.get("kind") not in {"B", "distance"} or
           d.get("edit_kind") not in {"formed", "broken"} for d in drivers):
        raise ValueError("NON_CONNECTIVITY_DRIVER")
