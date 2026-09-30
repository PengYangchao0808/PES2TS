"""G1 v2 classification: exclusive-edit taxonomy with stable fallback ids.

The v2 classifier consumes **only** v2 edit documents (self-contained graph
plus exclusive edits) and produces five levels:

- ``l0_edit_family`` -- the *directional* event-count signature
  ``F{f}B{b}O{o}H{h}`` over the exclusive edits (the scan-facing layer);
- ``l0u_undirected_family`` -- the direction-invariant chemical family
  (``formed``/``broken`` counts swap-normalized), so the same transformation
  written backwards lands in one family while the directional L0 keeps the
  scan-relevant distinction;
- ``l1_center_template`` / ``l2_context_r1`` / ``l3_context_r2`` -- labelled
  reaction-center and context templates under a canonical labeling whose
  result is the lexicographic minimum of the template and its R<->P reversal.

Two v1 defects are fixed here.  First, the individualization budget fallback
of v1 serialized by map-number insertion order, so the 868 budget-exhausted
records changed their L2/L3 cluster ids under pure map relabeling -- a
violation of the id's mapping-number invariance.  v2 never mixes partial
search results: when the budget is exhausted the whole search aborts and a
**group-based WL-stable serialization** (colour classes plus multisets of
connection labels, no map numbers anywhere) supplies a deterministic,
relabeling-invariant id; the document is still flagged.  Second, node roles
and the rule labels are computed from the *exclusive* edits, and H2 events
no longer receive the ``h_transfer`` label.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from pes2ts_core.g0.fetch import dataset_version
from pes2ts_core.g1.build import shard_name
from pes2ts_core.g1.reaction_class import (
    _TemplateGraph,
    _individualize,
    _refine_colors,
    _serialize,
    _stable_hash_int,
)
from pes2ts_core.g1.v2_build import v2_document_path, v2_settings
from pes2ts_core.g1.v2_schema import (
    AUDIT_EXCLUDED,
    CLASSES_DIRNAME,
    DEFAULT_V2_CANONICAL_BUDGET,
    EDIT_BROKEN,
    EDIT_FORMED,
    EDIT_ORDER_CHANGED,
    FAMILY_LABELS,
    FLAG_BUDGET,
    H_CHANGE_TRANSFER,
    LEVEL_L0,
    LEVEL_L0U,
    LEVEL_L1,
    LEVEL_L2,
    LEVEL_L3,
    V2_CLASS_MANIFEST_FILENAME,
    V2_CLASS_SCHEMA_VERSION,
    V2_CLASS_SUMMARY_FILENAME,
    V2_DIRNAME,
    V2_MANIFEST_SCHEMA_VERSION,
    V2_MIGRATION_FILENAME,
    V2_SUMMARY_FILENAME,
    V2_TAXONOMY_VERSION,
)
from pes2ts_core.g1.truth_schema import (
    P2_SUMMARY_FILENAME,
    STATUS_CLASSIFIED,
    STATUS_UNCLASSIFIABLE,
)
from pes2ts_core.utils.hashing import JSONValue, sha256_file
from pes2ts_core.utils.jsonio import read_json, write_json
from pes2ts_core.utils.parquet_io import read_parquet, write_parquet

#: G0 split assignment artifact (read-only reporting input).
SPLIT_ASSIGNMENT_FILENAME: Final[str] = "split_assignment.parquet"

#: Summary Parquet columns of the v2 classification.
V2_CLASS_SUMMARY_COLUMNS: Final[tuple[str, ...]] = (
    "reaction_id", "status", "p1_status", "audit_status",
    "l0_signature", "l0_cluster_id", "l0u_signature", "l0u_cluster_id",
    "l1_cluster_id", "l2_cluster_id", "l3_cluster_id",
    "family_labels", "n_center_atoms", "n_context_r1", "n_context_r2", "flags",
)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")

#: v2 node roles swapped when the reaction direction is reversed.
_REVERSED_ROLES: Final[dict[str, str]] = {"F": "B", "B": "F", "Hd": "Ha", "Ha": "Hd"}

#: Bond-order rendering inside signatures ("~" marks an absent side).
_ABSENT_ORDER: Final[str] = "~"


def _order_text(order: float | None) -> str:
    return _ABSENT_ORDER if order is None else f"{order:.2f}"


def _edge_label(order_r: float | None, order_p: float | None) -> str:
    return f"r{_order_text(order_r)}p{_order_text(order_p)}"


def node_roles(
    edits: Sequence[Mapping[str, Any]],
    hydrogen_changes: Sequence[Mapping[str, Any]],
) -> dict[int, set[str]]:
    """Return the exclusive-edit role set of every map number."""
    roles: dict[int, set[str]] = {}
    kind_to_role = {EDIT_FORMED: "F", EDIT_BROKEN: "B", EDIT_ORDER_CHANGED: "O"}
    for edit in edits:
        role = kind_to_role[str(edit["edit_kind"])]
        for map_number in edit["pair"]:
            roles.setdefault(int(map_number), set()).add(role)
    for change in hydrogen_changes:
        roles.setdefault(int(change["h"]), set()).add("H")
        if change.get("from") is not None:
            roles.setdefault(int(change["from"]), set()).add("Hd")
        if change.get("to") is not None:
            roles.setdefault(int(change["to"]), set()).add("Ha")
    return roles


def _reversed_template(graph: _TemplateGraph) -> _TemplateGraph:
    """Return the template with every R/P semantic swapped (v2 roles)."""
    flipped_nodes = {
        node: (element, tuple(sorted(
            _REVERSED_ROLES.get(role, role) for role in roles
        )))
        for node, (element, roles) in graph.node_labels.items()
    }
    flipped_edges = [
        (first, second, order_p, order_r)
        for first, second, order_r, order_p in graph._edges_with_labels()
    ]
    return _TemplateGraph(flipped_nodes, flipped_edges)


def _wl2_stable_serialization(graph: _TemplateGraph, colors: dict[int, int]) -> str:
    """Group-based canonical serialization under a non-discrete coloring.

    Nodes are grouped by WL colour; every group renders its label plus the
    sorted multiset of per-member connection-label multisets (neighbours
    referenced by colour rank, never by position).  No map number or
    insertion order can leak into the string, so the id is
    mapping-number-invariant by construction.
    """
    color_ranks = sorted(set(colors.values()))
    parts: list[str] = [f"n{len(colors)}wl2"]
    for rank in color_ranks:
        members = sorted(node for node, color in colors.items() if color == rank)
        label = graph.label_text(members[0])
        connections = sorted(
            sorted(f"{colors[neighbour]}#{edge}" for neighbour, edge in graph.adjacency[member])
            for member in members
        )
        rendered = ";".join(",".join(entry) for entry in connections)
        parts.append(f"g{rank}+{label}:{rendered}")
    return ";".join(parts)


def _canonical_serialization(
    graph: _TemplateGraph, *, budget: int
) -> tuple[str, bool]:
    """Return ``(canonical string, search_complete)``.

    When the individualization budget is exhausted mid-search the entire
    search aborts (partial minima are never mixed) and the group-based
    WL-stable serialization takes over deterministically.
    """
    label_colors = {
        label: _stable_hash_int(graph.label_text_of(label))
        for label in set(graph.node_labels.values())
    }
    colors = {node: label_colors[graph.node_labels[node]] for node in graph.adjacency}
    colors = _refine_colors(graph, colors)
    best: list[str] = []
    state = {"used": 0, "complete": True}

    def search(current: dict[int, int]) -> None:
        if len(set(current.values())) == len(current):
            best.append(_serialize(graph, current))
            return
        cells: dict[int, list[int]] = {}
        for node, color in current.items():
            cells.setdefault(color, []).append(node)
        # Branch on the smallest-*color* non-singleton cell: colour ranks are
        # canonical under relabeling, so the search tree (and therefore the
        # leaf set) is isomorphism-invariant.  Selecting by member ids would
        # make the leaves map-number dependent.
        target_color = min(color for color, members in cells.items() if len(members) > 1)
        for candidate in sorted(cells[target_color]):
            state["used"] += 1
            if state["used"] > budget:
                state["complete"] = False
                return
            search(_refine_colors(graph, _individualize(current, candidate)))
            if not state["complete"]:
                return

    search(colors)
    if state["complete"] and best:
        return min(best), True
    return _wl2_stable_serialization(graph, colors), False


def _direction_canonical(
    graph: _TemplateGraph, *, budget: int
) -> tuple[str, bool]:
    """Return the direction-invariant canonical string and completeness."""
    forward, forward_ok = _canonical_serialization(graph, budget=budget)
    backward, backward_ok = _canonical_serialization(_reversed_template(graph), budget=budget)
    return min(forward, backward), forward_ok and backward_ok


def _template_graph_for(
    nodes: set[int],
    elements: Mapping[int, str],
    roles: Mapping[int, set[str]],
    r_edges: Sequence[Sequence[float]],
    p_edges: Sequence[Sequence[float]],
) -> _TemplateGraph:
    """Build the labelled template graph over *nodes* from the v2 graph."""
    node_labels = {
        node: (elements.get(node, "?"), tuple(sorted(roles.get(node, ()))))
        for node in sorted(nodes)
    }
    edge_orders: dict[tuple[int, int], list[float | None]] = {}
    for side, edges in (("r", r_edges), ("p", p_edges)):
        for edge in edges:
            first, second = int(edge[0]), int(edge[1])
            if first not in nodes or second not in nodes:
                continue
            pair = (first, second) if first <= second else (second, first)
            edge_orders.setdefault(pair, [None, None])
            edge_orders[pair][0 if side == "r" else 1] = float(edge[2])
    return _TemplateGraph(
        node_labels,
        [
            (pair[0], pair[1], orders[0], orders[1])
            for pair, orders in sorted(edge_orders.items())
        ],
    )


def _shell_expansion(
    center: Sequence[int],
    r_edges: Sequence[Sequence[float]],
    p_edges: Sequence[Sequence[float]],
    shells: int,
) -> set[int]:
    adjacency: dict[int, set[int]] = {}
    for edge in [*r_edges, *p_edges]:
        first, second = int(edge[0]), int(edge[1])
        adjacency.setdefault(first, set()).add(second)
        adjacency.setdefault(second, set()).add(first)
    expanded = set(int(value) for value in center)
    for _ in range(max(shells, 0)):
        expanded |= {n for atom in expanded for n in adjacency.get(atom, ())}
    return expanded


def l0_signature(
    counts: Mapping[str, int],
) -> str:
    """Return the directional exclusive-edit count signature."""
    return (
        f"F{counts.get(EDIT_FORMED, 0)}"
        f"B{counts.get(EDIT_BROKEN, 0)}"
        f"O{counts.get(EDIT_ORDER_CHANGED, 0)}"
        f"H{counts.get('n_h_total', 0)}"
    )


def l0u_signature(counts: Mapping[str, int]) -> str:
    """Return the direction-invariant family signature (F/B swap-normalized)."""
    formed = counts.get(EDIT_FORMED, 0)
    broken = counts.get(EDIT_BROKEN, 0)
    return (
        f"F{max(formed, broken)}B{min(formed, broken)}"
        f"O{counts.get(EDIT_ORDER_CHANGED, 0)}"
        f"H{counts.get('n_h_total', 0)}"
    )


def _side_ring_count(
    edges: Sequence[Sequence[float]], n_nodes: int, n_components: int
) -> int:
    return max(len(edges) - n_nodes + n_components, 0)


def family_labels(
    edits: Sequence[Mapping[str, Any]],
    hydrogen_changes: Sequence[Mapping[str, Any]],
    graph: Mapping[str, Any],
) -> list[str]:
    """Return the rule labels computed from the *exclusive* edit counts."""
    n_formed = sum(1 for edit in edits if edit["edit_kind"] == EDIT_FORMED)
    n_broken = sum(1 for edit in edits if edit["edit_kind"] == EDIT_BROKEN)
    n_r = len(set(graph["r_components"].values()))
    n_p = len(set(graph["p_components"].values()))
    rings_r = _side_ring_count(graph["r_bonds"], len(graph["elements"]), n_r)
    rings_p = _side_ring_count(graph["p_bonds"], len(graph["elements"]), n_p)
    labels: list[str] = []
    if any(change["kind"] == H_CHANGE_TRANSFER for change in hydrogen_changes):
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


def classify_v2_reaction(
    v2_document: Mapping[str, Any],
    *,
    taxonomy_version: str = V2_TAXONOMY_VERSION,
    canonical_budget: int = DEFAULT_V2_CANONICAL_BUDGET,
) -> dict[str, Any] | None:
    """Return the complete v2 classification block of one v2 edit document.

    ``None`` marks an unclassifiable document (excluded records or documents
    without a usable graph); the caller records the typed status for those.
    """
    graph = v2_document.get("graph")
    edits = v2_document.get("edits")
    hydrogen_changes = v2_document.get("hydrogen_partner_changes")
    if not isinstance(graph, Mapping) or not isinstance(edits, list):
        return None
    elements = {int(k): str(v) for k, v in graph["elements"].items()}
    roles = node_roles(edits, hydrogen_changes or [])
    center = sorted(roles)
    if not center:
        return None
    r_edges = graph["r_bonds"]
    p_edges = graph["p_bonds"]
    counts = dict(v2_document.get("edit_counts") or {})
    h_total = sum(counts.get(kind, 0) for kind in (
        "transfer", "release", "capture", "to_hh", "from_hh",
        "hh_release", "hh_form_free", "hh_swap",
    ))
    counts["n_h_total"] = h_total

    def cluster_id(level: str, signature: str) -> str:
        digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:16]
        return f"{taxonomy_version}:{level}:{digest}"

    levels: dict[str, dict[str, Any]] = {
        LEVEL_L0: {
            "signature": l0_signature(counts),
            "cluster_id": cluster_id(LEVEL_L0, l0_signature(counts)),
        },
        LEVEL_L0U: {
            "signature": l0u_signature(counts),
            "cluster_id": cluster_id(LEVEL_L0U, l0u_signature(counts)),
        },
    }
    complete_all = True
    shell1 = sorted(_shell_expansion(center, r_edges, p_edges, 1))
    shell2 = sorted(_shell_expansion(center, r_edges, p_edges, 2))
    for level, nodes in ((LEVEL_L1, center), (LEVEL_L2, shell1), (LEVEL_L3, shell2)):
        template = _template_graph_for(set(nodes), elements, roles, r_edges, p_edges)
        signature, complete = _direction_canonical(template, budget=canonical_budget)
        levels[level] = {
            "signature_sha256": hashlib.sha256(signature.encode("utf-8")).hexdigest(),
            "cluster_id": cluster_id(level, signature),
            "n_nodes": len(nodes),
        }
        complete_all = complete_all and complete
    return {
        "levels": levels,
        "family_labels": family_labels(edits, hydrogen_changes or [], graph),
        "n_center_atoms": len(center),
        "n_context_r1": len(shell1),
        "n_context_r2": len(shell2),
        "flags": [] if complete_all else [FLAG_BUDGET],
    }


# ---------------------------------------------------------------------------
# Orchestration: classify every v2 edit document and build the migration
# report against the frozen v1 classification.
# ---------------------------------------------------------------------------
def v2_class_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the effective ``g1_v2`` classification settings."""
    raw = config.get("g1_v2")
    settings: Mapping[str, Any] = raw if isinstance(raw, Mapping) else {}
    return {
        "taxonomy_version": str(settings.get("taxonomy_version", V2_TAXONOMY_VERSION)),
        "canonical_budget": int(
            settings.get("canonical_budget", DEFAULT_V2_CANONICAL_BUDGET)
        ),
    }


def v2_classes_dir(interim_dir: str | Path) -> Path:
    return Path(interim_dir) / V2_DIRNAME / CLASSES_DIRNAME


def v2_class_document_path(
    interim_dir: str | Path, reaction_id: str, shard_size: int
) -> Path:
    return v2_classes_dir(interim_dir) / shard_name(reaction_id, shard_size) / f"{reaction_id}.json"


def _class_document(
    v2_document: Mapping[str, Any], settings: Mapping[str, Any]
) -> dict[str, JSONValue]:
    reaction_id = str(v2_document["reaction_id"])
    document: dict[str, JSONValue] = {
        "schema_version": V2_CLASS_SCHEMA_VERSION,
        "reaction_id": reaction_id,
        "generated_at": _now(),
        "taxonomy_version": str(settings["taxonomy_version"]),
        "status": STATUS_UNCLASSIFIABLE,
        "p1_status": str(v2_document.get("p1_status", "")),
        "audit_status": str(v2_document.get("audit_status", "")),
        "levels": {},
        "family_labels": [],
        "n_center_atoms": 0,
        "n_context_r1": 0,
        "n_context_r2": 0,
        "flags": [],
    }
    if document["audit_status"] == AUDIT_EXCLUDED:
        document["status_detail"] = "excluded_p1_status"
        return document
    classification = classify_v2_reaction(
        v2_document,
        taxonomy_version=str(settings["taxonomy_version"]),
        canonical_budget=int(settings["canonical_budget"]),
    )
    if classification is None:
        document["status_detail"] = "no_usable_events_or_graph"
        return document
    document["status"] = STATUS_CLASSIFIED
    document["levels"] = classification["levels"]
    document["family_labels"] = classification["family_labels"]
    document["n_center_atoms"] = classification["n_center_atoms"]
    document["n_context_r1"] = classification["n_context_r1"]
    document["n_context_r2"] = classification["n_context_r2"]
    document["flags"] = classification["flags"]
    return document


def _class_summary_row(document: Mapping[str, JSONValue]) -> dict[str, JSONValue]:
    levels: Mapping[str, Any] = document.get("levels") or {}
    l0 = levels.get(LEVEL_L0, {})
    l0u = levels.get(LEVEL_L0U, {})
    return {
        "reaction_id": document["reaction_id"],
        "status": document["status"],
        "p1_status": document["p1_status"],
        "audit_status": document["audit_status"],
        "l0_signature": l0.get("signature"),
        "l0_cluster_id": l0.get("cluster_id"),
        "l0u_signature": l0u.get("signature"),
        "l0u_cluster_id": l0u.get("cluster_id"),
        "l1_cluster_id": levels.get(LEVEL_L1, {}).get("cluster_id"),
        "l2_cluster_id": levels.get(LEVEL_L2, {}).get("cluster_id"),
        "l3_cluster_id": levels.get(LEVEL_L3, {}).get("cluster_id"),
        "family_labels": "|".join(document["family_labels"]),
        "n_center_atoms": document["n_center_atoms"],
        "n_context_r1": document["n_context_r1"],
        "n_context_r2": document["n_context_r2"],
        "flags": "|".join(document["flags"]),
    }


def build_migration_report(
    config: Mapping[str, Any],
    v2_rows: Sequence[Mapping[str, Any]],
) -> dict[str, JSONValue]:
    """Return the v1->v2 family/label migration report."""
    interim_dir = Path(config["paths"]["interim"])
    v1_rows = {
        str(row["reaction_id"]): row
        for row in read_parquet(interim_dir / P2_SUMMARY_FILENAME).to_pylist()
    }
    split_of: dict[str, str] = {}
    split_path = interim_dir / SPLIT_ASSIGNMENT_FILENAME
    if split_path.is_file():
        split_of = {
            str(row["reaction_id"]): str(row["split"])
            for row in read_parquet(split_path).to_pylist()
        }
    transitions: Counter[tuple[str, str]] = Counter()
    l0_v1: Counter[str] = Counter()
    l0_v2: Counter[str] = Counter()
    split_by_l0u: Counter[str] = Counter()
    for row in v2_rows:
        reaction_id = str(row["reaction_id"])
        if str(row["status"]) != STATUS_CLASSIFIED:
            continue
        v1_row = v1_rows.get(reaction_id)
        v1_labels = str(v1_row["family_labels"]) if v1_row else "<missing>"
        v2_labels = str(row["family_labels"])
        transitions[(v1_labels, v2_labels)] += 1
        if v1_row and str(v1_row.get("status")) == STATUS_CLASSIFIED:
            l0_v1[str(v1_row["edit_family_id"])] += 1
        l0_v2[str(row["l0_cluster_id"])] += 1
        split_by_l0u[f"{split_of.get(reaction_id, 'unassigned')}|{row['l0u_cluster_id']}"] += 1
    return {
        "schema_version": V2_MANIFEST_SCHEMA_VERSION,
        "n_classified_v2": sum(l0_v2.values()),
        "v1_l0_clusters": len(l0_v1),
        "v2_l0_clusters": len(l0_v2),
        "family_label_transitions": {
            f"{v1} => {v2}": count
            for (v1, v2), count in sorted(transitions.items(), key=lambda item: (-item[1], item[0]))
        },
        "n_label_transitions": sum(
            count for (v1, v2), count in transitions.items() if v1 != v2
        ),
        "split_by_l0u_pairs": len(split_by_l0u),
        "note": (
            "v1 counts come from the frozen v1 classification; v2 uses "
            "exclusive-edit semantics so directional L0 signatures are not "
            "comparable one-to-one (the transition matrix is the comparison)"
        ),
        "generated_at": _now(),
    }


def classify_v2(config: Mapping[str, Any]) -> dict[str, Any]:
    """Classify every v2 edit document and write class summary/manifests."""
    manifests_dir = Path(config["paths"]["manifests"])
    interim_dir = Path(config["paths"]["interim"])
    settings = v2_class_settings(config)
    shard_size = int(v2_settings(config)["shard_size"])
    summary_path = interim_dir / V2_SUMMARY_FILENAME
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"Missing v2 summary {summary_path}; run `g1 v2-build` first"
        )
    rows_in = read_parquet(summary_path).to_pylist()
    documents: list[dict[str, JSONValue]] = []
    for row in rows_in:
        reaction_id = str(row["reaction_id"])
        v2_document = read_json(v2_document_path(interim_dir, reaction_id, shard_size))
        document = _class_document(v2_document, settings)
        write_json(v2_class_document_path(interim_dir, reaction_id, shard_size), document)
        documents.append(document)
    rows = [_class_summary_row(document) for document in documents]
    class_summary_path = interim_dir / V2_CLASS_SUMMARY_FILENAME
    write_parquet(
        class_summary_path,
        {name: [row[name] for row in rows] for name in V2_CLASS_SUMMARY_COLUMNS},
    )
    status_histogram = Counter(str(row["status"]) for row in rows)
    family_histogram = Counter(
        label for row in rows for label in str(row["family_labels"]).split("|") if label
    )
    n_budget_flagged = sum(1 for row in rows if str(row["flags"]).find(FLAG_BUDGET) >= 0)
    clusters = {
        level: len({row[column] for row in rows if row[column]})
        for level, column in (
            (LEVEL_L0, "l0_cluster_id"),
            (LEVEL_L0U, "l0u_cluster_id"),
            (LEVEL_L1, "l1_cluster_id"),
            (LEVEL_L2, "l2_cluster_id"),
            (LEVEL_L3, "l3_cluster_id"),
        )
    }
    manifest: dict[str, JSONValue] = {
        "schema_version": V2_MANIFEST_SCHEMA_VERSION,
        "dataset_version": dataset_version(config),
        "taxonomy_version": settings["taxonomy_version"],
        "n_total": len(rows),
        "n_classified": status_histogram.get(STATUS_CLASSIFIED, 0),
        "n_unclassifiable": status_histogram.get(STATUS_UNCLASSIFIABLE, 0),
        "n_clusters_by_level": clusters,
        "family_labels": dict(sorted(family_histogram.items())),
        "canonical_budget_flagged": n_budget_flagged,
        "summary_sha256": sha256_file(class_summary_path),
        "config": dict(sorted(settings.items())),
        "generated_at": _now(),
    }
    manifest_path = manifests_dir / V2_CLASS_MANIFEST_FILENAME
    write_json(manifest_path, manifest)
    migration = build_migration_report(config, rows)
    write_json(manifests_dir / V2_MIGRATION_FILENAME, migration)
    return manifest


__all__ = [
    "FAMILY_LABELS",
    "classify_v2",
    "classify_v2_reaction",
    "family_labels",
    "l0_signature",
    "l0u_signature",
    "node_roles",
    "v2_class_document_path",
    "v2_class_settings",
    "v2_classes_dir",
]
