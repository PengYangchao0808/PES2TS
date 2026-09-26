"""P2 taxonomy fixtures: invariance, hierarchy, and readable labels."""

from __future__ import annotations

import copy
import json
from typing import Any

from pes2ts_core.g1.reaction_class import (
    FLAG_CANONICAL_BUDGET,
    LEVEL_L0,
    LEVEL_L1,
    LEVEL_L2,
    LEVEL_L3,
    classify_reaction,
    family_labels,
)

SN2 = {
    "bond_events": {
        "formed": [{"atoms": [2, 4], "order_r": None, "order_p": 1.0}],
        "broken": [{"atoms": [1, 2], "order_r": 1.0, "order_p": None}],
        "order_changed": [],
        "hydrogen_migration": [],
    },
    "graph": {
        "elements": {"1": "Cl", "2": "C", "3": "H", "4": "F"},
        "r_bonds": [[1, 2, 1.0], [2, 3, 1.0]],
        "p_bonds": [[2, 4, 1.0], [2, 3, 1.0]],
        "r_components": {"1": 0, "2": 0, "3": 0, "4": 1},
        "p_components": {"1": 0, "2": 1, "3": 1, "4": 1},
    },
    "irc_validation": {
        "orientation": "R_first", "endpoint_match": "pass",
        "synchrony": "synchronous", "n_support": 2, "n_weak": 0, "n_mismatch": 0,
    },
}


def _relabel(document: dict[str, Any], mapping: dict[int, int]) -> dict[str, Any]:
    out = copy.deepcopy(document)

    def m(value: int) -> int:
        return mapping.get(value, value)

    events = out["bond_events"]
    for kind in ("formed", "broken", "order_changed"):
        events[kind] = [
            {"atoms": [m(a) for a in entry["atoms"]], "order_r": entry["order_r"], "order_p": entry["order_p"]}
            for entry in events[kind]
        ]
    events["hydrogen_migration"] = [
        {"h": m(entry["h"]),
         "from": None if entry["from"] is None else m(entry["from"]),
         "to": None if entry["to"] is None else m(entry["to"])}
        for entry in events["hydrogen_migration"]
    ]
    graph = out["graph"]
    graph["elements"] = {str(m(int(k))): v for k, v in graph["elements"].items()}
    for side in ("r", "p"):
        graph[f"{side}_bonds"] = [[m(int(a)), m(int(b)), o] for a, b, o in graph[f"{side}_bonds"]]
        graph[f"{side}_components"] = {str(m(int(k))): v for k, v in graph[f"{side}_components"].items()}
    return out


def _reverse(document: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(document)
    events = out["bond_events"]
    formed = [
        {"atoms": entry["atoms"], "order_r": entry["order_p"], "order_p": entry["order_r"]}
        for entry in events["broken"]
    ]
    broken = [
        {"atoms": entry["atoms"], "order_r": entry["order_p"], "order_p": entry["order_r"]}
        for entry in events["formed"]
    ]
    events["formed"], events["broken"] = formed, broken
    events["order_changed"] = [
        {"atoms": entry["atoms"], "order_r": entry["order_p"], "order_p": entry["order_r"]}
        for entry in events["order_changed"]
    ]
    events["hydrogen_migration"] = [
        {"h": entry["h"], "from": entry["to"], "to": entry["from"]}
        for entry in events["hydrogen_migration"]
    ]
    graph = out["graph"]
    graph["r_bonds"], graph["p_bonds"] = graph["p_bonds"], graph["r_bonds"]
    graph["r_components"], graph["p_components"] = graph["p_components"], graph["r_components"]
    return out


def _levels(document: dict[str, Any]) -> dict[str, str]:
    result = classify_reaction(document)
    assert result is not None
    return {level: block["cluster_id"] for level, block in result["levels"].items()}


def test_map_relabeling_keeps_every_cluster() -> None:
    base = _levels(SN2)
    relabeled = _levels(_relabel(SN2, {1: 9, 2: 3, 3: 7, 4: 5}))
    assert relabeled == base


def test_direction_reversal_keeps_every_cluster() -> None:
    base = _levels(SN2)
    reversed_levels = _levels(_reverse(SN2))
    assert reversed_levels == base


def test_component_ordinal_shuffle_keeps_clusters() -> None:
    shuffled = copy.deepcopy(SN2)
    shuffled["graph"]["r_components"] = {"1": 3, "2": 3, "3": 3, "4": 0}
    shuffled["graph"]["p_components"] = {"1": 2, "2": 1, "3": 1, "4": 1}
    assert _levels(shuffled) == _levels(SN2)


def test_distinct_chemistry_lands_in_distinct_clusters() -> None:
    heavier = copy.deepcopy(SN2)
    heavier["graph"]["elements"] = {"1": "Br", "2": "C", "3": "H", "4": "F"}
    assert _levels(heavier)[LEVEL_L1] != _levels(SN2)[LEVEL_L1]


def test_context_levels_narrow_the_cluster() -> None:
    classification = classify_reaction(SN2)
    assert classification is not None
    assert classification["n_context_r2"] >= classification["n_context_r1"]
    assert classification["n_context_r1"] > classification["n_center_atoms"]
    assert classification["levels"][LEVEL_L0]["signature"] == "F1B1O0H0"


def test_context_difference_splits_l2_but_not_l1() -> None:
    # The shell atom (map 3, H on the central carbon) leaves the center, so
    # changing only its element must move L2 but never L1.
    heavier_shell = copy.deepcopy(SN2)
    heavier_shell["graph"]["elements"] = {"1": "Cl", "2": "C", "3": "Cl", "4": "F"}
    base = _levels(SN2)
    variant = _levels(heavier_shell)
    assert variant[LEVEL_L1] == base[LEVEL_L1]
    assert variant[LEVEL_L2] != base[LEVEL_L2]


def test_family_labels_rules() -> None:
    assert family_labels(SN2["bond_events"], SN2["graph"]) == ["substitution"]
    addition = {
        "formed": [{"atoms": [1, 2], "order_r": None, "order_p": 1.0}],
        "broken": [], "order_changed": [], "hydrogen_migration": [],
    }
    addition_graph = {
        "elements": {"1": "C", "2": "O"},
        "r_bonds": [], "p_bonds": [[1, 2, 1.0]],
        "r_components": {"1": 0, "2": 1}, "p_components": {"1": 0, "2": 0},
    }
    assert family_labels(addition, addition_graph) == ["addition"]
    fragmentation = {
        "formed": [], "broken": [{"atoms": [1, 2], "order_r": 1.0, "order_p": None}],
        "order_changed": [], "hydrogen_migration": [],
    }
    fragmentation_graph = {
        "elements": {"1": "C", "2": "O"},
        "r_bonds": [[1, 2, 1.0]], "p_bonds": [],
        "r_components": {"1": 0, "2": 0}, "p_components": {"1": 0, "2": 1},
    }
    assert family_labels(fragmentation, fragmentation_graph) == ["fragmentation"]
    ring = {
        "formed": [{"atoms": [1, 4], "order_r": None, "order_p": 1.0}],
        "broken": [], "order_changed": [], "hydrogen_migration": [],
    }
    ring_graph = {
        "elements": {"1": "C", "2": "C", "3": "C", "4": "C"},
        "r_bonds": [[1, 2, 1.0], [2, 3, 1.0], [3, 4, 1.0]],
        "p_bonds": [[1, 2, 1.0], [2, 3, 1.0], [3, 4, 1.0], [4, 1, 1.0]],
        "r_components": {"1": 0, "2": 0, "3": 0, "4": 0},
        "p_components": {"1": 0, "2": 0, "3": 0, "4": 0},
    }
    assert family_labels(ring, ring_graph) == ["ring_closure"]
    reverse_ring = {
        "formed": [], "broken": [{"atoms": [1, 4], "order_r": 1.0, "order_p": None}],
        "order_changed": [], "hydrogen_migration": [],
    }
    reverse_ring_graph = {
        "elements": ring_graph["elements"],
        "r_bonds": ring_graph["p_bonds"], "p_bonds": ring_graph["r_bonds"],
        "r_components": ring_graph["r_components"], "p_components": ring_graph["p_components"],
    }
    assert family_labels(reverse_ring, reverse_ring_graph) == ["ring_opening"]
    h_transfer = {
        "formed": [], "broken": [],
        "order_changed": [],
        "hydrogen_migration": [{"h": 3, "from": 1, "to": 2}],
    }
    h_graph = {
        "elements": {"1": "O", "2": "O", "3": "H"},
        "r_bonds": [[1, 3, 1.0], [1, 2, 1.0]],
        "p_bonds": [[2, 3, 1.0], [1, 2, 1.0]],
        "r_components": {"1": 0, "2": 0, "3": 0},
        "p_components": {"1": 0, "2": 0, "3": 0},
    }
    assert family_labels(h_transfer, h_graph) == ["h_transfer"]
    quiet = {"formed": [], "broken": [], "order_changed": [], "hydrogen_migration": []}
    assert family_labels(quiet, SN2["graph"]) == ["other"]


def test_canonical_budget_exhaustion_is_flagged() -> None:
    # A symmetric ring closure (C1-C2-C3-C4 closing 1-4) has tied center
    # atoms, so exhaustive canonical labeling needs individualization; a
    # budget of one branch cannot prove completeness.
    ring = {
        "bond_events": {
            "formed": [{"atoms": [1, 4], "order_r": None, "order_p": 1.0}],
            "broken": [], "order_changed": [], "hydrogen_migration": [],
        },
        "graph": {
            "elements": {"1": "C", "2": "C", "3": "C", "4": "C"},
            "r_bonds": [[1, 2, 1.0], [2, 3, 1.0], [3, 4, 1.0]],
            "p_bonds": [[1, 2, 1.0], [2, 3, 1.0], [3, 4, 1.0], [4, 1, 1.0]],
            "r_components": {"1": 0, "2": 0, "3": 0, "4": 0},
            "p_components": {"1": 0, "2": 0, "3": 0, "4": 0},
        },
        "irc_validation": {},
    }
    exhausted = classify_reaction(ring, canonical_budget=1)
    assert exhausted is not None
    assert FLAG_CANONICAL_BUDGET in exhausted["flags"]
    assert exhausted["levels"][LEVEL_L1]["cluster_id"]
    complete = classify_reaction(ring, canonical_budget=2000)
    assert complete is not None
    assert complete["flags"] == []
    assert complete["levels"][LEVEL_L1]["cluster_id"]


def test_missing_graph_is_unclassifiable() -> None:
    broken = json.loads(json.dumps(SN2))
    broken["graph"] = None
    assert classify_reaction(broken) is None
