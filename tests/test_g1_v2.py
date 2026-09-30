"""Offline tests for the G1 v2 exclusive-edit layer, taxonomy, and export.

The fixtures are synthetic map-space graphs (no RDKit, no data files).  They
lock the v2 contract: exactly one mutually exclusive edit per unordered map
pair, exhaustive hydrogen partner-change kinds (H2 events are never
transfers), aromatic-region annotation, the relabeling-invariant budget
fallback (the v1 L2/L3 instability regression), direction-invariant L0u,
and the whitelisted G2 export.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping

from pes2ts_core.g1.reaction_class import _TemplateGraph
from pes2ts_core.g1.v2_classify import (
    _canonical_serialization,
    classify_v2_reaction,
    family_labels,
    l0_signature,
    l0u_signature,
    node_roles,
)
from pes2ts_core.g1.v2_edits import (
    edit_counts,
    edit_signature,
    exclusive_edits,
    legacy_events_from_graph,
    legacy_reconciles,
    reaction_center,
    validate_edit_contract,
)
from pes2ts_core.g1.v2_build import build_v2_reaction
from pes2ts_core.g1.v2_gate import build_export_document
from pes2ts_core.g1.v2_schema import (
    EDIT_BROKEN,
    EDIT_FORMED,
    EDIT_ORDER_CHANGED,
)


def _block(elements: Mapping[int, str], r_bonds: list, p_bonds: list) -> dict:
    return exclusive_edits(elements, r_bonds, p_bonds)


ETHANE_ELEMENTS = {1: "C", 2: "C", 3: "H", 4: "H", 5: "H", 6: "H", 7: "H", 8: "H"}


def test_order_change_is_one_exclusive_event() -> None:
    block = _block(
        ETHANE_ELEMENTS,
        [[1, 2, 1.0], [1, 3, 1.0], [1, 4, 1.0], [1, 5, 1.0],
         [2, 6, 1.0], [2, 7, 1.0], [2, 8, 1.0]],
        [[1, 2, 2.0], [1, 3, 1.0], [1, 4, 1.0], [1, 5, 1.0],
         [2, 6, 1.0], [2, 7, 1.0], [2, 8, 1.0]],
    )
    edits = block["edits"]
    assert len(edits) == 1
    assert edits[0]["pair"] == [1, 2]
    assert edits[0]["edit_kind"] == EDIT_ORDER_CHANGED
    assert edits[0]["r_bond_order"] == 1.0
    assert edits[0]["p_bond_order"] == 2.0
    counts = edit_counts(block)
    assert counts[EDIT_FORMED] == 0
    assert counts[EDIT_BROKEN] == 0
    assert counts[EDIT_ORDER_CHANGED] == 1


def test_formation_and_breakage_are_exclusive() -> None:
    formed = _block({1: "C", 2: "C"}, [], [[1, 2, 1.0]])
    broken = _block({1: "C", 2: "C"}, [[1, 2, 1.0]], [])
    assert [e["edit_kind"] for e in formed["edits"]] == [EDIT_FORMED]
    assert formed["edits"][0]["r_bond_order"] is None
    assert [e["edit_kind"] for e in broken["edits"]] == [EDIT_BROKEN]
    assert broken["edits"][0]["p_bond_order"] is None


def test_aromatic_edits_carry_one_region_id() -> None:
    ring = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (1, 6)]
    r_bonds = [[a, b, 1.5] for a, b in ring]
    p_bonds = [
        [1, 2, 1.0], [2, 3, 2.0], [3, 4, 1.0],
        [4, 5, 2.0], [5, 6, 1.0], [1, 6, 1.0],
    ]
    block = _block({i: "C" for i in range(1, 7)}, r_bonds, p_bonds)
    regions = block["aromatic_regions"]
    assert len(regions) == 1
    region_id = next(iter(regions))
    assert {tuple(pair) for pair in regions[region_id]} == set(ring)
    for edit in block["edits"]:
        assert edit["aromatic_region"] == region_id
        assert edit["r_bond_order"] == 1.5


def test_block_is_stable_across_a_json_round_trip() -> None:
    import json

    ring = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (1, 6)]
    block = _block(
        {i: "C" for i in range(1, 7)},
        [[a, b, 1.5] for a, b in ring],
        [[1, 2, 1.0], [2, 3, 2.0], [3, 4, 1.0], [4, 5, 2.0], [5, 6, 1.0], [1, 6, 1.0]],
    )
    persisted = json.loads(json.dumps({
        "edits": block["edits"],
        "hydrogen_partner_changes": block["hydrogen_partner_changes"],
        "aromatic_regions": block["aromatic_regions"],
    }))
    assert persisted["edits"] == block["edits"]
    assert persisted["hydrogen_partner_changes"] == block["hydrogen_partner_changes"]
    assert persisted["aromatic_regions"] == block["aromatic_regions"]


def test_hydrogen_change_kinds_are_exhaustive_and_correct() -> None:
    transfer = _block({1: "H", 2: "O", 3: "C"}, [[1, 2, 1.0]], [[1, 3, 1.0]])
    assert [c["kind"] for c in transfer["hydrogen_partner_changes"]] == ["transfer"]

    release = _block({1: "H", 2: "O"}, [[1, 2, 1.0]], [])
    assert [c["kind"] for c in release["hydrogen_partner_changes"]] == ["release"]

    capture = _block({1: "H", 2: "O"}, [], [[1, 2, 1.0]])
    assert [c["kind"] for c in capture["hydrogen_partner_changes"]] == ["capture"]

    hh_formation = _block({1: "H", 2: "H"}, [], [[1, 2, 1.0]])
    kinds = [c["kind"] for c in hh_formation["hydrogen_partner_changes"]]
    assert kinds == ["hh_form_free", "hh_form_free"]
    assert "transfer" not in kinds

    hh_split = _block({1: "H", 2: "H", 3: "C"}, [[1, 2, 1.0]], [[2, 3, 1.0]])
    kinds = {c["h"]: c["kind"] for c in hh_split["hydrogen_partner_changes"]}
    assert kinds == {1: "hh_release", 2: "from_hh"}


def test_edit_signature_is_relabeling_invariant() -> None:
    r_bonds = [[1, 2, 1.0], [2, 3, 1.0], [1, 6, 1.5]]
    p_bonds = [[1, 2, 2.0], [2, 3, 1.0], [3, 4, 1.0]]
    elements = {1: "C", 2: "C", 3: "O", 4: "C", 5: "H", 6: "C"}
    relabel = {1: 10, 2: 20, 3: 30, 4: 40, 5: 50, 6: 60}
    assert edit_signature(elements, r_bonds, p_bonds) == edit_signature(
        {relabel[k]: v for k, v in elements.items()},
        [[relabel[a], relabel[b], o] for a, b, o in r_bonds],
        [[relabel[a], relabel[b], o] for a, b, o in p_bonds],
    )


def test_contract_validation_catches_tampering() -> None:
    elements = {1: "C", 2: "C", 3: "H"}
    r_bonds = [[1, 3, 1.0], [2, 3, 1.0]]
    p_bonds = [[1, 2, 1.0], [1, 3, 1.0]]
    block = _block(elements, r_bonds, p_bonds)
    center = reaction_center(
        block["edits"], block["hydrogen_partner_changes"], block["_adjacency"], 1,
    )
    assert validate_edit_contract(
        block["edits"], block["hydrogen_partner_changes"], center, block["multi_bond_pairs"],
    ) == []

    duplicate = [*block["edits"], dict(block["edits"][0])]
    assert validate_edit_contract(
        duplicate, block["hydrogen_partner_changes"], center, [],
    )

    wrong_kind = [dict(block["edits"][0], edit_kind=EDIT_ORDER_CHANGED)]
    assert validate_edit_contract(
        wrong_kind, block["hydrogen_partner_changes"], center, [],
    )

    multi = _block(elements, [[1, 2, 1.0], [1, 2, 2.0]], [])
    assert validate_edit_contract(
        multi["edits"], multi["hydrogen_partner_changes"],
        {"core": [1, 2], "with_shell": [1, 2]},
        multi["multi_bond_pairs"],
    )


def test_legacy_events_reconcile_with_v1_semantics() -> None:
    elements = ETHANE_ELEMENTS
    r_bonds = [[1, 2, 1.0], [1, 3, 1.0], [1, 4, 1.0], [1, 5, 1.0],
               [2, 6, 1.0], [2, 7, 1.0], [2, 8, 1.0]]
    p_bonds = [[1, 2, 2.0], [1, 3, 1.0], [1, 4, 1.0], [1, 5, 1.0],
               [2, 6, 1.0], [2, 7, 1.0], [2, 8, 1.0]]
    legacy = legacy_events_from_graph(elements, r_bonds, p_bonds)
    assert legacy["formed"] == [
        {"atoms": [1, 2], "order_r": None, "order_p": 2.0}
    ]
    assert legacy["broken"] == [
        {"atoms": [1, 2], "order_r": 1.0, "order_p": None}
    ]
    assert legacy["order_changed"] == [
        {"atoms": [1, 2], "order_r": 1.0, "order_p": 2.0}
    ]
    assert legacy_reconciles(legacy, legacy)
    tampered = copy.deepcopy(legacy)
    tampered["formed"] = []
    assert not legacy_reconciles(tampered, legacy)


def _ethane_v2_document() -> dict:
    elements = ETHANE_ELEMENTS
    r_bonds = [[1, 2, 1.0], [1, 3, 1.0], [1, 4, 1.0], [1, 5, 1.0],
               [2, 6, 1.0], [2, 7, 1.0], [2, 8, 1.0]]
    p_bonds = [[1, 2, 2.0], [1, 3, 1.0], [1, 4, 1.0], [1, 5, 1.0],
               [2, 6, 1.0], [2, 7, 1.0], [2, 8, 1.0]]
    return {
        "schema_version": "g1_v2_edits_v1",
        "reaction_id": "RXN_TEST",
        "p1_status": "resolved_unique",
        "audit_status": "clean",
        "issues": [],
        "dimensions": [],
        "edits": exclusive_edits(elements, r_bonds, p_bonds)["edits"],
        "hydrogen_partner_changes": [],
        "aromatic_regions": {},
        "reaction_center": {"core": [1, 2], "with_shell": [1, 2, 3, 4, 5, 6, 7, 8]},
        "edit_counts": {"formed": 0, "broken": 0, "order_changed": 1},
        "graph": {
            "elements": elements,
            "r_bonds": r_bonds,
            "p_bonds": p_bonds,
            "r_components": {m: 0 for m in elements},
            "p_components": {m: 0 for m in elements},
        },
        "irc_evidence": {"endpoint_match": "pass", "orientation": "R_first",
                         "n_event_mismatch": 0, "n_quality_flags": 0},
    }


def test_l0_directional_and_l0u_undirected() -> None:
    counts = {"formed": 2, "broken": 1, "order_changed": 0, "n_h_total": 1}
    assert l0_signature(counts) == "F2B1O0H1"
    assert l0u_signature(counts) == "F2B1O0H1"
    reversed_counts = {"formed": 1, "broken": 2, "order_changed": 0, "n_h_total": 1}
    assert l0_signature(reversed_counts) == "F1B2O0H1"
    assert l0u_signature(reversed_counts) == "F2B1O0H1"


def test_classification_is_relabeling_invariant() -> None:
    document = _ethane_v2_document()
    base = classify_v2_reaction(document)
    assert base is not None
    relabel = {1: 5, 2: 8, 3: 1, 4: 2, 5: 3, 6: 4, 7: 6, 8: 7}
    relabeled = copy.deepcopy(document)
    for key in ("edits",):
        relabeled[key] = [
            {**edit, "pair": [relabel[m] for m in edit["pair"]]} for edit in document[key]
        ]
    graph = relabeled["graph"]
    graph["elements"] = {relabel[m]: e for m, e in document["graph"]["elements"].items()}
    graph["r_bonds"] = [[relabel[a], relabel[b], o] for a, b, o in document["graph"]["r_bonds"]]
    graph["p_bonds"] = [[relabel[a], relabel[b], o] for a, b, o in document["graph"]["p_bonds"]]
    graph["r_components"] = {relabel[m]: v for m, v in document["graph"]["r_components"].items()}
    graph["p_components"] = {relabel[m]: v for m, v in document["graph"]["p_components"].items()}
    other = classify_v2_reaction(relabeled)
    assert other is not None
    for level in ("l1_center_template", "l2_context_r1", "l3_context_r2"):
        assert base["levels"][level]["cluster_id"] == other["levels"][level]["cluster_id"]


def test_classification_is_direction_invariant_except_l0() -> None:
    document = _ethane_v2_document()
    base = classify_v2_reaction(document)
    assert base is not None
    graph = document["graph"]
    reversed_doc = copy.deepcopy(document)
    reversed_doc["graph"] = {
        "elements": graph["elements"],
        "r_bonds": graph["p_bonds"],
        "p_bonds": graph["r_bonds"],
        "r_components": graph["p_components"],
        "p_components": graph["r_components"],
    }
    reversed_doc["edits"] = [{
        **edit,
        "edit_kind": EDIT_ORDER_CHANGED,
        "r_bond_order": edit["p_bond_order"],
        "p_bond_order": edit["r_bond_order"],
    } for edit in document["edits"]]
    other = classify_v2_reaction(reversed_doc)
    assert other is not None
    for level in ("l1_center_template", "l2_context_r1", "l3_context_r2"):
        assert base["levels"][level]["cluster_id"] == other["levels"][level]["cluster_id"]
    assert base["levels"]["l0u_undirected_family"]["cluster_id"] == (
        other["levels"]["l0u_undirected_family"]["cluster_id"]
    )


def test_component_reordering_keeps_cluster_ids() -> None:
    document = _ethane_v2_document()
    document["graph"]["r_components"] = {m: 0 for m in ETHANE_ELEMENTS}
    document["graph"]["p_components"] = {m: 0 for m in ETHANE_ELEMENTS}
    base = classify_v2_reaction(document)
    assert base is not None
    reordered = copy.deepcopy(document)
    graph = reordered["graph"]
    r_bonds, p_bonds = graph["r_bonds"], graph["p_bonds"]
    # Split into two components at the changing bond's endpoints: reorder by
    # assigning atoms {1,3,4,5} to component 1 and {2,6,7,8} to component 0
    # (ordinals swapped relative to first-appearance order).
    new_r = {m: (0 if m in (2, 6, 7, 8) else 1) for m in ETHANE_ELEMENTS}
    new_p = dict(new_r)
    graph["r_components"] = new_r
    graph["p_components"] = new_p
    other = classify_v2_reaction(reordered)
    assert other is not None
    for level in ("l0_edit_family", "l0u_undirected_family",
                  "l1_center_template", "l2_context_r1", "l3_context_r2"):
        assert base["levels"][level]["cluster_id"] == other["levels"][level]["cluster_id"]
    assert base["family_labels"] == other["family_labels"]


def test_budget_fallback_is_relabeling_invariant() -> None:
    symmetric = _TemplateGraph(
        {1: ("C", ("F",)), 2: ("C", ()), 3: ("O", ("F",)), 4: ("O", ("F",))},
        [(1, 2, None, 1.0), (2, 3, None, 1.0), (2, 4, None, 1.0)],
    )
    relabeled = _TemplateGraph(
        {10: ("C", ("F",)), 20: ("C", ()), 30: ("O", ("F",)), 40: ("O", ("F",))},
        [(10, 20, None, 1.0), (20, 30, None, 1.0), (20, 40, None, 1.0)],
    )
    first, first_complete = _canonical_serialization(symmetric, budget=1)
    second, second_complete = _canonical_serialization(relabeled, budget=1)
    assert first_complete is False
    assert second_complete is False
    assert first == second
    document = _ethane_v2_document()
    flagged = classify_v2_reaction(document, canonical_budget=1)
    assert flagged is not None
    reflagged = classify_v2_reaction(document, canonical_budget=1)
    assert flagged == reflagged


def test_family_labels_use_exclusive_counts() -> None:
    sn2_graph = {
        "elements": {1: "C", 2: "Br", 3: "O"},
        "r_bonds": [[1, 2, 1.0]],
        "p_bonds": [[1, 3, 1.0]],
        "r_components": {1: 0, 2: 1},
        "p_components": {1: 0, 3: 1},
    }
    sn2_edits = [
        {"pair": [1, 2], "edit_kind": EDIT_BROKEN},
        {"pair": [1, 3], "edit_kind": EDIT_FORMED},
    ]
    assert family_labels(sn2_edits, [], sn2_graph) == ["substitution"]

    order_only_graph = {
        "elements": {1: "C", 2: "C"},
        "r_bonds": [[1, 2, 1.0]],
        "p_bonds": [[1, 2, 2.0]],
        "r_components": {1: 0, 2: 0},
        "p_components": {1: 0, 2: 0},
    }
    assert family_labels(
        [{"pair": [1, 2], "edit_kind": EDIT_ORDER_CHANGED}], [], order_only_graph,
    ) == ["other"]

    h2_graph = {
        "elements": {1: "H", 2: "H"},
        "r_bonds": [],
        "p_bonds": [[1, 2, 1.0]],
        "r_components": {1: 0, 2: 1},
        "p_components": {1: 0, 2: 0},
    }
    labels = family_labels(
        [{"pair": [1, 2], "edit_kind": EDIT_FORMED}],
        [{"h": 1, "kind": "hh_form_free"}, {"h": 2, "kind": "hh_form_free"}],
        h2_graph,
    )
    assert labels == ["addition"]


def _inventory_row() -> dict:
    return {
        "reaction_id": "RXN_TEST",
        "dataset_version": "zenodo-test",
        "charge_total_reactants": 0,
        "charge_total_products": 0,
        "multiplicity_max": 1,
        "components": [
            {"tag": "R0", "atomic_numbers": [6, 6, 1, 1, 1, 1, 1, 1],
             "coordinates": [[float(i), 0.0, 0.0] for i in range(8)],
             "charge": 0, "multiplicity": 1},
            {"tag": "P0", "atomic_numbers": [6, 6, 1, 1, 1, 1, 1, 1],
             "coordinates": [[0.5 + i, 0.0, 0.0] for i in range(8)],
             "charge": 0, "multiplicity": 1},
        ],
    }


def _p1_document() -> dict:
    elements = ETHANE_ELEMENTS
    r_bonds = [[1, 2, 1.0], [1, 3, 1.0], [1, 4, 1.0], [1, 5, 1.0],
               [2, 6, 1.0], [2, 7, 1.0], [2, 8, 1.0]]
    p_bonds = [[1, 2, 2.0], [1, 3, 1.0], [1, 4, 1.0], [1, 5, 1.0],
               [2, 6, 1.0], [2, 7, 1.0], [2, 8, 1.0]]
    legacy = legacy_events_from_graph(elements, r_bonds, p_bonds)
    return {
        "schema_version": "g1_p1_truth_v1",
        "reaction_id": "RXN_TEST",
        "status": "resolved_unique",
        "mapping": {
            "algorithm": "ts_map_order_identity_v1",
            "map_to_atoms": [
                {"map": m, "element": elements[m],
                 "r_component": "R0", "r_geometry_tag": "R0", "r_local_index": m - 1,
                 "p_component": "P0", "p_geometry_tag": "P0", "p_local_index": m - 1,
                 "ts_irc_index": m - 1}
                for m in sorted(elements)
            ],
            "symmetry_collapsed": False,
        },
        "bond_events": legacy,
        "graph": {
            "elements": elements,
            "r_bonds": r_bonds,
            "p_bonds": p_bonds,
            "r_components": {m: 0 for m in elements},
            "p_components": {m: 0 for m in elements},
        },
        "irc_validation": {
            "orientation": "R_first", "endpoint_match": "pass",
            "n_mismatch": 0, "n_weak": 0, "n_support": 1, "quality_flags": [],
        },
    }


SETTINGS = {"shard_size": 1000, "collapse_audit_budget": 16, "center_shell": 1}


def test_build_v2_reaction_clean_and_audited() -> None:
    document = build_v2_reaction(
        _inventory_row(), _p1_document(), None, {"index_status": "unique"}, SETTINGS,
    )
    assert document["audit_status"] == "clean"
    assert document["issues"] == []
    assert document["mapping_audit"] == {
        "bijection_ok": True, "elements_ok": True, "component_assignment_ok": True,
    }
    assert document["legacy_reconciled"] is True
    assert document["collapse_state"] == "not_collapsed"
    assert document["edits"][0]["edit_kind"] == EDIT_ORDER_CHANGED


def test_build_v2_reaction_flags_irc_mismatch() -> None:
    p1 = _p1_document()
    p1["irc_validation"]["n_mismatch"] = 2
    document = build_v2_reaction(
        _inventory_row(), p1, None, {"index_status": "unique"}, SETTINGS,
    )
    assert document["audit_status"] == "issues"
    assert "irc_event_mismatch" in document["issues"]
    assert "irc_evidence_quality" in document["dimensions"]


def test_build_v2_reaction_flags_element_break() -> None:
    p1 = _p1_document()
    p1["mapping"]["map_to_atoms"][0]["element"] = "Cl"
    document = build_v2_reaction(
        _inventory_row(), p1, None, None, SETTINGS,
    )
    assert "element_sequence_mismatch" in document["issues"]
    assert document["mapping_audit"]["elements_ok"] is False


def test_build_v2_reaction_flags_legacy_drift() -> None:
    p1 = _p1_document()
    p1["bond_events"]["formed"] = []
    document = build_v2_reaction(
        _inventory_row(), p1, None, None, SETTINGS,
    )
    assert "legacy_event_inconsistency" in document["issues"]


def test_export_document_is_whitelisted() -> None:
    v2_document = build_v2_reaction(
        _inventory_row(), _p1_document(), None, {"index_status": "unique"}, SETTINGS,
    )
    classification = classify_v2_reaction(v2_document)
    assert classification is not None
    class_document = {
        "levels": classification["levels"],
        "family_labels": classification["family_labels"],
    }
    export = build_export_document(
        v2_document, _p1_document(), _inventory_row(), class_document,
    )
    assert export["mapping_provenance"] == "truth_assisted_p1"
    assert export["maps"] == list(range(1, 9))
    assert export["r_atomic_numbers"] == [6, 6, 1, 1, 1, 1, 1, 1]
    assert export["p_atomic_numbers"] == export["r_atomic_numbers"]
    assert len(export["r_coordinates"]) == 8 and len(export["r_coordinates"][0]) == 3
    assert export["r_coordinates"][3][0] == 3.0
    assert export["classification"]["l0_cluster_id"].startswith("g1p2_taxonomy_v2:")
    assert "ts_irc_index" not in __import__("json").dumps(export)
    assert "endpoint_match" not in __import__("json").dumps(export)
    assert "orientation" not in __import__("json").dumps(export)
