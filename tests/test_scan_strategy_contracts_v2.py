from __future__ import annotations

import copy
import math

import pytest

from pes2ts_core.contracts import FORBIDDEN_TRUTH_KEYS, ContractError
from pes2ts_core.g1.v2_verify import FORBIDDEN_EXPORT_KEYS
from pes2ts_core.generation.planning.contracts_v2 import (
    CANDIDATE_KIND_PATH,
    CANDIDATE_KIND_SCAN,
    FORBIDDEN_KEYS,
    dumps_v2_document,
    loads_v2_document,
    make_backend_capability,
    make_generation_plan,
    make_strategy_proposal,
    seal_document,
    validate_v2_document,
)

HEX_A = "a" * 64
HEX_B = "b" * 64
HEX_C = "c" * 64
HEX_D = "d" * 64
HEX_E = "e" * 64
HEX_F = "f" * 64


def _driver(kind: str = "B", maps: tuple[int, ...] = (2, 7), **extra):
    return {"kind": kind, "maps": list(maps), "unit": "angstrom", **extra}


def _scan_candidate(candidate_id: str = "plan:c001", mode: str = "SINGLE_1D", drivers=None, **extra):
    if drivers is None:
        drivers = [_driver()]
    candidate = {
        "candidate_kind": CANDIDATE_KIND_SCAN,
        "candidate_id": candidate_id,
        "mode": mode,
        "start_endpoint": "R",
        "direction": "R_to_P",
        "assembly_id": "asm-1",
        "anchor_reason": "LOCAL_CONNECTIVITY",
        "drivers": drivers,
        "monitors": [{"measurement": "O4-H13_distance", "target_test": "monotonic_decrease",
                      "evidence_available": True, "maps": [4, 13]}],
        "guards": [],
        "event_coverage": [{"event_id": "H1", "coverage": "direct"}],
        "lambda_values": [0.0, 0.5, 1.0],
        "schedule_id": "sched-linear-3",
        "required_capabilities": ["SINGLE_1D"],
        "budget": {"max_attempts": 1, "max_cpu_hours": 8.0, "max_wall_seconds": 14400},
        "failure_reasons": [],
    }
    candidate.update(extra)
    return candidate


def _proposal_candidate(candidate_id: str = "prop:c001", **extra):
    candidate = {
        "candidate_id": candidate_id,
        "mode": "SINGLE_1D",
        "start_endpoint": "R",
        "direction": "R_to_P",
        "assembly_id": "asm-1",
        "anchor_reason": "LOCAL_CONNECTIVITY",
        "drivers": [_driver()],
        "monitors": [],
        "guards": [],
        "event_coverage": [],
        "lambda_values": [0.0, 1.0],
        "schedule_id": "sched-linear-2",
        "required_capabilities": ["SINGLE_1D"],
        "capability_check": {"status": "pass", "missing": []},
        "budget": {"max_attempts": 1, "max_cpu_hours": 8.0, "max_wall_seconds": 14400},
        "failure_reasons": [],
    }
    candidate.update(extra)
    return candidate


def _path_candidate(candidate_id: str = "plan:p001", **extra):
    candidate = {
        "candidate_kind": CANDIDATE_KIND_PATH,
        "candidate_id": candidate_id,
        "n_atoms": 3,
        "endpoint_geometries": {
            "reactant": [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [3.5, 0.0, 0.0]],
            "product": [[0.0, 0.0, 0.0], [1.8, 0.0, 0.0], [2.8, 0.0, 0.0]],
        },
        "image_chain": {"n_images": 8},
        "start_endpoint": "R",
        "direction": "R_to_P",
        "anchor_reason": "CONNECTIVITY_EXCHANGE",
        "failure_reasons": [],
    }
    candidate.update(extra)
    return candidate


def _proposal_fields(**extra):
    fields = {
        "reaction_id": "RXN_SYNTH_0001",
        "case_id": "case:rxn-synth-0001",
        "split": "train",
        "source_case_sha256": HEX_A,
        "graph_input": {
            "endpoint_graph_sha256": HEX_B,
            "normalization_version": "norm-v1",
            "mapping_equivalence": {"status": "unique", "n_candidates": 1},
        },
        "graph_features": {
            "edit_components": [{"component_id": 0, "n_edits": 2}],
            "context_support": {},
            "events": [{"event_id": "H1", "kind": "H_TRANSFER"}],
            "typed_couplings": [],
        },
        "family": "h_transfer",
        "motif_tags": ["single_h"],
        "rule_trace": [{"rule_id": "ANCHOR_LOCAL_CONNECTIVITY", "evidence_level": "graph"}],
        "epistemic_status": "endpoint_hypothesis",
        "candidates": [_proposal_candidate()],
        "reasons": [{"code": "PROPOSED_FROM_ENDPOINT_HYPOTHESIS", "detail": "graph selector P0"}],
        "blocking_reasons": [],
    }
    fields.update(extra)
    return fields


def _plan_fields(**extra):
    fields = {
        "reaction_id": "RXN_SYNTH_0001",
        "case_id": "case:rxn-synth-0001",
        "split": "train",
        "plan_id": "plan:synth-0001",
        "plan_version": 1,
        "source_proposal_sha256": HEX_C,
        "source_case_sha256": HEX_A,
        "policy_hashes": {"rule_registry": HEX_D, "scan_strategy_policy": HEX_E},
        "backend": {
            "engine": "orca",
            "adapter_version": "0.1.0",
            "method": "B3LYP",
            "basis": "def2-TZVP",
            "resources": {"nproc": 4, "mem_gb": 8},
            "electronic_state": {"charge": 0, "multiplicity": 1},
        },
        "candidate_graph": {
            "nodes": [
                {"node_id": "n1", "state": "proposed", "terminal": False},
                {"node_id": "n2", "state": "frozen", "terminal": True},
            ],
            "edges": [{"from": "n1", "to": "n2"}],
        },
        "candidates": [_scan_candidate(), _path_candidate()],
        "budget": {"max_attempts": 2, "max_cpu_hours": 16.0, "max_wall_seconds": 28800},
        "compiled": {"kind": "hashes", "entries": [{"point_index": 0, "input_sha256": HEX_F}]},
        "quality_tests": [{"test_id": "endpoint_reach", "kind": "rmsd", "tolerance": 0.5}],
    }
    fields.update(extra)
    return fields


def _capability_fields(**extra):
    fields = {
        "engine_version": "6.1.0",
        "adapter_version": "0.1.0",
        "supported_modes": ["SINGLE_1D", "COUPLED_1D", "SCHEDULED_1D", "PATH_NEB"],
        "coordinate_kinds": ["B", "A", "D"],
        "max_scan_coordinates": 3,
        "point_limits": {"baseline": 9, "max": 101},
        "custom_schedule_support": True,
        "constraint_support": {"native_scan": True, "per_point_constraints": False, "simul_scan": True},
        "method_element_coverage": {"B3LYP": ["C", "H", "N", "O", "F", "Cl"]},
        "probe_receipts": [{"probe_id": "probe-1", "workflow": "PESsearch", "status": "completed",
                            "receipt_sha256": HEX_F}],
    }
    fields.update(extra)
    return fields


def test_strategy_proposal_make_seal_loads_roundtrip():
    doc = make_strategy_proposal("prop:synth-0001", "proposed", **_proposal_fields())
    restored = loads_v2_document(dumps_v2_document(doc))
    assert restored["content_sha256"] == doc["content_sha256"]
    assert restored["schema_version"] == "g1_strategy_proposal_v1"
    assert restored["candidates"] == doc["candidates"]
    assert restored["execution_eligible"] is False
    assert seal_document(doc)["content_sha256"] == doc["content_sha256"]


def test_generation_plan_roundtrip_with_scan_and_path_candidates():
    doc = make_generation_plan("plan:synth-0001", "frozen", **_plan_fields())
    restored = loads_v2_document(dumps_v2_document(doc))
    assert restored["content_sha256"] == doc["content_sha256"]
    kinds = [candidate["candidate_kind"] for candidate in restored["candidates"]]
    assert kinds == [CANDIDATE_KIND_SCAN, CANDIDATE_KIND_PATH]
    scan = restored["candidates"][0]
    assert scan["drivers"][0]["maps"] == [2, 7]
    path = restored["candidates"][1]
    assert path["image_chain"]["n_images"] == 8
    assert len(path["endpoint_geometries"]["reactant"]) == path["n_atoms"]


def test_backend_capability_make_seal_loads_roundtrip():
    doc = make_backend_capability("cap:orca-baseline", "active", **_capability_fields())
    restored = loads_v2_document(dumps_v2_document(doc))
    assert restored["content_sha256"] == doc["content_sha256"]
    assert restored["schema_version"] == "orca_capabilities_v1"
    assert restored["supported_modes"] == ["SINGLE_1D", "COUPLED_1D", "SCHEDULED_1D", "PATH_NEB"]
    assert restored["point_limits"] == {"baseline": 9, "max": 101}


def test_content_sha256_stable_for_identical_input_despite_volatile_created_at():
    first = make_strategy_proposal("prop:synth-0001", "proposed", **_proposal_fields())
    second = make_strategy_proposal("prop:synth-0001", "proposed", **_proposal_fields())
    assert first["content_sha256"] == second["content_sha256"]
    plan_first = make_generation_plan("plan:synth-0001", "frozen", **_plan_fields())
    plan_second = make_generation_plan("plan:synth-0001", "frozen", **_plan_fields())
    assert plan_first["content_sha256"] == plan_second["content_sha256"]


def test_content_sha256_changes_when_a_frozen_field_mutates():
    baseline = make_generation_plan("plan:synth-0001", "frozen", **_plan_fields())
    mutated = copy.deepcopy(baseline)
    mutated["candidates"][0]["drivers"][0]["maps"] = [2, 12]
    mutated = seal_document(mutated)
    assert mutated["content_sha256"] != baseline["content_sha256"]
    proposal = make_strategy_proposal("prop:synth-0001", "proposed", **_proposal_fields())
    proposal_mutated = copy.deepcopy(proposal)
    proposal_mutated["candidates"][0]["drivers"][0]["maps"] = [2, 12]
    proposal_mutated = seal_document(proposal_mutated)
    assert proposal_mutated["content_sha256"] != proposal["content_sha256"]


def test_content_sha256_unchanged_when_created_at_mutates():
    doc = make_strategy_proposal("prop:synth-0001", "proposed", **_proposal_fields())
    retimed = dict(doc)
    retimed["created_at"] = "2020-01-01T00:00:00+00:00"
    assert seal_document(retimed)["content_sha256"] == doc["content_sha256"]


@pytest.mark.parametrize("key", ["orientation", "irc_evidence", "endpoint_match"])
def test_validate_rejects_forbidden_truth_keys_at_any_depth(key):
    doc = make_strategy_proposal("prop:synth-0001", "proposed", **_proposal_fields())
    top = copy.deepcopy(doc)
    top[key] = "R_to_P"
    assert any("forbidden" in issue for issue in validate_v2_document(top))
    nested = copy.deepcopy(doc)
    nested["graph_features"]["events"][0][key] = {}
    assert any("forbidden" in issue for issue in validate_v2_document(nested))
    plan = make_generation_plan("plan:synth-0001", "frozen", **_plan_fields())
    in_candidate = copy.deepcopy(plan)
    in_candidate["candidates"][0]["failure_reasons"] = [{"code": "X", key: "leak"}]
    assert any("forbidden" in issue for issue in validate_v2_document(in_candidate))


def test_validate_rejects_forbidden_export_keys():
    doc = make_generation_plan("plan:synth-0001", "frozen", **_plan_fields())
    leaked = copy.deepcopy(doc)
    leaked["compiled"] = {"kind": "hashes", "entries": [
        {"point_index": 0, "input_sha256": HEX_F, "ts_irc_index": 0}]}
    issues = validate_v2_document(leaked)
    assert any("forbidden" in issue for issue in issues)


def test_forbidden_keys_are_a_single_source_union():
    assert set(FORBIDDEN_TRUTH_KEYS) <= set(FORBIDDEN_KEYS)
    lowered_export = {key.lower() for key in FORBIDDEN_EXPORT_KEYS}
    assert lowered_export <= set(FORBIDDEN_KEYS)
    for key in ("orientation", "endpoint_match", "irc_evidence", "ts_irc_index"):
        assert key in FORBIDDEN_KEYS


def test_proposal_execution_eligible_is_always_false():
    doc = make_strategy_proposal("prop:synth-0001", "proposed", **_proposal_fields())
    assert doc["execution_eligible"] is False
    with pytest.raises(ContractError, match="execution_eligible"):
        make_strategy_proposal("prop:synth-0001", "proposed",
                               **_proposal_fields(execution_eligible=True))
    forged = copy.deepcopy(doc)
    forged["execution_eligible"] = True
    assert any("execution_eligible" in issue for issue in validate_v2_document(forged))


def test_proposal_allows_needs_review_status_with_reasons():
    doc = make_strategy_proposal(
        "prop:synth-0001", "needs_review",
        **_proposal_fields(reasons=[{"code": "INPUT_NEEDS_REVIEW", "detail": "demo cohort"}]),
    )
    assert doc["status"] == "needs_review"
    assert doc["execution_eligible"] is False
    restored = loads_v2_document(dumps_v2_document(doc))
    assert restored["reasons"][0]["code"] == "INPUT_NEEDS_REVIEW"
    rejected = make_strategy_proposal(
        "prop:synth-0002", "rejected",
        **_proposal_fields(blocking_reasons=[{"code": "MAP_AMBIGUOUS", "detail": "symmetric maps"}]),
    )
    assert rejected["blocking_reasons"][0]["code"] == "MAP_AMBIGUOUS"
    with pytest.raises(ContractError, match="blocking_reasons"):
        make_strategy_proposal("prop:synth-0003", "rejected",
                               **_proposal_fields(blocking_reasons=[]))


def test_discriminated_union_rejects_wrong_kind_payloads():
    plan = make_generation_plan("plan:synth-0001", "frozen", **_plan_fields())
    bad_kind = copy.deepcopy(plan)
    bad_kind["candidates"][0]["candidate_kind"] = "scan"
    issues = validate_v2_document(bad_kind)
    assert any("candidate_kind" in issue for issue in issues)
    path_marked_scan = copy.deepcopy(plan)
    path_marked_scan["candidates"] = [_path_candidate(candidate_kind=CANDIDATE_KIND_SCAN)]
    issues = validate_v2_document(path_marked_scan)
    assert any("path-only" in issue or "unknown field" in issue or "required field missing" in issue
               for issue in issues)
    scan_marked_path = copy.deepcopy(plan)
    scan_marked_path["candidates"] = [_scan_candidate(candidate_kind=CANDIDATE_KIND_PATH)]
    issues = validate_v2_document(scan_marked_path)
    assert any("scan-only" in issue or "unknown field" in issue or "required field missing" in issue
               for issue in issues)


def test_scan_candidate_rejects_wrong_driver_arity_for_mode():
    coupled_ok = _scan_candidate(mode="COUPLED_1D", drivers=[_driver(maps=(2, 7)), _driver(maps=(7, 12))])
    doc = make_generation_plan("plan:synth-0001", "frozen",
                               **_plan_fields(candidates=[coupled_ok, _path_candidate()]))
    loads_v2_document(dumps_v2_document(doc))
    single_with_two = _scan_candidate(mode="SINGLE_1D",
                                      drivers=[_driver(maps=(2, 7)), _driver(maps=(7, 12))])
    bad = copy.deepcopy(doc)
    bad["candidates"][0] = single_with_two
    issues = validate_v2_document(bad)
    assert any("SINGLE_1D requires exactly one driver" in issue for issue in issues)
    coupled_with_one = _scan_candidate(mode="COUPLED_1D", drivers=[_driver()])
    bad["candidates"][0] = coupled_with_one
    issues = validate_v2_document(bad)
    assert any("COUPLED_1D requires 2..3 drivers" in issue for issue in issues)
    four = [ _driver(maps=(i, i + 1)) for i in range(1, 5)]
    bad["candidates"][0] = _scan_candidate(mode="COUPLED_1D", drivers=four)
    issues = validate_v2_document(bad)
    assert any("2..3 drivers" in issue for issue in issues)


def test_direction_must_start_from_start_endpoint():
    plan = make_generation_plan("plan:synth-0001", "frozen", **_plan_fields())
    bad = copy.deepcopy(plan)
    bad["candidates"][0]["start_endpoint"] = "P"
    bad["candidates"][0]["direction"] = "R_to_P"
    issues = validate_v2_document(bad)
    assert any("must start from start_endpoint" in issue for issue in issues)


def test_path_candidate_requires_geometry_rows_to_match_n_atoms():
    plan = make_generation_plan("plan:synth-0001", "frozen", **_plan_fields())
    bad = copy.deepcopy(plan)
    bad["candidates"][1]["endpoint_geometries"]["product"] = [[0.0, 0.0, 0.0], [1.8, 0.0, 0.0]]
    issues = validate_v2_document(bad)
    assert any("n_atoms=3" in issue for issue in issues)


def test_plan_requires_non_empty_candidates_and_valid_graph():
    with pytest.raises(ContractError, match="candidates"):
        make_generation_plan("plan:synth-0001", "frozen", **_plan_fields(candidates=[]))
    bad_graph = copy.deepcopy(_plan_fields())
    bad_graph["candidate_graph"] = {"nodes": [{"node_id": "n1", "state": "frozen", "terminal": True}],
                                    "edges": [{"from": "n1", "to": "ghost"}]}
    with pytest.raises(ContractError, match="unknown node"):
        make_generation_plan("plan:synth-0001", "frozen", **bad_graph)


def test_validate_rejects_nonfinite_numbers_and_unknown_fields():
    plan = make_generation_plan("plan:synth-0001", "frozen", **_plan_fields())
    nonfinite = copy.deepcopy(plan)
    nonfinite["candidates"][0]["lambda_values"] = [0.0, math.nan, 1.0]
    issues = validate_v2_document(nonfinite)
    assert any("non-finite" in issue for issue in issues)
    unknown = copy.deepcopy(plan)
    unknown["unexpected_field"] = 1
    issues = validate_v2_document(unknown)
    assert any("unknown field" in issue for issue in issues)
    bad_candidate = copy.deepcopy(plan)
    bad_candidate["candidates"][0]["surprise"] = True
    issues = validate_v2_document(bad_candidate)
    assert any("candidates[0].surprise" in issue for issue in issues)


def test_loads_rejects_tampered_content_sha256():
    doc = make_backend_capability("cap:orca-baseline", "active", **_capability_fields())
    payload = dumps_v2_document(doc)
    tampered = payload.replace(doc["content_sha256"], "0" * 64)
    with pytest.raises(ContractError, match="digest mismatch"):
        loads_v2_document(tampered)


def test_capability_rejects_unknown_modes_and_inverted_point_limits():
    bad_modes = _capability_fields(supported_modes=["SINGLE_1D", "GRID_2D"])
    with pytest.raises(ContractError, match="supported_modes"):
        make_backend_capability("cap:orca-baseline", "active", **bad_modes)
    bad_limits = _capability_fields(point_limits={"baseline": 101, "max": 9})
    with pytest.raises(ContractError, match="point_limits"):
        make_backend_capability("cap:orca-baseline", "active", **bad_limits)
