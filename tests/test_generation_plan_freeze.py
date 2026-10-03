"""Tests for pes2ts_core.generation.planning.plan_freeze (plan task 23).

Covers: freeze producing a contract-valid ``g1_generation_plan_v2``;
``content_sha256`` covering every execution-behavior dimension (rule
versions, complete graph hash, atom order, assembly, schedule, direction,
method, budget, quality tests, fallback tree); the closed failure-tree
vocabulary with computable predicates; budget exhaustion preserving the
spent/max denominator and obtained paths; ``verify_generation_plan`` catching
tampering; the contracts_v2 ``supersedes`` validation fix; and the CLI
``v2-scan-freeze`` gate-pass path writing plans + manifest while the
refusal path (including needs_review proposals) stays unchanged.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml
from pes2ts_core.cli import EXIT_G1_BUILD_FAILED, main
from pes2ts_core.config_loader import load_config
from pes2ts_core.contracts import seal_document
from pes2ts_core.generation.planning.contracts_v2 import (
    make_generation_plan,
    make_strategy_proposal,
    validate_v2_document,
)
from pes2ts_core.generation.planning.plan_freeze import (
    BUDGET_ACCOUNTING_CATEGORIES,
    FAILURE_PREDICATES,
    FAILURE_TREE_CODES,
    BudgetLedger,
    FailureState,
    PlanFreezeError,
    active_failure_codes,
    assert_failure_code,
    compiled_binding_from_candidate,
    freeze_generation_plan,
    freeze_path_candidate_plan,
    is_failure_code,
    verify_generation_plan,
)
from pes2ts_core.utils.hashing import sha256_file
from pes2ts_core.utils.jsonio import read_json, write_json
from pes2ts_core.utils.parquet_io import write_parquet

# ---------------------------------------------------------------------------
# Synthetic proposal fixtures (contracts_v2 shape, capability pass).
# ---------------------------------------------------------------------------
SHA_B = "b" * 64


def _driver() -> dict[str, Any]:
    return {"kind": "B", "maps": [1, 2], "unit": "angstrom"}


def _candidate(**overrides: Any) -> dict[str, Any]:
    candidate: dict[str, Any] = {
        "candidate_id": "cand-0000",
        "mode": "SINGLE_1D",
        "start_endpoint": "R",
        "direction": "R_to_P",
        "assembly_id": "asm-r-c1",
        "anchor_reason": "layered_direction_comparison_v1",
        "drivers": [_driver()],
        "monitors": [],
        "guards": [],
        "event_coverage": [{"event_id": "evt-0001", "coverage": "direct"}],
        "lambda_values": [0.0, 0.5, 1.0],
        "schedule_id": "sched-linear-000",
        "schedule_kind": "linear",
        "required_capabilities": ["SINGLE_1D"],
        "capability_check": {"status": "pass", "missing": []},
        "budget": {"max_attempts": 1, "max_cpu_hours": 0.0, "max_wall_seconds": 0},
        "failure_reasons": [],
        "fallback_ids": ["LOCAL_CONNECTIVITY"],
    }
    candidate.update(overrides)
    return candidate


def _proposal(candidate: dict[str, Any] | None = None, **overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "reaction_id": "RXN_0000000001",
        "case_id": "case-0001",
        "split": "train",
        "source_case_sha256": "a" * 64,
        "graph_input": {
            "endpoint_graph_sha256": SHA_B,
            "normalization_version": "rdkit_default_unkekulized",
            "mapping_equivalence": {
                "map_ids": [1, 2],
                "element_conserved": True,
                "isotope_conserved": True,
                "basis": "synthetic_fixture",
            },
        },
        "graph_features": {
            "edit_components": [[1, 2]],
            "context_support": {"context_radius": 2, "edits": []},
            "events": [{"event_id": "evt-0001", "event_type": "CONNECTIVITY_EXCHANGE"}],
            "typed_couplings": {
                "strong_components": [[1, 2]],
                "weak_links": [],
                "hydrogen_events": [],
            },
        },
        "family": "LOCAL_CONNECTIVITY",
        "motif_tags": ["R_EVENT_FORMED"],
        "rule_trace": [{"rule_id": "R_SELECTOR_PIPELINE_V1"}],
        "epistemic_status": "endpoint_hypothesis",
        "candidates": [candidate if candidate is not None else _candidate()],
        "reasons": [{"code": "PROPOSED_FROM_ENDPOINT_HYPOTHESIS", "detail": ""}],
        "blocking_reasons": [],
    }
    fields.update(overrides)
    return make_strategy_proposal(fields["reaction_id"], str(fields.pop("status", "proposed")), **fields)


def _config() -> dict[str, Any]:
    return {
        "scan_strategy": {
            "freeze": {
                "max_attempts": 2,
                "max_cpu_hours": 4.0,
                "max_wall_seconds": 3600,
            },
        },
        "g1_v2": {"shard_size": 1000},
    }


def _frozen_plan(**freeze_kwargs: Any) -> dict[str, Any]:
    proposal = _proposal()
    candidate = proposal["candidates"][0]
    compiled = {
        "kind": "recipe",
        "recipe": {
            "recipe_kind": "pes2ts_candidate_recipe_v1",
            "engine": "orca",
            "method": "B3LYP-D3",
        },
        "atom_rows": [1, 2],
        "candidate_id": "cand-0000",
    }
    return freeze_generation_plan(proposal, candidate, compiled, _config(), **freeze_kwargs)


# ---------------------------------------------------------------------------
# Freeze validity and version-bump semantics.
# ---------------------------------------------------------------------------
def test_freeze_produces_valid_document() -> None:
    plan = _frozen_plan()
    assert validate_v2_document(plan) == []
    assert plan["schema_version"] == "g1_generation_plan_v2"
    assert plan["status"] == "frozen"
    assert plan["plan_id"] == "RXN_0000000001:gp-v1"
    assert plan["candidates"][0]["candidate_kind"] == "ScanCandidateV2"
    assert plan["candidates"][0]["failure_reasons"] == []
    assert verify_generation_plan(plan) == []


def test_freeze_is_deterministic_over_same_inputs() -> None:
    first = _frozen_plan()
    second = _frozen_plan()
    assert first["content_sha256"] == second["content_sha256"]


def test_superseding_freeze_carries_new_version_and_reference() -> None:
    plan = _frozen_plan(plan_version=2, supersedes="RXN_0000000001:gp-v1")
    assert plan["plan_version"] == 2
    assert plan["supersedes"] == "RXN_0000000001:gp-v1"
    assert plan["plan_id"] == "RXN_0000000001:gp-v2"
    assert verify_generation_plan(plan) == []


# ---------------------------------------------------------------------------
# Hash coverage over every execution-behavior dimension.
# ---------------------------------------------------------------------------
def _mutate_rule_version(doc: dict[str, Any]) -> None:
    doc["policy_hashes"]["registry_version"] = "strategy_registry_v2"


def _mutate_graph_hash(doc: dict[str, Any]) -> None:
    doc["extensions"]["freeze"]["graph_sha256"] = "0" * 64


def _mutate_atom_order(doc: dict[str, Any]) -> None:
    doc["extensions"]["freeze"]["atom_order"] = [2, 1]


def _mutate_assembly(doc: dict[str, Any]) -> None:
    doc["candidates"][0]["assembly_id"] = "asm-other"


def _mutate_schedule(doc: dict[str, Any]) -> None:
    doc["candidates"][0]["schedule_id"] = "sched-other"
    doc["candidates"][0]["lambda_values"] = [0.0, 1.0]


def _mutate_direction(doc: dict[str, Any]) -> None:
    doc["candidates"][0]["direction"] = "P_to_R"
    doc["candidates"][0]["start_endpoint"] = "P"


def _mutate_method(doc: dict[str, Any]) -> None:
    doc["backend"]["method"] = "HF"


def _mutate_budget(doc: dict[str, Any]) -> None:
    doc["budget"]["max_attempts"] = 3


def _mutate_quality_tests(doc: dict[str, Any]) -> None:
    doc["quality_tests"][0]["test_id"] = "mutated_quality_test"


def _mutate_fallback_tree(doc: dict[str, Any]) -> None:
    doc["candidates"][0]["fallback_ids"] = ["H_TRANSFER"]
    doc["extensions"]["freeze"]["fallback_tree"]["cand-0000"] = ["H_TRANSFER"]


@pytest.mark.parametrize(
    ("dimension", "mutate"),
    [
        ("rule_version", _mutate_rule_version),
        ("graph_hash", _mutate_graph_hash),
        ("atom_order", _mutate_atom_order),
        ("assembly", _mutate_assembly),
        ("schedule", _mutate_schedule),
        ("direction", _mutate_direction),
        ("method", _mutate_method),
        ("budget", _mutate_budget),
        ("quality_tests", _mutate_quality_tests),
        ("fallback_tree", _mutate_fallback_tree),
    ],
)
def test_plan_hash_covers_all_execution_dimensions(
    dimension: str, mutate: Any
) -> None:
    plan = _frozen_plan()
    original = str(plan["content_sha256"])
    mutated = copy.deepcopy(plan)
    mutate(mutated)
    assert seal_document(mutated)["content_sha256"] != original, dimension


# ---------------------------------------------------------------------------
# Failure tree: closed vocabulary + computable predicates.
# ---------------------------------------------------------------------------
def test_failure_tree_predicates_match_vocabulary() -> None:
    assert set(FAILURE_PREDICATES) == FAILURE_TREE_CODES


def test_failure_tree_codes_are_computable() -> None:
    assert active_failure_codes(FailureState()) == ()
    assert active_failure_codes(FailureState(mapping_invalid=True)) == ("MAP_INVALID",)
    assert active_failure_codes(FailureState(gate_pass=False)) == ("GATE_REFUSED",)
    ledger = BudgetLedger(max_attempts=1).record(attempts=1)
    assert active_failure_codes(FailureState(budget=ledger)) == ("BUDGET_EXHAUSTED",)
    multi = FailureState(opt_failed=True, constraint_residual=True)
    assert active_failure_codes(multi) == ("OPT_FAILED", "CONSTRAINT_RESIDUAL")
    assert active_failure_codes(
        FailureState(wrong_connectivity=True, path_discontinuity=True)
    ) == ("WRONG_CONNECTIVITY", "PATH_DISCONTINUITY")


def test_unknown_failure_codes_are_rejected() -> None:
    assert is_failure_code("BUDGET_EXHAUSTED") is True
    assert is_failure_code("效果不好") is False
    with pytest.raises(PlanFreezeError) as excinfo:
        assert_failure_code("效果不好")
    assert excinfo.value.code == "FAILURE_CODE_UNKNOWN"


# ---------------------------------------------------------------------------
# Budget discipline.
# ---------------------------------------------------------------------------
def test_budget_exhaustion_preserves_denominator_and_paths() -> None:
    ledger = BudgetLedger(max_attempts=2, max_wall_seconds=100)
    assert ledger.exhausted() is False
    ledger = ledger.record(attempts=1, directions=1, schedules=1, assemblies=1)
    assert ledger.exhausted() is False
    ledger = ledger.record(attempts=1, retries=1, endpoint_preparations=1)
    assert ledger.exhausted() is True
    record = ledger.exhaustion_record(
        obtained_paths=[{"path_id": "path-1", "candidate_id": "cand-0000"}]
    )
    assert record["code"] == "BUDGET_EXHAUSTED"
    assert record["budget_spent"]["attempts"] == 2
    assert record["budget_spent"]["retries"] == 1
    assert record["budget_spent"]["endpoint_preparations"] == 1
    assert record["budget_spent"]["directions"] == 1
    assert record["budget_max"]["max_attempts"] == 2
    assert record["obtained_paths"] == [{"path_id": "path-1", "candidate_id": "cand-0000"}]


def test_budget_record_rejects_negative_increments() -> None:
    ledger = BudgetLedger(max_attempts=2)
    with pytest.raises(PlanFreezeError) as excinfo:
        ledger.record(attempts=-1)
    assert excinfo.value.code == "PLAN_BUDGET_RECORD_INVALID"


def test_frozen_plan_budget_is_stricter_of_sources() -> None:
    plan = _frozen_plan()
    assert plan["budget"]["max_attempts"] == 1
    assert plan["budget"]["max_cpu_hours"] == 4.0
    assert plan["budget"]["max_wall_seconds"] == 3600
    assert plan["budget"]["accounting"] == list(BUDGET_ACCOUNTING_CATEGORIES)


# ---------------------------------------------------------------------------
# Verify: tampering and semantic checks.
# ---------------------------------------------------------------------------
def test_verify_catches_tampering() -> None:
    plan = _frozen_plan()
    assert verify_generation_plan(plan) == []

    digest_tampered = copy.deepcopy(plan)
    digest_tampered["candidates"][0]["lambda_values"] = [0.0, 1.0]
    problems = verify_generation_plan(digest_tampered)
    assert any("content_sha256" in problem for problem in problems)

    vocabulary_tampered = copy.deepcopy(plan)
    vocabulary_tampered["candidates"][0]["failure_reasons"] = [{"code": "效果不好"}]
    vocabulary_tampered = seal_document(vocabulary_tampered)
    problems = verify_generation_plan(vocabulary_tampered)
    assert any("unknown failure-tree code" in problem for problem in problems)
    assert any("clean failure_reasons" in problem for problem in problems)

    graph_tampered = copy.deepcopy(plan)
    for node in graph_tampered["candidate_graph"]["nodes"]:
        node["terminal"] = False
    graph_tampered = seal_document(graph_tampered)
    problems = verify_generation_plan(graph_tampered)
    assert any("terminal" in problem for problem in problems)

    atom_order_tampered = copy.deepcopy(plan)
    atom_order_tampered["extensions"]["freeze"]["atom_order"] = [1, 1]
    atom_order_tampered = seal_document(atom_order_tampered)
    problems = verify_generation_plan(atom_order_tampered)
    assert any("atom_order" in problem for problem in problems)


def test_verify_rejects_non_generation_plan_documents() -> None:
    assert verify_generation_plan({"schema_name": "StrategyProposal"}) == [
        "$.schema_name: expected 'GenerationPlan'"
    ]
    assert verify_generation_plan(None) == ["$: document root must be an object"]


# ---------------------------------------------------------------------------
# contracts_v2 supersedes validation fix (todo-18 recorded defect).
# ---------------------------------------------------------------------------
def _minimal_plan_fields() -> dict[str, Any]:
    proposal = _proposal()
    candidate = proposal["candidates"][0]
    payload = copy.deepcopy(candidate)
    payload["candidate_kind"] = "ScanCandidateV2"
    payload.pop("capability_check", None)
    return {
        "reaction_id": "RXN_0000000001",
        "case_id": "case-0001",
        "split": "train",
        "plan_id": "RXN_0000000001:gp-v2",
        "plan_version": 2,
        "source_proposal_sha256": str(proposal["content_sha256"]),
        "source_case_sha256": "a" * 64,
        "policy_hashes": {"registry_version": "strategy_registry_v1"},
        "backend": {"engine": "orca", "adapter_version": "acp-adapter-v0", "method": "B3LYP-D3"},
        "candidate_graph": {
            "nodes": [{"node_id": "n1", "state": "frozen", "terminal": True}],
            "edges": [],
        },
        "candidates": [payload],
        "budget": {"max_attempts": 1, "max_cpu_hours": 0.0, "max_wall_seconds": 0},
        "compiled": {"kind": "recipe", "recipe": {"recipe_kind": "pes2ts_candidate_recipe_v1"}},
        "quality_tests": [{"test_id": "endpoint_reached"}],
    }


def test_superseded_plan_validates_with_supersedes_field() -> None:
    plan = make_generation_plan(
        "RXN_0000000001:gp-v2",
        "superseded",
        supersedes="RXN_0000000001:gp-v3",
        **_minimal_plan_fields(),
    )
    assert validate_v2_document(plan) == []


def test_superseded_plan_without_reference_is_rejected() -> None:
    with pytest.raises(Exception) as excinfo:
        make_generation_plan(
            "RXN_0000000001:gp-v2", "superseded", **_minimal_plan_fields()
        )
    assert "supersedes" in str(excinfo.value)


def test_optional_supersedes_rejects_non_string_values() -> None:
    plan = make_generation_plan("RXN_0000000001:gp-v1", "frozen", **_minimal_plan_fields() | {"plan_id": "RXN_0000000001:gp-v1", "plan_version": 1})
    tampered = copy.deepcopy(plan)
    tampered["supersedes"] = 42
    tampered = seal_document(tampered)
    problems = validate_v2_document(tampered)
    assert any("supersedes" in problem for problem in problems)


# ---------------------------------------------------------------------------
# Freeze refusal discipline.
# ---------------------------------------------------------------------------
def test_freeze_refuses_candidate_with_failure_reasons() -> None:
    proposal = _proposal(
        _candidate(failure_reasons=[{"code": "PRUNED_BY_BUDGET", "detail": "budget"}])
    )
    with pytest.raises(PlanFreezeError) as excinfo:
        freeze_generation_plan(proposal, proposal["candidates"][0], None, _config())
    assert excinfo.value.code == "CANDIDATE_NOT_CLEAN"


def test_freeze_refuses_candidate_without_capability_pass() -> None:
    proposal = _proposal(
        _candidate(capability_check={"status": "unknown", "missing": ["SINGLE_1D"]})
    )
    with pytest.raises(PlanFreezeError) as excinfo:
        freeze_generation_plan(proposal, proposal["candidates"][0], None, _config())
    assert excinfo.value.code == "CAPABILITY_NOT_PASSED"


def test_freeze_refuses_blocking_proposal() -> None:
    proposal = _proposal(blocking_reasons=[{"code": "NO_REACTION_CHANGE"}])
    with pytest.raises(PlanFreezeError) as excinfo:
        freeze_generation_plan(proposal, proposal["candidates"][0], None, _config())
    assert excinfo.value.code == "PROPOSAL_INVALID"


def test_compiled_hashes_binding_projects_entries_and_atom_order() -> None:
    proposal = _proposal()
    candidate = proposal["candidates"][0]
    compiled = {
        "kind": "hashes",
        "point_input_sha256": ["1" * 64, "2" * 64, "3" * 64],
        "atom_rows": [2, 1],
        "candidate_id": "cand-0000",
    }
    plan = freeze_generation_plan(proposal, candidate, compiled, _config())
    assert plan["compiled"]["kind"] == "hashes"
    assert plan["compiled"]["entries"] == [
        {"point_index": 0, "input_sha256": "1" * 64},
        {"point_index": 1, "input_sha256": "2" * 64},
        {"point_index": 2, "input_sha256": "3" * 64},
    ]
    assert plan["extensions"]["freeze"]["atom_order"] == [2, 1]
    assert verify_generation_plan(plan) == []


def test_compiled_none_seals_candidate_recipe() -> None:
    proposal = _proposal()
    candidate = proposal["candidates"][0]
    binding = compiled_binding_from_candidate(candidate)
    assert binding["kind"] == "recipe"
    assert binding["recipe"]["mode"] == "SINGLE_1D"
    assert binding["recipe"]["lambda_values"] == [0.0, 0.5, 1.0]
    plan = freeze_generation_plan(proposal, candidate, None, _config())
    assert plan["compiled"]["kind"] == "recipe"
    assert plan["compiled"]["recipe"]["recipe_kind"] == "pes2ts_candidate_recipe_v1"
    assert verify_generation_plan(plan) == []


# ---------------------------------------------------------------------------
# Path freezes: NEB keeps the image-chain recipe; XTB_PATH seals an ACP
# recipe binding without an image chain (ADR-0001/ADR-0002, X3'-C).
# ---------------------------------------------------------------------------
def _path_candidate(method_kind: str | None, **extra: Any) -> dict[str, Any]:
    candidate: dict[str, Any] = {
        "candidate_kind": "PathCandidateV1",
        "candidate_id": "cand-path-0000",
        "n_atoms": 2,
        "endpoint_geometries": {
            "reactant": [[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]],
            "product": [[0.0, 0.0, 0.0], [1.50, 0.0, 0.0]],
        },
        "start_endpoint": "R",
        "direction": "R_to_P",
        "anchor_reason": "PATH_PROTOCOL",
        "failure_reasons": [],
    }
    if method_kind is not None:
        candidate["method_kind"] = method_kind
    candidate.update(extra)
    return candidate


def _freeze_path(candidate: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    kwargs.setdefault("endpoint_graph_sha256", SHA_B)
    return freeze_path_candidate_plan(
        candidate,
        reaction_id="RXN_0000000001",
        case_id="case-0001",
        split="train",
        source_case_sha256="a" * 64,
        **kwargs,
    )


def test_freeze_neb_path_seals_image_chain_recipe() -> None:
    candidate = _path_candidate("NEB", image_chain={"n_images": 5})
    plan = _freeze_path(candidate)
    assert validate_v2_document(plan) == []
    assert verify_generation_plan(plan) == []
    recipe = plan["compiled"]["recipe"]
    assert plan["compiled"]["kind"] == "recipe"
    assert recipe["method_kind"] == "NEB"
    assert recipe["n_images"] == 5
    assert recipe["image_chain"] == {"n_images": 5}
    assert plan["backend"]["engine"] == "orca"


def test_freeze_neb_path_without_method_kind_still_seals_image_chain() -> None:
    candidate = _path_candidate(None, image_chain={"n_images": 5})
    plan = _freeze_path(candidate)
    assert verify_generation_plan(plan) == []
    assert plan["compiled"]["recipe"]["method_kind"] == "NEB"


def test_freeze_xtb_path_seals_acp_recipe_without_image_chain() -> None:
    candidate = _path_candidate(
        "XTB_PATH",
        path_recipe={"path_inp_text": "$path\n nrun=1\n npoint=50\n$end\n", "gfn_level": 2},
    )
    assert "image_chain" not in candidate
    plan = _freeze_path(candidate)
    assert validate_v2_document(plan) == []
    assert verify_generation_plan(plan) == []
    assert plan["compiled"]["kind"] == "recipe"
    recipe = plan["compiled"]["recipe"]
    assert recipe["method_kind"] == "XTB_PATH"
    assert recipe["engine"] == "xtb"
    assert recipe["adapter_version"] == "pes2ts_xtb_path_request_v1"
    assert recipe["path_recipe"]["gfn_level"] == 2
    assert "image_chain" not in recipe
    assert "n_images" not in recipe
    assert plan["backend"]["engine"] == "xtb"
    assert plan["backend"]["method"] == "GFN2-xTB"


# ---------------------------------------------------------------------------
# CLI: gate-pass path writes plans + manifest; refusal path unchanged.
# ---------------------------------------------------------------------------
def _cli_config_path(tmp_path: Path) -> Path:
    config = load_config()
    config["paths"]["interim"] = str(tmp_path / "interim")
    config["paths"]["manifests"] = str(tmp_path / "manifests")
    path = tmp_path / "fixture.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _cli(config_path: Path, *argv: str) -> int:
    return main(["--config", str(config_path), *argv])


def _seed_proposal_tree(tmp_path: Path, proposal: dict[str, Any]) -> Path:
    config_path = _cli_config_path(tmp_path)
    proposals_dir = tmp_path / "interim" / "g1_v2" / "scan_proposals" / "00000"
    proposals_dir.mkdir(parents=True)
    write_json(proposals_dir / f"{proposal['reaction_id']}.json", proposal)
    summary = tmp_path / "interim" / "g1_v2_scan_proposal_summary.parquet"
    write_parquet(
        summary,
        {
            "reaction_id": [proposal["reaction_id"]],
            "status": [proposal["status"]],
            "family": [proposal["family"]],
            "execution_eligible": [proposal["execution_eligible"]],
            "n_candidates": [len(proposal["candidates"])],
            "content_sha256": [proposal["content_sha256"]],
        },
    )
    write_json(
        tmp_path / "manifests" / "g1_v2_scan_proposal_manifest.json",
        {
            "schema_version": "g1_v2_scan_proposal_manifest_v1",
            "n_total": 1,
            "n_files": 1,
            "n_summary_rows": 1,
            "summary_sha256": sha256_file(summary),
            "generated_at": "2026-10-01T00:00:00+00:00",
        },
    )
    return config_path


def test_cli_freeze_gate_pass_writes_plans_and_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = _seed_proposal_tree(tmp_path, _proposal())
    capsys.readouterr()

    assert _cli(config_path, "g1", "v2-scan-freeze") == 0
    out = capsys.readouterr().out
    assert "plan_gate_pass=true" in out

    plan_path = tmp_path / "interim" / "g1_v2" / "scan_plans" / "00000" / "RXN_0000000001.json"
    assert plan_path.is_file()
    plan = read_json(plan_path)
    assert verify_generation_plan(plan) == []

    plan_manifest = read_json(tmp_path / "manifests" / "g1_v2_scan_plan_manifest.json")
    assert plan_manifest["schema_version"] == "g1_v2_scan_plan_manifest_v1"
    assert plan_manifest["n_total"] == 1
    assert plan_manifest["n_files"] == 1

    freeze_manifest = read_json(tmp_path / "manifests" / "g1_v2_scan_freeze.json")
    assert freeze_manifest["plan_gate_pass"] is True
    assert freeze_manifest["n_plans"] == 1
    assert freeze_manifest["reasons"] == []

    plan_summary = tmp_path / "interim" / "g1_v2_scan_plan_summary.parquet"
    assert plan_summary.is_file()

    capsys.readouterr()
    assert _cli(config_path, "g1", "v2-scan-verify") == 0
    out = capsys.readouterr().out
    assert "problems=0" in out


def test_cli_freeze_refusal_path_unchanged(tmp_path: Path) -> None:
    config_path = _cli_config_path(tmp_path)
    assert _cli(config_path, "g1", "v2-scan-plan") == 0
    plans_dir = tmp_path / "interim" / "g1_v2" / "scan_plans"
    plan_manifest = tmp_path / "manifests" / "g1_v2_scan_plan_manifest.json"

    assert _cli(config_path, "g1", "v2-scan-freeze") == EXIT_G1_BUILD_FAILED
    manifest = read_json(tmp_path / "manifests" / "g1_v2_scan_freeze.json")
    assert manifest["plan_gate_pass"] is False
    assert "NO_ELIGIBLE_PROPOSALS" in manifest["reasons"]
    assert manifest["n_plans"] == 0
    assert plans_dir.is_dir()
    assert list(plans_dir.rglob("*.json")) == []
    assert not plan_manifest.exists()


def test_cli_freeze_refuses_needs_review_proposals(tmp_path: Path) -> None:
    needs_review = _proposal(status="needs_review")
    config_path = _seed_proposal_tree(tmp_path, needs_review)

    assert _cli(config_path, "g1", "v2-scan-freeze") == EXIT_G1_BUILD_FAILED
    manifest = read_json(tmp_path / "manifests" / "g1_v2_scan_freeze.json")
    assert manifest["plan_gate_pass"] is False
    assert "NO_ELIGIBLE_PROPOSALS" in manifest["reasons"]
    plans_dir = tmp_path / "interim" / "g1_v2" / "scan_plans"
    assert list(plans_dir.rglob("*.json")) == []
