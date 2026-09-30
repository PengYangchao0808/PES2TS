"""Stage-by-stage ValidationResult assembly, provenance, and cost accounting."""
from __future__ import annotations

import pytest

from pes2ts_core.contracts import ContractError, seal_document
from pes2ts_core.demo import synthetic_objects
from pes2ts_core.integration.validation import build_validation_result
from pes2ts_core.ranking import rank_path_bundle


def _fixtures():
    docs = synthetic_objects()
    path = seal_document({**docs["PathBundle"], "status": "usable"})
    proposal = rank_path_bundle(path, rule="highest_scan_energy", top_k=3)
    frame_id = proposal["selected_frames"][0]["frame_id"]
    optts_geometry_digest = "d" * 64
    stages = {
        "optts": {"status": "converged", "acp_task_id": "acp:optts-1", "first_order_saddle": True,
                  "source_frame_id": frame_id, "ts_structure_ref": "RESULT/ts.xyz",
                  "optimized_geometry_sha256": optts_geometry_digest, "protocol_sha256":"e" * 64,
                  "result_manifest_sha256":"a" * 64},
        "frequency": {"status": "passed", "acp_task_id": "acp:frequency-1", "imaginary_mode_count": 1,
                      "spectrum_ref": "RESULT/frequencies.json",
                      "source_ts_geometry_sha256": optts_geometry_digest, "protocol_sha256":"f" * 64,
                      "result_manifest_sha256":"b" * 64},
        "irc_forward": {"status": "matched", "acp_task_id": "acp:irc-forward-1", "endpoint_reached": "reactant",
                         "endpoint_ref": "RESULT/irc-r.xyz",
                         "source_ts_geometry_sha256": optts_geometry_digest, "protocol_sha256":"1" * 64,
                         "result_manifest_sha256":"c" * 64},
        "irc_reverse": {"status": "matched", "acp_task_id": "acp:irc-reverse-1", "endpoint_reached": "product",
                         "endpoint_ref": "RESULT/irc-p.xyz",
                         "source_ts_geometry_sha256": optts_geometry_digest, "protocol_sha256":"2" * 64,
                         "result_manifest_sha256":"d" * 64},
    }
    return path, proposal, stages


def test_passed_result_requires_full_evidence_and_accounts_for_retries():
    path, proposal, stages = _fixtures()
    attempts = {
        "optts": [
            {"attempt_id": "opt-1", "status": "failed", "cpu_seconds": 3,
             "failure_code": "scf_nonconvergence"},
            {"attempt_id": "opt-2", "status": "completed", "cpu_seconds": 12},
        ],
        "frequency": [{"attempt_id": "freq-1", "status": "completed", "cpu_seconds": 4}],
        "irc_forward": [{"attempt_id": "irc-r-1", "status": "completed", "cpu_seconds": 8}],
        "irc_reverse": [{"attempt_id": "irc-p-1", "status": "completed", "cpu_seconds": 9}],
    }

    result = build_validation_result(validation_id="validation:pass", path=path, proposal=proposal,
        **stages, stage_attempts=attempts)

    assert result["status"] == "passed"
    ledger = result["extensions"]["pes2ts.validation_costs.v1"]
    assert ledger["cost_complete"] is True
    assert ledger["total_cpu_seconds"] == 36
    assert len(ledger["stages"]["optts"]["attempts"]) == 2

    contradictory_attempts = {"optts": [{"attempt_id": "opt-failed", "status": "failed",
                                         "cpu_seconds": 2, "failure_code": "not_converged"}]}
    with pytest.raises(ContractError, match="must end with a completed attempt"):
        build_validation_result(validation_id="validation:contradictory", path=path,
            proposal=proposal, **stages, stage_attempts=contradictory_attempts)


def test_missing_stage_cost_remains_explicit_without_changing_physical_status():
    path, proposal, stages = _fixtures()
    attempts = {stage: [{"attempt_id": f"{stage}-1", "status": "completed", "cpu_seconds": 1}]
                for stage in stages}
    attempts["irc_reverse"] = [{"attempt_id": "irc-p-1", "status": "completed", "cpu_seconds": None}]

    result = build_validation_result(validation_id="validation:unknown-cost", path=path,
        proposal=proposal, **stages, stage_attempts=attempts)

    ledger = result["extensions"]["pes2ts.validation_costs.v1"]
    assert result["status"] == "passed"
    assert ledger["cost_complete"] is False
    assert ledger["total_cpu_seconds"] is None


def test_incomplete_failed_and_not_run_stages_are_replayed_separately():
    path, proposal, stages = _fixtures()
    active = {**stages, "optts": {"status": "running", "acp_task_id": "acp:optts-active",
                                   "source_frame_id":
                                   proposal["selected_frames"][0]["frame_id"]}}
    active.update({stage: {"status": "not_run"} for stage in
                   ("frequency", "irc_forward", "irc_reverse")})
    assert build_validation_result(validation_id="validation:active", path=path,
        proposal=proposal, **active)["status"] == "incomplete"

    failed = {**stages, "frequency": {"status": "passed", "acp_task_id": "acp:frequency-bad",
                                       "imaginary_mode_count": 2,
                                       "result_manifest_sha256": "9" * 64}}
    assert build_validation_result(validation_id="validation:bad-frequency", path=path,
        proposal=proposal, **failed)["status"] == "failed"

    not_run = {stage: {"status": "not_run"} for stage in stages}
    result = build_validation_result(validation_id="validation:not-run", path=path,
        proposal=proposal, **not_run)
    assert result["status"] == "not_run"
    assert result["extensions"]["pes2ts.validation_costs.v1"]["total_cpu_seconds"] == 0


def test_irc_sign_order_can_swap_but_endpoints_must_be_distinct():
    path, proposal, stages = _fixtures()
    swapped = {**stages,
               "irc_forward": {**stages["irc_forward"], "endpoint_reached": "product"},
               "irc_reverse": {**stages["irc_reverse"], "endpoint_reached": "reactant"}}
    assert build_validation_result(validation_id="validation:swapped", path=path,
        proposal=proposal, **swapped)["status"] == "passed"

    same = {**stages, "irc_reverse": {**stages["irc_reverse"], "endpoint_reached": "reactant"}}
    assert build_validation_result(validation_id="validation:same-end", path=path,
        proposal=proposal, **same)["status"] == "failed"


def test_stale_proposal_wrong_initial_frame_and_bad_stage_costs_are_rejected():
    path, proposal, stages = _fixtures()
    changed_frames = [dict(frame) for frame in path["frames"]]
    changed_frames[-1] = {**changed_frames[-1], "geometry": [[0.1, 0, 0], *changed_frames[-1]["geometry"][1:]]}
    changed_path = seal_document({**path, "frames": changed_frames})
    with pytest.raises(ContractError, match="accepted proposal bound"):
        build_validation_result(validation_id="validation:stale", path=changed_path,
            proposal=proposal, **stages)

    wrong_source = {**stages, "optts": {**stages["optts"], "source_frame_id": "other-frame"}}
    result = build_validation_result(validation_id="validation:wrong-source", path=path,
        proposal=proposal, **wrong_source)
    assert result["status"] == "failed"

    stale_frames = [dict(frame) for frame in path["frames"]]
    selected_id = proposal["selected_frames"][0]["frame_id"]
    selected_index = next(i for i, frame in enumerate(stale_frames) if frame["frame_id"] == selected_id)
    stale_frames[selected_index] = {**stale_frames[selected_index], "geometry": [[0.1, 0, 0],
        *stale_frames[selected_index]["geometry"][1:]]}
    stale_path = seal_document({**path, "frames": stale_frames})
    stale_proposal = seal_document({**proposal, "path_content_sha256": stale_path["content_sha256"]})
    with pytest.raises(ContractError, match="geometry is stale"):
        build_validation_result(validation_id="validation:stale-frame", path=stale_path,
            proposal=stale_proposal, **stages)

    attempts = {"optts": [{"attempt_id": "attempt-1", "status": "failed", "cpu_seconds": 1}]}
    with pytest.raises(ContractError, match="failure_code"):
        build_validation_result(validation_id="validation:bad-cost", path=path,
            proposal=proposal, **stages, stage_attempts=attempts)


def test_cli_validation_stages_bind_execution_attempt_and_completed_manifest():
    path, proposal, stages = _fixtures()
    cli_stages = {}
    for stage_name, stage in stages.items():
        cli_stages[stage_name] = {
            key: value for key, value in stage.items() if key != "acp_task_id"
        }
        cli_stages[stage_name].update({
            "execution_id": f"exec:{stage_name}",
            "attempt_id": f"attempt:{stage_name}",
        })
    result = build_validation_result(validation_id="validation:cli", path=path,
        proposal=proposal, **cli_stages)
    assert result["status"] == "passed"

    missing_digest = {**cli_stages,
        "irc_reverse": {**cli_stages["irc_reverse"], "result_manifest_sha256": None}}
    with pytest.raises(ContractError, match="result manifest SHA256"):
        build_validation_result(validation_id="validation:cli-unbound", path=path,
            proposal=proposal, **missing_digest)
