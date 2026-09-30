"""S7 metrics use ReactionCase grain, separate labels, and honest cost coverage."""
from __future__ import annotations

import json
import hashlib
import subprocess
import sys
from pathlib import Path
import pytest

from pes2ts_core.contracts import ContractError, make_document, seal_document
from pes2ts_core.utils.hashing import stable_json_dumps
from pes2ts_core.demo import synthetic_objects
from pes2ts_core.demo_metrics import evaluate_demo_runs, load_demo_run_bundle
from pes2ts_core.integration.acp.quality import apply_scan_path_quality
from pes2ts_core.ranking import rank_path_bundle


def _run(split: str = "train"):
    docs = synthetic_objects()
    case, plan, execution = docs["ReactionCase"], docs["ScanPlan"], docs["ExecutionRecord"]
    if split != case["split"]:
        case = seal_document({**case, "split": split})
        plan = seal_document({**plan, "split": split, "source_case_sha256": case["content_sha256"]})
    points = plan["candidates"][0]["coordinates"][0]["points"]
    seed_frames = docs["PathBundle"]["frames"]
    frames = []
    for index, target in enumerate(points):
        frame = seed_frames[min(index, len(seed_frames) - 1)]
        geometry = [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [1.1 + target, 0.0, 0.0]]
        frames.append({**frame, "frame_id": f"metrics-frame-{index}", "frame_index": index,
                       "geometry": geometry, "actual_coordinate": target, "constraint_residuals": {"driver-01": 0.0},
                       "target_coordinate": target, "converged": True,
                       "energies": {"scan_electronic": {"value": -20 + index / 100,
                           "unit": "hartree", "method_id": "xtb:gfn2-xTB"}}})
    path = seal_document({**docs["PathBundle"], "frames": frames})
    path = apply_scan_path_quality(plan=plan, execution=execution, path=path)["path_bundle"]
    proposal = rank_path_bundle(path, rule="highest_scan_energy", top_k=3)
    validation = {**docs["ValidationResult"], "proposal_id": proposal["object_id"]}
    validation = seal_document(validation)
    return {"case": case, "plan": plan, "execution": execution, "path": path,
            "proposals": [proposal], "validation": validation}


def _label(case_id: str, frame_ids: list[str]) -> dict:
    return {"case_id": case_id, "acceptable_frame_ids": frame_ids,
            "label_source": "synthetic_fixture", "method": "explicit_test_label",
            "source_sha256": "a" * 64}


def test_metrics_are_case_grain_and_report_all_three_outcomes():
    run = _run()
    top_frame = run["proposals"][0]["selected_frames"][0]["frame_id"]
    result = evaluate_demo_runs(runs=[run], cohort_cases=[run["case"]],
        ranking_labels=[_label(run["case"]["case_id"], [top_frame])],
        evaluation_split="train")

    assert result["measurement_grain"] == "reaction_id"
    assert result["case_snapshot_unit"] == "ReactionCase"
    assert result["cohort"]["case_count"] == 1
    assert result["generation"]["by_split"]["train"]["usable_path_coverage"] == 1
    assert result["ranking"]["recall_at_k"] == {"1": 1.0, "3": 1.0}
    assert result["ranking"]["mean_reciprocal_rank"] == 1.0
    assert result["end_to_end"]["by_split"]["train"]["validation_status_counts"] == {"not_run": 1}
    assert result["costs"]["total_cpu_seconds"] == 60


def test_ranking_labels_are_separate_split_scoped_and_missing_labels_are_not_fabricated():
    run = _run()
    unmeasured = evaluate_demo_runs(runs=[run], cohort_cases=[run["case"]])
    assert unmeasured["ranking"]["status"] == "not_measured"
    assert unmeasured["ranking"]["recall_at_k"] == {"1": None, "3": None}

    with pytest.raises(ContractError, match="outside evaluation split"):
        evaluate_demo_runs(runs=[run], cohort_cases=[run["case"]], ranking_labels=[_label(
            run["case"]["case_id"], [run["path"]["frames"][0]["frame_id"]])])


def test_default_valid_ranking_metrics_use_case_level_recall_and_mrr():
    run = _run(split="valid")
    second_choice = run["proposals"][0]["selected_frames"][1]["frame_id"]
    result = evaluate_demo_runs(runs=[run], cohort_cases=[run["case"]],
        ranking_labels=[_label(run["case"]["case_id"], [second_choice])])

    assert result["ranking"]["status"] == "measured"
    assert result["ranking"]["evaluation_split"] == "valid"
    assert result["ranking"]["recall_at_k"] == {"1": 0.0, "3": 1.0}
    assert result["ranking"]["mean_reciprocal_rank"] == 0.5


def test_missing_costs_remain_unknown_and_failure_reasons_are_counted():
    run = _run()
    attempts = [dict(item) for item in run["execution"]["attempts"]]
    attempts[-1] = {**attempts[-1], "cpu_seconds": None}
    execution = seal_document({**run["execution"], "attempts": attempts,
                               "total_cpu_seconds": None, "cost_complete": False})
    run["execution"] = execution
    result = evaluate_demo_runs(runs=[run], cohort_cases=[run["case"]], evaluation_split="train")
    assert result["costs"]["total_cpu_seconds"] is None
    assert result["costs"]["known_cpu_seconds_lower_bound"] == 12
    assert result["costs"]["cost_incomplete_cases"] == 1

    docs = synthetic_objects()
    failed_attempts = [dict(item) for item in docs["ExecutionRecord"]["attempts"]]
    failed_attempts[-1] = {**failed_attempts[-1], "status": "failed", "failure_code": "scan_failed"}
    failed = seal_document({**docs["ExecutionRecord"], "status": "failed",
                            "attempts": failed_attempts, "failure_retained": True})
    docs["ExecutionRecord"] = failed
    failed_run = {"case": docs["ReactionCase"], "plan": docs["ScanPlan"], "execution": failed}
    failures = evaluate_demo_runs(runs=[failed_run], cohort_cases=[failed_run["case"]], evaluation_split="train")
    assert failures["generation"]["by_split"]["train"]["failure_analysis"]["execution_failure_codes"] == {
        "scan_failed": 1, "synthetic_timeout": 1}


def test_cli_wall_time_is_measured_from_attempt_receipts_not_plan_budgets():
    run = _run()
    attempts = run["execution"]["attempts"]
    receipts = [{"attempt_id": item["attempt_id"], "wall_seconds": seconds}
        for item, seconds in zip(attempts, (3.5, 7.25), strict=True)]
    execution = seal_document({**run["execution"], "extensions": {
        "pes2ts.acp_cli_attempt_receipts.v1": {"schema":"pes2ts_acp_cli_attempt_receipts_v1",
            "wall_time_unit":"second", "cpu_time_available":False, "attempts":receipts}}})
    run["execution"] = execution
    report = evaluate_demo_runs(runs=[run], cohort_cases=[run["case"]], evaluation_split="train")
    assert report["costs"]["wall_time_complete_cases"] == 1
    assert report["costs"]["total_wall_seconds"] == 10.75

    receipts[0]["wall_seconds"] = -1
    execution = seal_document({**execution, "extensions": {
        "pes2ts.acp_cli_attempt_receipts.v1": {"schema":"pes2ts_acp_cli_attempt_receipts_v1",
            "wall_time_unit":"second", "cpu_time_available":False, "attempts":receipts}}})
    run["execution"] = execution
    report = evaluate_demo_runs(runs=[run], cohort_cases=[run["case"]], evaluation_split="train")
    assert report["costs"]["total_wall_seconds"] is None
    assert report["costs"]["known_wall_seconds_lower_bound"] == 7.25


def test_proposal_must_match_path_content_and_duplicate_case_runs_are_rejected():
    run = _run()
    changed = seal_document({**run["path"], "status": "needs_review"})
    with pytest.raises(ContractError, match="stale or bound"):
        evaluate_demo_runs(runs=[{**run, "path": changed}], cohort_cases=[run["case"]])
    with pytest.raises(ContractError, match="multiple run rows"):
        evaluate_demo_runs(runs=[run, run], cohort_cases=[run["case"]])


def test_validation_cannot_be_reused_after_path_and_top1_change():
    run = _run()
    proposal = run["proposals"][0]
    frame = next(row for row in run["path"]["frames"]
                 if row["frame_id"] == proposal["selected_frames"][0]["frame_id"])
    geometry_digest = hashlib.sha256(stable_json_dumps(frame["geometry"]).encode()).hexdigest()
    run["validation"] = make_document("ValidationResult", "validation:old-top1", "passed",
        reaction_id=run["case"]["reaction_id"], case_id=run["case"]["case_id"],
        path_id=run["path"]["object_id"], proposal_id=proposal["object_id"],
        path_content_sha256=run["path"]["content_sha256"],
        proposal_content_sha256=proposal["content_sha256"],
        source_frame_id=frame["frame_id"], source_geometry_sha256=geometry_digest,
        optts={"status":"converged", "first_order_saddle":True,
               "acp_task_id":"acp:optts-synthetic", "source_frame_id":frame["frame_id"],
               "optimized_geometry_sha256":"d" * 64,
               "protocol_sha256":"e" * 64, "result_manifest_sha256":"a" * 64},
        frequency={"status":"passed", "acp_task_id":"acp:freq-synthetic", "imaginary_mode_count":1,
                   "source_ts_geometry_sha256":"d" * 64, "protocol_sha256":"f" * 64,
                   "result_manifest_sha256":"b" * 64},
        irc_forward={"status":"matched", "acp_task_id":"acp:irc-f-synthetic", "endpoint_reached":"reactant",
                      "source_ts_geometry_sha256":"d" * 64, "protocol_sha256":"1" * 64,
                      "result_manifest_sha256":"c" * 64},
        irc_reverse={"status":"matched", "acp_task_id":"acp:irc-r-synthetic", "endpoint_reached":"product",
                      "source_ts_geometry_sha256":"d" * 64, "protocol_sha256":"2" * 64,
                      "result_manifest_sha256":"d" * 64},
        validation_note="bound synthetic test evidence")

    frames = [dict(item) for item in run["path"]["frames"]]
    # Change the energy surface so a newly ranked top-1 uses another frame.
    frames[0] = {**frames[0], "energies":{"scan_electronic":{
        "value": 0.0, "unit":"hartree", "method_id":"xtb:gfn2-xTB"}}}
    changed_path = seal_document({**run["path"], "frames":frames})
    changed_path = apply_scan_path_quality(plan=run["plan"], execution=run["execution"],
                                           path=changed_path)["path_bundle"]
    changed_proposal = rank_path_bundle(changed_path, rule="highest_scan_energy", top_k=3)
    stale_run = {**run, "path":changed_path, "proposals":[changed_proposal]}

    with pytest.raises(ContractError, match="ValidationResult is stale"):
        evaluate_demo_runs(runs=[stale_run], cohort_cases=[run["case"]], evaluation_split="train")


def test_rejected_plan_remains_in_generation_denominator():
    docs = synthetic_objects()
    run = {"case": docs["ReactionCase"], "plan": docs["RejectedPlan"]}
    result = evaluate_demo_runs(runs=[run], cohort_cases=[run["case"]], evaluation_split="train")
    train = result["generation"]["by_split"]["train"]
    assert train["case_count"] == 1
    assert train["ready_plan_cases"] == 0
    assert train["rejected_plan_cases"] == 1
    assert train["failure_analysis"]["plan_rejection_reasons"]


def test_missing_case_run_is_kept_in_frozen_cohort_denominator():
    run = _run()
    absent = seal_document({**run["case"], "object_id": "case:synthetic-not-run",
                            "case_id": "case:synthetic-not-run", "reaction_id": "RXN_SYNTH_NOT_RUN"})

    report = evaluate_demo_runs(runs=[run], cohort_cases=[run["case"], absent], evaluation_split="train")

    assert report["cohort"]["case_count"] == 2
    assert report["cohort"]["run_record_coverage"] == 0.5
    assert report["cohort"]["missing_run_record_cases"] == 1
    assert report["generation"]["by_split"]["train"]["ready_plan_rate"] == 0.5
    assert report["costs"]["cost_incomplete_cases"] == 1

    duplicate_reaction = seal_document({**absent, "object_id": "case:duplicate-snapshot",
        "case_id": "case:duplicate-snapshot", "reaction_id": run["case"]["reaction_id"]})
    with pytest.raises(ContractError, match="one ReactionCase per reaction_id"):
        evaluate_demo_runs(runs=[run], cohort_cases=[run["case"], duplicate_reaction])


def test_run_bundle_loader_reads_referenced_artifacts_and_rejects_path_escape(tmp_path):
    run = _run()
    root = tmp_path / "bundle"
    for directory in ("cases", "plans", "executions", "paths", "proposals", "validations", "evaluation"):
        (root / directory).mkdir(parents=True)
    refs = {}
    for key in ("case", "plan", "execution", "path", "validation"):
        name = f"{key}.json"
        (root / f"{ {'case':'cases','plan':'plans','execution':'executions','path':'paths','validation':'validations'}[key] }" / name).write_text(
            json.dumps(run[key]), encoding="utf-8")
        refs[key] = f"{ {'case':'cases','plan':'plans','execution':'executions','path':'paths','validation':'validations'}[key] }/{name}"
    (root / "proposals" / "rank.json").write_text(json.dumps(run["proposals"][0]), encoding="utf-8")
    label = _label(run["case"]["case_id"], [run["path"]["frames"][0]["frame_id"]])
    (root / "evaluation" / "labels.json").write_text(json.dumps([label]), encoding="utf-8")
    bundle = {"schema_version": "pes2ts_demo_run_bundle_v1", "cohort_cases": [refs["case"]],
              "ranking_labels": "evaluation/labels.json",
              "runs": [{**refs, "proposals": ["proposals/rank.json"]}]}
    bundle_path = root / "run_bundle.json"
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")

    loaded_runs, loaded_labels, cohort_cases = load_demo_run_bundle(bundle_path)
    report = evaluate_demo_runs(runs=loaded_runs, cohort_cases=cohort_cases,
        ranking_labels=loaded_labels, evaluation_split="train")
    assert report["cohort"]["case_count"] == 1
    assert report["ranking"]["status"] == "measured"

    metrics_path = root / "metrics.json"
    script = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_demo_runs.py"
    cli = subprocess.run([sys.executable, str(script), str(bundle_path), "--output", str(metrics_path),
                          "--evaluation-split", "train"], capture_output=True, text=True, check=False)
    assert cli.returncode == 0, cli.stderr
    assert json.loads(metrics_path.read_text(encoding="utf-8"))["cohort"]["case_count"] == 1

    bundle["runs"][0]["case"] = "../outside.json"
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
    with pytest.raises(ContractError, match="escapes the bundle root"):
        load_demo_run_bundle(bundle_path)
