"""Train-first Demo cohort runner tests using the fake ACP CLI only."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import pytest

from pes2ts_core.contracts import ContractError, make_document, seal_document
from pes2ts_core.demo import synthetic_objects
from pes2ts_core.planning import build_minimal_scan_plan
from pes2ts_core.utils.jsonio import write_json
from pes2ts_core.demo_runner import run_demo_execution_manifest


def _fake_acp(root: Path) -> Path:
    package = root / "src" / "acp"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "cli.py").write_text(r'''import json, pathlib, sys
args = sys.argv
call_counter = pathlib.Path(__file__).resolve().parents[2] / "CALL_COUNT"
count = int(call_counter.read_text(encoding="utf-8")) if call_counter.exists() else 0
call_counter.write_text(str(count + 1), encoding="utf-8")
request = json.loads(pathlib.Path(args[args.index("--scan-config") + 1]).read_text())
failure_marker = pathlib.Path(__file__).resolve().parents[2] / "EXIT_CODE"
if failure_marker.exists():
    raise SystemExit(int(failure_marker.read_text(encoding="utf-8")))
retry_marker = pathlib.Path(__file__).resolve().parents[2] / "FAIL_FIRST"
if retry_marker.exists():
    remaining = int(retry_marker.read_text(encoding="utf-8"))
    if remaining > 0:
        retry_marker.write_text(str(remaining - 1), encoding="utf-8")
        raise SystemExit(17)
output = pathlib.Path(args[args.index("--output") + 1])
work = output / "WORK" / "07_PATH" / "pes_scan_001"
result = output / "RESULT" / "pes_search"
frames_dir = work / "frames"
frames_dir.mkdir(parents=True, exist_ok=True)
result.mkdir(parents=True, exist_ok=True)
source = request["source"]["xyz_text"].splitlines()
n = int(source[0]); symbols = [line.split()[0] for line in source[2:2+n]]
xyz = [[float(x) for x in line.split()[1:4]] for line in source[2:2+n]]
coord = request["coordinate"]; frames = []
for i in range(coord["n_points"]):
    target = coord["start"] + (coord["end"]-coord["start"])*i/(coord["n_points"]-1)
    geom = [row[:] for row in xyz]; a,b = coord["atoms"]
    geom[b] = [geom[a][0]+target, geom[a][1], geom[a][2]]
    name = f"frame_{i:04d}.xyz"
    (frames_dir/name).write_text(str(n)+"\nfake\n"+"".join(
        f"{symbols[j]} {geom[j][0]} {geom[j][1]} {geom[j][2]}\n" for j in range(n)))
    method = request["protocol"]["scan_optimizer"]
    frames.append({"index":i,"target_coordinate":target,"actual_coordinate":target,
        "geometry_path":"frames/"+name,"scan_energy_hartree":-20+i/100,
        "single_point_energy_hartree":None,"optimization_converged":True,
        "constraint_residuals":{"driver-01":0.0},"retry_history":[],
        "optimizer_level":{"method":method["method"],"basis":method.get("basis"),
        "dispersion":method.get("dispersion"),"solvent_model":method.get("solvent_model","none"),
        "solvent":method.get("solvent"),"ri_approximation":method.get("ri_approximation","none")},
        "optimizer_engine":"orca"})
profile={"schema_version":"pes_profile_v2","workflow":"PESsearch","mode":"bond_length_scan",
    "status":"completed","scan_dir":"WORK/07_PATH/pes_scan_001","frames":frames}
(result/"pes_profile.json").write_text(json.dumps(profile))
(result/"pes_recommendations.json").write_text(json.dumps({"schema_version":"pes_recommendations_v1"}))
manifest={"version":2,"task_id":"","workflow":"PESsearch","status":"completed","products":[
    {"id":"pes_profile","label":"profile","path":"pes_search/pes_profile.json","kind":"pes_profile"},
    {"id":"pes_recommendations","label":"recommendations","path":"pes_search/pes_recommendations.json","kind":"report"}]}
(output/"RESULT"/"result_manifest.json").write_text(json.dumps(manifest))
''', encoding="utf-8")
    return root


def _reviewed_train_case():
    original = synthetic_objects()["ReactionCase"]
    review_id = "review:test-train"
    case = seal_document({**original, "source": {**original["source"],
        "review_record_id": review_id, "reviewed_from_case_sha256": original["content_sha256"]}})
    reviewers = [{"reviewer": name, "decision": "accept", "dimensions": {
        "reaction_center": "confirmed", "atom_mapping": "confirmed", "charge_spin": "confirmed",
        "geometry_assembly": "confirmed"}, "scan_feasibility": "1D",
        "multiplicities": {"reactant": 1, "product": 1}, "note": "test-only"}
        for name in ("reviewer-A", "reviewer-B")]
    review = make_document("ReviewRecord", review_id, "accepted",
        dataset_version=case["dataset_version"], reaction_id=case["reaction_id"],
        case_id=case["case_id"], split=case["split"], case_sha256=original["content_sha256"],
        workbook_sha256="a"*64, reviewers=reviewers,
        adjudication={"decision":"accepted","adjudicator":"test","reactant_multiplicity":1,
        "product_multiplicity":1,"resolution_note":""}, decision_source="human",
        extensions={"pes2ts.review_output.v1":{"reviewed_case_sha256":case["content_sha256"]}})
    return case, build_minimal_scan_plan(case), review


def test_demo_runner_executes_train_and_keeps_valid_held_out(tmp_path):
    case, _, review = _reviewed_train_case()
    valid = seal_document({**synthetic_objects()["ReactionCase"], "reaction_id":"RXN_SYNTH_VALID",
        "case_id":"case:synthetic-valid", "split":"valid"})
    inputs = tmp_path / "inputs"; inputs.mkdir()
    write_json(inputs / "train.json", case); write_json(inputs / "review.json", review)
    write_json(inputs / "valid.json", valid)
    write_json(inputs / "labels.json", [{"case_id":valid["case_id"],
        "acceptable_frame_ids":["manual:expected-frame"], "label_source":"independent validation",
        "method":"review protocol v1", "source_sha256":"b"*64}])
    write_json(inputs / "manifest.json", {"schema_version":"pes2ts_demo_execution_manifest_v1",
        "experiment_id":"demo-test", "n_points":5, "ranking_labels":"labels.json",
        "cohort_cases":["train.json", "valid.json"],
        "runs":[{"case":"train.json","review_record":"review.json"}, {"case":"valid.json"}]})
    result = run_demo_execution_manifest(manifest_path=inputs/"manifest.json",
        output_root=tmp_path/"out", acp_root=_fake_acp(tmp_path/"fake-acp"),
        python_executable=sys.executable)
    assert result["status_counts"] == {"path_collected":1, "valid_held_out":1}
    assert result["cohort_case_count"] == 2
    rows = {row["split"]: row for row in result["runs"]}
    assert rows["valid"]["execution_ref"] is None
    assert rows["train"]["path_ref"]
    assert len(rows["train"]["proposal_refs"]) == 2
    metrics = json.loads((tmp_path/"out"/"DemoMetrics.json").read_text(encoding="utf-8"))
    assert metrics["cohort"]["split_counts"] == {"train":1,"valid":1,"test":0,"unassigned":0}
    assert metrics["cohort"]["run_record_cases"] == 2
    assert metrics["cohort"]["missing_run_record_cases"] == 0
    assert metrics["generation"]["by_split"]["train"]["usable_path_cases"] == 1
    assert metrics["generation"]["by_split"]["valid"]["usable_path_cases"] == 0
    assert metrics["costs"]["known_wall_seconds_lower_bound"] > 0
    assert metrics["costs"]["total_wall_seconds"] == metrics["costs"]["known_wall_seconds_lower_bound"]
    assert metrics["ranking"]["labeled_case_count"] == 1
    assert metrics["ranking"]["labeled_cases_without_usable_path"] == 1
    bundle = json.loads((tmp_path/"out"/"DemoRunBundle.json").read_text(encoding="utf-8"))
    assert bundle["ranking_labels"] == "RankingLabels.json"
    assert len(bundle["runs"]) == 2
    assert next(row for row in bundle["runs"] if row["run_status"] == "valid_held_out")["execution"] is None
    assert (tmp_path/"out"/"RankingLabels.json").is_file()
    from pes2ts_core.demo_metrics import load_demo_run_bundle
    loaded_runs, loaded_labels, cohort = load_demo_run_bundle(tmp_path/"out"/"DemoRunBundle.json")
    assert len(loaded_runs) == len(cohort) == 2
    assert len(loaded_labels) == 1
    with pytest.raises(ContractError, match="requires --train-index"):
        run_demo_execution_manifest(manifest_path=inputs/"manifest.json",
            output_root=tmp_path/"invalid-valid-run", acp_root=tmp_path/"fake-acp",
            python_executable=sys.executable, include_valid=True)
    assert not (tmp_path/"invalid-valid-run").exists()
    label_path = inputs/"labels.json"
    frozen_label_text = label_path.read_text(encoding="utf-8")
    changed_labels = json.loads(frozen_label_text); changed_labels[0]["label_source"] = "changed after train"
    write_json(label_path, changed_labels)
    with pytest.raises(ContractError, match="exact frozen input manifest"):
        run_demo_execution_manifest(manifest_path=inputs/"manifest.json",
            output_root=tmp_path/"changed-input-valid-run", acp_root=tmp_path/"fake-acp",
            python_executable=sys.executable, include_valid=True,
            train_index_path=tmp_path/"out"/"DemoExecutionIndex.json")
    assert not (tmp_path/"changed-input-valid-run").exists()
    label_path.write_text(frozen_label_text, encoding="utf-8")
    valid_result = run_demo_execution_manifest(manifest_path=inputs/"manifest.json",
        output_root=tmp_path/"valid-evaluation", acp_root=tmp_path/"fake-acp",
        python_executable=sys.executable, include_valid=True,
        train_index_path=tmp_path/"out"/"DemoExecutionIndex.json")
    valid_rows = {row["split"]:row for row in valid_result["runs"]}
    assert valid_rows["valid"]["status"] == "blocked_review"
    old_exec = json.loads((tmp_path/"out"/rows["train"]["execution_ref"]).read_text(encoding="utf-8"))["object_id"]
    new_exec = json.loads((tmp_path/"valid-evaluation"/valid_rows["train"]["execution_ref"]).read_text(encoding="utf-8"))["object_id"]
    assert old_exec == new_exec
    assert (tmp_path/"fake-acp"/"CALL_COUNT").read_text(encoding="utf-8") == "1"


def test_demo_runner_keeps_failed_execution_and_unknown_cost_in_bundle(tmp_path):
    case, _, review = _reviewed_train_case()
    inputs = tmp_path / "inputs"; inputs.mkdir()
    write_json(inputs / "train.json", case); write_json(inputs / "review.json", review)
    write_json(inputs / "manifest.json", {"schema_version":"pes2ts_demo_execution_manifest_v1",
        "experiment_id":"demo-failure-test", "n_points":5,
        "cohort_cases":["train.json"],
        "runs":[{"case":"train.json","review_record":"review.json"}]})
    acp_root = _fake_acp(tmp_path / "fake-acp")
    (acp_root / "EXIT_CODE").write_text("17", encoding="utf-8")
    result = run_demo_execution_manifest(manifest_path=inputs/"manifest.json",
        output_root=tmp_path/"out", acp_root=acp_root, python_executable=sys.executable)
    row = result["runs"][0]
    assert row["status"] == "execution_failed"
    assert row["execution_ref"]
    execution = json.loads((tmp_path/"out"/row["execution_ref"]).read_text(encoding="utf-8"))
    assert execution["status"] == "failed"
    assert execution["failure_retained"] is True
    assert execution["attempts"][-1]["failure_code"] == "acp_cli_failed"
    assert execution["cost_complete"] is False
    metrics = json.loads((tmp_path/"out"/"DemoMetrics.json").read_text(encoding="utf-8"))
    assert metrics["generation"]["by_split"]["train"]["failed_or_cancelled_execution_cases"] == 1
    assert metrics["costs"]["cost_incomplete_cases"] == 1
    assert metrics["costs"]["wall_time_complete_cases"] == 1


def test_demo_runner_retries_with_new_attempt_id_and_retains_failure_receipt(tmp_path):
    case, _, review = _reviewed_train_case()
    inputs = tmp_path / "inputs"; inputs.mkdir()
    write_json(inputs / "train.json", case); write_json(inputs / "review.json", review)
    write_json(inputs / "manifest.json", {"schema_version":"pes2ts_demo_execution_manifest_v1",
        "experiment_id":"demo-retry-test", "n_points":5,
        "budget":{"max_attempts":2,"max_cpu_hours":0.1,"max_wall_seconds":60},
        "cohort_cases":["train.json"],
        "runs":[{"case":"train.json","review_record":"review.json"}]})
    acp_root = _fake_acp(tmp_path / "fake-acp")
    (acp_root / "FAIL_FIRST").write_text("1", encoding="utf-8")
    result = run_demo_execution_manifest(manifest_path=inputs/"manifest.json",
        output_root=tmp_path/"out", acp_root=acp_root, python_executable=sys.executable)
    row = result["runs"][0]
    assert row["status"] == "path_collected"
    execution = json.loads((tmp_path/"out"/row["execution_ref"]).read_text(encoding="utf-8"))
    assert execution["status"] == "completed"
    assert [attempt["status"] for attempt in execution["attempts"]] == ["failed", "completed"]
    assert len({attempt["attempt_id"] for attempt in execution["attempts"]}) == 2
    assert execution["failure_retained"] is True
    receipts = execution["extensions"]["pes2ts.acp_cli_attempt_receipts.v1"]["attempts"]
    assert [receipt["returncode"] for receipt in receipts] == [17, 0]
    assert all(receipt["request_sha256"] for receipt in receipts)
    for receipt in receipts:
        assert (tmp_path/"out"/Path(row["execution_ref"]).parent/receipt["log_ref"]).is_file()
