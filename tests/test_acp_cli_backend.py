"""Local ACP CLI process lifecycle, result recovery, and geometry collection."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
import threading
import time

import pytest

from pes2ts_core.contracts import dumps_document, make_document, seal_document
from pes2ts_core.demo import synthetic_objects
from pes2ts_core.integration.acp.adapter import scan_plan_to_acp_request
from pes2ts_core.planning import build_minimal_scan_plan
from pes2ts_core.integration.acp.cli_backend import (
    ACPCLIBackend, ACPCLIError, collect_cli_path_bundle,
    cli_result_to_execution_record, validate_cli_result,
)
from pes2ts_core.integration.acp.quality import apply_scan_path_quality


def _fake_acp(tmp_path: Path) -> Path:
    root = tmp_path / "fake_acp"
    package = root / "src" / "acp"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "cli.py").write_text(r'''import json, pathlib, sys, time
args = sys.argv
request = json.loads(pathlib.Path(args[args.index("--scan-config") + 1]).read_text(encoding="utf-8"))
output = pathlib.Path(args[args.index("--output") + 1])
delay = request.get("protocol", {}).get("test_delay_seconds", 0)
time.sleep(delay)
if request.get("protocol", {}).get("test_exit_code"):
    raise SystemExit(int(request["protocol"]["test_exit_code"]))
work = output / "WORK" / "07_PATH" / "pes_scan_001"
result = output / "RESULT" / "pes_search"
frames_dir = work / "frames"
frames_dir.mkdir(parents=True, exist_ok=True)
result.mkdir(parents=True, exist_ok=True)
source = request["source"]["xyz_text"].splitlines()
n_atoms = int(source[0])
symbols = [line.split()[0] for line in source[2:2+n_atoms]]
xyz = [[float(x) for x in line.split()[1:4]] for line in source[2:2+n_atoms]]
coord = request["coordinate"]
frames = []
for i in range(coord["n_points"]):
    target = coord["start"] + (coord["end"] - coord["start"]) * i / (coord["n_points"] - 1)
    geom = [row[:] for row in xyz]
    a, b = coord["atoms"]
    geom[b] = [geom[a][0] + target, geom[a][1], geom[a][2]]
    frame_name = f"frame_{i:04d}.xyz"
    frame_path = frames_dir / frame_name
    frame_path.write_text(str(n_atoms) + "\nfake ACP frame\n" + "".join(
        f"{symbols[j]} {geom[j][0]:.8f} {geom[j][1]:.8f} {geom[j][2]:.8f}\n" for j in range(n_atoms)), encoding="utf-8")
    frames.append({"index":i, "target_coordinate":target, "actual_coordinate":target,
        "geometry_path":"frames/" + frame_name, "scan_energy_hartree":-20.0 + i/100,
        "single_point_energy_hartree":None, "optimization_converged":True,
        "constraint_residuals":{"driver-01":0.0}, "retry_history":[],
        "optimizer_level":{"method":request["protocol"]["scan_optimizer"]["method"],
            "basis":request["protocol"]["scan_optimizer"].get("basis"),
            "dispersion":request["protocol"]["scan_optimizer"].get("dispersion"),
            "solvent_model":request["protocol"]["scan_optimizer"].get("solvent_model", "none"),
            "solvent":request["protocol"]["scan_optimizer"].get("solvent"),
            "ri_approximation":request["protocol"]["scan_optimizer"].get("ri_approximation", "none")},
        "optimizer_engine":"orca"})
profile = {"schema_version":"pes_profile_v2", "workflow":"PESsearch", "mode":"bond_length_scan",
    "status":"completed", "scan_dir":"WORK/07_PATH/pes_scan_001", "frames":frames}
(result / "pes_profile.json").write_text(json.dumps(profile), encoding="utf-8")
(result / "pes_recommendations.json").write_text(json.dumps({"schema_version":"pes_recommendations_v1"}), encoding="utf-8")
manifest = {"version":2, "task_id":"", "workflow":"PESsearch", "status":"completed", "products":[
    {"id":"pes_profile", "label":"profile", "path":"pes_search/pes_profile.json", "kind":"pes_profile"},
    {"id":"pes_recommendations", "label":"recommendations", "path":"pes_search/pes_recommendations.json", "kind":"report"}]}
(output / "RESULT" / "result_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
''', encoding="utf-8")
    return root


def _request(*, delay: float = 0.0, exit_code: int = 0):
    docs = synthetic_objects()
    case, plan = docs["ReactionCase"], docs["ScanPlan"]
    payload = scan_plan_to_acp_request(case, plan)["scan_request"]
    payload["protocol"]["test_delay_seconds"] = delay
    payload["protocol"]["test_exit_code"] = exit_code
    return case, plan, payload


def _reviewed_case_plan():
    original = synthetic_objects()["ReactionCase"]
    review_id = f"review:{original['reaction_id']}:test"
    case = seal_document({**original, "source":{**original["source"],
        "review_record_id":review_id, "reviewed_from_case_sha256":original["content_sha256"]}})
    reviewers = [{"reviewer":name, "decision":"accept", "dimensions":{
        "reaction_center":"confirmed", "atom_mapping":"confirmed", "charge_spin":"confirmed",
        "geometry_assembly":"confirmed"}, "scan_feasibility":"1D",
        "multiplicities":{"reactant":1,"product":1}, "note":"test-only review record"}
        for name in ("reviewer-A", "reviewer-B")]
    review = make_document("ReviewRecord", review_id, "accepted",
        dataset_version=case["dataset_version"], reaction_id=case["reaction_id"],
        case_id=case["case_id"], split=case["split"], case_sha256=original["content_sha256"],
        workbook_sha256="a" * 64, reviewers=reviewers,
        adjudication={"decision":"accepted", "adjudicator":"test-adjudicator",
            "reactant_multiplicity":1, "product_multiplicity":1, "resolution_note":""},
        decision_source="human",
        extensions={"pes2ts.review_output.v1":{"reviewed_case_sha256":case["content_sha256"]}})
    return case, build_minimal_scan_plan(case), review


def test_cli_attempt_executes_verifies_and_recovers_without_duplicate_submission(tmp_path):
    root = _fake_acp(tmp_path)
    case, plan, request = _request()
    backend = ACPCLIBackend(acp_root=root, python_executable=sys.executable)

    result = backend.run_scan(execution_id="execution-test-1", attempt_id="attempt-001",
        scan_request=request, output_root=tmp_path / "run", timeout_seconds=10)
    recovered = backend.run_scan(execution_id="execution-test-1", attempt_id="attempt-001",
        scan_request=request, output_root=tmp_path / "run", timeout_seconds=10)

    assert result.status == "completed" and result.returncode == 0
    assert result.manifest_sha256 and Path(result.attempt_dir, "WORK", "pes2ts", "acp_cli.log").is_file()
    assert recovered.reused is True
    verified = validate_cli_result(result.attempt_dir)
    assert verified["profile"]["frames"]
    assert verified["manifest_sha256"] == result.manifest_sha256
    frame_product = next(product for product in verified["products"]
                         if product["metadata"].get("pes2ts_role") == "scan_frame_geometry")
    frame_geometry = Path(result.attempt_dir, "RESULT", frame_product["path"])
    assert frame_geometry.is_file()
    assert frame_product["sha256"] == frame_product["metadata"]["sha256"]
    frame_products = [product for product in verified["products"]
                      if product["metadata"].get("pes2ts_role") == "scan_frame_geometry"]
    assert len(frame_products) == len(verified["profile"]["frames"])
    assert len(frame_products) == len(plan["candidates"][0]["coordinates"][0]["points"])
    execution = cli_result_to_execution_record(case=case, plan=plan,
        candidate_id=plan["candidates"][0]["candidate_id"], result=result)
    path = collect_cli_path_bundle(output_dir=result.attempt_dir, case=case, plan=plan,
        execution_id=result.execution_id)
    quality = apply_scan_path_quality(plan=plan, execution=execution, path=path)
    assert execution["acp_task_id"] is None
    assert execution["cost_complete"] is False
    assert quality["path_bundle"]["status"] == "usable"
    assert quality["path_bundle"]["frames"][0]["geometry_ref"] == "RESULT/pes2ts/frames/frame_00000.xyz"
    assert all(frame["geometry_ref"].startswith("RESULT/pes2ts/frames/")
               for frame in quality["path_bundle"]["frames"])

    receipt_path = Path(result.attempt_dir, "WORK", "pes2ts", "cli_receipt.json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt.update({"status":"running", "pid":99999999, "returncode":None,
                    "finished_at":None, "manifest_sha256":None})
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    recovered_after_parent_exit = backend.run_scan(execution_id="execution-test-1", attempt_id="attempt-001",
        scan_request=request, output_root=tmp_path / "run", timeout_seconds=10)
    assert recovered_after_parent_exit.reused is True


def test_cli_timeout_and_nonzero_exit_are_retained_as_failed_attempts(tmp_path):
    root = _fake_acp(tmp_path)
    _, _, slow_request = _request(delay=0.5)
    backend = ACPCLIBackend(acp_root=root, python_executable=sys.executable)
    timed_out = backend.run_scan(execution_id="execution-timeout", attempt_id="attempt-timeout",
        scan_request=slow_request, output_root=tmp_path / "timeout", timeout_seconds=0.05)
    assert timed_out.status == "failed" and timed_out.timed_out is True

    _, _, bad_request = _request(exit_code=9)
    failed = backend.run_scan(execution_id="execution-failed", attempt_id="attempt-failed",
        scan_request=bad_request, output_root=tmp_path / "failed", timeout_seconds=10)
    assert failed.status == "failed" and failed.returncode == 9
    record = cli_result_to_execution_record(case=synthetic_objects()["ReactionCase"],
        plan=synthetic_objects()["ScanPlan"], candidate_id=synthetic_objects()["ScanPlan"]["candidates"][0]["candidate_id"],
        result=failed)
    assert record["status"] == "failed"
    assert record["attempts"][0]["failure_code"] == "acp_cli_failed"


def test_cli_refuses_path_traversal_and_reuse_with_changed_request(tmp_path):
    root = _fake_acp(tmp_path)
    _, _, request = _request()
    backend = ACPCLIBackend(acp_root=root, python_executable=sys.executable)
    complete = backend.run_scan(execution_id="execution-safe", attempt_id="attempt-safe",
        scan_request=request, output_root=tmp_path / "safe", timeout_seconds=10)
    manifest_file = Path(complete.attempt_dir, "RESULT", "result_manifest.json")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    manifest["products"][0]["path"] = "..\\outside.json"
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ACPCLIError, match="unsafe path component"):
        validate_cli_result(complete.attempt_dir)

    changed = {**request, "protocol":{**request["protocol"], "scan_optimizer":{"method":"OTHER"}}}
    with pytest.raises(ACPCLIError, match="different execution or request"):
        backend.run_scan(execution_id="execution-safe", attempt_id="attempt-safe",
            scan_request=changed, output_root=tmp_path / "safe", timeout_seconds=10)


def test_result_geometry_must_match_the_source_work_frame_even_with_updated_manifest_hash(tmp_path):
    root = _fake_acp(tmp_path)
    case, plan, request = _request()
    backend = ACPCLIBackend(acp_root=root, python_executable=sys.executable)
    result = backend.run_scan(execution_id="execution-geometry-binding", attempt_id="attempt-geometry-binding",
        scan_request=request, output_root=tmp_path / "geometry-binding", timeout_seconds=10)
    manifest_path = Path(result.attempt_dir, "RESULT", "result_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    product = next(item for item in manifest["products"]
                   if item.get("metadata", {}).get("pes2ts_role") == "scan_frame_geometry"
                   and item["metadata"]["frame_index"] == 0)
    geometry_path = Path(result.attempt_dir, "RESULT", product["path"])
    rows = geometry_path.read_text(encoding="utf-8").splitlines()
    fields = rows[2].split()
    fields[1] = "99.0000000000"
    rows[2] = " ".join(fields)
    geometry_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    digest = hashlib.sha256(geometry_path.read_bytes()).hexdigest()
    product["metadata"]["sha256"] = digest
    product["metadata"]["size_bytes"] = geometry_path.stat().st_size
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ACPCLIError, match="differs from its registered RESULT geometry"):
        collect_cli_path_bundle(output_dir=result.attempt_dir, case=case, plan=plan,
                                execution_id=result.execution_id)


def test_pes2ts_acp_run_command_writes_execution_path_and_quality_records(tmp_path):
    root = _fake_acp(tmp_path)
    case, plan, review = _reviewed_case_plan()
    case_path, plan_path = tmp_path / "case.json", tmp_path / "plan.json"
    review_path = tmp_path / "review.json"
    case_path.write_text(dumps_document(case), encoding="utf-8")
    plan_path.write_text(dumps_document(plan), encoding="utf-8")
    review_path.write_text(dumps_document(review), encoding="utf-8")
    project_root = Path(__file__).resolve().parents[1]
    command = [sys.executable, str(project_root / "bin" / "pes2ts"), "acp-run",
        "--case", str(case_path), "--plan", str(plan_path), "--review-record", str(review_path), "--acp-root", str(root),
        "--python", sys.executable, "--output-root", str(tmp_path / "cli-run"),
        "--execution-id", "execution-cli", "--attempt-id", "attempt-cli", "--timeout", "10"]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    completed = subprocess.run(command, capture_output=True, text=True, env=env, check=False)

    audit = tmp_path / "cli-run" / "attempts" / "attempt-cli" / "PES2TS"
    assert completed.returncode == 0, completed.stderr
    assert json.loads((audit / "ExecutionRecord.json").read_text(encoding="utf-8"))["acp_task_id"] is None
    assert json.loads((audit / "PathBundle.json").read_text(encoding="utf-8"))["status"] == "usable"
    assert json.loads((audit / "PathQuality.json").read_text(encoding="utf-8"))["checks"]["no_atom_collisions"] is True
