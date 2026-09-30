"""Read-only validation evidence extraction from ACP v2 results."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest

from pes2ts_core.demo import synthetic_objects
from pes2ts_core.integration.acp.cli_backend import ACPCLIError
from pes2ts_core.integration.acp.stage_results import (
    _aligned_rmsd,
    _pair_distance_rms,
    collect_acp_validation_result,
    collect_batch_ts_frequency_evidence,
    collect_irc_evidence,
)
from pes2ts_core.contracts import dumps_document, make_document, seal_document
from pes2ts_core.ranking import rank_path_bundle
from pes2ts_core.utils.hashing import stable_json_dumps


def _write_xyz(path: Path, elements: list[str], geometry: list[list[float]], title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [str(len(elements)), title]
    lines.extend(f"{element} {xyz[0]:.10f} {xyz[1]:.10f} {xyz[2]:.10f}"
                 for element, xyz in zip(elements, geometry))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _manifest(root: Path, workflow: str, products: list[dict]) -> None:
    result = root / "RESULT"
    result.mkdir(parents=True, exist_ok=True)
    (result / "result_manifest.json").write_text(json.dumps({
        "version": 2, "task_id": "", "workflow": workflow,
        "status": "completed", "products": products,
    }), encoding="utf-8")


def _case():
    return synthetic_objects()["ReactionCase"]


def _reviewed_case(case: dict):
    review_id = f"review:{case['reaction_id']}:stage-results-test"
    original_digest = case["content_sha256"]
    ready = seal_document({**case, "status": "ready", "source": {
        **case.get("source", {}), "review_record_id": review_id,
        "reviewed_from_case_sha256": original_digest}})
    reviewers = [{"reviewer": name, "decision": "accept", "dimensions": {
        "reaction_center": "confirmed", "atom_mapping": "confirmed",
        "charge_spin": "confirmed", "geometry_assembly": "confirmed"},
        "scan_feasibility": "1D", "multiplicities": {"reactant": 1, "product": 1},
        "note": "synthetic test evidence"} for name in ("reviewer-A", "reviewer-B")]
    review = make_document("ReviewRecord", review_id, "accepted",
        dataset_version=case["dataset_version"], reaction_id=case["reaction_id"],
        case_id=case["case_id"], split=case["split"], case_sha256=original_digest,
        workbook_sha256="a" * 64, reviewers=reviewers,
        adjudication={"decision": "accepted", "adjudicator": "test-adjudicator",
            "reactant_multiplicity": 1, "product_multiplicity": 1, "resolution_note": ""},
        decision_source="human",
        extensions={"pes2ts.review_output.v1": {"reviewed_case_sha256": ready["content_sha256"]}})
    return ready, review


def _case_geometry_digest(geometry: list[list[float]]) -> str:
    return hashlib.sha256(stable_json_dumps(geometry).encode()).hexdigest()


def _make_irc_result(tmp_path: Path, case: dict, ts_file_digest: str) -> None:
    elements = [atom["element"] for atom in case["atoms"]]
    endpoints = {"forward": case["reactant"]["geometry"],
                 "reverse": case["product"]["geometry"]}
    products = [{"id": "irc_report", "path": "irc/irc_report.json", "kind": "report"}]
    report_endpoints = {}
    for direction, geometry in endpoints.items():
        relative = f"irc/irc_{direction}.xyz"
        _write_xyz(tmp_path / "RESULT" / relative, elements, geometry, f"{direction} endpoint")
        work_path = tmp_path / "WORK/07_PATH/ORCA" / f"irc_{direction}.xyz"
        _write_xyz(work_path, elements, geometry, f"{direction} endpoint")
        report_endpoints[direction] = str(work_path)
        products.append({"id": f"irc_{direction}_endpoint", "path": relative,
                         "kind": "irc_endpoint"})
    report_path = tmp_path / "RESULT/irc/irc_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({
        "workflow": "irc", "status": "completed", "directions": ["forward", "reverse"],
        "method": "B3LYP", "basis": "def2-SVP", "maxpoints": 100, "step": 0.1,
        "input_role": "transition_state", "endpoints": report_endpoints,
        "ts_source": {"schema": "irc_ts_source_v1", "geometry_sha256": ts_file_digest,
                      "method": "B3LYP", "basis": "def2-SVP", "charge": 0,
                      "multiplicity": 1},
    }), encoding="utf-8")
    _manifest(tmp_path, "irc", products)


def _make_batch_result(tmp_path: Path, *, wrong_geometry_binding: bool = False,
                       source_geometry: list[list[float]] | None = None) -> tuple[dict, list[str], list[list[float]], str]:
    case = _case()
    elements = [atom["element"] for atom in case["atoms"]]
    source_geometry = source_geometry or case["product"]["geometry"]
    source_digest = _case_geometry_digest(source_geometry)
    _write_xyz(tmp_path / "input.xyz", elements, source_geometry, "TS source")
    optimized = [[xyz[0] + 0.02, xyz[1], xyz[2]] for xyz in source_geometry]
    _write_xyz(tmp_path / "RESULT/structures/ts001__TAG_TS__optimized.xyz",
               elements, optimized, "TS optimized")
    structure_id = "batch_ts001"
    modes_id = "batch_ts001_normal_modes"
    mode_geometry_id = "batch_wrong" if wrong_geometry_binding else structure_id
    (tmp_path / "RESULT/frequencies/ts001__normal_modes.json").parent.mkdir(
        parents=True, exist_ok=True)
    frequencies = [-120.0] + [25.0] * (3 * len(elements) - 5 - 1)
    (tmp_path / "RESULT/frequencies/ts001__normal_modes.json").write_text(json.dumps({
        "schema_version": "normal_modes_v1", "units": {"frequency": "cm-1"},
        "atom_count": len(elements), "geometry_product_id": None, "warnings": [],
        "modes": [{"mode_index": index, "frequency_cm1": value,
                   "imaginary": value < 0,
                   "vectors": [[1.0, 0.0, 0.0], *[[0.0, 0.0, 0.0] for _ in elements[1:]]]}
                  for index, value in enumerate(frequencies)],
    }), encoding="utf-8")
    (tmp_path / "RESULT/batch_provenance.json").write_text(json.dumps({
        "schema": "batch_provenance_v1", "profile": "opt_freq",
        "workflow": "BatchOptimize", "items": [{"item_id": "ts001", "role": "ts",
            "effective_config": {"method": "B3LYP", "basis": "def2-SVP",
                                 "opt_level": "tight"}}],
    }), encoding="utf-8")
    _manifest(tmp_path, "BatchOptimize", [
        {"id": structure_id, "path": "structures/ts001__TAG_TS__optimized.xyz", "kind": "structure"},
        {"id": modes_id, "path": "frequencies/ts001__normal_modes.json", "kind": "frequency_modes",
         "metadata": {"geometry_product_id": mode_geometry_id}},
    ])
    return case, elements, source_geometry, source_digest


def test_collects_batch_optts_and_frequency_bound_to_same_geometry(tmp_path: Path) -> None:
    case, elements, source_geometry, source_digest = _make_batch_result(tmp_path)
    optts, frequency = collect_batch_ts_frequency_evidence(
        task_root=tmp_path, item_id="ts001", expected_elements=elements,
        source_frame_id="frame:top1", source_geometry=source_geometry,
        source_geometry_sha256=source_digest, execution_id="exec:ts",
        attempt_id="attempt:ts", expected_method="B3LYP", expected_basis="def2-SVP")
    assert optts["status"] == "converged"
    assert optts["first_order_saddle"] is True
    assert frequency["status"] == "passed"
    assert frequency["imaginary_mode_count"] == 1
    assert frequency["source_ts_geometry_sha256"] == optts["optimized_geometry_sha256"]
    assert frequency["result_manifest_sha256"] == optts["result_manifest_sha256"]
    assert len(optts["protocol_sha256"]) == 64


def test_batch_collector_rejects_geometry_binding_method_and_input_mismatch(tmp_path: Path) -> None:
    case, elements, source_geometry, source_digest = _make_batch_result(
        tmp_path, wrong_geometry_binding=True)
    with pytest.raises(ACPCLIError, match="not bound"):
        collect_batch_ts_frequency_evidence(task_root=tmp_path, item_id="ts001",
            expected_elements=elements, source_frame_id="frame:top1", source_geometry=source_geometry,
            source_geometry_sha256=source_digest, execution_id="exec:ts", attempt_id="attempt:ts")

    _make_batch_result(tmp_path)
    with pytest.raises(ACPCLIError, match="method differs"):
        collect_batch_ts_frequency_evidence(task_root=tmp_path, item_id="ts001",
            expected_elements=elements, source_frame_id="frame:top1", source_geometry=source_geometry,
            source_geometry_sha256=source_digest, execution_id="exec:ts", attempt_id="attempt:ts",
            expected_method="HF", expected_basis="def2-SVP")

    _write_xyz(tmp_path / "input.xyz", elements,
               [[xyz[0] + 0.5, xyz[1], xyz[2]] for xyz in source_geometry], "wrong source")
    with pytest.raises(ACPCLIError, match="input geometry differs"):
        collect_batch_ts_frequency_evidence(task_root=tmp_path, item_id="ts001",
            expected_elements=elements, source_frame_id="frame:top1", source_geometry=source_geometry,
            source_geometry_sha256=source_digest, execution_id="exec:ts", attempt_id="attempt:ts")


def test_batch_collector_rejects_non_single_imaginary_mode(tmp_path: Path) -> None:
    case, elements, source_geometry, source_digest = _make_batch_result(tmp_path)
    modes_path = tmp_path / "RESULT/frequencies/ts001__normal_modes.json"
    modes = json.loads(modes_path.read_text(encoding="utf-8"))
    modes["modes"][1]["frequency_cm1"] = -25.0
    modes["modes"][1]["imaginary"] = True
    modes_path.write_text(json.dumps(modes), encoding="utf-8")
    optts, frequency = collect_batch_ts_frequency_evidence(task_root=tmp_path,
        item_id="ts001", expected_elements=elements, source_frame_id="frame:top1",
        source_geometry=source_geometry, source_geometry_sha256=source_digest,
        execution_id="exec:ts", attempt_id="attempt:ts")
    assert optts["first_order_saddle"] is False
    assert frequency["status"] == "failed"
    assert frequency["imaginary_mode_count"] == 2


@pytest.mark.parametrize("corruption, message", [
    ("zero", "identically zero"),
    ("shape", "invalid coordinates"),
])
def test_batch_collector_rejects_malformed_mode_displacements(
    tmp_path: Path, corruption: str, message: str,
) -> None:
    case, elements, source_geometry, source_digest = _make_batch_result(tmp_path)
    modes_path = tmp_path / "RESULT/frequencies/ts001__normal_modes.json"
    modes = json.loads(modes_path.read_text(encoding="utf-8"))
    if corruption == "zero":
        modes["modes"][0]["vectors"] = [[0.0, 0.0, 0.0] for _ in elements]
    else:
        modes["modes"][0]["vectors"][0] = [0.0, 0.0]
    modes_path.write_text(json.dumps(modes), encoding="utf-8")
    with pytest.raises(ACPCLIError, match=message):
        collect_batch_ts_frequency_evidence(task_root=tmp_path, item_id="ts001",
            expected_elements=elements, source_frame_id="frame:top1", source_geometry=source_geometry,
            source_geometry_sha256=source_digest, execution_id="exec:ts", attempt_id="attempt:ts")


def test_collects_irc_directions_and_matches_registered_endpoint_xyz(tmp_path: Path) -> None:
    case = _case()
    elements = [atom["element"] for atom in case["atoms"]]
    optimized_file = tmp_path / "RESULT/structures/ts.xyz"
    optimized_geometry = case["product"]["geometry"]
    _write_xyz(optimized_file, elements, optimized_geometry, "optimized TS")
    ts_file_digest = hashlib.sha256(optimized_file.read_bytes()).hexdigest()
    endpoints = {
        "forward": case["reactant"]["geometry"],
        "reverse": case["product"]["geometry"],
    }
    products = [{"id": "irc_report", "path": "irc/irc_report.json", "kind": "report"}]
    report_endpoints = {}
    for direction, geometry in endpoints.items():
        relative = f"irc/irc_{direction}.xyz"
        _write_xyz(tmp_path / "RESULT" / relative, elements, geometry, f"{direction} endpoint")
        work_path = tmp_path / "WORK/07_PATH/ORCA" / f"irc_{direction}.xyz"
        _write_xyz(work_path, elements, geometry, f"{direction} endpoint")
        report_endpoints[direction] = str(work_path)
        products.append({"id": f"irc_{direction}_endpoint", "path": relative,
                         "kind": "irc_endpoint"})
    report_path = tmp_path / "RESULT/irc/irc_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({
        "workflow": "irc", "status": "completed", "directions": ["forward", "reverse"],
        "method": "B3LYP", "basis": "def2-SVP", "maxpoints": 100, "step": 0.1,
        "input_role": "transition_state", "endpoints": report_endpoints,
        "ts_source": {"schema": "irc_ts_source_v1", "geometry_sha256": ts_file_digest,
                      "method": "B3LYP", "basis": "def2-SVP", "charge": 0,
                      "multiplicity": 1},
    }), encoding="utf-8")
    _manifest(tmp_path, "irc", products)
    forward, reverse = collect_irc_evidence(task_root=tmp_path, case=case,
        optimized_ts_geometry_sha256=hashlib.sha256(
            stable_json_dumps(optimized_geometry).encode()).hexdigest(),
        optimized_ts_file_sha256=ts_file_digest, execution_id="exec:irc",
        attempt_id="attempt:irc")
    assert (forward["status"], reverse["status"]) == ("matched", "matched")
    assert {forward["endpoint_reached"], reverse["endpoint_reached"]} == {"reactant", "product"}
    assert forward["source_ts_geometry_sha256"] == reverse["source_ts_geometry_sha256"]


def test_end_to_end_synthetic_acp_artifacts_build_validation_result(tmp_path: Path) -> None:
    docs = synthetic_objects()
    case, review_record = _reviewed_case(docs["ReactionCase"])
    path = seal_document({**docs["PathBundle"], "status": "usable"})
    proposal = rank_path_bundle(path, rule="highest_scan_energy", top_k=3)
    selected = proposal["selected_frames"][0]
    frame = next(row for row in path["frames"] if row["frame_id"] == selected["frame_id"])

    batch_root = tmp_path / "batch"
    _make_batch_result(batch_root, source_geometry=frame["geometry"])
    ts_file = batch_root / "RESULT/structures/ts001__TAG_TS__optimized.xyz"
    ts_file_digest = hashlib.sha256(ts_file.read_bytes()).hexdigest()
    irc_root = tmp_path / "irc"
    _make_irc_result(irc_root, case, ts_file_digest)

    result = collect_acp_validation_result(
        validation_id="validation:synthetic-acp-e2e", case=case, review_record=review_record, path=path,
        proposal=proposal, batch_task_root=batch_root, batch_item_id="ts001",
        batch_execution_id="exec:batch", batch_attempt_id="attempt:batch",
        irc_task_root=irc_root, irc_execution_id="exec:irc", irc_attempt_id="attempt:irc",
        expected_method="B3LYP", expected_basis="def2-SVP")
    assert result["status"] == "passed"
    assert result["source_frame_id"] == selected["frame_id"]
    assert result["optts"]["source_frame_id"] == selected["frame_id"]
    assert result["frequency"]["source_ts_geometry_sha256"] == result["optts"]["optimized_geometry_sha256"]
    assert result["irc_forward"]["source_ts_geometry_sha256"] == result["optts"]["optimized_geometry_sha256"]
    assert result["irc_reverse"]["source_ts_geometry_sha256"] == result["optts"]["optimized_geometry_sha256"]

    wrong_endpoint = [row[:] for row in case["reactant"]["geometry"]]
    wrong_endpoint[1][0] += 4.0
    for path_ref in (irc_root / "RESULT/irc/irc_forward.xyz",
                     irc_root / "WORK/07_PATH/ORCA/irc_forward.xyz"):
        _write_xyz(path_ref, [atom["element"] for atom in case["atoms"]],
                   wrong_endpoint, "unmatched endpoint")
    failed = collect_acp_validation_result(
        validation_id="validation:synthetic-acp-fail", case=case, review_record=review_record, path=path,
        proposal=proposal, batch_task_root=batch_root, batch_item_id="ts001",
        batch_execution_id="exec:batch", batch_attempt_id="attempt:batch",
        irc_task_root=irc_root, irc_execution_id="exec:irc-fail", irc_attempt_id="attempt:irc-fail",
        expected_method="B3LYP", expected_basis="def2-SVP")
    assert failed["status"] == "failed"
    assert failed["irc_forward"]["status"] == "failed"


def test_acp_collect_validation_cli_writes_result(tmp_path: Path) -> None:
    from pes2ts_core.cli import main

    docs = synthetic_objects()
    case, review_record = _reviewed_case(docs["ReactionCase"])
    path = seal_document({**docs["PathBundle"], "status": "usable"})
    proposal = rank_path_bundle(path, rule="highest_scan_energy", top_k=3)
    selected = proposal["selected_frames"][0]
    frame = next(row for row in path["frames"] if row["frame_id"] == selected["frame_id"])
    batch_root = tmp_path / "batch"
    _make_batch_result(batch_root, source_geometry=frame["geometry"])
    ts_file = batch_root / "RESULT/structures/ts001__TAG_TS__optimized.xyz"
    irc_root = tmp_path / "irc"
    _make_irc_result(irc_root, case, hashlib.sha256(ts_file.read_bytes()).hexdigest())

    input_paths = {}
    for name, document in (("case", case), ("review", review_record),
                           ("path", path), ("proposal", proposal)):
        input_path = tmp_path / f"{name}.json"
        input_path.write_text(dumps_document(document), encoding="utf-8")
        input_paths[name] = input_path
    output = tmp_path / "ValidationResult.json"
    code = main([
        "acp-collect-validation", "--case", str(input_paths["case"]),
        "--review-record", str(input_paths["review"]), "--path", str(input_paths["path"]),
        "--proposal", str(input_paths["proposal"]), "--batch-result", str(batch_root),
        "--batch-item", "ts001", "--batch-execution-id", "exec:batch",
        "--batch-attempt-id", "attempt:batch", "--irc-result", str(irc_root),
        "--irc-execution-id", "exec:irc", "--irc-attempt-id", "attempt:irc",
        "--method", "B3LYP", "--basis", "def2-SVP", "--validation-id", "validation:cli",
        "--output", str(output),
    ])
    assert code == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["schema_name"] == "ValidationResult"
    assert result["status"] == "passed"


def test_acp_validation_runner_enforces_review_gate_before_creating_attempts(tmp_path: Path) -> None:
    from pes2ts_core.integration.acp.stage_cli import ACPValidationCLIBackend

    docs = synthetic_objects()
    case, review_record = _reviewed_case(docs["ReactionCase"])
    path = seal_document({**docs["PathBundle"], "status": "usable"})
    proposal = rank_path_bundle(path, rule="highest_scan_energy", top_k=3)
    review_record = seal_document({**review_record, "status": "rejected",
        "adjudication": {**review_record["adjudication"], "decision": "rejected"}})
    fake_acp = tmp_path / "acp"
    (fake_acp / "src/acp").mkdir(parents=True)
    (fake_acp / "src/acp/cli.py").write_text("", encoding="utf-8")
    backend = ACPValidationCLIBackend(acp_root=fake_acp)
    with pytest.raises(ACPCLIError, match="accepted human review"):
        backend.run_validation(case=case, review_record=review_record, path=path,
            proposal=proposal, source_frame_id=proposal["selected_frames"][0]["frame_id"],
            expected_method="B3LYP", expected_basis="def2-SVP",
            validation_id="validation:gate", output_root=tmp_path / "attempts", batch_execution_id="exec-batch",
            batch_attempt_id="attempt-batch", irc_execution_id="exec-irc",
            irc_attempt_id="attempt-irc", batch_timeout_seconds=60,
            irc_timeout_seconds=60)
    assert not (tmp_path / "attempts").exists()


def test_fake_acp_runs_full_optts_frequency_irc_chain(tmp_path: Path, monkeypatch) -> None:

    docs = synthetic_objects()
    case, review_record = _reviewed_case(docs["ReactionCase"])
    path = seal_document({**docs["PathBundle"], "status": "usable"})
    proposal = rank_path_bundle(path, rule="highest_scan_energy", top_k=3)
    source_frame_id = proposal["selected_frames"][0]["frame_id"]
    case_file = tmp_path / "case.json"
    case_file.write_text(dumps_document(case), encoding="utf-8")

    fake_root = tmp_path / "fake_acp"
    fake_package = fake_root / "src/acp"
    fake_package.mkdir(parents=True)
    (fake_package / "__init__.py").write_text("", encoding="utf-8")
    (fake_package / "cli.py").write_text(r'''import hashlib, json, os, shutil, sys
from pathlib import Path
args = sys.argv[1:]
workflow = args[1]
def arg(key): return args[args.index(key) + 1]
root = Path(arg("--output"))
result = root / "RESULT"
result.mkdir(parents=True, exist_ok=True)
def manifest(name, products):
    (result / "result_manifest.json").write_text(json.dumps({
        "version": 2, "workflow": name, "status": "completed", "products": products}))
def xyz(path, symbols, coords, comment):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [str(len(symbols)), comment]
    rows += [f"{s} {p[0]:.10f} {p[1]:.10f} {p[2]:.10f}" for s,p in zip(symbols,coords)]
    path.write_text("\n".join(rows) + "\n")
if workflow == "BatchOptimize":
    req = json.loads(Path(arg("--items-file")).read_text())
    item = req["items"][0]
    input_path = Path(item["geometry"])
    shutil.copyfile(input_path, root / "input.xyz")
    lines = input_path.read_text().splitlines()
    symbols = [r.split()[0] for r in lines[2:]]
    coords = [[float(v) for v in r.split()[1:4]] for r in lines[2:]]
    structure_id = "batch_" + item["id"]
    mode_id = structure_id + "_normal_modes"
    optimized_rel = f"structures/{item['id']}__TAG_TS__optimized.xyz"
    modes_rel = f"frequencies/{item['id']}__normal_modes.json"
    xyz(result / optimized_rel, symbols, coords, "TS optimized")
    mode_count = 3 * len(symbols) - 5
    modes = [{"mode_index": i, "frequency_cm1": -120.0 if i == 0 else 25.0 + i,
              "imaginary": i == 0,
              "vectors": [[1.0,0.0,0.0]] + [[0.0,0.0,0.0] for _ in symbols[1:]]}
             for i in range(mode_count)]
    (result / modes_rel).parent.mkdir(parents=True, exist_ok=True)
    (result / modes_rel).write_text(json.dumps({"schema_version":"normal_modes_v1",
        "atom_count":len(symbols), "warnings":[], "modes":modes}))
    (result / "batch_provenance.json").write_text(json.dumps({"workflow":"BatchOptimize",
        "items":[{"item_id":item["id"], "role":"ts", "effective_config":{
            "method":arg("--transition-state-method"),
            "basis":arg("--transition-state-basis")}}]}))
    manifest("BatchOptimize", [
        {"id":structure_id,"kind":"structure","path":optimized_rel},
        {"id":mode_id,"kind":"frequency_modes","path":modes_rel,
         "metadata":{"geometry_product_id":structure_id}}])
elif workflow == "irc":
    case = json.loads(Path(os.environ["PES2TS_FAKE_CASE"]).read_text())
    proof = json.loads(Path(arg("--ts-provenance")).read_text())
    symbols = [a["element"] for a in case["atoms"]]
    endpoints = {"forward":case["reactant"]["geometry"],
                 "reverse":case["product"]["geometry"]}
    products = [{"id":"irc_report","kind":"report","path":"irc/irc_report.json"}]
    report_endpoints = {}
    for direction, coords in endpoints.items():
        rel = f"irc/irc_{direction}.xyz"
        xyz(result / rel, symbols, coords, direction)
        work = root / "WORK/07_PATH/ORCA" / f"irc_{direction}.xyz"
        xyz(work, symbols, coords, direction)
        report_endpoints[direction] = str(work)
        products.append({"id":f"irc_{direction}_endpoint","kind":"irc_endpoint","path":rel})
    report = {"workflow":"irc","status":"completed","directions":["forward","reverse"],
        "method":proof["method"],"basis":proof["basis"],"maxpoints":int(arg("--maxpoints")),
        "step":float(arg("--step")),"input_role":"transition_state",
        "endpoints":report_endpoints,"ts_source":proof}
    (result / "irc").mkdir(exist_ok=True)
    (result / "irc/irc_report.json").write_text(json.dumps(report))
    manifest("irc", products)
''', encoding="utf-8")

    monkeypatch.setenv("PES2TS_FAKE_CASE", str(case_file))
    input_paths = {}
    for name, document in (("review", review_record), ("path", path),
                           ("proposal", proposal)):
        input_path = tmp_path / f"{name}.json"
        input_path.write_text(dumps_document(document), encoding="utf-8")
        input_paths[name] = input_path
    output_root = tmp_path / "results"
    output_file = tmp_path / "ValidationResult.json"
    from pes2ts_core.cli import main
    code = main(["acp-validate-run", "--case", str(case_file),
        "--review-record", str(input_paths["review"]), "--path", str(input_paths["path"]),
        "--proposal", str(input_paths["proposal"]), "--source-frame-id", source_frame_id,
        "--acp-root", str(fake_root), "--python", sys.executable,
        "--output-root", str(output_root), "--batch-execution-id", "exec-batch",
        "--batch-attempt-id", "attempt-batch", "--irc-execution-id", "exec-irc",
        "--irc-attempt-id", "attempt-irc", "--method", "B3LYP", "--basis", "def2-SVP",
        "--batch-timeout", "20", "--irc-timeout", "20",
        "--validation-id", "validation:fake-chain", "--output", str(output_file)])
    assert code == 0
    result = json.loads(output_file.read_text(encoding="utf-8"))
    assert result["status"] == "passed"
    assert result["source_frame_id"] == source_frame_id
    assert result["frequency"]["imaginary_mode_count"] == 1
    assert result["irc_forward"]["status"] == "matched"
    assert result["irc_reverse"]["status"] == "matched"
    run_receipts = result["extensions"]["pes2ts.acp_stage_receipts.v1"]
    assert [row["workflow"] for row in run_receipts["attempts"]] == ["BatchOptimize", "irc"]
    assert all(row["wall_seconds"] >= 0 and row["result_manifest_sha256"]
               for row in run_receipts["attempts"])
    assert run_receipts["cpu_time_available"] is False
    assert (output_root / "batch/attempts/attempt-batch/WORK/pes2ts/stage_cli_receipt.json").is_file()
    assert (output_root / "irc/attempts/attempt-irc/WORK/pes2ts/stage_cli_receipt.json").is_file()


def test_irc_collector_rejects_report_source_mismatch_and_path_escape(tmp_path: Path) -> None:
    case = _case()
    elements = [atom["element"] for atom in case["atoms"]]
    endpoint = case["reactant"]["geometry"]
    product_file = "irc/irc_forward.xyz"
    _write_xyz(tmp_path / "RESULT" / product_file, elements, endpoint, "endpoint")
    work_file = tmp_path / "WORK/forward.xyz"
    _write_xyz(work_file, elements, endpoint, "endpoint")
    _write_xyz(tmp_path / "RESULT/irc/irc_reverse.xyz", elements,
               case["product"]["geometry"], "reverse endpoint")
    reverse_work = tmp_path / "WORK/reverse.xyz"
    _write_xyz(reverse_work, elements, case["product"]["geometry"], "reverse endpoint")
    report = tmp_path / "RESULT/irc/irc_report.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({"workflow": "irc", "status": "completed",
        "directions": ["forward", "reverse"], "method": "B3LYP", "basis": "",
        "maxpoints": 100, "step": 0.1,
        "endpoints": {"forward": str(work_file), "reverse": str(reverse_work)},
        "ts_source": {"schema": "irc_ts_source_v1", "geometry_sha256": "x" * 64}}),
        encoding="utf-8")
    _manifest(tmp_path, "irc", [
        {"id": "irc_report", "path": "irc/irc_report.json", "kind": "report"},
        {"id": "irc_forward_endpoint", "path": product_file, "kind": "irc_endpoint"},
        {"id": "irc_reverse_endpoint", "path": "irc/irc_reverse.xyz", "kind": "irc_endpoint"},
    ])
    with pytest.raises(ACPCLIError, match="source proof"):
        collect_irc_evidence(task_root=tmp_path, case=case,
            optimized_ts_geometry_sha256="a" * 64, optimized_ts_file_sha256="b" * 64,
            execution_id="exec:irc", attempt_id="attempt:irc")


def test_endpoint_alignment_does_not_allow_mirror_reflections():
    tetrahedron = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    reflected = [[-point[0], point[1], point[2]] for point in tetrahedron]
    assert _pair_distance_rms(tetrahedron, reflected) == pytest.approx(0.0)
    assert _aligned_rmsd(tetrahedron, reflected) > 0.35
