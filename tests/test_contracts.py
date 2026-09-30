from __future__ import annotations

import copy
import json
import subprocess
import sys

import pytest

from pes2ts_core.contracts import ContractError, dumps_document, loads_document, make_document, seal_document, validate_artifact_ref, validate_document
from pes2ts_core.integration.acp.adapter import (ACPMappingError, acp_execution_to_record, acp_result_to_path_bundle,
    acp_s2_profile_to_path_bundle, build_acp_result_manifest,
    scan_plan_to_acp_job_payload, scan_plan_to_acp_request)
from pes2ts_core.planning import build_minimal_scan_plan, build_scan_plan_with_rejection
from pes2ts_core.g1.reaction_case import reaction_case_from_g1_export
from pes2ts_core.g1.export_sanitize import ExportSanitizeError, sanitize_export_document
from pes2ts_core.ranking import rank_path_bundle
from pes2ts_core.demo import synthetic_objects, write_synthetic_bundle
from pes2ts_core.viewer import render_path_viewer


def test_cli_import_does_not_eagerly_load_the_optional_drfp_encoder():
    result = subprocess.run([sys.executable, "-c",
        "import sys, pes2ts_core.cli; assert 'drfp' not in sys.modules"],
        check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def simple_case():
    return make_document("ReactionCase", "case:rxn-demo-v1", "ready",
        dataset_version="synthetic-v1", reaction_id="RXN_DEMO_0001", case_id="case:rxn-demo-v1", split="train",
        atoms=[{"atom_map_id": 2, "element": "C"}, {"atom_map_id": 7, "element": "H"}, {"atom_map_id": 12, "element": "N"}],
        reactant={"charge": 0, "multiplicity": 1, "geometry": [[0,0,0],[1.1,0,0],[3.5,0,0]]},
        product={"charge": 0, "multiplicity": 1, "geometry": [[0,0,0],[1.8,0,0],[2.8,0,0]]},
        edits=[{"kind":"broken", "atom_map_ids":[2,7]}, {"kind":"formed", "atom_map_ids":[7,12]}],
        hydrogen_transfers=[{"hydrogen_map_id":7, "old_partner_map_id":2, "new_partner_map_id":12}],
        source={"dataset":"synthetic", "mapping_provenance":"endpoint_only"})


def test_contract_round_trip_preserves_atom_identity_and_plan():
    case = simple_case()
    case["edits"].append({"kind":"order_changed", "atom_map_ids":[2, 12]})
    case = seal_document(case)
    plan = build_minimal_scan_plan(case)
    restored = loads_document(dumps_document(plan))
    assert restored["atom_map_ids"] == [2, 7, 12]
    candidate = restored["candidates"][0]
    assert candidate["coordinates"][0]["atom_map_ids"] == [7, 12]
    assert candidate["coordinates"][0]["atom_indices"] == [1, 2]
    assert candidate["observer_pairs_map"] == [[2, 7], [2, 12]]
    request = scan_plan_to_acp_request(case, restored)
    assert request["metadata"]["atom_map_ids"] == [2, 7, 12]
    assert request["scan_request"]["coordinate"] == {"kind":"distance", "atoms":[1,2], "unit":"angstrom",
        "start":1.0, "end":2.4, "n_points":9}
    assert request["scan_request"]["source"]["source_type"] == "xyz_text"
    assert request["scan_request"]["protocol"]["scan_driver"]["software"] == "xtb"
    assert request["method_levels"]["scan_coordinate"]["scan_coordinate_points"] == 9
    assert request["method_levels"]["scan_driver"]["scan_retry_count"] == 1
    assert request["method_levels"]["scan_optimizer"]["scan_optimizer_method"] == "GFN2-xTB"
    assert request["metadata"]["retry_policy"]["reuse_previous_geometry"] is True
    job = scan_plan_to_acp_job_payload(case, restored, resources={"nproc": 1, "mem_gb": 4})
    assert job["workflow"] == "PESsearch"
    assert job["method"]["schema_id"] == "pes_scan"
    assert job["method"]["levels"] == request["method_levels"]
    assert set(job["input"]) == {"source", "coordinate", "protocol"}
    assert "output_dir" not in job  # ACP remains the owner of WORK/RESULT
    assert "max_wall_seconds" in job["remark"]
    manifest = build_acp_result_manifest(task_id="job-1", workflow="PESsearch", status="completed",
        products=[{"id":"path","label":"Path bundle","path":"pes2ts/path_bundle.json","kind":"trajectory"}])
    assert manifest["version"] == 2 and manifest["products"][0]["path"] == "pes2ts/path_bundle.json"
    with pytest.raises(ACPMappingError):
        build_acp_result_manifest(task_id="job-1", workflow="PESsearch", status="completed",
            products=[{"id":"escape","label":"bad","path":"../outside.xyz","kind":"file"}])


def test_acp_request_cli_exports_a_preflighted_non_submitted_payload(tmp_path):
    case = simple_case()
    plan = build_minimal_scan_plan(case)
    case_path = tmp_path / "case.json"
    plan_path = tmp_path / "plan.json"
    resources_path = tmp_path / "resources.json"
    output_path = tmp_path / "acp_request.json"
    case_path.write_text(dumps_document(case), encoding="utf-8")
    plan_path.write_text(dumps_document(plan), encoding="utf-8")
    resources_path.write_text('{"nproc":1,"mem_gb":4}', encoding="utf-8")

    result = subprocess.run([sys.executable, "bin/pes2ts", "acp-request",
        "--case", str(case_path), "--plan", str(plan_path),
        "--resources", str(resources_path), "--output", str(output_path)],
        check=False, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["workflow"] == "PESsearch"
    assert payload["input"]["source"]["source_type"] == "xyz_text"
    assert payload["input"]["coordinate"] == {
        "kind": "distance", "atoms": [1, 2], "unit": "angstrom",
        "start": 1.0, "end": 2.4, "n_points": 9}
    assert "not submitted" in result.stdout


def test_truth_fields_nonfinite_and_bad_indices_are_rejected():
    case = simple_case()
    leaked = copy.deepcopy(case)
    leaked["status_summary"] = {"orientation": "R_to_P"}
    assert any("forbidden" in issue for issue in validate_document(leaked, production_input=True))
    plan = build_minimal_scan_plan(case)
    plan["candidates"][0]["coordinates"][0]["atom_indices"] = [1, 3]
    assert any("out of bounds" in issue for issue in validate_document(plan))
    bad = copy.deepcopy(plan)
    bad["candidates"][0]["coordinates"][0]["points"][2] = float("nan")
    assert any("non-finite" in issue for issue in validate_document(bad))
    assert any("unknown field" in issue for issue in validate_document({**case, "unexpected":1}))


def test_scan_plan_identity_covers_frozen_method_and_coordinate_policy():
    case = simple_case()
    baseline = build_minimal_scan_plan(case)
    repeated = build_minimal_scan_plan(case)
    longer = build_minimal_scan_plan(case, n_points=11)
    alternate_method = build_minimal_scan_plan(case, method={
        "engine": "xtb", "method": "GFN1-xTB", "basis": None,
        "solvent": None, "engine_version": None, "parameter_sha256": None,
    })

    assert baseline["plan_id"] == repeated["plan_id"]
    assert baseline["plan_id"] != longer["plan_id"]
    assert baseline["plan_id"] != alternate_method["plan_id"]
    changed_geometry = seal_document({**case, "reactant": {**case["reactant"],
        "geometry": [[0.1, 0, 0], *case["reactant"]["geometry"][1:]]}})
    changed_plan = build_minimal_scan_plan(changed_geometry)
    assert baseline["plan_id"] != changed_plan["plan_id"]
    assert baseline["candidates"][0]["request_id"] != changed_plan["candidates"][0]["request_id"]
    candidate = baseline["candidates"][0]
    assert baseline["plan_frozen"] is True
    assert candidate["direction"] == "P_to_R"
    assert candidate["coordinates"][0]["points"][0] == 1.0
    assert candidate["coordinates"][0]["points"][-1] == 2.4
    assert candidate["retry_policy"] == {
        "failure_policy": "retry_previous", "scan_retry_count": 1,
        "optimizer_retries": 1, "reuse_previous_geometry": True,
    }
    assert candidate["budget"] == {"max_attempts": 1, "max_cpu_hours": 8.0, "max_wall_seconds": 14400}


def test_acp_method_mapping_preserves_supported_level_fields_and_refuses_unknowns():
    case = simple_case()
    method = {"engine":"orca", "engine_version":"6.0", "method":"B3LYP",
              "basis":"def2-TZVP", "dispersion":"D3BJ", "solvent_model":"CPCM",
              "solvent":"water", "parameter_sha256":"a" * 64}
    plan = build_minimal_scan_plan(case, method=method)
    request = scan_plan_to_acp_request(case, plan)
    optimizer = request["scan_request"]["protocol"]["scan_optimizer"]
    assert optimizer["method"] == "B3LYP"
    assert optimizer["basis"] == "def2-TZVP"
    assert optimizer["solvent_model"] == "CPCM" and optimizer["solvent"] == "water"
    assert optimizer["dispersion"] == "D3BJ"
    assert request["metadata"]["planned_method"] == method
    assert request["method_levels"]["scan_optimizer"]["scan_optimizer_basis"] == "def2-TZVP"

    unsupported = build_minimal_scan_plan(case, method={**method, "unmapped_field":"must-not-drop"})
    with pytest.raises(ACPMappingError, match="cannot be silently dropped"):
        scan_plan_to_acp_request(case, unsupported)


def test_scientific_digest_ignores_volatile_metadata():
    document = simple_case()
    changed = {**document, "created_at":"2030-01-01T00:00:00+00:00", "producer":{"name":"other"}}
    assert seal_document(document)["content_sha256"] == seal_document(changed)["content_sha256"]


def test_path_bundle_keeps_frame_ids_and_separates_energy_channels():
    case = simple_case()
    plan = build_minimal_scan_plan(case)
    frames = [
        {"frame_id":"native-10", "acp_frame_id":"native-10", "atom_map_ids":[2,7,12],
         "geometry_angstrom":case["product"]["geometry"], "scan_energy_hartree":-10.2,
         "scan_method_id":"xtb:gfn2", "refined_energy_hartree":-10.3,
         "refined_method_id":"orca:sp-b3lyp", "converged":True},
        {"frame_id":"native-11", "acp_frame_id":"native-11", "atom_map_ids":[2,7,12],
         "geometry_angstrom":case["product"]["geometry"], "scan_energy_hartree":-10.1,
         "scan_method_id":"xtb:gfn2", "refined_energy_hartree":None,
         "refined_method_id":"orca:sp-b3lyp", "converged":True},
    ]
    bundle = acp_result_to_path_bundle(case=case, plan=plan, execution_id="exec-1", acp_task_id="job-1", frames=frames)
    readback = loads_document(dumps_document(bundle))
    assert [x["frame_id"] for x in readback["frames"]] == ["native-10", "native-11"]
    assert readback["frames"][0]["energies"]["scan_electronic"]["method_id"] == "xtb:gfn2"
    assert readback["frames"][0]["energies"]["refined_electronic"]["method_id"] == "orca:sp-b3lyp"
    proposal = rank_path_bundle(readback)
    assert proposal["selected_frames"][0]["frame_id"] == "native-11"
    assert proposal["selection_source"] == "ranking"
    assert proposal["status"] == "needs_review" and proposal["path_status"] == "unchecked"
    bad = copy.deepcopy(bundle)
    bad["frames"][1]["energies"]["scan_electronic"]["method_id"] = "other-method"
    assert any("mixed methods" in issue for issue in validate_document(bad))
    bad_order = copy.deepcopy(bundle)
    bad_order["frames"][0]["atom_map_ids"] = [12, 7, 2]
    assert any("order must exactly match" in issue for issue in validate_document(bad_order))
    bad_geometry = copy.deepcopy(bundle)
    bad_geometry["frames"][0]["geometry"] = [["0", 0, 0], [1, 0, 0], [2, 0]]
    assert any("finite N×3" in issue for issue in validate_document(bad_geometry))


def test_artifact_refs_reject_windows_parent_traversal():
    assert validate_artifact_ref({"artifact_id":"x", "media_type":"chemical/x-xyz",
        "relative_path":r"..\outside.xyz", "sha256":"a" * 64})


def test_validation_pass_requires_saddle_frequency_and_bidirectional_irc_evidence():
    evidence = {
        "reaction_id": "RXN_SYNTH_VALIDATION", "case_id": "case:validation",
        "path_id": "path:validation", "proposal_id": "proposal:validation",
        "path_content_sha256": "a" * 64, "proposal_content_sha256": "b" * 64,
        "source_frame_id": "frame:initial-guess", "source_geometry_sha256": "c" * 64,
            "optts": {"status": "converged", "source_frame_id": "frame:initial-guess",
                      "execution_id": "exec:optts", "attempt_id": "attempt:optts",
                      "first_order_saddle": True, "optimized_geometry_sha256":"d" * 64,
                      "protocol_sha256":"e" * 64, "result_manifest_sha256":"a" * 64},
            "frequency": {"status": "passed", "imaginary_mode_count": 1,
                          "execution_id": "exec:frequency", "attempt_id": "attempt:frequency",
                          "source_ts_geometry_sha256":"d" * 64, "protocol_sha256":"f" * 64,
                          "result_manifest_sha256":"b" * 64},
            "irc_forward": {"status": "matched", "endpoint_reached": "reactant",
                             "execution_id": "exec:irc-forward", "attempt_id": "attempt:irc-forward",
                             "source_ts_geometry_sha256":"d" * 64, "protocol_sha256":"1" * 64,
                             "result_manifest_sha256":"c" * 64},
            "irc_reverse": {"status": "matched", "endpoint_reached": "product",
                             "execution_id": "exec:irc-reverse", "attempt_id": "attempt:irc-reverse",
                             "source_ts_geometry_sha256":"d" * 64, "protocol_sha256":"2" * 64,
                             "result_manifest_sha256":"d" * 64},
        "validation_note": "synthetic contract test",
    }
    passed = make_document("ValidationResult", "validation:strict-pass", "passed", **evidence)
    assert not validate_document(passed)
    unbound = {key: value for key, value in evidence.items()
               if key not in {"path_content_sha256", "proposal_content_sha256", "source_frame_id", "source_geometry_sha256"}}
    with pytest.raises(ContractError, match="bind passed validation"):
        make_document("ValidationResult", "validation:unbound", "passed", **unbound)

    false_saddle = copy.deepcopy(evidence)
    false_saddle["optts"]["first_order_saddle"] = False
    with pytest.raises(ContractError, match="first-order saddle"):
        make_document("ValidationResult", "validation:false-saddle", "passed", **false_saddle)

    wrong_irc = copy.deepcopy(evidence)
    wrong_irc["irc_reverse"]["endpoint_reached"] = "reactant"
    with pytest.raises(ContractError, match="distinct R/P endpoints"):
        make_document("ValidationResult", "validation:wrong-irc", "passed", **wrong_irc)

    missing_cli_attempt = copy.deepcopy(evidence)
    del missing_cli_attempt["irc_reverse"]["attempt_id"]
    with pytest.raises(ContractError, match="immutable attempt ID"):
        make_document("ValidationResult", "validation:missing-attempt", "passed", **missing_cli_attempt)


def test_minimal_plan_rejects_unsupported_coupled_bond_edits():
    case = simple_case()
    case["edits"].append({"kind":"formed", "atom_map_ids":[2, 12]})
    case = seal_document(case)
    with pytest.raises(ContractError, match="additional bond formation/breaking"):
        build_minimal_scan_plan(case)
    refusal = build_scan_plan_with_rejection(case)
    assert refusal["status"] == "rejected"
    assert refusal["candidates"] == []
    assert refusal["reject_reasons"] == ["H transfer is coupled to additional bond formation/breaking; use a complex-path strategy"]
    assert not validate_document(refusal)


def test_acp_native_scanframe_geometry_uses_injected_resolver():
    case = simple_case()
    plan = build_minimal_scan_plan(case)
    native_frame = {"index":4, "target_coordinate":2.1, "actual_coordinate":2.08,
        "coordinate_unit":"angstrom", "geometry_path":"pes_search/scan/frame_0004.xyz",
        "scan_energy_hartree":-20.1, "single_point_energy_hartree":-20.2,
        "optimization_converged":True, "single_point_status":"completed",
        "optimizer_level":{"method":"GFN2-xTB"}, "retry_history":[]}
    bundle = acp_result_to_path_bundle(case=case, plan=plan, execution_id="exec-native", acp_task_id="job-native",
        frames=[native_frame], refined_method_id="orca:B97-3c",
        geometry_loader=lambda path: case["product"]["geometry"] if path == native_frame["geometry_path"] else None)
    frame = bundle["frames"][0]
    assert frame["frame_id"] == "exec-native:f00004"
    assert frame["source_acp_frame_id"] == 4
    assert frame["actual_coordinate"] == 2.08
    assert frame["energies"]["refined_electronic"]["method_id"] == "orca:B97-3c"


def test_acp_s2_profile_and_frame_geometries_project_to_review_path_bundle():
    case = simple_case()
    plan = build_minimal_scan_plan(case)
    profile = {
        "job_id": "job-s2-profile", "mode": "bond_length_scan", "status": "completed",
        "stationary_point_claimed": False,
        "frames": [
            {"index": 0, "target_coordinate": 1.0, "actual_coordinate": 1.02,
             "geometry_path": "pes_search/scan/frame_0000.xyz", "scan_energy_hartree": -20.0,
             "single_point_energy_hartree": -20.1, "optimization_converged": True},
            {"index": 1, "target_coordinate": 1.175, "actual_coordinate": 1.18,
             "geometry_path": "pes_search/scan/frame_0001.xyz", "scan_energy_hartree": -19.8,
             "single_point_energy_hartree": None, "optimization_converged": True},
        ],
    }
    geometries = {0: case["reactant"]["geometry"], 1: case["product"]["geometry"]}

    bundle = acp_s2_profile_to_path_bundle(case=case, plan=plan,
        execution_id="exec-s2-profile", acp_task_id="job-s2-profile", profile=profile,
        geometry_angstrom_by_index=geometries, refined_method_id="orca:B97-3c")

    assert bundle["status"] == "needs_review"
    assert bundle["acp_task_id"] == "job-s2-profile"
    assert [frame["source_acp_frame_id"] for frame in bundle["frames"]] == [0, 1]
    assert bundle["frames"][0]["frame_id"] == "job-s2-profile:frame:00000"
    assert bundle["frames"][0]["geometry_ref"] == "RESULT/pes_search/scan/frame_0000.xyz"
    assert bundle["frames"][0]["energies"]["refined_electronic"]["method_id"] == "orca:B97-3c"
    assert "refined_electronic" not in bundle["frames"][1]["energies"]
    assert not validate_document(bundle)

    with pytest.raises(ACPMappingError, match="does not match acp_task_id"):
        acp_s2_profile_to_path_bundle(case=case, plan=plan,
            execution_id="exec-wrong-task", acp_task_id="job-wrong", profile=profile,
            geometry_angstrom_by_index=geometries)
    with pytest.raises(ACPMappingError, match="exactly match profile frame indices"):
        acp_s2_profile_to_path_bundle(case=case, plan=plan,
            execution_id="exec-missing-frame", acp_task_id="job-s2-profile", profile=profile,
            geometry_angstrom_by_index={0: geometries[0]})


def test_acp_result_path_bundle_preserves_the_executed_candidate_identity_and_method():
    case = simple_case()
    plan = build_minimal_scan_plan(case)
    first = plan["candidates"][0]
    second = copy.deepcopy(first)
    second.update({"candidate_id":f"{plan['plan_id']}:c002", "request_id":f"{plan['plan_id']}:c002:attempt1"})
    second["method"] = {**first["method"], "method":"GFN1-xTB"}
    plan = seal_document({**plan, "candidates":[first, second]})

    bundle = acp_result_to_path_bundle(case=case, plan=plan, execution_id="exec-c002",
        acp_task_id="job-c002", candidate_id=second["candidate_id"], frames=[{
            "geometry_angstrom":case["product"]["geometry"], "scan_energy_hartree":-12.3,
            "optimizer_level":{"engine":"xtb", "method":"GFN1-xTB"}, "optimization_converged":True,
        }])
    assert bundle["candidate_id"] == second["candidate_id"]
    assert bundle["frames"][0]["energies"]["scan_electronic"]["method_id"] == "xtb:GFN1-xTB"

    unknown_method = acp_result_to_path_bundle(case=case, plan=plan, execution_id="exec-unknown-method",
        acp_task_id="job-unknown-method", candidate_id=second["candidate_id"], frames=[{
            "geometry_angstrom":case["product"]["geometry"], "scan_energy_hartree":-12.3,
            "optimization_converged":True,
        }])
    assert unknown_method["frames"][0]["energies"]["scan_electronic"]["method_id"] is None
    assert not validate_document(bundle)

    with pytest.raises(ACPMappingError, match="unknown candidate_id"):
        acp_result_to_path_bundle(case=case, plan=plan, execution_id="exec-bad",
            acp_task_id="job-bad", candidate_id="missing", frames=[])

    rejected_case = seal_document({**case, "edits":[{"kind":"order_changed", "atom_map_ids":[2, 12]}],
                                   "hydrogen_transfers":[]})
    rejected = build_scan_plan_with_rejection(rejected_case)
    with pytest.raises(ACPMappingError, match="ready ScanPlan"):
        acp_result_to_path_bundle(case=rejected_case, plan=rejected, execution_id="exec-rejected",
            acp_task_id="job-rejected", frames=[])


def test_acp_execution_record_preserves_task_state_retries_costs_and_identity():
    case = simple_case()
    plan = build_minimal_scan_plan(case)
    candidate = plan["candidates"][0]
    record = acp_execution_to_record(case=case, plan=plan, candidate_id=candidate["candidate_id"],
        execution_id="exec-retry", acp_task_id="job-retry", task_status="completed",
        attempts=[
            {"acp_attempt_id":"native-1", "status":"failed", "failure_code":"scan_timeout", "cpu_seconds":12},
            {"acp_attempt_id":"native-2", "status":"completed", "cpu_seconds":48},
        ], work_ref="WORK/attempts", result_ref="RESULT/path_bundle.json")
    assert record["status"] == "completed"
    assert record["candidate_id"] == candidate["candidate_id"]
    assert record["request_id"] == candidate["request_id"]
    assert [attempt["status"] for attempt in record["attempts"]] == ["failed", "completed"]
    assert record["failure_retained"] is True
    assert record["total_cpu_seconds"] == 60
    assert record["cost_complete"] is True
    assert not validate_document(record)
    bad_total = seal_document({**record, "total_cpu_seconds":61})
    assert any("sum of attempt CPU costs" in issue for issue in validate_document(bad_total))

    with pytest.raises(ACPMappingError, match="unsupported ACP task status"):
        acp_execution_to_record(case=case, plan=plan, candidate_id=candidate["candidate_id"],
            execution_id="exec-unknown", acp_task_id="job-unknown", task_status="mystery",
            attempts=[], work_ref="WORK/attempts", result_ref="RESULT/path_bundle.json")

    with pytest.raises(ACPMappingError, match="beneath RESULT"):
        acp_execution_to_record(case=case, plan=plan, candidate_id=candidate["candidate_id"],
            execution_id="exec-path", acp_task_id="job-path", task_status="queued",
            attempts=[], work_ref="WORK/attempts", result_ref="RESULT/../outside")

    partial_cost = acp_execution_to_record(case=case, plan=plan, candidate_id=candidate["candidate_id"],
        execution_id="exec-cost-unknown", acp_task_id="job-cost-unknown", task_status="completed",
        attempts=[{"status":"completed", "cpu_seconds":None}],
        work_ref="WORK/attempts", result_ref="RESULT/path_bundle.json")
    assert partial_cost["cost_complete"] is False
    assert partial_cost["total_cpu_seconds"] is None
    assert not validate_document(partial_cost)


def test_acp_adapter_rejects_plan_case_mismatch_and_artifact_path_traversal():
    case = simple_case()
    plan = build_minimal_scan_plan(case)
    with pytest.raises(ACPMappingError):
        scan_plan_to_acp_request(seal_document({**case, "case_id":"other"}), plan)
    with pytest.raises(ContractError, match="content_sha256"):
        scan_plan_to_acp_request({**case, "reaction_id":"RXN_TAMPERED"}, plan)
    bad_indices = copy.deepcopy(plan)
    bad_indices["candidates"][0]["coordinates"][0]["atom_indices"] = [0, 2]
    bad_indices = seal_document(bad_indices)
    with pytest.raises(ACPMappingError, match="atom maps and zero-based indices"):
        scan_plan_to_acp_request(case, bad_indices)
    unsupported_engine = copy.deepcopy(plan)
    unsupported_engine["candidates"][0]["method"]["engine"] = "nwchem"
    unsupported_engine = seal_document(unsupported_engine)
    with pytest.raises(ACPMappingError, match="backend is unsupported"):
        scan_plan_to_acp_request(case, unsupported_engine)
    coupled = simple_case()
    coupled["edits"].append({"kind":"formed", "atom_map_ids":[2, 12]})
    coupled = seal_document(coupled)
    refused = build_scan_plan_with_rejection(coupled)
    with pytest.raises(ACPMappingError, match="only a ready ScanPlan"):
        scan_plan_to_acp_request(coupled, refused)
    from pes2ts_core.contracts import validate_artifact_ref
    assert validate_artifact_ref({"artifact_id":"a", "relative_path":"../escape", "sha256":"0"*64, "media_type":"application/json"})


def test_g1_migration_uses_whitelist_and_requires_explicit_spin():
    raw = {
        "schema_version":"g1_v2_export_v1", "reaction_id":"RXN_TEST", "dataset_version":"test-v1",
        "mapping_provenance":"truth_assisted_p1", "maps":[1,2], "elements":["C","H"],
        "r_coordinates":[[0,0,0],[1.1,0,0]], "p_coordinates":[[0,0,0],[2.2,0,0]],
        "charge_total_reactants":0, "charge_total_products":0,
        "edits":[{"edit_kind":"broken", "pair":[1,2], "r_bond_order":1.0, "p_bond_order":None}],
        "hydrogen_partner_changes":[], "status_summary":{"endpoint_match":"pass", "orientation":"R_to_P"},
    }
    case = reaction_case_from_g1_export(raw, split="train", reactant_multiplicity=1, product_multiplicity=1)
    assert "endpoint_match" not in str(case)
    assert "orientation" not in str(case)
    assert case["source"]["mapping_provenance"] == "truth_assisted_p1"
    with pytest.raises(ContractError):
        reaction_case_from_g1_export({**raw, "truth_label":True}, split="train", reactant_multiplicity=1, product_multiplicity=1)


def test_unresolved_spin_case_is_valid_for_review_but_cannot_be_planned():
    case = simple_case()
    case["status"] = "needs_review"
    case["reactant"]["multiplicity"] = None
    case["review_reasons"] = ["reactant spin multiplicity requires chemistry review"]
    case = seal_document(case)

    assert not validate_document(case, production_input=True)
    with pytest.raises(ContractError, match="must be ready"):
        build_minimal_scan_plan(case)
    with pytest.raises(ContractError, match="must be ready"):
        build_scan_plan_with_rejection(case)
    invalid_spin = simple_case()
    invalid_spin["reactant"]["multiplicity"] = True
    invalid_spin = seal_document(invalid_spin)
    assert any("multiplicity: must be positive" in issue for issue in validate_document(invalid_spin))


def test_legacy_export_sanitizer_removes_nested_truth_fields():
    raw = {"schema_version":"g1_v2_export_v1", "reaction_id":"RXN_TEST", "dataset_version":"test-v1",
        "mapping_provenance":"truth_assisted_p1", "maps":[1], "elements":["H"],
        "r_coordinates":[[0,0,0]], "p_coordinates":[[0,0,0]], "edits":[],
        "hydrogen_partner_changes":[], "status_summary":{"endpoint_match":"pass", "orientation":"R_first", "audit_status":"clean"}}
    clean, removed = sanitize_export_document(raw)
    assert clean["status_summary"] == {"audit_status":"clean"}
    assert removed["endpoint_match"] == 1 and removed["orientation"] == 1
    with pytest.raises(ExportSanitizeError):
        sanitize_export_document({**raw, "unrecognized_field":True})


def test_synthetic_seven_object_bundle_round_trips():
    docs = synthetic_objects()
    assert set(docs) == {"ExperimentManifest", "ReactionCase", "ScanPlan", "ExecutionRecord",
                         "PathBundle", "SeedProposal", "ValidationResult", "RejectedPlan"}
    assert docs["PathBundle"]["frames"][4]["energies"] == {}
    assert docs["ExecutionRecord"]["attempts"][0]["status"] == "failed"
    assert docs["ScanPlan"]["candidates"][0]["direction"] == "P_to_R"
    assert docs["ValidationResult"]["status"] == "not_run"
    highest = rank_path_bundle(docs["PathBundle"], rule="highest_scan_energy")
    peak = rank_path_bundle(docs["PathBundle"], rule="internal_scan_peak")
    assert highest["selected_frames"][0]["frame_id"] == "frame-syn-000"
    assert peak["selected_frames"][0]["frame_id"] == "frame-syn-002"
    assert highest["selected_frames"][0]["frame_id"] != peak["selected_frames"][0]["frame_id"]
    for document in docs.values():
        loads_document(dumps_document(document))


def test_synthetic_bundle_persists_both_ranking_rules(tmp_path):
    import json

    root = write_synthetic_bundle(tmp_path / "synthetic-acp-demo")
    highest = json.loads((root / "SeedProposal.json").read_text(encoding="utf-8"))
    internal_peak = json.loads(
        (root / "SeedProposal.internal_scan_peak.json").read_text(encoding="utf-8")
    )
    assert highest["rule"] == "highest_scan_energy"
    assert internal_peak["rule"] == "internal_scan_peak"
    assert highest["selected_frames"][0]["frame_id"] != internal_peak["selected_frames"][0]["frame_id"]
    assert not validate_document(internal_peak)


def test_path_viewer_uses_existing_frames_and_rejects_mixed_energy_channels():
    from pathlib import Path

    path = synthetic_objects()["PathBundle"]
    output = Path(__file__).with_name(".viewer-test-output.html")
    tampered = copy.deepcopy(path)
    tampered["frames"][0]["energies"]["scan_electronic"]["value"] += 0.01
    with pytest.raises(ContractError, match="content_sha256"):
        rank_path_bundle(tampered)
    with pytest.raises(ContractError, match="content_sha256"):
        render_path_viewer(tampered, output)
    try:
        render_path_viewer(path, output, top_k=2)
        page = output.read_text(encoding="utf-8")
        assert "frame-syn-000" in page and "frame-syn-002" in page
        assert "internal_scan_peak" in page and "未做 TS 验证" in page
        assert "Plotly" not in page  # fully offline, no remote runtime dependency

        mixed = copy.deepcopy(path)
        mixed["frames"][2]["energies"]["scan_electronic"]["method_id"] = "other-method"
        mixed = seal_document(mixed)
        with pytest.raises(ContractError, match="mixed methods|one comparable scan energy"):
            render_path_viewer(mixed, output)
    finally:
        output.unlink(missing_ok=True)


def test_ranking_preserves_path_usability_and_refuses_unusable_paths():
    path = synthetic_objects()["PathBundle"]
    provisional = rank_path_bundle(path)
    assert provisional["status"] == "needs_review"
    assert provisional["path_status"] == "needs_review"
    assert "provisional" in provisional["review_note"]

    usable = seal_document({**path, "status":"usable"})
    assert rank_path_bundle(usable)["status"] == "accepted"

    unusable = seal_document({**path, "status":"unusable"})
    rejected = rank_path_bundle(unusable)
    assert rejected["status"] == "rejected"
    assert rejected["selected_frames"] == []
