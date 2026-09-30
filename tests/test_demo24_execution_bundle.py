"""Synthetic-only tests for freezing reviewed Demo24 input snapshots."""
from __future__ import annotations

import json
from pathlib import Path

from pes2ts_core.contracts import make_document, seal_document
from pes2ts_core.demo import synthetic_objects
from pes2ts_core.utils.jsonio import write_json
from scripts.build_demo24_execution_bundle import build_demo24_execution_bundle


def _review_pair(original, reviewed):
    review_id = f"review:{original['reaction_id']}:synthetic-test"
    reviewers = [{"reviewer": name, "decision":"accept", "dimensions": {
        "reaction_center":"confirmed", "atom_mapping":"confirmed", "charge_spin":"confirmed",
        "geometry_assembly":"confirmed"}, "scan_feasibility":"1D",
        "multiplicities":{"reactant":1,"product":1}, "note":"synthetic fixture only"}
        for name in ("reviewer-A", "reviewer-B")]
    return make_document("ReviewRecord", review_id, "accepted",
        dataset_version=reviewed["dataset_version"], reaction_id=reviewed["reaction_id"],
        case_id=reviewed["case_id"], split=reviewed["split"],
        case_sha256=original["content_sha256"], workbook_sha256="a"*64,
        reviewers=reviewers, adjudication={"decision":"accepted","adjudicator":"fixture",
            "reactant_multiplicity":1,"product_multiplicity":1,"resolution_note":""},
        decision_source="human", extensions={"pes2ts.review_output.v1":{
            "reviewed_case_sha256":reviewed["content_sha256"]}})


def test_demo24_execution_bundle_freezes_exact_16_8_reviewed_snapshots(tmp_path):
    original_template = synthetic_objects()["ReactionCase"]
    source_records = []
    reviewed_records = []
    cases_root = tmp_path / "reviewed"; cases_root.mkdir()
    for index in range(24):
        split = "train" if index < 16 else "valid"
        reaction_id = f"RXN_FIXTURE_{index:04d}"
        case_id = f"case:fixture-{index:04d}"
        original = seal_document({**original_template, "reaction_id":reaction_id,
            "case_id":case_id, "split":split, "status":"needs_review",
            "source":{"dataset":"synthetic-fixture","mapping_provenance":"endpoint_only"}})
        review_id = f"review:{reaction_id}:synthetic-test"
        reviewed = seal_document({**original, "status":"ready", "source":{
            **original["source"], "review_record_id":review_id,
            "reviewed_from_case_sha256":original["content_sha256"]}})
        review = _review_pair(original, reviewed)
        case_ref = f"{split}/{reaction_id}.json"
        review_ref = f"review_records/{reaction_id}.json"
        write_json(cases_root / case_ref, reviewed)
        write_json(cases_root / review_ref, review)
        source_records.append({"reaction_id":reaction_id,"case_id":case_id,"split":split,
            "relative_path":case_ref,"content_sha256":original["content_sha256"]})
        reviewed_records.append({"reaction_id":reaction_id,"case_id":case_id,"split":split,
            "case_status":"ready","case_sha256":reviewed["content_sha256"],
            "case_relative_path":case_ref,"review_object_id":review["object_id"],
            "review_sha256":review["content_sha256"],"review_relative_path":review_ref})

    source_manifest = tmp_path / "source_manifest.json"
    reviewed_manifest = tmp_path / "reviewed_manifest.json"
    write_json(source_manifest, {"schema_version":"demo24_reaction_case_manifest_v1",
        "n_cases":24,"records":source_records})
    write_json(reviewed_manifest, {"schema_version":"demo24_reviewed_case_manifest_v1",
        "status":"completed_human_review_import","n_cases":24,"workbook_sha256":"a"*64,
        "records":reviewed_records})
    labels = tmp_path / "labels.json"
    write_json(labels, [{"case_id":"case:fixture-0023","acceptable_frame_ids":["fixture-frame"],
        "label_source":"explicit test label","method":"test protocol","source_sha256":"b"*64}])

    receipt = build_demo24_execution_bundle(reviewed_cases_root=cases_root,
        reviewed_manifest_path=reviewed_manifest, source_manifest_path=source_manifest,
        output_root=tmp_path/"execution_input", budget={"max_attempts":2,
        "max_cpu_hours":0.5,"max_wall_seconds":120}, ranking_labels_path=labels)

    bundle_root = tmp_path / "execution_input"
    manifest = json.loads((bundle_root/"DemoExecutionManifest.json").read_text(encoding="utf-8"))
    assert len(manifest["cohort_cases"]) == len(manifest["runs"]) == 24
    assert manifest["budget"]["max_attempts"] == 2
    assert manifest["ranking_labels"] == "evaluation/ranking_labels.json"
    assert receipt["split_counts"] == {"train":16,"valid":8}
    assert receipt["execution_manifest_sha256"]
    assert len(json.loads((bundle_root/manifest["ranking_labels"]).read_text(encoding="utf-8"))) == 1

