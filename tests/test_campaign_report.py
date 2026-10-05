"""Feedback report builder tests (G2-AB2 WP-6, synthetic campaign)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pes2ts_core.generation.campaign import (
    CampaignBackends,
    build_campaign_manifest,
    run_campaign,
)
from pes2ts_core.generation.campaign_report import (
    CAPABILITY_STATUS,
    FEEDBACK_REPORT_SCHEMA,
    build_feedback_report,
)
from pes2ts_core.generation.planning.continuation import (
    ContinuationPolicy,
    project_geometry,
)
from pes2ts_core.generation.planning.origin_preparation import prepare_origin

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "p0_demo24" / "records"


def synthetic_policy() -> ContinuationPolicy:
    return ContinuationPolicy(max_frames=14, max_attempts=80,
                              max_seconds=600., cumulative_rmsd_limit=10.0)


def identity_origin(geometry, elements, bond_indices, charge, multiplicity, case_root):
    def evaluate_free(x, evaluation_id):
        return {"success": True, "failure_class": None,
                "coordinates": __import__("numpy").asarray(x, float).tolist(),
                "energy": -1.0, "duration_seconds": 0.01}
    return prepare_origin(geometry, elements, bond_indices, evaluate_free,
                          method_context={"method": "synthetic"})


class SyntheticCorrector:
    def __init__(self, plan):
        self.plan = plan
        self.coordinates = plan["drivers"] + plan.get("guards", [])

    def set_branch_reference(self, geometry, frame_id):
        pass

    def on_accept(self, result):
        pass

    def __call__(self, guess, targets, attempt_id):
        import numpy as np
        x, _ = project_geometry(np.asarray(guess, float), self.coordinates, targets,
                                self.plan["masses"], ContinuationPolicy())
        if x is None:
            return {"success": False, "converged": False, "failure_class": "OPT_FAILED",
                    "duration_seconds": 0.001, "n_gradient_evaluations": 1,
                    "acp": {"reused": False}}
        return {"success": True, "converged": True, "coordinates": x.tolist(),
                "energy": 1e-6*float(np.sum(x**2)), "duration_seconds": 0.002,
                "physical_gradient_status": "not_collected",
                "n_gradient_evaluations": 2,
                "acp": {"reused": False, "wall_seconds": 0.002}}


@pytest.fixture(scope="module")
def campaign_root(tmp_path_factory):
    directory = tmp_path_factory.mktemp("campaign")
    snapshots = [FIXTURES / "RXN_0000017762.json", FIXTURES / "RXN_0000079731.json"]
    manifest_path = directory / "manifest.json"
    build_campaign_manifest(
        snapshots, output_path=manifest_path, continuation_policy=synthetic_policy(),
        budget={"max_frames_per_path": 14, "max_attempts_per_path": 80,
                "max_wall_seconds_per_path": 600.0},
        reference_failures={"RXN_0000017762": {"last_accepted_lambda": 0.5}})
    root = directory / "root"
    run_campaign(manifest_path, root,
                 backends=CampaignBackends(prepare_origin=identity_origin,
                                           make_corrector=lambda p, r: SyntheticCorrector(p)))
    return manifest_path, root


def test_report_schema_and_nine_content_items(campaign_root):
    manifest_path, root = campaign_root
    report = build_feedback_report(root, manifest_path=manifest_path)
    assert report["schema_version"] == FEEDBACK_REPORT_SCHEMA
    for item in ("funnel", "typed_failure_histogram", "b1_comparison",
                 "branch_evidence", "candidates_and_validation", "cost_summary",
                 "minimal_j_g", "ab3_improvement_list", "capability_status"):
        assert item in report


def test_four_class_denominators_never_fold_rejections(campaign_root):
    _manifest_path, root = campaign_root
    report = build_feedback_report(root)
    denominators = report["funnel"]["four_class_denominators"]
    assert sum(denominators.values()) == report["n_cases"]
    for forbidden in ("success_rate", "n_success"):
        assert forbidden not in report
    assert report["claims"]["m0_through_m5_claimed"] is False


def test_b1_comparison_uses_reference_and_typed_origin(campaign_root):
    manifest_path, root = campaign_root
    report = build_feedback_report(root, manifest_path=manifest_path)
    per_case = {row["reaction_id"]: row for row in report["b1_comparison"]["per_case"]}
    reference_row = per_case["RXN_0000017762"]
    assert reference_row["reference_last_accepted_lambda"] == 0.5
    assert reference_row["current_last_accepted_lambda"] is not None
    assert isinstance(reference_row["b1_4_candidates_extracted_not_dropped"], bool)
    assert report["b1_comparison"]["b1_1_identity_conflicts_total"] == 0


def test_cost_summary_counts_gradients_and_cache_hits(campaign_root):
    _manifest_path, root = campaign_root
    report = build_feedback_report(root)
    costs = report["cost_summary"]
    assert costs["N_gradient_total"] > 0
    assert costs["most_expensive_case"]["reaction_id"].startswith("RXN_")
    assert "r5_rule" in costs


def test_minimal_j_g_is_explicit_no_data_without_validated(campaign_root):
    _manifest_path, root = campaign_root
    report = build_feedback_report(root)
    assert report["minimal_j_g"]["available"] is False
    assert "G3.0" in report["minimal_j_g"]["note"]


def test_capability_status_table_is_honest_no_smoke_claims():
    for capability, row in CAPABILITY_STATUS.items():
        assert row["status"] in {"unknown", "implemented", "fixture_passed",
                                 "smoke_passed"}, capability
        assert row["status"] != "smoke_passed" or capability == "__never__"
        assert row["evidence"]


def test_ab3_list_derives_from_histogram_with_evidence(campaign_root):
    _manifest_path, root = campaign_root
    report = build_feedback_report(root)
    for item in report["ab3_improvement_list"]:
        assert item["n_cases"] >= 1
        assert item["evidence_cases"]
        assert item["candidate_fix_direction"]


def test_report_refuses_non_campaign_roots(tmp_path):
    with pytest.raises(ValueError, match="NOT_A_CAMPAIGN_ROOT"):
        build_feedback_report(tmp_path)
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "campaign_identity.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="NO_TERMINAL_STATES"):
        build_feedback_report(empty)


def test_report_json_is_stably_serializable(campaign_root):
    _manifest_path, root = campaign_root
    report = build_feedback_report(root)
    text = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2,
                      sort_keys=True)
    assert "NaN" not in text and "Infinity" not in text
    parsed = json.loads(text)
    assert parsed["n_cases"] == report["n_cases"]
