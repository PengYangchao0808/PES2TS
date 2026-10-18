"""WP-0 audit-fix regression tests (G2-AB3).

New file only — does not edit the existing ``test_campaign_report.py`` /
``test_cost_capture.py`` suites.  Locks:

* campaign_report ``_branch_evidence`` never pollsutes current statistics
  with crash directories that carry a ``continuation_result.json`` but no
  verified terminal (P2-2);
* campaign_report ``_b1_checks`` treats a missing reference/current lambda as
  not_comparable (null + typed reason), never "progressed beyond"; both
  present -> strict ``>`` (P2-3);
* cost_capture probe records without an explicit evaluation count raise
  ``ValueError`` (never zero-filled); with ``n_gradient_evaluations`` (or the
  ``n_evaluations`` fallback) the ledger counts them once (P1).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pes2ts_core.generation.campaign import TERMINAL_STATE_SCHEMA
from pes2ts_core.generation.campaign_report import (
    FEEDBACK_REPORT_SCHEMA,
    _b1_checks,
    _branch_evidence,
    build_feedback_report,
)
from pes2ts_core.generation.planning.cost_capture import capture_continuation_costs


# ---------------------------------------------------------------------------
# campaign_report: branch-evidence isolation (WP-0 P2-2)
# ---------------------------------------------------------------------------

def continuation_doc(rid, *, n_extra_frames=1, drift=0.2, probes=1):
    frames = [
        {"convergence_mode": "converged",
         "quality": {"cumulative_drift_angstrom": drift}}
    ]
    for _ in range(n_extra_frames):
        frames.append(
            {"convergence_mode": "biased_only",
             "quality": {"cumulative_drift_angstrom": drift}}
        )
    attempts = [{"curvature_probe": True} for _ in range(probes)]
    return {"reaction_id": rid, "frames": frames, "attempts": attempts}


def test_branch_evidence_iterates_only_the_verified_case_root(tmp_path):
    good = tmp_path / "RXN_A"
    good.mkdir()
    (good / "continuation_result.json").write_text(
        json.dumps(continuation_doc("RXN_A")), encoding="utf-8"
    )
    crashed = tmp_path / "RXN_X.crashed_v1"
    crashed.mkdir()
    (crashed / "continuation_result.json").write_text(
        json.dumps(continuation_doc("RXN_X.crashed_v1", n_extra_frames=3,
                                    drift=9.0, probes=9)),
        encoding="utf-8",
    )
    # case_root carries ONLY the verified case — the crash dir is absent.
    stats = _branch_evidence({"RXN_A": good})
    assert stats["n_cases_with_continuation"] == 1
    assert stats["accepted_frames_beyond_origin_per_case"] == [1]
    assert stats["convergence_mode_histogram"] == {"biased_only": 1}
    assert stats["curvature_probes_total"] == 1
    assert stats["cumulative_drift_angstrom"]["max"] == pytest.approx(0.2)
    assert stats["biased_only_rate"] == 1.0


def test_feedback_report_never_pollutes_branch_evidence_with_crash_dirs(tmp_path):
    root = tmp_path / "campaign"
    case = root / "RXN_A"
    case.mkdir(parents=True)
    (root / "campaign_identity.json").write_text("{}", encoding="utf-8")
    terminal = {
        "schema_version": TERMINAL_STATE_SCHEMA,
        "reaction_id": "RXN_A",
        "terminal_class": "partial_prefix",
        "stage_reached": "continuation",
        "last_accepted_lambda": 0.6,
        "identity_conflict_count": 0,
        "origin_status": "prepared",
        "candidates": {"n": 0, "evidence_classes": [], "completed_interval": False},
        "validation": {"status": "blocked_on_human_review", "needs": []},
    }
    (case / "terminal_state.json").write_text(json.dumps(terminal), encoding="utf-8")
    (case / "continuation_result.json").write_text(
        json.dumps(continuation_doc("RXN_A")), encoding="utf-8"
    )
    crashed = root / "RXN_B.crashed_v1"
    crashed.mkdir()
    (crashed / "continuation_result.json").write_text(
        json.dumps(continuation_doc("RXN_B", n_extra_frames=3, drift=9.0, probes=9)),
        encoding="utf-8",
    )
    report = build_feedback_report(root)
    assert report["schema_version"] == FEEDBACK_REPORT_SCHEMA
    assert report["n_cases"] == 1
    branch = report["branch_evidence"]
    assert branch["n_cases_with_continuation"] == 1
    assert branch["accepted_frames_beyond_origin_per_case"] == [1]
    assert branch["curvature_probes_total"] == 1
    assert branch["cumulative_drift_angstrom"]["max"] == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# campaign_report: B1-2 not-comparable semantics (WP-0 P2-3)
# ---------------------------------------------------------------------------

def terminal_doc(rid, last_lambda):
    return {
        "reaction_id": rid,
        "last_accepted_lambda": last_lambda,
        "identity_conflict_count": 0,
        "origin_status": "prepared",
        "candidates": {"n": 0},
        "validation": {"status": "blocked_on_human_review"},
    }


def b1(terminals, references):
    case_root = {doc["reaction_id"]: Path("/nonexistent-case") for doc in terminals}
    return _b1_checks(terminals, case_root, references)


def test_b1_missing_reference_lambda_is_not_comparable():
    report = b1([terminal_doc("RXN_A", 0.6)], {})
    row = report["per_case"][0]
    assert row["b1_2_progressed_beyond_reference"] is None
    assert row["b1_2_not_comparable_reason"] == "reference_last_accepted_lambda_missing"
    assert row["reference_last_accepted_lambda"] is None
    assert row["current_last_accepted_lambda"] == 0.6
    assert report["b1_2_progressed_count"] == 0
    assert report["b1_2_not_comparable_count"] == 1
    assert report["b1_2_cases_with_reference"] == 0


def test_b1_missing_current_lambda_is_not_comparable():
    references = {"RXN_A": {"last_accepted_lambda": 0.4}}
    report = b1([terminal_doc("RXN_A", None)], references)
    row = report["per_case"][0]
    assert row["b1_2_progressed_beyond_reference"] is None
    assert row["b1_2_not_comparable_reason"] == "current_last_accepted_lambda_missing"
    assert row["reference_last_accepted_lambda"] == 0.4
    assert row["current_last_accepted_lambda"] is None
    assert report["b1_2_progressed_count"] == 0
    assert report["b1_2_not_comparable_count"] == 1
    assert report["b1_2_cases_with_reference"] == 1


def test_b1_progressed_requires_strictly_greater_current_lambda():
    references = {"RXN_A": {"last_accepted_lambda": 0.4}}
    report = b1([terminal_doc("RXN_A", 0.6)], references)
    assert report["per_case"][0]["b1_2_progressed_beyond_reference"] is True
    assert report["b1_2_progressed_count"] == 1
    assert report["b1_2_not_comparable_count"] == 0

    report = b1([terminal_doc("RXN_A", 0.4)], references)
    assert report["per_case"][0]["b1_2_progressed_beyond_reference"] is False

    report = b1([terminal_doc("RXN_A", 0.3)], references)
    assert report["per_case"][0]["b1_2_progressed_beyond_reference"] is False


# ---------------------------------------------------------------------------
# cost_capture: probe evaluation-count contract (WP-0 P1)
# ---------------------------------------------------------------------------

def test_probe_without_gradient_count_raises_never_zero_fills():
    result = {"attempts": [{"attempt_id": "attempt-0000",
                            "curvature_probe": {"duration_seconds": 0.4}}]}
    with pytest.raises(ValueError, match="lacks n_gradient_evaluations"):
        capture_continuation_costs(result, reaction_id="RXN_1")


def test_probe_without_either_count_key_raises():
    result = {"attempts": [{"attempt_id": "attempt-0000",
                            "curvature_probe": {"wall_seconds": 0.4}}]}
    with pytest.raises(ValueError, match="lacks n_gradient_evaluations"):
        capture_continuation_costs(result, reaction_id="RXN_1")


def test_probe_n_evaluations_fallback_is_counted():
    probe = {"n_evaluations": 3, "duration_seconds": 0.5}
    result = {"attempts": [{"attempt_id": "attempt-0000", "curvature_probe": probe}]}
    bundle = capture_continuation_costs(result, reaction_id="RXN_1")
    probe_rows = [
        row for row in bundle["ledgers"]
        if str(row["cost_source"]).startswith("curvature_probe")
    ]
    assert len(probe_rows) == 1
    assert probe_rows[0]["N_gradient"] == 3
    assert probe_rows[0]["cpu_seconds"] == 0.5


def test_probe_with_n_gradient_evaluations_and_duration_counts_once():
    probe = {"n_gradient_evaluations": 4, "duration_seconds": 2.5}
    result = {
        "attempts": [
            {
                "attempt_id": "attempt-0000",
                "backend_evidence": {
                    "n_gradient_evaluations": 1, "duration_seconds": 1.0,
                    "acp": {"reused": False},
                },
                "curvature_probe": probe,
            }
        ]
    }
    bundle = capture_continuation_costs(result, reaction_id="RXN_1")
    by_source = {row["cost_source"]: row for row in bundle["ledgers"]}
    assert by_source["curvature_probe"]["N_gradient"] == 4
    assert by_source["curvature_probe"]["cpu_seconds"] == 2.5
    assert by_source["acp_orca_gradient_attempt"]["N_gradient"] == 1
    # Probe gradients counted once, in the probe ledger, plus the parent.
    assert bundle["aggregate"]["N_gradient"] == 5
